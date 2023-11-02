# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from email.policy import default
import os

import torch

from fairseq.data import LanguagePairDataset, MultiSourceTranslationDataset, noising
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq import utils,tokenizer

from fairseq.data import BertWordpieceDictionary,Dictionary


@register_task('translation_lev')
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """


    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_delete_shuffle', 'random_mask', 'no_noise', 'full_mask'])
        
        parser.add_argument('--random-seed', default=1, type=int)
        
        parser.add_argument('--cached_features_dir',help='the path of cached features')
        parser.add_argument('--tokenizer_dir',help='the path of tokenizer')
        # parser.add_argument('--model_class',default='bert2joint')
        # parser.add_argument('--dataset_class',default='kp20k')
        # parser.add_argument('--pretrain_model',default='roberta')
        parser.add_argument('--encoder_adapter_dimention',default=2048)
        parser.add_argument('--decoder_input',default='target')
        parser.add_argument('--kpe',default=False,action ='store_true')

        parser.add_argument('--return_num',default = 1.0,type=float,help= 'keywords_nums that need to generate in stage 1')
        parser.add_argument('--return_mode',default = 'shers',help= 'keywords_mode that need to generate in stage 1')
        parser.add_argument('--constraint_file',default = None ,help= 'constraint_file_path')

    def __init__(self, args,src_dict, tgt_dict):
        super().__init__(args,src_dict, tgt_dict)
        
        self.decoder_input = args.decoder_input
        self.need_cached_examples = False
        self.args = args

        if getattr(args,'kpe',False):
            self.need_cached_examples = True
            
            if getattr(args,'return_mode',False) is not None:
                self.return_mode = getattr(args,'return_mode','shers')
            if getattr(args,'return_num',False) is not None:
                self.return_num = getattr(args,'return_num',1.0)
            if getattr(args,'constraint_file',None) is not None:
                constraint_path = getattr(args,'constraint_file')
                self.constraint_file = os.path.join(constraint_path,'constraint.json')
            

    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = BertWordpieceDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def load_dictionary(cls, filename):
        return BertWordpieceDictionary.load(filename)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict,
            tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            cached_examples_dir =self.args.cached_features_dir,
            tokenizer_dir=self.args.tokenizer_dir,
            prepend_bos=True,
            need_cached_examples=self.need_cached_examples
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_shuffle(target_tokens, p, max_shuffle_distance):
            word_shuffle = noising.WordShuffle(self.tgt_dict)
            target_mask = target_tokens.eq(self.tgt_dict.pad())
            target_length = target_mask.size(1) - target_mask.long().sum(1)
            prev_target_tokens, _ = word_shuffle.noising(
                target_tokens.t().cpu(), target_length.cpu(), max_shuffle_distance)
            prev_target_tokens = prev_target_tokens.to(target_tokens.device).t()
            masks = (target_tokens.clone().sum(dim=1, keepdim=True).float()
                .uniform_(0, 1) < p)
            prev_target_tokens = masks * prev_target_tokens + (~masks) * target_tokens
            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete_shuffle':
            return _random_shuffle(_random_delete(target_tokens), 0.5, 3)
            #return _random_shuffle(_random_delete(target_tokens), 0.8, 5)
        elif self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, args):
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        if self.need_cached_examples == True:
            return IterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
                del_reward=getattr(args, 'iter_decode_deletion_reward', 0.0),
                max_iter=getattr(args, 'iter_decode_max_iter', 10),
                beam_size=getattr(args, 'iter_decode_with_beam', 1),
                reranking=getattr(args, 'iter_decode_with_external_reranker', False),
                decoding_format=getattr(args, 'decoding_format', None),
                adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
                retain_history=getattr(args, 'retain_iter_history', False),
                constrained_decoding=getattr(args, 'constrained_decoding', False),
                hard_constrained_decoding=getattr(args, 'hard_constrained_decoding', False),
                random_seed=getattr(args, 'random_seed', 1),
                return_mode =self.return_mode,
                return_num = self.return_num,
                constraint_file= self.constraint_file,
                )
        else:
            return IterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
                del_reward=getattr(args, 'iter_decode_deletion_reward', 0.0),
                max_iter=getattr(args, 'iter_decode_max_iter', 10),
                beam_size=getattr(args, 'iter_decode_with_beam', 1),
                reranking=getattr(args, 'iter_decode_with_external_reranker', False),
                decoding_format=getattr(args, 'decoding_format', None),
                adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
                retain_history=getattr(args, 'retain_iter_history', False),
                constrained_decoding=getattr(args, 'constrained_decoding', False),
                hard_constrained_decoding=getattr(args, 'hard_constrained_decoding', False),
                random_seed=getattr(args, 'random_seed', 1),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, tgt_tokens=None, tgt_lengths=None, num_source_inputs=1):
        if num_source_inputs == 1:
            return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, tgt=tgt_tokens, tgt_sizes=tgt_lengths, append_bos=True,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            )
        else:
            return MultiSourceTranslationDataset(src_tokens, src_lengths, self.source_dictionary, tgt=tgt_tokens, tgt_sizes=tgt_lengths, append_bos=True, 
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            )

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()
        if self.decoder_input == 'target':
            sample['prev_target'] = self.inject_noise(sample['target'])
            # if getattr(self.args,'kpe',False):
            #         print('WARNING: kpe usually match with keywords instead of target.')
        else:
            sample['prev_target'] = self.inject_noise(sample['cached_features'][8])
            #sample['prev_target'] = sample['cached_features'][8]
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if self.decoder_input == 'target':
                sample['prev_target'] = self.inject_noise(sample['target'])
                # sample['prev_target'] = self.inject_noise(sample['target'],sample['cached_features'][8])
                # if getattr(self.args,'kpe',False) and :
                #     print('WARNING: kpe usually match with keywords instead of target.')
            else :
                sample['prev_target'] = self.inject_noise(sample['cached_features'][8])
                #sample['prev_target'] = sample['cached_features'][8]
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
    
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if getattr(self.args,'kpe',False):
                return generator.generate_kpe(models, sample, prefix_tokens=prefix_tokens)
            else:
                return generator.generate(models, sample, prefix_tokens=prefix_tokens)
