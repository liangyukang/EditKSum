# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import cache
from fairseq.logging.metrics import state_dict
import numpy as np

from time import perf_counter
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options,utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoderLayer,
    base_architecture
)

from fairseq.models.fairseq_encoder import EncoderOut
import json

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

from fairseq.models.nat import (
    FairseqNATModel,
    FairseqNATDecoder,
    FairseqNATEncoder,
    ensemble_decoder
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
#from transformers import RobertaConfig
from datas.Roberta2Joint import RobertaForChunkTFRanking,RobertaForCnnGramKernelRanking

from datas.Rank2Phrase import rank2phrase,rank2phrase_keep_order
#from datas.utils import pred_arranger

from .levenshtein_utils import (
    _skip, _skip_encoder_out, _fill,
    _get_advanced_ins_targets, _get_advanced_reposition_targets,
    _apply_ins_masks, _apply_ins_words, _apply_reposition_words,
)

@register_model("kpe_editor_transformer")
class KPEEDITORTransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action='store_true',
            help='instead of argmax, use sampling to predict the tokens'
        )
        parser.add_argument('--cache_dir',help='pretrained model dir for encoder')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = EDITORTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls,  args , src_dict):
        model_config=RobertaConfig.from_pretrained(args.cache_dir,max_position_embeddings=args.max_source_positions)
        encoder = RobertaForChunkTFRanking.from_pretrained(args.cache_dir,config= model_config,src_dict=src_dict)

        return encoder  

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args,src_dict)
        decoder = cls.build_decoder(args,tgt_dict,decoder_embed_tokens)
        return KPEEDITORTransformerModel(args,encoder,decoder)

    def forward(
        self, src_tokens, src_lengths,cached_features, prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."
        
        # start_time = perf_counter()

        # input_ids = src_tokens
        # attention_mask = src_tokens.eq(self.dict.pad())

        input_ids = cached_features[0]

        # assert src_tokens == input_ids

        attention_mask = cached_features[1] 
        valid_ids = cached_features[2]

        

        active_mask = cached_features[3]
        valid_output = cached_features[4]
        labels = cached_features[5]
        chunk_labels = cached_features[6]
        chunk_mask=cached_features[7]

        #mention_list=cached_features[9]

        # encoding
        encoder_output,kpe_loss = self.encoder(input_ids,attention_mask,
                                    valid_ids,
                                    active_mask,
                                    valid_output,
                                    labels,
                                    chunk_labels,
                                    chunk_mask,
                                    )

        encoder_out = EncoderOut(encoder_out=encoder_output.transpose(0,1),
                    encoder_padding_mask=1-attention_mask,
                    encoder_embedding=None,
                    encoder_states=None,
                    src_tokens=None,
                    src_lengths=None)
        # end_time = perf_counter()
        # print("encoder time:%f" % (end_time-start_time))


        if random.random() < 0.5:  # insertion -> reposition & deletion
            # generate training labels for insertion
            masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
                prev_output_tokens, tgt_tokens, self.pad, self.unk
            )
            mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
            mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

            mask_ins_out, _ = self.decoder.forward_mask_ins(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out
            )
            word_ins_out, _ = self.decoder.forward_word_ins(
                normalize=False,
                prev_output_tokens=masked_tgt_tokens,
                encoder_out=encoder_out
            )

            # make online prediction
            if self.decoder.sampling_for_deletion:
                word_predictions = torch.multinomial(
                    F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                        word_ins_out.size(0), -1)
            else:
                word_predictions = F.log_softmax(word_ins_out, dim=-1).max(-1)[1]

            word_predictions.masked_scatter_(
                ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
            )

            # generate training labels for reposition
            word_reposition_targets = _get_advanced_reposition_targets(word_predictions, tgt_tokens, self.pad)
            word_reposition_out, _ = self.decoder.forward_word_reposition(
                normalize=False,
                prev_output_tokens=word_predictions,
                encoder_out=encoder_out)
            word_reposition_masks = word_predictions.ne(self.pad)
        else:  # reposition & deletion -> insertion
            # generate training labels for deletion and substitution
            word_reposition_targets = _get_advanced_reposition_targets(
                prev_output_tokens, tgt_tokens, self.pad
            )
            word_reposition_out, _ = self.decoder.forward_word_reposition(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out)
            word_reposition_masks = prev_output_tokens.ne(self.pad)

            # make online prediction
            word_predictions = F.log_softmax(word_reposition_out, dim=-1).max(-1)[1]

            word_predictions, _, _, _ = _apply_reposition_words(
                prev_output_tokens,
                None,
                None,
                None,
                word_predictions,
                self.pad,
                self.bos,
                self.eos,
            )

            # generate training labels for insertion
            masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
                word_predictions, tgt_tokens, self.pad, self.unk
            )
            mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
            mask_ins_masks = word_predictions[:, 1:].ne(self.pad)

            mask_ins_out, _ = self.decoder.forward_mask_ins(
                normalize=False,
                prev_output_tokens=word_predictions,
                encoder_out=encoder_out
            )
            word_ins_out, _ = self.decoder.forward_word_ins(
                normalize=False,
                prev_output_tokens=masked_tgt_tokens,
                encoder_out=encoder_out
            )

        return {
            "mask_ins": {
                "out": mask_ins_out, "tgt": mask_ins_targets,
                "mask": mask_ins_masks, "ls": 0.01,
            },
            "word_ins_ml": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": masked_tgt_masks, "ls": self.args.label_smoothing,
                "nll_loss": True, "factor": 1.0,
            },
            "word_reposition": {
                "out": word_reposition_out, "tgt": word_reposition_targets,
                "mask": word_reposition_masks,
            },
            "kpe":{"loss":kpe_loss,}
        }

    def forward_decoder(
        self, decoder_out, encoder_out, hard_constrained_decoding=False, eos_penalty=0.0, del_reward=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (1-encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # reposition words
        # do not apply if it is <s> </s>
        can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_reposition_word.sum() != 0:
            word_reposition_score, word_reposition_attn = self.decoder.forward_word_reposition(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_reposition_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_reposition_word)
            )

            if hard_constrained_decoding:
                no_del_mask = output_marks[can_reposition_word].ne(0)
                word_del_score = word_reposition_score[:, :, 0]
                word_del_score.masked_fill_(no_del_mask, -float('Inf'))
                word_reposition_score = torch.cat([word_del_score.unsqueeze(2), word_reposition_score[:, :, 1:]], 2)

            word_reposition_score[:, :, 0] = word_reposition_score[:, :, 0] + del_reward
            word_reposition_pred = word_reposition_score.max(-1)[1]
            num_deletion = word_reposition_pred.eq(0).sum().item() - word_reposition_pred.size(0)
            total_deletion_ops += num_deletion
            total_reposition_ops += word_reposition_pred.ne(
                torch.arange(word_reposition_pred.size(1), device=word_reposition_pred.device)
                .unsqueeze(0)).sum().item() - num_deletion

            _tokens, _marks, _scores, _attn = _apply_reposition_words(
                output_tokens[can_reposition_word],
                output_marks[can_reposition_word],
                output_scores[can_reposition_word],
                word_reposition_attn,
                word_reposition_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_reposition_word, _tokens, self.pad)
            output_marks = _fill(output_marks, can_reposition_word, _marks, 0)
            output_scores = _fill(output_scores, can_reposition_word, _scores, 0)
            attn = _fill(attn, can_reposition_word, _attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask)
            )
            if eos_penalty > 0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(
                mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            )

            if hard_constrained_decoding:
                no_ins_mask = output_marks[can_ins_mask][:, :-1].eq(1)
                mask_ins_pred.masked_fill_(no_ins_mask, 0)

            total_insertion_ops += mask_ins_pred.sum().item()

            _tokens, _marks, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_marks[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )

            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_marks = _fill(output_marks, can_ins_mask, _marks, 0)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word)
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_marks=output_marks,
            output_scores=output_scores,
            attn=attn,
            num_ops=(total_reposition_ops, total_deletion_ops, total_insertion_ops),
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, initial_tokens, initial_marks):
        # initial_tokens = initial_tokens.tolist() if initial_tokens is not None else None
        # initial_marks = initial_marks.tolist() if initial_marks is not None else None

        initial_tokens = initial_tokens if initial_tokens is not None else None
        initial_marks = initial_marks if initial_marks is not None else None
        max_num_constraints = max([len(seq) for seq in initial_tokens]) if initial_tokens else 0
        initial_output_marks = src_tokens.new_zeros(src_tokens.size(0), max_num_constraints + 2)
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), max_num_constraints + 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        if initial_tokens:
            for i, seq in enumerate(initial_tokens):
                for j, tok in enumerate(seq):
                    initial_output_tokens[i, j + 1] = tok
                initial_output_tokens[i, len(seq) + 1] = self.eos
                for j in range(len(seq) + 2, max_num_constraints + 2):
                    initial_output_tokens[i, j] = self.pad

        if initial_marks:
            for i, seq in enumerate(initial_marks):
                for j, mark in enumerate(seq):
                    initial_output_marks[i, j + 1] = mark

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_marks=initial_output_marks,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            num_ops=(0, 0, 0),
            history=None
        )

    def forward_encoder(self,src_tokens,src_lengths,cached_examples):
        inputs = {
            "input_ids":cached_examples[0],
            "attention_mask":cached_examples[1],
            "valid_ids":cached_examples[2],
            "active_mask":cached_examples[3],
            "valid_output":cached_examples[4],
        }
        indices,numbers = cached_examples[-1],cached_examples[-2]
        phrase_list = cached_examples[-3]

        encoder_output,logits = self.encoder(**inputs)

        assert (logits.shape[0] == len(numbers)) and (logits.shape[1] == max(numbers))
        logits = logits.data.cpu().tolist()

        logit_lists = []
        for batch_id, num in enumerate(numbers):
            logit_lists.append(logits[batch_id][:num])
        
        params = {
            "examples": phrase_list,
            "logit_lists": logit_lists,
            "indices": indices,
            "return_num": 40,
            "stem_flag": False,
        }

        #constraints,scores = rank2phrase(**params)
        constraints,scores =rank2phrase_keep_order(**params)
        
        attention_mask =cached_examples[1]

        encoder_out = EncoderOut(encoder_out=encoder_output.transpose(0,1),
            encoder_padding_mask=1-attention_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None)

        return encoder_out,constraints
    







# from torch.nn import MarginRankingLoss, CrossEntropyLoss
# class KPETransformerEncoder(RobertaForCnnGramKernelRanking,FairseqNATEncoder):
#     def __init__(self,args,dictionary,embed_tokens):
        
#         RobertaForCnnGramKernelRanking.__init__(self,args)
#         FairseqNATEncoder.__init__(self,args,dictionary,embed_tokens)
#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         valid_ids,
#         active_mask,
#         valid_output,
#         labels=None,
#         chunk_labels=None,
#         chunk_mask=None,
#     ):
#         """
#         active_mask : mention_mask for ngrams = torch.LongTensor([[1,2,1,3,4,5,4], [1,2,3,0,4,4,0]])
#         laebls : for ngrams labels = torch.LongTensor([[1,-1,-1,1,-1], [1,-1,-1,1,0]])
#         """
#         # --------------------------------------------------------------------------------
#         # Bert Embedding Outputs
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

#         sequence_output = outputs[0]

#         # --------------------------------------------------------------------------------
#         # Valid Outputs : get first token vector
#         batch_size = sequence_output.size(0)
#         for i in range(batch_size):
#             valid_num = sum(valid_ids[i]).item()

#             vectors = sequence_output[i][valid_ids[i] == 1]
#             valid_output[i, :valid_num].copy_(vectors)

#         # --------------------------------------------------------------------------------
#         # Dropout
#         sequence_output = self.dropout(valid_output)

#         # --------------------------------------------------------------------------------
#         # CNN Outputs
#         cnn_outputs = self.cnn2gram(
#             sequence_output
#         )  # shape = (batch_size, max_gram_num, 512)

#         # --------------------------------------------------------------------------------
#         # Classifier 512 to 1
#         classifier_scores = self.classifier(
#             cnn_outputs
#         )  # shape = (batch_size, max_gram_num, 1)
#         classifier_scores = classifier_scores.squeeze(
#             -1
#         )  # shape = (batch_size, max_gram_num)

#         classifier_scores = classifier_scores.unsqueeze(1).expand(
#             active_mask.size()
#         )  # shape = (batch_size, max_diff_ngram_num, max_gram_num)
#         classifier_scores = classifier_scores.masked_fill(
#             mask=active_mask, value=-float("inf")
#         )

#         # --------------------------------------------------------------------------------
#         # Merge TF : # shape = (batch_size * max_diff_ngram_num * max_gram_num) to (batch_size * max_diff_ngram_num)
#         total_scores, indices = torch.max(classifier_scores, dim=-1)

#         # --------------------------------------------------------------------------------
#         # --------------------------------------------------------------------------------
#         # Total Loss Compute
#         if labels is not None and chunk_labels is not None:
#             # *************************************************************************************
#             # *************************************************************************************
#             # [1] Chunk Loss
#             Chunk_Loss_Fct = CrossEntropyLoss(reduction="mean")

#             active_chunk_loss = chunk_mask.view(-1) != -1
#             chunk_logits = self.chunk_classifier(
#                 cnn_outputs
#             )  # shape = (batch_size * num_gram, 2)
#             active_chunk_logits = chunk_logits.view(-1, self.num_labels)[
#                 active_chunk_loss
#             ]

#             active_chunk_label_loss = chunk_labels.view(-1) != -1
#             active_chunk_labels = chunk_labels.view(-1)[active_chunk_label_loss]

#             chunk_loss = Chunk_Loss_Fct(active_chunk_logits, active_chunk_labels)

#             # *************************************************************************************
#             # *************************************************************************************
#             # [2] Rank Loss
#             Rank_Loss_Fct = MarginRankingLoss(margin=1, reduction="mean")

#             device = torch.device("cuda", total_scores.get_device())
#             flag = torch.FloatTensor([1]).to(device)

#             rank_losses = []
#             for i in range(batch_size):

#                 score = total_scores[i]
#                 label = labels[i]

#                 true_score = score[label == 1]
#                 neg_score = score[label == -1]
#                 rank_losses.append(
#                     Rank_Loss_Fct(
#                         true_score.unsqueeze(-1), neg_score.unsqueeze(0), flag
#                     )
#                 )

#             rank_loss = torch.mean(torch.stack(rank_losses))
#             # *************************************************************************************
#             # *************************************************************************************
#             # [3] Total Loss
#             tot_loss = rank_loss + chunk_loss
#             return tot_loss

#         else:
#             return total_scores  # shape = (batch_size * max_differ_gram_num)

class EDITORTransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList([
                                TransformerDecoderLayer(args, no_encoder_attn)
                                for _ in range(self.early_exit[1])
                            ])
        self.layers_reposition = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_reposition = nn.ModuleList([
                                TransformerDecoderLayer(args, no_encoder_attn)
                                for _ in range(self.early_exit[0])
                            ])

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(args, "no_share_discriminator", False), "must set saperate discriminator"
            self.layers_msk = self.layers_reposition

    def extract_features(
        self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the EDITORTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        
        for _, layer in enumerate(layers[: early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[1], layers=self.layers_msk, **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[2], layers=self.layers, **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_reposition(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[2], layers=self.layers_reposition, **unused
        )
        prev_output_embed = self.embed_tokens(prev_output_tokens)
        # B x T x T
        decoder_out = torch.bmm(features, prev_output_embed.transpose(1, 2))
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

@register_model_architecture("kpe_editor_transformer", "kpe_editor_transformer")
def kpe_editor_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
    args.cached_dir = getattr(args, "cache_dir", "/data/yukangliang/数据集/cnn_dm_KPE_Interaction/pretrain_model/roberta-base")

