from fairseq.models import register_model,register_model_architecture
from fairseq.models import (
    FairseqEncoderDecoderModel
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils

from fairseq.modules import(
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from transformers import BertTokenizer
from transformers import BertEmbeddings, BertAttention, BertIntermediate, BertOutput, BertPreTrainedModel, BertOnlyMLMHead
import base_architecture



DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model('KPE_EDITOR_transformer_with_adapter')
class KPEEDITORTransfomerWithAdapter(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder,berttokenizer,tgt_berttokenizer,args):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.berttokenizer = berttokenizer
        self.tgt_berttokenizer = tgt_berttokenizer
        self.max_source_position = args.max_source_position
        self.max_target_positions = args.max_target_position

    @staticmethod
    def add_args(parser):
        return super().add_args(parser)

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

        src_berttokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
        tgt_berttokenizer = BertTokenizer.from_pretrained(args.decoder_bert_model_name)
        assert src_berttokenizer.pad()==tgt_berttokenizer.pad()

        bertdecoder = BertAdapterDecoderFull



        return super().build_model(args, task)
    
    def build_encoder(cls,args)

class BertAdapterEDITORTransformerDecoder(BertPreTrainedModel):
    def __init__(self,config,args,dictionary,embed_tokens): 
        super(BertAdapterEDITORTransformerDecoder,self).__init__(config,args)

class BertAdapterDecoderLayer(nn.Module):
    def __init__(self, config, args, layer_num):
        super(BertAdapterDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.top_layer_adapter = getattr(args,'top_layer_adapter', -1)

        export = getattr(args, 'char_inputs', False)

        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            kdim=getattr(args, 'encoder_embed_dim', None),
            vdim=getattr(args, 'encoder_embed_dim', None),
            dropout=args.attention_dropout, encoder_decoder_attention=True
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn_fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.encoder_attn_fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.encoder_attn_final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = False

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        self_attn_mask=None,
        position_embedding=None,
        targets_padding=None,
        layer_num=-1,
    ):
        x = self.attention(x, self_attn_mask)

        intermediate_output = self.intermediate(x)
        x = self.output(intermediate_output, x)

        x = x.transpose(0, 1)
        
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_final_layer_norm, x, before=True)
        x = self.activation_fn(self.encoder_attn_fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.encoder_attn_fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        layer_output = self.maybe_layer_norm(self.encoder_attn_final_layer_norm, x, after=True)
        layer_output = layer_output.transpose(0,1)
        return layer_output

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def without_self_mask(self, tensor):
        dim = tensor.size(0)
        eye_matrix = torch.eye(dim)
        eye_matrix[eye_matrix == 1.0] = float('-inf')
        return eye_matrix.cuda()

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m