import logging
## TODO check weight init
from typing import Optional

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers  # roughly equivalent to torch functional

from .position_encoding import PositionEmbeddingSine
# TODO torch implementation also includes Registry in detectron2.utils, which is
# TODO for mapping string identrifiers to Callables


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return layers.ReLU()
    # TODO is is necessary to impl. GELU/GLU as a layer?
    # if activation == "gelu":
    #     return layers.ge
    # if activation == "glu":
    #     return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(layers.Layer):  # torch imple. uses nn.Module
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = layers.MultiHeadAttention(key_dim=d_model,
                                                   num_heads=nhead,
                                                   dropout=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-5)  # modify eps to torch default
        # self.dropout = tf.nn.Dropout(dropout)
        self.dropout = dropout
        # torch impl. used a string with a get func: _get_activation_fn(activation)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # torch: call reset_parameters method to initialize weights and bias to xavier

    @staticmethod
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # need to check this to ensure dimension works

    def forward_post(self,
                     tgt,
                     tgt_mask: Optional[Tensor] = None,
                     # TODO is key_padding_mask needed?
                     # tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)  # is position embed is None pass tgt along
        # torch: param order: q,k,v but in TF order is q, v, k
        # torch implementation only needs to get the attn_output returned
        tgt2 = self.self_attn(q, tgt, k, attention_mask=tgt_mask)
        # TODO consider changing nn.Dropout for nn.experimental.stateless_dropout
        tgt = tgt + tf.nn.dropout(tgt2, self.dropout)

        return tgt

    def forward_pre(self,
                    tgt,
                    tgt_mask: Optional[Tensor] = None,
                    # TODO is key_padding_mask needed?
                    # tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt, query_pos)
        print(k.shape, tgt.shape)
        # torch: param order: q, k, v but in TF order is q, v, k
        # torch implementation only needs to get the attn_output returned
        # tgt2 = self.self_attn(q, tgt, k, attention_mask=tgt_mask)
        tgt2 = self.self_attn(q, tgt, k, attention_mask=tgt_mask)
        # TODO consider changing nn.Dropout for nn.experimental.stateless_dropout
        tgt = tgt + tf.nn.dropout(tgt2, self.dropout)

        return tgt

    def call(self,
             tgt,
             tgt_mask: Optional[Tensor] = None,
             # TODO is key_padding_mask needed?
             # tgt_key_padding_mask: Optional[Tensor] = None,
             query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, query_pos)

        return self.forward_post(tgt, tgt_mask, query_pos)


class CrossAttentionLayer(layers.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False,
                 **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.multihead_attn = layers.MultiHeadAttention(num_heads=nhead,
                                                        key_dim=d_model)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.dropout = dropout
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # torch: call reset_parameters method to initialize weights and bias to xavier

    @staticmethod
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # need to check this to ensure dimension works

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        query = self.with_pos_embed(tgt, query_pos)
        key = self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(query=query, key=key, value=memory,
                                   attention_mask=memory_mask)
        # TODO consider changing nn.Dropout for nn.experimental.stateless_dropout
        tgt = tgt + tf.nn.dropout(tgt2, self.dropout)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        query = self.with_pos_embed(tgt, query_pos)
        key = self.with_pos_embed(memory, pos)
        tgt2 = self.multihead_attn(query=query, key=key, value=memory,
                                   attention_mask=memory_mask)
        # TODO consider changing nn.Dropout for nn.experimental.stateless_dropout
        tgt = tgt + tf.nn.dropout(tgt2, self.dropout)

        return tgt

    def call(self,
             tgt,
             memory,
             memory_mask: Optional[Tensor] = None,
             memory_key_padding_mask: Optional[Tensor] = None,
             pos: Optional[Tensor] = None,
             query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(layers.Layer):
    def __init__(self,
                 d_model,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False,
                 **kwargs):
        super(FFNLayer, self).__init__(**kwargs)
        # Feedforward model, ie. basically MLP; torch linear takes (input_shape, output_shape)
        # but tf only needs output shape
        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = dropout
        self.linear2 = layers.Dense(d_model)

        # set normalization and activation
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # TODO init weights to xavier unifrom

    @staticmethod
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # need to check this to ensure dimension works

    def forward_post(self, tgt):
        x = self.linear1(tgt)
        x = self.activation(x)
        x = tf.nn.dropout(x, self.dropout)
        tgt2 = self.linear2(x)
        # equivalent to a residual connection between layers
        tgt = tgt + tf.nn.dropout(tgt2, self.dropout)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        x = self.norm(tgt)
        x = self.linear1(x)
        x = self.activation(x)
        x = tf.nn.dropout(x, self.dropout)
        tgt2 = self.linear2(x)
        tgt = tgt + tf.nn.dropout(tgt2, self.dropout)
        return tgt

    def call(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(layers.Layer):
    # simple MLP with logit output
    def __init__(self,
                 # input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.h = [hidden_dim] * (num_layers - 1)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.dense_layers = [layers.Dense(d) for d in (self.h + self.output_dim)]
        self.built = True

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = layers.ReLU()(x) if i < self.num_layers - 1 else layer(x)
        return x


# consider implementing a string identifier for this
class MultiScaleMaskedTransformerDecoder(layers.Layer):
    __version__ = 1  # 1 for TF impl. but corresponds to torch ver 2

    # TODO impl. load from state dict
    def __init__(self,
                 in_channels,
                 mask_classification=True,
                 *,
                 num_classes: int,
                 hidden_dim: int,
                 num_queries: int,
                 nheads: int,
                 dim_feedforward: int,
                 dec_layers: int,
                 pre_norm: bool,
                 mask_dim: int,
                 enforce_input_project: bool,
                 **kwargs):
        super(MultiScaleMaskedTransformerDecoder, self).__init__(**kwargs)

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer Encoder; put this into build() call or not?
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = []
        self.transformer_cross_attention_layers = []
        self.transformer_ffn_layers = []

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = layers.LayerNormalization(axis=hidden_dim, epsilon=1e-5)
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = layers.Embedding(num_queries, hidden_dim)
        # learnable query p_e
        self.query_embed = layers.Embedding(num_queries, hidden_dim)

        # level embedding (use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = layers.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = []
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(
                    layers.Conv2D(hidden_dim, kernel_size=1)  # TODO check if padding is the same default to TF
                )
            else:
                self.input_proj.append(tf.keras.Sequential())

        # output FFNS
        if self.mask_classification:
            self.class_embed = layers.Dense(num_classes + 1)
        self.mask_embed = MLP(hidden_dim, mask_dim, 3)

    # TODO implement a from_config method for reading yaml configs into model

    def call(self,
             x,
             mask_features,
             mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # does this affect anything???
        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2])
            pos.append(tf.squeeze(self.pe_layer(x[i], None), axis=2))
            src.append(tf.squeeze(self.input_proj[i](x[i]), axis=2)
                       + self.level_embed.weights[i][None, :, None])

            # flatten NxCxHxW
            pos[-1] = tf.transpose(pos[-1], perm=(2, 0, 1))
            src[-1] = tf.transpose(src[-1], perm=(2, 0, 1))

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = tf.expand_dims(self.query_embed.weights, axis=1)
        query_embed = tf.repeat(query_embed, repeat=(1, bs, 1))  # TODO check TF equivalence
        output = tf.expand_dims(self.query_feat.weights, axis=1)
        output = tf.repeat(output, repeat=(1, bs, 1))

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                               mask_features,
                                                                               attn_mask_target=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            sum_ = tf.reduce_sum(attn_mask, axis=-1)
            attn_mask[tf.where(sum_ == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            attn_mask_target_size = size_list[(i + 1) % self.num_feature_levels]
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                   mask_features,
                                                                                   attn_mask_target_size)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def call_prediction_heads(self,
                              output,
                              mask_features,
                              attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = tf.transpose(decoder_output, perm=(0, 1))
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = tf.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # TODO check

        # Note: prediction is of higher resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        ## TODO implement interpolate
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    # check if dictionary containing tensor and list is supported in tensorflow
    # if not find a workaround
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


target = tf.keras.Input(shape=[8, 16])
source = tf.keras.Input(shape=[4, 16])
print(MLP(32, [3,] , 2)(source))
# CrossAttentionLayer(128, 12, dropout=.1, activation="relu", normalize_before=True)(source, target)
# print(layers.MultiHeadAttention(128, 12)(target, source))