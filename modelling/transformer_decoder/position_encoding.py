import math

import tensorflow as tf
from tensorflow.keras import layers

class PositionEmbeddingSine(layers.Layer):
    """
    Similar to Attention is all you need paper, generalized
    for images.
    """
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None,
                 **kwargs):
        super(PositionEmbeddingSine, self).__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, x, mask=None):
        if mask is None:
            mask = tf.zeros(shape=(x.shape[0], x.shape[2], x.shape[3]), dtype=tf.bool)
        not_mask = ~mask
        x_embed = tf.cast(tf.cumsum(not_mask, axis=2), dtype=tf.float32)
        y_embed = tf.cast(tf.cumsum(not_mask, axis=1), dtype=tf.float32)

        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

        dim_t = tf.range(start=0, limit=self.num_pos_feats, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = tf.stack(
            (tf.sin(pos_x[:, :, :, 0::2]), tf.cos(pos_x[:, :, :, 1::2])), axis=4
        )
        pos_x = tf.squeeze(pos_x, axis=3)
        pos_y = tf.stack(
            (tf.sin(pos_y[:, :, :, 0::2]), tf.cos(pos_y[:, :, :, 1::2])), axis=4
        )
        pos_y = tf.squeeze(pos_y, axis=3)
        pos = tf.transpose(tf.concat((pos_y, pos_x), axis=3), perm=(0, 3, 1, 2))
        return pos

    # TODO check __repr__ method
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

target = tf.keras.Input(shape=[100, 100, 3])
# print(target)
print(PositionEmbeddingSine())