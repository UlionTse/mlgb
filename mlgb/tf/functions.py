# coding=utf-8
# author=uliontse

"""
Copyright 2020 UlionTse

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from mlgb.tf.configs import (
    numpy,
    tf,
    Embedding,
    Activation,
    BatchNormalization,
    L1L2,
    Flatten,
)
from mlgb.error import MLGBError


__all__ = [
    'IdentityLayer',
    'ActivationLayer',
    'InitializerLayer',
    'OneHotLayer',
    'SparseEmbeddingLayer',
    'DenseEmbeddingLayer',
    'PositionalEncoding',
    'BiasEncoding',
    'MaskLayer',
    'SimplePoolingLayer',
    'MultiValuedPoolingLayer',
    'KMaxPoolingLayer',
    'FlattenAxesLayer',
    'TransposeLayer',
]


class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self, if_stop_gradient=False):
        super().__init__()
        self.if_stop_gradient = if_stop_gradient

    @tf.function
    def call(self, inputs):
        x = inputs
        x = tf.stop_gradient(x) if self.if_stop_gradient else x  # tf.constant(x) without tf.function
        return x


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, perm=None):
        super().__init__()
        self.perm = perm

    @tf.function
    def call(self, inputs):
        x = inputs
        x = tf.transpose(x, perm=self.perm)
        return x


class FlattenAxesLayer(tf.keras.layers.Layer):
    def __init__(self, axes=(1, 2)):
        super().__init__()
        if len(axes) != 2:
            raise MLGBError
        if 0 in axes:
            raise MLGBError

        self.axes = axes

    def build(self, input_shape):
        if input_shape.rank != 4:
            raise MLGBError

        _, self.p1, self.p2, self.p3 = input_shape
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        if list(self.axes) == [1, 2]:
            x = tf.reshape(x, shape=[-1, self.p1 * self.p2, self.p3])
        elif list(self.axes) == [2, 3]:
            x = tf.reshape(x, shape=[-1, self.p1, self.p2 * self.p3])
        elif list(self.axes) == [1, 3]:
            x = tf.reshape(x, shape=[-1, self.p1 * self.p3, self.p2])
        else:
            raise MLGBError
        return x


class DiceLayer(tf.keras.layers.Layer):
    def __init__(self, dice_if_official_bn=True):
        super().__init__()
        self.epsilon = 1e-8
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer='zeros',
            regularizer=None,
            trainable=True,
        )
        self.official_bn_fn = BatchNormalization(
            axis=-1,
            epsilon=self.epsilon,
            center=False,
            scale=False,
        )
        self.bn_fn = self.official_bn_fn if dice_if_official_bn else self.unofficial_bn_fn

    @tf.function
    def call(self, inputs):
        # swish(x) = silu(x) = sigmoid(x) * x
        # prelu(x) = tf.where(x > 0, x, self.alpha * x)
        # dice(x) = tf.where(x > 0, sigmoid(x) * x, (1 - sigmoid(x)) * self.alpha * x)
        x = inputs
        p = self.bn_fn(x)
        p = tf.nn.sigmoid(p)
        x = p * x + (1 - p) * self.alpha * x
        return x

    def unofficial_bn_fn(self, x):
        diff = x - tf.reduce_mean(x, axis=0, keepdims=True)
        var = tf.math.reduce_variance(x, axis=0, keepdims=True)
        x = diff / tf.sqrt(var + self.epsilon)
        return x


class SquashLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0
        self.beta = 1e-8

    @tf.function
    def call(self, inputs):
        x = inputs
        l1 = self.p_norm(x, p=1, if_root=True)
        l2 = self.p_norm(x, p=2, if_root=True)
        x = x * (l2 / (l2 + self.alpha)) * (1 / (l1 + self.beta))
        return x

    def p_norm(self, x, p=2, if_root=True):  # tf.norm()
        lp = tf.reduce_sum(tf.abs(x) ** p, axis=None, keepdims=False) ** ((1 / p) if if_root else 1)
        return lp


class ActivationLayer(tf.keras.layers.Layer):
    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation
        if self.activation == 'dice':
            self.activation_fn = DiceLayer()
        elif self.activation == 'squash':
            self.activation_fn = SquashLayer()
        elif isinstance(self.activation, tf.keras.layers.Activation):
            self.activation_fn = self.activation
        else:
            self.activation_fn = Activation(self.activation)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.activation_fn(x)
        return x


class InitializerLayer:
    def __init__(self, initializer=None, activation=None, seed=None):
        super().__init__()
        self.initializer = initializer
        self.activation = activation
        self.initializer_map = {
            'glorot_normal': tf.keras.initializers.glorot_normal(seed=seed),  # xavier normal
            'glorot_uniform': tf.keras.initializers.glorot_uniform(seed=seed),
            'xavier_normal': tf.keras.initializers.glorot_normal(seed=seed),  # xavier normal
            'xavier_uniform': tf.keras.initializers.glorot_uniform(seed=seed),
            'he_normal': tf.keras.initializers.he_normal(seed=seed),  # kaiming normal
            'he_uniform': tf.keras.initializers.he_uniform(seed=seed),
            'kaiming_normal': tf.keras.initializers.he_normal(seed=seed),  # kaiming normal
            'kaiming_uniform': tf.keras.initializers.he_uniform(seed=seed),
            'lecun_normal': tf.keras.initializers.lecun_normal(seed=seed),
            'lecun_uniform': tf.keras.initializers.lecun_uniform(seed=seed),
            'random_normal': tf.keras.initializers.random_normal(mean=0.0, stddev=1.0, seed=seed),
            'random_uniform': tf.keras.initializers.random_uniform(seed=seed),
            'truncated_normal': tf.keras.initializers.truncated_normal(mean=0.0, stddev=1.0, seed=seed),
            'orthogonal': tf.keras.initializers.orthogonal(seed=seed),
            'zeros': tf.keras.initializers.zeros(),
            'ones': tf.keras.initializers.ones(),
        }
        self.activation_initializer_map = {
            'relu': self.initializer_map['he_normal'],
            'gelu': self.initializer_map['he_normal'],
            'silu': self.initializer_map['he_normal'],
            'prelu': self.initializer_map['he_normal'],
            'selu': self.initializer_map['lecun_normal'],
            'tanh': self.initializer_map['glorot_normal'],
            'sigmoid': self.initializer_map['glorot_normal'],
            'dice': self.initializer_map['glorot_normal'],
            'squash': self.initializer_map['random_normal'],
        }

    def get(self):
        if isinstance(self.initializer, (tf.initializers.Initializer, tf.keras.initializers.Initializer)):
            initializer_fn = self.initializer
        elif not (self.initializer in self.initializer_map or self.activation in self.activation_initializer_map):
            initializer_fn = self.initializer_map['glorot_normal']
        elif self.initializer:
            initializer_fn = self.initializer_map[self.initializer]
        else:
            initializer_fn = self.activation_initializer_map[self.activation]
        return initializer_fn


class OneHotLayer(tf.keras.layers.Layer):
    def __init__(self, sparse_feature_names, onehot_dim=None):
        super().__init__()
        self.sparse_feature_names = sparse_feature_names
        self.onehot_dim = onehot_dim

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.fields_width = input_shape[1]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        sparse_tensor = inputs

        onehot_tensor_list = []
        for i in range(self.fields_width):
            depth = self.onehot_dim if self.onehot_dim else self.sparse_feature_names[i]['feature_nunique']
            sparse_i_tensor = tf.gather(params=sparse_tensor, indices=tf.constant(i, dtype=tf.int32), axis=1)
            onehot_i_tensor = tf.one_hot(indices=sparse_i_tensor, depth=depth)
            onehot_tensor_list.append(onehot_i_tensor)
        onehot_tensor = tf.concat(onehot_tensor_list, axis=1)
        return onehot_tensor


class SparseEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, sparse_feature_names, embed_dim=None, embed_l2=0.0, embed_initializer=None,
                 embed_if_output2d=False, seed=None):
        super().__init__()
        self.sparse_feature_names = sparse_feature_names
        self.embed_dim = embed_dim
        self.embed_if_output2d = embed_if_output2d
        self.embed_initializer = InitializerLayer(
            initializer=embed_initializer,
            activation=None,
            seed=seed,
        ).get()

        self.embed_fn_map = {
            f'{i}': Embedding(
                input_dim=feat_dict['embed_feature_nunique'],
                output_dim=feat_dict['embed_dim'] if not self.embed_dim else self.embed_dim,
                mask_zero=feat_dict['mask_zero'],
                input_length=feat_dict['input_length'],
                embeddings_initializer=self.embed_initializer,
                embeddings_regularizer=L1L2(l1=0.0, l2=embed_l2),
            )
            for i, feat_dict in enumerate(self.sparse_feature_names)
        }  # debug: save_model need str key.

    def build(self, input_shape):
        if input_shape.rank not in (2, 3):
            raise MLGBError

        self.fields_width = input_shape[1]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        sparse_tensor = inputs
        embed_tensor_list = [
            self.embed_fn_map[f'{i}'](tf.gather(params=sparse_tensor, indices=tf.constant(i, dtype=tf.int32), axis=1))
            for i in range(self.fields_width)
        ]
        d23_tensor = tf.concat(embed_tensor_list, axis=1)

        if self.embed_if_output2d:
            return d23_tensor, d23_tensor

        if not self.embed_dim:
            raise MLGBError

        d34_tensor = tf.stack(embed_tensor_list, axis=1)
        return d23_tensor, d34_tensor


class DenseEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=32, embed_l2=0.0, embed_initializer=None, seed=None):
        super().__init__()
        if not embed_dim:
            raise MLGBError

        self.embed_dim = embed_dim
        self.embed_l2 = embed_l2
        self.embed_initializer = InitializerLayer(
            initializer=embed_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.embed_weight = self.add_weight(
            name='embed_weight',
            shape=[input_shape[1], self.embed_dim],
            initializer=self.embed_initializer,
            regularizer=L1L2(0.0, self.embed_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = tf.einsum('bf,fe->bfe', x, self.embed_weight)
        return x


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, att_if_pe=True):
        super().__init__()
        self.att_if_pe = att_if_pe
        self.wave_length = 1e4  # 1e4 * pi

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        _, f, e = input_shape

        pos_f = numpy.arange(f).reshape(-1, 1)
        pos_e = numpy.arange(e) // 2 * 2
        pos_x = pos_f / numpy.power(self.wave_length, pos_e / e)  # (f, e)

        pos_x[:, 0::2] = numpy.sin(pos_x[:, 0::2])
        pos_x[:, 1::2] = numpy.cos(pos_x[:, 1::2])

        self.pe = tf.convert_to_tensor(pos_x.reshape(1, f, e), dtype=tf.float32)
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        if self.att_if_pe:
            x += self.pe
        return x


class BiasEncoding(tf.keras.layers.Layer):
    def __init__(self, if_bias=True, bias_l2=0.0, bias_initializer='zeros', seed=None):
        super().__init__()
        self.if_bias = if_bias
        self.bias_l2 = bias_l2
        self.bias_initializer = InitializerLayer(
            initializer=bias_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        if input_shape.rank < 2:
            raise MLGBError

        if self.if_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=input_shape[1:],
                initializer='zeros',
                regularizer=L1L2(0.0, self.bias_l2),
                trainable=True,
            )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        if self.if_bias:
            x += self.bias
        return x


class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, att_if_mask=True):
        super().__init__()
        self.min_inf = -4294967295  # float32: -2 ** 32 + 1
        self.att_if_mask = att_if_mask

    @tf.function
    def call(self, inputs):
        x = inputs
        if self.att_if_mask:
            x = tf.where(tf.not_equal(x, 0), x, self.min_inf)
        return x


class SimplePoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_mode='Pooling:average', pool_axis=-1, pool_axis_if_keep=False):
        super().__init__()
        if pool_mode not in ('Pooling:max', 'Pooling:average', 'Pooling:sum'):
            raise MLGBError

        self.pool_mode = pool_mode
        self.pool_axis = pool_axis
        self.pool_axis_if_keep = pool_axis_if_keep

    def build(self, input_shape):
        if input_shape.rank not in (2, 3, 4):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.pool_fn(x, self.pool_mode, self.pool_axis, self.pool_axis_if_keep)
        return x

    def pool_fn(self, x, pool_mode, pool_axis, pool_axis_if_keep):
        pool_fn_map = {
            'Pooling:average': lambda x: tf.reduce_mean(x, axis=pool_axis, keepdims=pool_axis_if_keep),
            'Pooling:sum': lambda x: tf.reduce_sum(x, axis=pool_axis, keepdims=pool_axis_if_keep),
            'Pooling:max': lambda x: tf.reduce_max(x, axis=pool_axis, keepdims=pool_axis_if_keep),
        }
        x = pool_fn_map[pool_mode](x)
        return x


class MultiValuedPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_mode='Attention', pool_axis=2, pool_if_output2d=False, pool_l2=0.0, pool_initializer=None, seed=None):
        super().__init__()
        if pool_mode not in ('Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum'):
            raise MLGBError
        if pool_axis not in (1, 2, 3, -1, -2, -3):
            raise MLGBError

        self.pool_mode = pool_mode
        self.pool_axis = pool_axis
        self.pool_if_output2d = pool_if_output2d
        self.pool_l2 = pool_l2
        self.pool_initializer = InitializerLayer(
            initializer=pool_initializer,
            activation=None,
            seed=seed,
        ).get()
        if self.pool_mode.startswith('Pooling:'):
            self.pool_fn = SimplePoolingLayer(
                pool_mode=pool_mode,
                pool_axis=pool_axis,
            )
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 4:
            raise MLGBError

        _, self.fields_width, self.sequence_length, _ = input_shape

        if self.pool_mode in ('Weighted', 'Attention'):
            self.pool_weight = self.add_weight(
                name='pool_weight',
                shape=[self.fields_width, self.sequence_length],
                initializer=self.pool_initializer,
                regularizer=L1L2(0.0, self.pool_l2),
                trainable=True,
            )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs  # (batch_size, fields_width, sequence_length, embed_dim)

        if self.pool_mode.startswith('Pooling:'):
            x = self.pool_fn(x)
        elif self.pool_mode in ('Weighted', 'Attention'):
            x_w = tf.einsum('ijkl,jk->ijkl', x, self.pool_weight)
            if self.pool_mode == 'Attention':
                x_w = MaskLayer(att_if_mask=True)(x_w)
                x_w = x * tf.nn.softmax(x_w, axis=self.pool_axis)
            x = tf.reduce_sum(x_w, axis=self.pool_axis, keepdims=False)
        else:
            raise MLGBError

        if self.pool_if_output2d:
            x = self.flatten_fn(x)
        return x  # (b, f, e) or (b, f*e)


class KMaxPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_axis=-1):
        super().__init__()
        if pool_axis == 0:
            raise MLGBError

        self.axis = pool_axis

    def build(self, input_shape):
        self.dim = input_shape.rank
        if self.dim not in (3, 4):
            raise MLGBError

        self.axis = list(range(self.dim))[self.axis]
        self.if_transpose = True if (self.axis != self.dim - 1) else False
        self.perm_dim2axis_dict = {
            '3': {
                '1': [0, 2, 1]
            },
            '4': {
                '1': [0, 3, 2, 1],
                '2': [0, 1, 3, 2]
            }
        }  # debug: save_model need str key.
        self.built = True
        return

    @tf.function
    def call(self, inputs, k=1):
        x = inputs
        if self.if_transpose:
            perm = self.perm_dim2axis_dict[str(self.dim)][str(self.axis)]
            x = tf.transpose(x, perm=perm)
            x = tf.math.top_k(x, k=k, sorted=True)[0]  # axis=-1 only.
            x = tf.transpose(x, perm=perm)  # (b, f, e, 1)
        else:
            x = tf.math.top_k(x, k=k, sorted=True)[0]
        return x






