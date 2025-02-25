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
    L1L2, 
    Dropout,
)
from mlgb.tf.functions import (
    ActivationLayer, 
    InitializerLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'AllFieldBinaryInteractionLayer',
    'AttentionalFieldBinaryInteractionLayer',
    'TwoAllFieldBinaryInteractionLayer',
    'GroupedAllFieldWiseBinaryInteractionLayer',
]


class BinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank not in (2, 3):
            raise MLGBError

        self.fbi_if_keepdim = True if input_shape.rank == 2 else False
        self.fbi_weight = self.add_weight(
            name='fbi_weight',
            shape=input_shape.as_list()[1:],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x = x * self.fbi_weight
        square_of_sum = tf.square(tf.reduce_sum(x, axis=1, keepdims=self.fbi_if_keepdim))
        sum_of_square = tf.reduce_sum(tf.square(x), axis=1, keepdims=self.fbi_if_keepdim)
        x = 0.5 * tf.subtract(square_of_sum, sum_of_square)  # (batch_size, embed_dim) or (batch_size, 1)
        return x


class HigherOrderBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_hofm_order=3, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if not (fbi_hofm_order >= 3):
            raise MLGBError

        self.fbi_hofm_order = fbi_hofm_order

        self.fm_fn = BinaryInteractionLayer(
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError
        if self.fbi_hofm_order > input_shape[1]:
            raise MLGBError

        self.fields_width = input_shape[1]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x_fm = self.fm_fn(x)
        x = x * self.fm_fn.fbi_weight
        x_hofm = self.get_anova_kernel(x, self.fbi_hofm_order)
        x = tf.concat([x_fm, x_hofm], axis=1)
        return x

    def get_anova_kernel(self, x, hofm_order):
        a = tf.ones_like(x)
        a = tf.gather(a, indices=tf.constant([0] * (self.fields_width + 1), dtype=tf.int32), axis=1)  # (b, f+1, e)

        ak_list = []
        for i in range(hofm_order):
            a_i_0 = tf.zeros_like(x)[:, :i+1, :]
            a_i = x[:, i:, :] * a[:, i:-1, :]
            a_i = tf.concat([a_i_0, a_i], axis=1)
            a = tf.math.cumsum(a_i, axis=1)  # (b, f+1, e)

            if i >= 2:
                ak_list.append(a[:, -1, :])  # (b, e)

        ak = tf.concat(ak_list, axis=1)  # (b, (order-2)*e)
        return ak


class FieldBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if fbi_mode not in ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.fbi_unit = fbi_unit  # fbi_weight__length_of_latent_vector
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        _, self.fields_width, self.embed_dim = input_shape
        self.product_width = int(self.fields_width * (self.fields_width - 1) // 2)
        self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T
        # self.ij_ids = numpy.array(list(itertools.combinations(range(self.fields_width), 2)))

        self.weight_shape_map = {
            'FwFM': [self.product_width],
            'FFM': [self.fields_width, self.embed_dim, self.fbi_unit],
            'PNN:inner_product': [self.product_width, self.embed_dim, self.fbi_unit],
            'PNN:outer_product': [self.product_width, self.embed_dim, self.embed_dim, self.fbi_unit],
        }
        self.fbi_weight = self.add_weight(
            name='fbi_weight',
            shape=self.weight_shape_map[self.fbi_mode],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_i = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)

        if self.fbi_mode == 'FFM':
            w_i = tf.gather(params=self.fbi_weight, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=0)
            w_j = tf.gather(params=self.fbi_weight, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=0)
            w_ij = w_i * w_j
        else:
            w_ij = self.fbi_weight

        if self.fbi_mode == 'FwFM':
            x_ij = x_i * x_j
            x = tf.einsum('bfe,f->be', x_ij, w_ij)  # (batch_size, embed_dim)
        elif self.fbi_mode in ('FFM', 'PNN:inner_product'):
            x_ij = x_i * x_j
            x = tf.einsum('bfe,feu->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        elif self.fbi_mode == 'PNN:outer_product':
            x_i = tf.expand_dims(x_i, axis=3)
            x_j = tf.expand_dims(x_j, axis=2)
            x_ij = x_i @ x_j  # (bfe1,bf1e->bfee)
            x = tf.einsum('bfij,fiju->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        return x


class FieldProductBothInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='PNN:both', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if fbi_mode != 'PNN:both':
            raise MLGBError

        self.ip_fn = FieldBinaryInteractionLayer(
            fbi_mode='PNN:inner_product',
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            seed=seed,
        )
        self.op_fn = FieldBinaryInteractionLayer(
            fbi_mode='PNN:outer_product',
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x_ip = self.ip_fn(x)
        x_op = self.op_fn(x)
        x = tf.concat([x_ip, x_op], axis=1)  # (batch_size, fbi_unit * 2)
        return x


class BilinearInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='Bilinear:field_interaction', fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if fbi_mode not in ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        _, self.fields_width, self.embed_dim = input_shape
        self.product_width = int(self.fields_width * (self.fields_width - 1) // 2)
        self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T

        self.weight_num_map = {
            'Bilinear:field_all': 1,
            'Bilinear:field_each': self.fields_width,
            'Bilinear:field_interaction': self.product_width,
            'FEFM': self.product_width,
            'FvFM': self.product_width,
            'FmFM': self.product_width,
        }

        if self.fbi_mode == 'FvFM':
            self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim]
        else:
            self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim, self.embed_dim]

        self.fbi_weight = self.add_weight(
            name='fbi_weight',
            shape=self.fbi_weight_shape,
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_i = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)

        if self.fbi_mode == 'Bilinear:field_all':
            w = tf.concat([self.fbi_weight] * self.product_width, axis=0)
        elif self.fbi_mode == 'Bilinear:field_each':
            w = tf.gather(params=self.fbi_weight, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=0)
        elif self.fbi_mode in ('Bilinear:field_interaction', 'FmFM', 'FvFM'):
            w = self.fbi_weight
        elif self.fbi_mode == 'FEFM':
            w1 = self.fbi_weight
            w2 = tf.transpose(self.fbi_weight, perm=[0, 2, 1])
            w = (w1 + w2) * 0.5  # symmetric matrix
        else:
            raise MLGBError

        if self.fbi_mode == 'FvFM':
            x_i_mid = x_i * w  # (bfe,fe->bfe)
        else:
            x_i_mid = tf.einsum('bfe,fee->bfe', x_i, w)

        if self.fbi_mode == 'FEFM':
            x = tf.einsum('bme,bne->bmn', x_i_mid, x_j)  # bmm, (batch_size, product_width, product_width)
        else:
            x = x_i_mid * x_j  # (batch_size, product_width, embed_dim)

        x = tf.reduce_sum(x, axis=2, keepdims=False)  # (batch_size, product_width)
        return x


class AttentionalFieldBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_afm_activation='relu', fbi_afm_dropout=0.0, fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 seed=None):
        super().__init__()
        self.fbi_unit = fbi_unit
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=fbi_afm_activation,
            seed=seed,
        ).get()
        self.activation_fn = ActivationLayer(fbi_afm_activation)
        self.drop_fn = Dropout(fbi_afm_dropout)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        _, self.fields_width, self.embed_dim = input_shape
        self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T

        self.att_weight = self.add_weight(
            name='attention_weight',
            shape=[self.embed_dim, self.fbi_unit],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.att_bias = self.add_weight(
            name='attention_bias',
            shape=[self.fbi_unit],
            initializer='zeros',
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.proj_h = self.add_weight(
            name='projection_h',
            shape=[self.fbi_unit, 1],
            initializer=self.attention_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.proj_p = self.add_weight(
            name='projection_p',
            shape=[self.embed_dim, 1],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_i = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)
        x = x_i * x_j  # (batch_size, product_width, embed_dim)

        w = x @ self.att_weight + self.att_bias  # (batch_size, product_width, att_unit)
        w = self.activation_fn(w)
        w = w @ self.proj_h  # (batch_size, product_width, 1)
        w = tf.nn.softmax(w, axis=1)

        x = x * w  # (batch_size, product_width, embed_dim)
        x = self.drop_fn(x)
        x = x @ self.proj_p  # (batch_size, product_width, 1)
        x = tf.squeeze(x, axis=2)  # (batch_size, product_width)
        return x


class AllFieldBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0, seed=None):
        super().__init__()
        self.bi_mode_list = ('FM', 'FM3D',)
        self.ho_model_list = ('HOFM',)
        self.at_model_list = ('AFM',)
        self.pb_mode_list = ('PNN:both',)
        self.fb_mode_list = ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product')
        self.bl_mode_list = ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM')
        self.fbi_mode_list = self.bi_mode_list + self.ho_model_list + self.at_model_list + self.pb_mode_list + self.fb_mode_list + self.bl_mode_list
        if fbi_mode not in self.fbi_mode_list:
            raise MLGBError

        self.fbi_mode = fbi_mode

        if fbi_mode in self.pb_mode_list:
            self.fbi_fn = FieldProductBothInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.fb_mode_list:
            self.fbi_fn = FieldBinaryInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.bl_mode_list:
            self.fbi_fn = BilinearInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.at_model_list:
            self.fbi_fn = AttentionalFieldBinaryInteractionLayer(
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.ho_model_list:
            self.fbi_fn = HigherOrderBinaryInteractionLayer(
                fbi_hofm_order=fbi_hofm_order,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        else:
            self.fbi_fn = BinaryInteractionLayer(
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank not in (2, 3):
            raise MLGBError
        if input_shape.rank == 2 and self.fbi_mode != 'FM':
            raise MLGBError
        if input_shape.rank != 2 and self.fbi_mode == 'FM':
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.fbi_fn(x)
        return x


class TwoBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError

        input_0_shape = tf.TensorShape(input_shape[0]) if not isinstance(input_shape[0], tf.TensorShape) else input_shape[0]
        input_1_shape = tf.TensorShape(input_shape[1]) if not isinstance(input_shape[1], tf.TensorShape) else input_shape[1]

        if input_0_shape.rank not in (2, 3):
            raise MLGBError
        if input_0_shape.rank != input_1_shape.rank:
            raise MLGBError
        if input_0_shape.rank == 3 and (input_0_shape[2] != input_1_shape[2]):  # embed_dim
            raise MLGBError

        self.fbi_if_keepdim = True if input_0_shape.rank == 2 else False

        self.fields_i_width, self.fields_j_width = input_0_shape[1], input_1_shape[1]
        self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)
        # self.ij_ids = numpy.array([[i, j] for i in range(self.fields_i_width) for j in range(self.fields_j_width)])

        self.fbi_i_weight = self.add_weight(
            name='fbi_i_weight',
            shape=input_shape[0].as_list()[1:],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.fbi_j_weight = self.add_weight(
            name='fbi_j_weight',
            shape=input_shape[1].as_list()[1:],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_i, x_j = inputs

        x_i = x_i * self.fbi_j_weight
        x_j = x_j * self.fbi_j_weight
        x_i = tf.gather(params=x_i, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x_j, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)

        x = x_i * x_j
        x = tf.reduce_sum(x, axis=1, keepdims=self.fbi_if_keepdim)  # (batch_size, embed_dim) or (batch_size, 1)
        return x


class TwoFieldBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if fbi_mode not in ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.fbi_unit = fbi_unit  # fbi_weight__length_of_latent_vector
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError

        input_0_shape = tf.TensorShape(input_shape[0]) if not isinstance(input_shape[0], tf.TensorShape) else input_shape[0]
        input_1_shape = tf.TensorShape(input_shape[1]) if not isinstance(input_shape[1], tf.TensorShape) else input_shape[1]

        if input_0_shape.rank != 3:
            raise MLGBError
        if input_0_shape.rank != input_1_shape.rank:
            raise MLGBError
        if input_0_shape.rank == 3 and (input_0_shape[2] != input_1_shape[2]):  # embed_dim
            raise MLGBError

        self.embed_dim = input_0_shape[2]
        self.fields_i_width, self.fields_j_width = input_0_shape[1], input_1_shape[1]
        self.product_width = int(self.fields_i_width * self.fields_j_width)
        self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)

        self.weight_shape_map = {
            'FwFM': [self.product_width],
            'FFM': [self.fields_i_width + self.fields_j_width, self.embed_dim, self.fbi_unit],
            'PNN:inner_product': [self.product_width, self.embed_dim, self.fbi_unit],
            'PNN:outer_product': [self.product_width, self.embed_dim, self.embed_dim, self.fbi_unit],
        }
        self.fbi_weight = self.add_weight(
            name='fbi_weight',
            shape=self.weight_shape_map[self.fbi_mode],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_i, x_j = inputs
        x_i = tf.gather(params=x_i, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x_j, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)

        if self.fbi_mode == 'FFM':
            w_i = tf.gather(params=self.fbi_weight, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=0)
            w_j = tf.gather(params=self.fbi_weight, indices=tf.constant(self.ij_ids[:, 1] + self.fields_i_width, dtype=tf.int32), axis=0)  # shift
            w_ij = w_i * w_j
        else:
            w_ij = self.fbi_weight

        if self.fbi_mode == 'FwFM':
            x_ij = x_i * x_j
            x = tf.einsum('bfe,f->be', x_ij, w_ij)  # (batch_size, embed_dim)
        elif self.fbi_mode in ('FFM', 'PNN:inner_product'):
            x_ij = x_i * x_j
            x = tf.einsum('bfe,feu->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        elif self.fbi_mode == 'PNN:outer_product':
            x_i = tf.expand_dims(x_i, axis=3)
            x_j = tf.expand_dims(x_j, axis=2)
            x_ij = x_i @ x_j  # (bfe1,bf1e->bfee)
            x = tf.einsum('bfij,fiju->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        else:
            raise MLGBError
        return x


class TwoFieldProductBothInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='PNN:both', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if fbi_mode != 'PNN:both':
            raise MLGBError

        self.ip_fn = TwoFieldBinaryInteractionLayer(
            fbi_mode='PNN:inner_product',
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            seed=seed,
        )
        self.op_fn = TwoFieldBinaryInteractionLayer(
            fbi_mode='PNN:outer_product',
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError

        input_0_shape = tf.TensorShape(input_shape[0]) if not isinstance(input_shape[0], tf.TensorShape) else input_shape[0]
        input_1_shape = tf.TensorShape(input_shape[1]) if not isinstance(input_shape[1], tf.TensorShape) else input_shape[1]

        if input_0_shape.rank != 3:
            raise MLGBError
        if input_0_shape.rank != input_1_shape.rank:
            raise MLGBError
        if input_0_shape.rank == 3 and (input_0_shape[2] != input_1_shape[2]):  # embed_dim
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_i, x_j = inputs

        x_ip = self.ip_fn([x_i, x_j])
        x_op = self.op_fn([x_i, x_j])
        x = tf.concat([x_ip, x_op], axis=1)  # (batch_size, fbi_unit * 2)
        return x


class TwoBilinearInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='Bilinear:field_interaction', fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        if fbi_mode not in ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError

        input_0_shape = tf.TensorShape(input_shape[0]) if not isinstance(input_shape[0], tf.TensorShape) else input_shape[0]
        input_1_shape = tf.TensorShape(input_shape[1]) if not isinstance(input_shape[1], tf.TensorShape) else input_shape[1]

        if input_0_shape.rank != 3:
            raise MLGBError
        if input_0_shape.rank != input_1_shape.rank:
            raise MLGBError
        if input_0_shape.rank == 3 and (input_0_shape[2] != input_1_shape[2]):  # embed_dim
            raise MLGBError

        self.embed_dim = input_0_shape[2]
        self.fields_i_width, self.fields_j_width = input_0_shape[1], input_1_shape[1]
        self.product_width = int(self.fields_i_width * self.fields_j_width)
        self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)

        self.weight_num_map = {
            'Bilinear:field_all': 1,
            'Bilinear:field_each': self.fields_width,
            'Bilinear:field_interaction': self.product_width,
            'FEFM': self.product_width,
            'FvFM': self.product_width,
            'FmFM': self.product_width,
        }

        if self.fbi_mode == 'FvFM':
            self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim]
        else:
            self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim, self.embed_dim]

        self.fbi_weight = self.add_weight(
            name='fbi_weight',
            shape=self.fbi_weight_shape,
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_i, x_j = inputs
        x_i = tf.gather(params=x_i, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x_j, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)

        if self.fbi_mode == 'Bilinear:field_all':
            w = tf.concat([self.fbi_weight] * self.product_width, axis=0)
        elif self.fbi_mode == 'Bilinear:field_each':
            w = tf.gather(params=self.fbi_weight, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=0)
        elif self.fbi_mode in ('Bilinear:field_interaction', 'FmFM'):
            w = self.fbi_weight
        elif self.fbi_mode == 'FEFM':
            w1 = self.fbi_weight
            w2 = tf.transpose(self.fbi_weight, perm=[0, 2, 1])
            w = (w1 + w2) * 0.5  # symmetric matrix
        else:
            raise MLGBError

        if self.fbi_mode == 'FvFM':
            x_i_mid = x_i * w  # (bfe,fe->bfe)
        else:
            x_i_mid = tf.einsum('bfe,fee->bfe', x_i, w)

        if self.fbi_mode == 'FEFM':
            x = tf.einsum('bme,bne->bmn', x_i_mid, x_j)  # bmm, (batch_size, product_width, product_width)
        else:
            x = x_i_mid * x_j  # (batch_size, product_width, embed_dim)

        x = tf.reduce_sum(x, axis=2, keepdims=False)  # (batch_size, product_width)
        return x


class TwoAttentionalFieldBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_afm_activation='relu', fbi_afm_dropout=0.0, fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 seed=None):
        super().__init__()
        self.fbi_unit = fbi_unit
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=fbi_afm_activation,
            seed=seed,
        ).get()
        self.activation_fn = ActivationLayer(fbi_afm_activation)
        self.dropout_fn = Dropout(fbi_afm_dropout)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError

        input_0_shape = tf.TensorShape(input_shape[0]) if not isinstance(input_shape[0], tf.TensorShape) else input_shape[0]
        input_1_shape = tf.TensorShape(input_shape[1]) if not isinstance(input_shape[1], tf.TensorShape) else input_shape[1]

        if input_0_shape.rank != 3:
            raise MLGBError
        if input_0_shape.rank != input_1_shape.rank:
            raise MLGBError
        if input_0_shape.rank == 3 and (input_0_shape[2] != input_1_shape[2]):  # embed_dim
            raise MLGBError

        self.embed_dim = input_0_shape[2]
        self.fields_i_width, self.fields_j_width = input_0_shape[1], input_1_shape[1]
        self.product_width = int(self.fields_i_width * self.fields_j_width)
        self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)

        self.att_weight = self.add_weight(
            name='attention_weight',
            shape=[self.embed_dim, self.fbi_unit],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.att_bias = self.add_weight(
            name='attention_bias',
            shape=[self.fbi_unit],
            initializer='zeros',
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.proj_h = self.add_weight(
            name='projection_h',
            shape=[self.fbi_unit, 1],
            initializer=self.attention_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.proj_p = self.add_weight(
            name='projection_p',
            shape=[self.embed_dim, 1],
            initializer=self.fbi_initializer,
            regularizer=L1L2(0.0, self.fbi_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_i = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        x_j = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)
        x = x_i * x_j  # (batch_size, product_width, embed_dim)

        w = x @ self.att_weight + self.att_bias  # (batch_size, product_width, att_unit)
        w = self.activation_fn(w)
        w = w @ self.proj_h  # (batch_size, product_width, 1)
        w = tf.nn.softmax(w, axis=1)

        x = x * w  # (batch_size, product_width, embed_dim)
        x = self.drop_fn(x)
        x = x @ self.proj_p  # (batch_size, product_width, 1)
        x = tf.squeeze(x, axis=2)  # (batch_size, product_width)
        return x


class TwoAllFieldBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0, seed=None):
        super().__init__()
        self.bi_mode_list = ('FM', 'FM3D',)
        self.at_mode_list = ('AFM',)
        self.pb_mode_list = ('PNN:both',)
        self.fb_mode_list = ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product')
        self.bl_mode_list = ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM')
        self.fbi_mode_list = self.bi_mode_list + self.at_model_list + self.pb_mode_list + self.fb_mode_list + self.bl_mode_list
        if fbi_mode not in self.fbi_mode_list:
            raise MLGBError

        self.fbi_mode = fbi_mode
        if fbi_mode in self.pb_mode_list:
            self.two_fbi_fn = TwoFieldProductBothInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.fb_mode_list:
            self.two_fbi_fn = TwoFieldBinaryInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.bl_mode_list:
            self.two_fbi_fn = TwoBilinearInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        elif fbi_mode in self.at_model_list:
            self.two_fbi_fn = TwoAttentionalFieldBinaryInteractionLayer(
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )
        else:
            self.two_fbi_fn = TwoBinaryInteractionLayer(
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                seed=seed,
            )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError

        input_0_shape = tf.TensorShape(input_shape[0]) if not isinstance(input_shape[0], tf.TensorShape) else input_shape[0]
        input_1_shape = tf.TensorShape(input_shape[1]) if not isinstance(input_shape[1], tf.TensorShape) else input_shape[1]

        if input_0_shape.rank not in (2, 3):
            raise MLGBError
        if input_0_shape.rank != input_1_shape.rank:
            raise MLGBError
        if input_0_shape.rank == 3 and (input_0_shape[2] != input_1_shape[2]):  # embed_dim
            raise MLGBError
        if input_0_shape.rank == 2 and self.fbi_mode != 'FM':
            raise MLGBError
        if input_0_shape.rank != 2 and self.fbi_mode == 'FM':
            raise MLGBError

    @tf.function
    def call(self, inputs):
        x_i, x_j = inputs
        x = self.two_fbi_fn([x_i, x_j])
        return x


class GroupedAllFieldWiseBinaryInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, group_indices=((0, 1), (2, 3), (4, 5, 6)),
                 fbi_fm_mode='FM3D', fbi_mf_mode='FwFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 seed=None):
        super().__init__()
        if group_indices and len(group_indices) < 2:
            raise MLGBError

        self.group_indices = group_indices  # (user, item, context)
        self.group_width = len(self.group_indices)
        self.product_width = int(self.group_width * (self.group_width - 1) // 2)
        self.ij_ids = numpy.array(numpy.triu_indices(n=self.group_width, k=1)).T

        self.fbi_fm_fn_list = [
            AllFieldBinaryInteractionLayer(
                fbi_mode=fbi_fm_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(self.group_width)
        ]
        self.fbi_mf_fn_list = [
            TwoAllFieldBinaryInteractionLayer(
                fbi_mode=fbi_mf_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(self.product_width)
        ]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape) if not isinstance(input_shape, tf.TensorShape) else input_shape
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        fbi_fm_pool, fbi_mf_pool = [], []
        for k in range(self.group_width):
            x_k = tf.gather(x, indices=tf.constant(self.group_indices[k], dtype=tf.int32), axis=1)
            fbi_fm_k_outputs = self.fbi_fm_fn_list[k](x_k)
            fbi_fm_pool.append(fbi_fm_k_outputs)

        for k, (i, j) in enumerate(self.ij_ids):
            x_i = tf.gather(x, indices=tf.constant(self.group_indices[i], dtype=tf.int32), axis=1)
            x_j = tf.gather(x, indices=tf.constant(self.group_indices[j], dtype=tf.int32), axis=1)
            fbi_mf_k_ouputs = self.fbi_mf_fn_list[k]([x_i, x_j])
            fbi_mf_pool.append(fbi_mf_k_ouputs)

        fbi_pool = fbi_fm_pool + fbi_mf_pool
        x = tf.concat(fbi_pool, axis=1)
        return x























