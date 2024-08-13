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
    tf,
    Flatten,
    SampleModeList
)
from mlgb.tf.inputs import InputsLayer
from mlgb.tf.components.linears import DeepNeuralNetworkLayer
from mlgb.error import MLGBError


__all__ = [
    'SampledSoftmaxLossLayer',
    'BaseInputsEmbeddingLayer',
]


# @tf.keras.utils.register_keras_serializable(package='SampledSoftmaxLossLayer')
class SampledSoftmaxLossLayer(tf.keras.layers.Layer):
    def __init__(self, sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 seed=None, **kwargs):  # debug1: param kwargs is must. If not, load_model will show bug.
        """
        Paper Link:
            https://arxiv.org/pdf/1310.4546.pdf,
            https://arxiv.org/pdf/1412.2007.pdf,
            https://www.tensorflow.org/extras/candidate_sampling.pdf
        """
        super().__init__()
        if sample_mode not in SampleModeList:
            raise MLGBError

        self.sample_mode = sample_mode
        self.sample_num = sample_num
        self.sample_item_distribution_list = sample_item_distribution_list
        self.sample_fixed_unigram_frequency_list = sample_fixed_unigram_frequency_list
        self.sample_fixed_unigram_distortion = sample_fixed_unigram_distortion
        self.seed = seed

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        self.check_shape(y_true, y_pred)

        if self.sample_mode != 'Sample:all':
            y_true_int = tf.cast(y_true, dtype=tf.int64)
            _, multiclass_num = y_true_int.shape
            if self.sample_num and self.sample_num > multiclass_num:
                raise MLGBError('sample_num')
            self.sample_num = self.sample_num if self.sample_num else multiclass_num

            if self.sample_mode == 'Sample:uniform':
                item_indices = tf.random.uniform_candidate_sampler(
                    true_classes=y_true_int,
                    num_true=multiclass_num,
                    num_sampled=self.sample_num,
                    range_max=multiclass_num,
                    unique=True,
                    seed=self.seed,
                )[0]
            elif self.sample_mode == 'Sample:log_uniform':
                item_indices = tf.random.log_uniform_candidate_sampler(
                    true_classes=y_true_int,
                    num_true=multiclass_num,
                    num_sampled=self.sample_num,
                    range_max=multiclass_num,
                    unique=True,
                    seed=self.seed,
                )[0]
            elif self.sample_mode == 'Sample:fixed_unigram':
                if not self.sample_fixed_unigram_frequency_list:
                    self.sample_fixed_unigram_frequency_list = [1.0 / multiclass_num] * multiclass_num
                item_indices = tf.random.fixed_unigram_candidate_sampler(
                    true_classes=y_true_int,
                    num_true=multiclass_num,
                    num_sampled=self.sample_num,
                    range_max=multiclass_num,
                    unigrams=self.sample_fixed_unigram_frequency_list,
                    distortion=self.sample_fixed_unigram_distortion,
                    unique=True,
                    seed=self.seed,
                )[0]
            elif self.sample_mode == 'Sample:learned_unigram':
                item_indices = tf.random.learned_unigram_candidate_sampler(
                    true_classes=y_true_int,
                    num_true=multiclass_num,
                    num_sampled=self.sample_num,
                    range_max=multiclass_num,
                    unique=True,
                    seed=self.seed,
                )[0]
            elif self.sample_mode == 'Sample:batch':
                item_indices = tf.reshape(tf.where(tf.reduce_sum(y_true_int, axis=0, keepdims=False) > 0), shape=(-1,))
            else:
                raise MLGBError

            if self.sample_item_distribution_list:
                q = tf.expand_dims(tf.convert_to_tensor(self.sample_item_distribution_list, dtype=tf.float32), axis=0)
            else:
                q = self.get_item_distribution(y_true)

            q = tf.gather(q, indices=item_indices, axis=1)
            y_true = tf.gather(y_true, indices=item_indices, axis=1)
            y_pred = tf.gather(y_pred, indices=item_indices, axis=1)
            y_pred -= tf.math.log(q)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            axis=-1,
        )
        loss = tf.reduce_mean(loss, axis=None, keepdims=False)
        return loss

    def get_item_distribution(self, y_true, add_bias=1):  # Batch Q like BN, not global Q.
        q_top = tf.reduce_sum(y_true, axis=0, keepdims=True)
        q_bottom = tf.reduce_sum(y_true, axis=None, keepdims=False)
        q = (q_top + add_bias) / (q_bottom + add_bias)  # add 1 avoid log(0) = -inf.
        return q

    def check_shape(self, y_true, y_pred):
        if not (y_true.shape.rank == y_pred.shape.rank == 2):
            raise MLGBError
        if y_true.shape != y_pred.shape:
            raise MLGBError
        if y_true.shape[1] <= 2:
            raise MLGBError
        return

    def get_config(self):  # debug2: get_config is also must. If not, load_model will show bug.
        config = super().get_config()
        config.update(
            {
                'sample_mode': self.sample_mode,
                'sample_num': self.sample_num,
                'sample_item_distribution_list': self.sample_item_distribution_list,
                'sample_fixed_unigram_frequency_list': self.sample_fixed_unigram_frequency_list,
                'sample_fixed_unigram_distortion': self.sample_fixed_unigram_distortion,
                'seed': self.seed,
            }
        )
        return config


class BaseInputsEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, user_feature_names,
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False, user_inputs_if_embed_dense=True,
                 embed_dim=32, embed_l2=0.0, embed_initializer=None,
                 pool_mv_mode='Attention', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None,
                 pool_seq_mode='Attention', pool_seq_axis=2, pool_seq_l2=0.0, pool_seq_initializer=None,
                 user_dnn_hidden_units=(64, 32), dnn_activation=None, dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 tower_embeds_flatten_mode='flatten', tower_embeds_if_l2_norm=True,
                 seed=None):
        super().__init__()
        if tower_embeds_flatten_mode not in ('flatten', 'sum'):
            raise MLGBError

        self.user_dnn_hidden_units = user_dnn_hidden_units
        self.tower_embeds_flatten_mode = tower_embeds_flatten_mode
        self.tower_embeds_if_l2_norm = tower_embeds_if_l2_norm

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,
            inputs_if_embed_dense=user_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_2d_dim=None,
            embed_cate_if_output2d=False,  #
            embed_l2=embed_l2,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_l2=pool_mv_l2,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_l2=pool_seq_l2,
            pool_seq_initializer=pool_seq_initializer,
            seed=seed,
        )
        if self.user_dnn_hidden_units:
            self.dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=user_dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed,
            )
        self.flatten_fn = Flatten()

    def build(self, input_shape):  # OriginInputs
        if len(input_shape) not in (2, 3, 4):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        _, x = self.user_input_fn(inputs)

        if self.tower_embeds_flatten_mode == 'flatten':
            x = self.flatten_fn(x)
        else:
            x = tf.reduce_sum(x, axis=1, keepdims=False)

        if self.user_dnn_hidden_units:
            x = self.dnn_fn(x)

        if self.tower_embeds_if_l2_norm:
            x = tf.nn.l2_normalize(x, axis=1)
        return x












