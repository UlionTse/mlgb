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
)
from mlgb.tf.inputs import InputsLayer
from mlgb.tf.components.linears import (
    DeepNeuralNetworkLayer,
)
from mlgb.tf.components.retrieval import (
    BaseInputsEmbeddingLayer,
    CapsuleNetworkLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'DeepStructuredSemanticModelUserEmbeddingLayer',
    'DeepStructuredSemanticModelItemEmbeddingLayer',
    'NeuralCollaborativeFilteringLayer',
    'MultiInterestNetworkWithDynamicRoutingLayer',
    'UserInputs_MultiInterestNetworkWithDynamicRoutingLayer',
]


class DeepStructuredSemanticModelUserEmbeddingLayer(BaseInputsEmbeddingLayer):
    def __init__(self, user_feature_names,
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False, user_inputs_if_embed_dense=True,
                 embed_dim=32, embed_l2=0.0, embed_initializer=None,
                 pool_mv_mode='Attention', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None,
                 pool_seq_mode='Attention', pool_seq_axis=2, pool_seq_l2=0.0, pool_seq_initializer=None,
                 user_dnn_hidden_units=(64, 32), dnn_activation='tanh', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 tower_embeds_flatten_mode='flatten', tower_embeds_if_l2_norm=True,
                 seed=None):
        super().__init__(
            user_feature_names,
            user_inputs_if_multivalued, user_inputs_if_sequential, user_inputs_if_embed_dense,
            embed_dim, embed_l2, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_l2, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_l2, pool_seq_initializer,
            user_dnn_hidden_units, dnn_activation, dnn_dropout,
            dnn_if_bn, dnn_if_ln, dnn_l2, dnn_initializer,
            tower_embeds_flatten_mode, tower_embeds_if_l2_norm,
            seed,
        )


class DeepStructuredSemanticModelItemEmbeddingLayer(BaseInputsEmbeddingLayer):
    def __init__(self, item_feature_names,
                 item_inputs_if_multivalued=False, item_inputs_if_sequential=False, item_inputs_if_embed_dense=True,
                 embed_dim=32, embed_l2=0.0, embed_initializer=None,
                 pool_mv_mode='Attention', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None,
                 pool_seq_mode='Attention', pool_seq_axis=2, pool_seq_l2=0.0, pool_seq_initializer=None,
                 item_dnn_hidden_units=(64, 32), dnn_activation='tanh', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 tower_embeds_flatten_mode='flatten', tower_embeds_if_l2_norm=True,
                 seed=None):
        super().__init__(
            item_feature_names,
            item_inputs_if_multivalued, item_inputs_if_sequential, item_inputs_if_embed_dense,
            embed_dim, embed_l2, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_l2, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_l2, pool_seq_initializer,
            item_dnn_hidden_units, dnn_activation, dnn_dropout,
            dnn_if_bn, dnn_if_ln, dnn_l2, dnn_initializer,
            tower_embeds_flatten_mode, tower_embeds_if_l2_norm,
            seed,
        )


class NeuralCollaborativeFilteringLayer(tf.keras.layers.Layer):
    def __init__(self, dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == input_shape[3].rank == 2):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_user_mf, x_item_mf, x_user_dnn, x_item_dnn = inputs

        x_dnn = tf.concat([x_user_dnn, x_item_dnn], axis=1)
        dnn_outputs = self.dnn_fn(x_dnn)
        mf_outputs = x_user_mf * x_item_mf

        ncf_outputs = tf.concat([mf_outputs, dnn_outputs], axis=1)
        return ncf_outputs


class MultiInterestNetworkWithDynamicRoutingLayer(tf.keras.layers.Layer):
    def __init__(self, capsule_num=3, capsule_activation='squash', capsule_l2=0.0, capsule_initializer=None,
                 capsule_interest_num_if_dynamic=False, capsule_input_sequence_pad_mode='pre',
                 capsule_routing_initializer='random_normal',
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.caps_net_fn = CapsuleNetworkLayer(
            capsule_num=capsule_num,
            capsule_activation=capsule_activation,
            capsule_l2=capsule_l2,
            capsule_initializer=capsule_initializer,
            capsule_interest_num_if_dynamic=capsule_interest_num_if_dynamic,
            capsule_input_sequence_pad_mode=capsule_input_sequence_pad_mode,
            capsule_routing_initializer=capsule_routing_initializer,
            seed=seed,
        )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.flatten_fn_list = [Flatten() for _ in range(2)]

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_user_fea, x_user_seq = inputs

        x_user_seq = self.caps_net_fn(x_user_seq)
        x_user_seq = self.flatten_fn_list[0](x_user_seq)
        x_user_fea = self.flatten_fn_list[1](x_user_fea)
        x = tf.concat([x_user_fea, x_user_seq], axis=1)
        x = self.dnn_fn(x)
        return x


class UserInputs_MultiInterestNetworkWithDynamicRoutingLayer(tf.keras.layers.Layer):
    def __init__(self, user_feature_names,
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 embed_dim=32, embed_l2=0.0, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=2, pool_seq_l2=0.0, pool_seq_initializer=None,
                 tower_embeds_if_l2_norm=True,
                 capsule_num=3, capsule_activation='squash', capsule_l2=0.0, capsule_initializer=None,
                 capsule_interest_num_if_dynamic=False, capsule_input_sequence_pad_mode='pre',
                 capsule_routing_initializer='random_normal',
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.tower_embeds_if_l2_norm = tower_embeds_if_l2_norm

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,
            inputs_if_embed_dense=True,
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
            pool_seq_axis=pool_seq_axis,  #
            pool_seq_l2=pool_seq_l2,
            pool_seq_initializer=pool_seq_initializer,
            seed=seed,
        )
        self.user_mind_fn = MultiInterestNetworkWithDynamicRoutingLayer(
            capsule_num=capsule_num,
            capsule_activation=capsule_activation,
            capsule_l2=capsule_l2,
            capsule_initializer=capsule_initializer,
            capsule_interest_num_if_dynamic=capsule_interest_num_if_dynamic,
            capsule_input_sequence_pad_mode=capsule_input_sequence_pad_mode,
            capsule_routing_initializer=capsule_routing_initializer,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
    
    def build(self, input_shape):  # OriginInputs
        if len(input_shape) not in (2, 3, 4):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        _, user_fea_3d_inputs, user_seq_3d_inputs = self.user_input_fn(inputs)
        
        x = self.user_mind_fn([user_fea_3d_inputs, user_seq_3d_inputs])
        if self.tower_embeds_if_l2_norm:
            x = tf.nn.l2_normalize(x, axis=1)
        return x








































