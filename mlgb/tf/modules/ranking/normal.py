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
    Flatten,
    BatchNormalization,
    LayerNormalization,
    FBIModeList,
)
from mlgb.tf.functions import (
    IdentityLayer,
    ActivationLayer,
    InitializerLayer,
    KMaxPoolingLayer,
    TransposeLayer,
    FlattenAxesLayer,
)
from mlgb.tf.components.linears import (
    LinearLayer,
    DeepNeuralNetworkLayer,
    DNN3dParallelLayer,
    Linear2dParallelLayer,
    ConvolutionalNeuralNetworkLayer,
)
from mlgb.tf.components.fbis import (
    AllFieldBinaryInteractionLayer,
    GroupedAllFieldWiseBinaryInteractionLayer,
)
from mlgb.tf.components.trms import (
    InteractingLayer,
)
from mlgb.tf.components.base import (
    MaskBlockLayer,
    ResidualUnitLayer,
    CrossNetworkLayer,
    RegulationModuleLayer,
    BridgeModuleLayer,
    FactorEstimatingNetworkLayer,
    LogarithmicTransformationLayer,
    CompressedInteractionNetworkLayer,
    SqueezeExcitationNetworkLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'LinearOrLogisticRegressionLayer',
    'MultiLayerPerceptronLayer',
    'PiecewiseLinearModelLayer',
    'DeepLearningRecommendationModelLayer',
    'MaskNetLayer',

    'DeepCrossingModelLayer',
    'DeepCrossNetworkLayer',
    'EnhancedDeepCrossNetworkLayer',
    'AllFactorizationMachineLayer',
    'FactorizationMachineLayer',
    'FieldFactorizationMachineLayer',
    'FieldWeightedFactorizationMachineLayer',
    'FieldEmbeddedFactorizationMachineLayer',
    'FieldVectorizedFactorizationMachineLayer',
    'FieldMatrixedFactorizationMachineLayer',
    'AttentionalFactorizationMachineLayer',
    'LorentzFactorizationMachineLayer',
    'InteractionMachineLayer',
    'AllInputFactorizationMachineLayer',
    'InputFactorizationMachineLayer',
    'DualInputFactorizationMachineLayer',

    'FactorizationMachineNeuralNetworkLayer',
    'ProductNeuralNetworkLayer',
    'ProductNetworkInNetworkLayer',
    'OperationNeuralNetworkLayer',
    'AdaptiveFactorizationNetworkLayer',

    'NeuralFactorizationMachineLayer',
    'WideDeepLearningLayer',
    'DeepFactorizationMachineLayer',
    'DeepFieldEmbeddedFactorizationMachineLayer',
    'FieldLeveragedEmbeddingNetworkLayer',

    'ConvolutionalClickPredictionModelLayer',
    'FeatureGenerationByConvolutionalNeuralNetworkLayer',
    'ExtremeDeepFactorizationMachineLayer',
    'FeatureImportanceBilinearInteractionNetworkLayer',
    'AutomaticFeatureInteractionLearningLayer',
]


class LinearOrLogisticRegressionLayer(LinearLayer):
    def __init__(self, linear_if_bias=True, linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            linear_if_bias=linear_if_bias,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )


class MultiLayerPerceptronLayer(DeepNeuralNetworkLayer):
    def __init__(self, dnn_hidden_units=(64, 32), dnn_activation='tanh', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_if_bias=True, dnn_l2=0.0, dnn_initializer=None, seed=None):
        super().__init__(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_if_bias=dnn_if_bias,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )


class PiecewiseLinearModelLayer(tf.keras.layers.Layer):
    def __init__(self, plm_task='binary', plm_piece_num=2,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__()
        self.task_activation_dict = {
            'regression': None,
            'binary': 'sigmoid',
        }
        self.linear2d_parallel_fn = Linear2dParallelLayer(
            linear_parallel_num=plm_piece_num,
            linear_activation=None,
            linear_if_bias=True,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )
        self.task2d_parallel_fn = Linear2dParallelLayer(
            linear_parallel_num=plm_piece_num,
            linear_activation=self.task_activation_dict[plm_task],
            linear_if_bias=True,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == 2):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        piece_dense_2d_tensor, base_dense_2d_tensor = inputs

        piece_tensor = self.linear2d_parallel_fn(piece_dense_2d_tensor)  # (batch_size, plm_piece_num)
        base_tensor = self.task2d_parallel_fn(base_dense_2d_tensor)  # (batch_size, plm_piece_num)
        piece_w_tensor = tf.nn.softmax(piece_tensor, axis=1)
        plm_outputs = base_tensor * piece_w_tensor
        plm_outputs = tf.reduce_sum(plm_outputs, axis=1, keepdims=True)
        return plm_outputs


class DeepLearningRecommendationModelLayer(tf.keras.layers.Layer):
    def __init__(self, dnn_bottom_hidden_units=(64, 32), dnn_top_hidden_units=(64, 32),
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None, seed=None):
        super().__init__()
        self.dnn_bottom_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_bottom_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.dnn_top_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_top_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == 2):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_2d_tensor = inputs

        bottom_outputs = self.dnn_bottom_fn(dense_2d_tensor)
        top_inputs = tf.concat([bottom_outputs, embed_2d_tensor], axis=1)
        dlrm_outputs = self.dnn_top_fn(top_inputs)
        return dlrm_outputs


class MaskNetLayer(tf.keras.layers.Layer):
    def __init__(self, mask_mode='MaskNet:serial', mask_block_num=4,
                 block_activation='relu', block_if_bn=False, block_dropout=0.0, block_l2=0.0, block_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu',
                 dnn_if_bn=False, dnn_if_ln=False, dnn_dropout=0.0, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if mask_mode not in ('MaskNet:serial', 'MaskNet:parallel'):
            raise MLGBError
        if not mask_block_num > 0:
            raise MLGBError

        self.mask_mode = mask_mode

        self.block_fn_list = [
            MaskBlockLayer(
                ffn_activation=block_activation,
                ffn_if_bn=block_if_bn,
                ffn_dropout=block_dropout,
                ffn_l2=block_l2,
                ffn_initializer=block_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(mask_block_num)
        ]
        if self.mask_mode == 'MaskNet:parallel':
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
        self.ln_emb_fn = LayerNormalization(axis=2)  #
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        embed_3d_tensor = inputs

        x_e = self.ln_emb_fn(embed_3d_tensor)
        x_m = embed_3d_tensor

        if self.mask_mode == 'MaskNet:parallel':
            x_pool = [block_fn([x_e, x_m]) for block_fn in self.block_fn_list]
            x = tf.concat(x_pool, axis=1)
            x = self.flatten_fn(x)
            x = self.dnn_fn(x)
        else:
            for block_fn in self.block_fn_list:
                x_e = block_fn([x_e, x_m])
            x = self.flatten_fn(x_e)
        return x


class DeepCrossingModelLayer(tf.keras.layers.Layer):
    def __init__(self, dcm_if_dnn=True, dcm_if_ln=False,
                 dnn_hidden_units=(64, 32), dnn_activation='relu',
                 dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.dcm_layer_length = len(dnn_hidden_units)
        self.dcm_fn_list = [
            ResidualUnitLayer(
                dcm_if_dnn=dcm_if_dnn,
                dcm_if_ln=dcm_if_ln,
                dnn_hidden_unit=dnn_hidden_units[i],
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(self.dcm_layer_length)
        ]

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        for dcm_fn in self.dcm_fn_list:
            x = dcm_fn(x)
        return x


class DeepCrossNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, dcn_version='v2', dcn_l2=0.0, dcn_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.cross_fn = CrossNetworkLayer(
            dcn_layer_num=len(dnn_hidden_units),
            dcn_version=dcn_version,
            dcn_l2=dcn_l2,
            dcn_initializer=dcn_initializer,
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

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        cn_outputs = self.cross_fn(x)
        dnn_outputs = self.dnn_fn(x)
        outputs = tf.concat([cn_outputs, dnn_outputs], axis=-1)
        return outputs


class EnhancedDeepCrossNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, edcn_layer_num=2,
                 bdg_mode='EDCN:attention_pooling', bdg_layer_num=1,
                 rgl_tau_ratio=1.0, rgl_initializer='ones', rgl_l2=0.0,
                 dcn_version='v2', dcn_l2=0.0, dcn_initializer=None,
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.edcn_layer_num = edcn_layer_num
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.dnn_if_bn = dnn_if_bn
        self.dnn_if_ln = dnn_if_ln
        self.dnn_l2 = dnn_l2
        self.dnn_initializer = dnn_initializer
        self.seed = seed

        self.rgl_cross_fn_list = [
            RegulationModuleLayer(
                fgu_tau_ratio=rgl_tau_ratio,
                fgu_initializer=rgl_initializer,
                fgu_l2=rgl_l2,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(edcn_layer_num+1)
        ]
        self.rgl_deep_fn_list = [
            RegulationModuleLayer(
                fgu_tau_ratio=rgl_tau_ratio,
                fgu_initializer=rgl_initializer,
                fgu_l2=rgl_l2,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(edcn_layer_num+1)
        ]
        self.bdg_fn_list = [
            BridgeModuleLayer(
                bdg_mode=bdg_mode,
                bdg_layer_num=bdg_layer_num,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(edcn_layer_num+1)
        ]
        self.cross_fn_list = [
            CrossNetworkLayer(
                dcn_layer_num=1,
                dcn_version=dcn_version,
                dcn_l2=dcn_l2,
                dcn_initializer=dcn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(edcn_layer_num*2)
        ]

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        _, self.fields_width, self.embed_dim = input_shape
        self.inputs_width = int(self.fields_width * self.embed_dim)

        self.dnn_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=[self.inputs_width],
                dnn_activation=self.dnn_activation,
                dnn_dropout=self.dnn_dropout,
                dnn_if_bn=self.dnn_if_bn,
                dnn_l2=self.dnn_l2,
                dnn_initializer=self.dnn_initializer,
                seed=self.seed + i if isinstance(self.seed, int) else self.seed,
            ) for i in range(self.edcn_layer_num*2)
        ]
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_c = self.rgl_cross_fn_list[-1](x)  # last
        x_d = self.rgl_deep_fn_list[-1](x)  # last

        for i in range(self.edcn_layer_num):
            x_c = self.cross_fn_list[i*2](x_c)
            x_d = self.dnn_fn_list[i*2](x_d)

            x_bdg = self.bdg_fn_list[i]([x_c, x_d])
            x = tf.reshape(x_bdg, shape=[-1, self.fields_width, self.embed_dim])

            x_c = self.rgl_cross_fn_list[i](x)
            x_d = self.rgl_deep_fn_list[i](x)
            x_c = self.cross_fn_list[i*2+1](x_c)
            x_d = self.dnn_fn_list[i*2+1](x_d)

        x_bdg = self.bdg_fn_list[-1]([x_c, x_d])  # last
        x = tf.concat([x_c, x_d, x_bdg], axis=1)
        return x


class AllFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 seed=None):
        super().__init__()
        self.fbi_fn = AllFieldBinaryInteractionLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            seed=seed,
        )
        self.linear_fn = LinearLayer(
            linear_initializer=linear_initializer,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank not in (2, 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        lr_outputs = self.linear_fn(dense_2d_tensor)
        fbi_outputs = self.fbi_fn(embed_3d_tensor)
        fm_outputs = tf.concat([lr_outputs, fbi_outputs], axis=-1)
        return fm_outputs


class FactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='FM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode not in ('FM', 'FM3D'):
            raise MLGBError


class FieldFactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'FFM':
            raise MLGBError


class HigherOrderFactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='HOFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            fbi_hofm_order, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'HOFM':
            raise MLGBError


class FieldWeightedFactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='FwFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'FwFM':
            raise MLGBError


class FieldEmbeddedFactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='FEFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'FEFM':
            raise MLGBError


class FieldVectorizedFactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='FvFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'FvFM':
            raise MLGBError


class FieldMatrixedFactorizationMachineLayer(AllFactorizationMachineLayer):
    def __init__(self, fbi_mode='FmFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None, seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, 'relu', 0.0,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'FmFM':
            raise MLGBError


class AttentionalFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='AFM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 seed=None):
        super().__init__(
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            3, fbi_afm_activation, fbi_afm_dropout,
            linear_l1, linear_l2, linear_initializer, seed,
        )
        if fbi_mode != 'AFM':
            raise MLGBError


class LorentzFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, lfm_beta=1.0, fbi_l2=0.0, fbi_initializer=None, seed=None):
        super().__init__()
        self.lfm_beta = lfm_beta
        self.fbi_l2 = fbi_l2
        self.fbi_initializer = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
            seed=seed,
        ).get()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.fields_width = input_shape[1]
        self.product_width = int(self.fields_width * (self.fields_width-1) // 2)
        self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T

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

        u_i = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 0], dtype=tf.int32), axis=1)
        v_i = tf.gather(params=x, indices=tf.constant(self.ij_ids[:, 1], dtype=tf.int32), axis=1)
        u_0 = self.lorentzian_norm_x(u_i)
        v_0 = self.lorentzian_norm_x(v_i)
        uv_lip = self.lorentzian_inner_product(u_i, v_i, u_0, v_0)

        x = self.triangle_pooling(uv_lip, u_0, v_0)
        return x

    def lorentzian_norm_x(self, x):
        x = tf.square(tf.reduce_sum(tf.pow(x, 2), axis=2, keepdims=False) + self.lfm_beta)
        return x

    def lorentzian_inner_product(self, u_i, v_i, u_0, v_0):
        x = tf.reduce_sum(u_i * v_i, axis=2, keepdims=False) - u_0 * v_0
        return x

    def triangle_pooling(self, uv_lip, u_0, v_0):
        x = (1 - uv_lip - u_0 - v_0) / (u_0 * v_0)  # (batch_size, product_width)
        return x


class InteractionMachineLayer(tf.keras.layers.Layer):
    def __init__(self, im_mode='IM', im_order=3, im_l2=0.0, im_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if im_mode not in ('IM', 'DeepIM'):
            raise MLGBError
        if not (1 <= im_order <= 5):
            raise MLGBError

        self.im_mode = im_mode
        self.im_order = im_order
        self.im_l2 = im_l2
        self.im_initializer = InitializerLayer(
            initializer=im_initializer,
            activation=None,
            seed=seed,
        ).get()
        if self.im_mode == 'DeepIM':
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
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.im_weight = self.add_weight(
            name='im_weight',
            shape=input_shape.as_list()[1:],
            initializer=self.im_initializer,
            regularizer=L1L2(0.0, self.im_l2),
            trainable=True,
        )
        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_3d = inputs
        x_3d = x_3d * self.im_weight

        p = self.get_x_list(x_3d, n=self.im_order)
        x_pool = [self.order_fn(p, i) for i in range(self.im_order)]
        x_im = tf.concat(x_pool, axis=1)  # (b, order * e)

        if self.im_mode == 'DeepIM':
            x_2d = self.flatten_fn(x_3d)
            x_dnn = self.dnn_fn(x_2d)
            x_im = tf.concat([x_im, x_dnn], axis=1)
        return x_im

    def order_fn(self, p, i):
        order_fn_list = [
            lambda p: p[0],
            lambda p: (p[0] ** 2 - p[1]) / 2,
            lambda p: (p[0] ** 3 - 3 * p[0] * p[1] + 2 * p[2]) / 6,
            lambda p: (p[0] ** 4 - 6 * p[0] ** 2 * p[1] + 3 * p[1] ** 2 + 8 * p[0] * p[2] - 6 * p[3]) / 24,
            lambda p: (p[0] ** 5 - 10 * p[0] ** 3 * p[1] + 20 * p[0] ** 2 * p[2] - 30 * p[0] * p[3]
                       - 20 * p[1] * p[2] + 15 * p[0] * p[1] ** 2 + 24 * p[4]) / 120,
        ]
        return order_fn_list[i](p)

    def get_x_list(self, x, n):
        x = tf.stack([x] * n, axis=1)
        x = tf.math.cumprod(x, axis=1)
        x = tf.reduce_sum(x, axis=2, keepdims=False)
        x_list = tf.unstack(x, n, axis=1)  # (b, e)
        return x_list


class AllInputFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, ifm_mode_if_dual=False,
                 fen_hidden_units=(64, 32), fen_activation='relu', fen_dropout=0.0,
                 fen_if_bn=False, fen_if_ln=False, fen_l2=0.0, fen_initializer=None,
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=False, trm_mha_l2=0.0, trm_mha_initializer=None, trm_residual_dropout=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 seed=None):
        super().__init__()
        if fbi_mode == 'FM':
            raise MLGBError

        self.ifm_mode_if_dual = ifm_mode_if_dual

        if self.ifm_mode_if_dual:
            self.trm_fn = InteractingLayer(
                trm_mha_head_num=trm_mha_head_num,
                trm_mha_head_dim=trm_mha_head_dim,
                trm_mha_if_mask=trm_mha_if_mask,
                trm_mha_l2=trm_mha_l2,
                trm_mha_initializer=trm_mha_initializer,
                trm_residual_dropout=trm_residual_dropout,
                seed=seed,
            )
        self.fen_fn = FactorEstimatingNetworkLayer(
            ifm_mode_if_dual=ifm_mode_if_dual,
            dnn_hidden_units=fen_hidden_units,
            dnn_activation=fen_activation,
            dnn_dropout=fen_dropout,
            dnn_if_bn=fen_if_bn,
            dnn_if_ln=fen_if_ln,
            dnn_l2=fen_l2,
            dnn_initializer=fen_initializer,
            seed=seed,
        )
        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        w_bit = self.fen_fn(embed_3d_tensor)  # (b, f, 1)
        if self.ifm_mode_if_dual:
            w_vec = self.trm_fn([embed_3d_tensor, embed_3d_tensor])  # (b, f, e)
            w_ifm = w_vec + w_bit
        else:
            w_ifm = w_bit

        embed_3d_tensor = embed_3d_tensor * w_ifm
        ifm_outputs = self.fm_fn([dense_2d_tensor, embed_3d_tensor])
        return ifm_outputs


class InputFactorizationMachineLayer(AllInputFactorizationMachineLayer):
    def __init__(self, ifm_mode_if_dual=False,
                 fen_hidden_units=(64, 32), fen_activation='relu', fen_dropout=0.0,
                 fen_if_bn=False, fen_if_ln=False, fen_l2=0.0, fen_initializer=None,
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=False, trm_mha_l2=0.0, trm_mha_initializer=None, trm_residual_dropout=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 seed=None):
        super().__init__(
            ifm_mode_if_dual,
            fen_hidden_units, fen_activation, fen_dropout,
            fen_if_bn, fen_if_ln, fen_l2, fen_initializer,
            trm_mha_head_num, trm_mha_head_dim,
            trm_mha_if_mask, trm_mha_l2, trm_mha_initializer, trm_residual_dropout,
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            fbi_hofm_order, fbi_afm_activation, fbi_afm_dropout,
            linear_l1, linear_l2, linear_initializer,
            seed,
        )
        if ifm_mode_if_dual:
            raise MLGBError


class DualInputFactorizationMachineLayer(AllInputFactorizationMachineLayer):
    def __init__(self, ifm_mode_if_dual=True,
                 fen_hidden_units=(64, 32), fen_activation='relu', fen_dropout=0.0,
                 fen_if_bn=False, fen_if_ln=False, fen_l2=0.0, fen_initializer=None,
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=False, trm_mha_l2=0.0, trm_mha_initializer=None, trm_residual_dropout=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 seed=None):
        super().__init__(
            ifm_mode_if_dual,
            fen_hidden_units, fen_activation, fen_dropout,
            fen_if_bn, fen_if_ln, fen_l2, fen_initializer,
            trm_mha_head_num, trm_mha_head_dim,
            trm_mha_if_mask, trm_mha_l2, trm_mha_initializer, trm_residual_dropout,
            fbi_mode, fbi_unit, fbi_l2, fbi_initializer,
            fbi_hofm_order, fbi_afm_activation, fbi_afm_dropout,
            linear_l1, linear_l2, linear_initializer,
            seed,
        )
        if not ifm_mode_if_dual:
            raise MLGBError


class FactorizationMachineNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FM', fbi_unit=16, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='tanh', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
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

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank not in (2, 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        linear_tensor, embed_3d_tensor = inputs

        fm_outputs = self.fm_fn([linear_tensor, embed_3d_tensor])
        fnn_outputs = self.dnn_fn(fm_outputs)
        return fnn_outputs


class ProductNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='PNN:both', fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.dnn_length = len(dnn_hidden_units)
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_dropouts = [dnn_dropout] * self.dnn_length if isinstance(dnn_dropout, float) else dnn_dropout
        self.dnn_bns = [dnn_if_bn] * self.dnn_length if isinstance(dnn_if_bn, bool) else dnn_if_bn
        self.dnn_lns = [dnn_if_ln] * self.dnn_length if isinstance(dnn_if_ln, bool) else dnn_if_ln

        self.dnn_first_layer = [
            BatchNormalization(axis=1) if self.dnn_bns[0] else IdentityLayer(),
            LayerNormalization(axis=1) if self.dnn_lns[0] else IdentityLayer(),
            Dropout(self.dnn_dropouts[0]),
            ActivationLayer(dnn_activation),
        ]
        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=self.dnn_hidden_units[0],
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=self.dnn_hidden_units[1:],
            dnn_dropout=self.dnn_dropouts[1:],
            dnn_if_bn=self.dnn_bns[1:],
            dnn_if_ln=self.dnn_lns[1:],
            dnn_activation=dnn_activation,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        x = self.fm_fn([dense_2d_tensor, embed_3d_tensor])
        for fn in self.dnn_first_layer:
            x = fn(x)
        x = self.dnn_fn(x)
        return x


class ProductNetworkInNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, pin_parallel_num=4,
                 fbi_mode='PNN:both', fbi_unit=16, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=True,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if fbi_mode not in FBIModeList:
            raise MLGBError

        self.pin_parallel_num = pin_parallel_num

        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )
        self.dnn_parallel_fn = DNN3dParallelLayer(
            dnn_if_output2d=True,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,  #
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        x = self.fm_fn([dense_2d_tensor, embed_3d_tensor])
        x = tf.stack([x] * self.pin_parallel_num, axis=1)
        x = self.dnn_parallel_fn(x)
        return x


class OperationNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FFM', fbi_unit=16, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=True, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.fbi_fn = AllFieldBinaryInteractionLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            seed=seed,
        )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_activation=dnn_activation,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        embed_3d_tensor = inputs
        embed_2d_tensor = self.flatten_fn(embed_3d_tensor)

        fbi_outputs = self.fbi_fn(embed_3d_tensor)
        ffm_outputs = tf.concat([fbi_outputs, embed_2d_tensor], axis=1)
        nffm_outputs = self.dnn_fn(ffm_outputs)
        return nffm_outputs


class AdaptiveFactorizationNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, afn_mode_if_ensemble=True,
                 ltl_clip_min=1e-4, ltl_unit=32, ltl_l2=0.0, ltl_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=True, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 ensemble_dnn_hidden_units=(64, 32), ensemble_dnn_activation='relu', ensemble_dnn_dropout=0.0,
                 ensemble_dnn_if_bn=True, ensemble_dnn_if_ln=False, ensemble_dnn_l2=0.0, ensemble_dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.afn_mode_if_ensemble = afn_mode_if_ensemble

        self.ltl_fn = LogarithmicTransformationLayer(
            ltl_clip_min=ltl_clip_min,
            ltl_unit=ltl_unit,
            ltl_l2=ltl_l2,
            ltl_initializer=ltl_initializer,
            seed=seed,
        )
        self.dnn_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_activation=dnn_activation,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        if self.afn_mode_if_ensemble:
            self.ensemble_dnn_fn = DeepNeuralNetworkLayer(
                dnn_hidden_units=ensemble_dnn_hidden_units,
                dnn_dropout=ensemble_dnn_dropout,
                dnn_if_bn=ensemble_dnn_if_bn,
                dnn_if_ln=ensemble_dnn_if_ln,
                dnn_activation=ensemble_dnn_activation,
                dnn_l2=ensemble_dnn_l2,
                dnn_initializer=ensemble_dnn_initializer,
                seed=seed,
            )
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        embed_3d_tensor = inputs
        embed_2d_tensor = self.flatten_fn(embed_3d_tensor)

        ltl_outputs = self.ltl_fn(embed_3d_tensor)
        afn_outputs = self.dnn_fn(ltl_outputs)

        if self.afn_mode_if_ensemble:
            dnn_outputs = self.ensemble_dnn_fn(embed_2d_tensor)
            afn_outputs = tf.concat([afn_outputs, dnn_outputs], axis=1)
        return afn_outputs


class NeuralFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FM3D', fbi_unit=16, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.linear_fn = LinearLayer(
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
            seed=seed,
        )
        self.fbi_fn = AllFieldBinaryInteractionLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        linear_outputs = self.linear_fn(dense_2d_tensor)
        fbi_tensor = self.fbi_fn(embed_3d_tensor)
        dnn_outputs = self.dnn_fn(fbi_tensor)
        nfm_outputs = tf.concat([linear_outputs, dnn_outputs], axis=1)
        return nfm_outputs


class WideDeepLearningLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.fbi_fn = AllFieldBinaryInteractionLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank not in (2, 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        wide_outputs = self.fbi_fn(embed_3d_tensor)
        deep_outputs = self.dnn_fn(dense_2d_tensor)
        wdl_outputs = tf.concat([wide_outputs, deep_outputs], axis=1)
        return wdl_outputs


class DeepFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FM', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            linear_initializer=linear_initializer,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
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

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank not in (2, 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        fm_outputs = self.fm_fn([dense_2d_tensor, embed_3d_tensor])
        deep_outputs = self.dnn_fn(dense_2d_tensor)
        dfm_outputs = tf.concat([fm_outputs, deep_outputs], axis=1)
        return dfm_outputs


class DeepFieldEmbeddedFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, fbi_mode='FEFM', fbi_unit=1, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
            linear_initializer=linear_initializer,
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

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank not in (2, 3):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        fm_outputs = self.fm_fn([dense_2d_tensor, embed_3d_tensor])
        dnn_outputs = self.dnn_fn(fm_outputs)
        deepfefm_outputs = tf.concat([fm_outputs, dnn_outputs], axis=1)
        return deepfefm_outputs


class FieldLeveragedEmbeddingNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, flen_group_indices=((0, 1), (2, 3), (4, 5, 6)),
                 fbi_fm_mode='FM3D', fbi_mf_mode='FwFM', fbi_unit=16, fbi_l2=0.0, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.linear_fn = LinearLayer(
            linear_initializer=linear_initializer,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
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
        self.fwbi_fn = GroupedAllFieldWiseBinaryInteractionLayer(
            group_indices=flen_group_indices,
            fbi_fm_mode=fbi_fm_mode,
            fbi_mf_mode=fbi_mf_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            seed=seed,
        )

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        fwbi_outputs = self.fwbi_fn(embed_3d_tensor)
        dnn_ouputs = self.dnn_fn(dense_2d_tensor)
        linear_outputs = self.linear_fn(dense_2d_tensor)
        flen_outputs = tf.concat([fwbi_outputs, dnn_ouputs, linear_outputs], axis=1)
        return flen_outputs


class ConvolutionalClickPredictionModelLayer(tf.keras.layers.Layer):
    def __init__(self, cnn_filter_nums=(64, 32), cnn_kernel_sizes=(64, 32), cnn_activation='tanh',
                 cnn_l2=0.0, cnn_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.p_min = 3
        if min(cnn_filter_nums) < self.p_min or min(cnn_kernel_sizes) < self.p_min:
            raise MLGBError

        self.cnn_length = len(cnn_filter_nums)

        self.cnn_fn_list = [
            ConvolutionalNeuralNetworkLayer(
                cnn_conv_mode='Conv2D',
                cnn_filter_nums=[cnn_filter_nums[i]],
                cnn_kernel_heights=[cnn_kernel_sizes[i]],
                cnn_kernel_widths=1,
                cnn_activation=cnn_activation,
                cnn_l2=cnn_l2,
                cnn_initializer=cnn_initializer,
                cnn_if_max_pool=False,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(self.cnn_length)
        ]
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
        self.k_max_pooling_fn = KMaxPoolingLayer(pool_axis=1)
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 3:
            raise MLGBError
        if input_shape[1] < self.p_min:  # fields_width
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x = tf.expand_dims(x, axis=3)  # (b, f, e, 1) == (b, h, w, i_c)
        for i in range(self.cnn_length):
            x = self.cnn_fn_list[i](x)  # (b, f, e, o_c)
            p = self.flexible_p(i=i+1, j=self.cnn_length, n=x.shape[1])  # n = flexible features width
            x = self.k_max_pooling_fn(x, k=p)

        x = self.flatten_fn(x)
        x = self.dnn_fn(x)
        return x

    def flexible_p(self, i, j, n):
        if i < 1 or j < 1 or i > j:
            raise MLGBError

        p = max(int((1 - (i / j) ** (j - i)) * n), self.p_min)
        return p


class FeatureGenerationByConvolutionalNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, cnn_filter_nums=(64, 32), cnn_kernel_sizes=(64, 32), cnn_pool_sizes=(2, 2),
                 cnn_activation='tanh', cnn_l2=0.0, cnn_initializer=None,
                 recomb_dnn_hidden_units=(64, 32), recomb_dnn_activation='tanh',
                 fbi_mode='PNN:inner_product', fbi_unit=32, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.cnn_length = len(cnn_filter_nums)
        if not (self.cnn_length == len(cnn_kernel_sizes) == len(cnn_pool_sizes) == len(recomb_dnn_hidden_units)):
            raise MLGBError('self.cnn_length == len(cnn_kernel_sizes) == len(cnn_pool_sizes) == len(recomb_dnn_units)')
        if fbi_mode == 'FM':
            raise MLGBError

        self.cpr_fn_list = [
            [
                ConvolutionalNeuralNetworkLayer(
                    cnn_conv_mode='Conv2D',
                    cnn_filter_nums=[cnn_filter_nums[i]],
                    cnn_kernel_heights=[cnn_kernel_sizes[i]],
                    cnn_kernel_widths=1,
                    cnn_activation=cnn_activation,
                    cnn_l2=cnn_l2,
                    cnn_initializer=cnn_initializer,
                    cnn_if_max_pool=False,
                    seed=seed + i if isinstance(seed, int) else seed,
                ),  # (b, f, e, 1) -> (b, (f - p + 1) // p, e, cnn_filter_num)
                FlattenAxesLayer(axes=[1, 3]),  # (b, cnn_filter_num * (f - p + 1) // p, e)
                TransposeLayer(perm=[0, 2, 1]),
                DNN3dParallelLayer(
                    dnn_hidden_units=[recomb_dnn_hidden_units[i]],
                    dnn_activation=recomb_dnn_activation,
                    dnn_if_output2d=False,
                    seed=seed + i if isinstance(seed, int) else seed,
                ),
                TransposeLayer(perm=[0, 2, 1]),  # (b, recomb_dnn_hidden_unit, e)
            ]
            for i in range(self.cnn_length)
        ]
        self.fbi_fn = AllFieldBinaryInteractionLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_l2=fbi_l2,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        cnn_3d_outputs = self.cpr_fn(embed_3d_tensor)
        cnn_2d_outputs = self.flatten_fn(cnn_3d_outputs)

        all_3d_features = tf.concat([embed_3d_tensor, cnn_3d_outputs], axis=1)
        fbi_outputs = self.fbi_fn(all_3d_features)

        all_dense_features = tf.concat([dense_2d_tensor, cnn_2d_outputs, fbi_outputs], axis=1)
        fgcnn_outputs = self.dnn_fn(all_dense_features)
        return fgcnn_outputs

    def cpr_fn(self, x):  # Conv_Pool_Recombination
        x_list = []
        for cpr_fn in self.cpr_fn_list:
            x_i = tf.expand_dims(x, axis=3)
            for _fn in cpr_fn:
                x_i = _fn(x_i)
            x_list.append(x_i)
        x = tf.concat(x_list, axis=1)
        return x


class ExtremeDeepFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, cin_interaction_num=4, cin_interaction_ratio=1.0,
                 cnn_filter_num=64, cnn_kernel_size=64, cnn_activation='relu',
                 cnn_l2=0.0, cnn_initializer=None,
                 linear_l1=0.0, linear_l2=0.0, linear_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if not (0.5 <= cin_interaction_ratio <= 1.0):
            raise MLGBError('0.5 <= cin_interaction_ratio <= 1.0')

        self.cin_fn = CompressedInteractionNetworkLayer(
            cin_interaction_num=cin_interaction_num,
            cin_interaction_ratio=cin_interaction_ratio,
            cnn_filter_num=cnn_filter_num,
            cnn_kernel_size=cnn_kernel_size,
            cnn_activation=cnn_activation,
            cnn_l2=cnn_l2,
            cnn_initializer=cnn_initializer,
            seed=seed,
        )
        self.linear_fn = LinearLayer(
            linear_initializer=linear_initializer,
            linear_l1=linear_l1,
            linear_l2=linear_l2,
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

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_3d_tensor = inputs

        linear_outputs = self.linear_fn(dense_2d_tensor)
        dnn_outputs = self.dnn_fn(dense_2d_tensor)
        cin_outputs = self.cin_fn(embed_3d_tensor)
        xdeepfm_outputs = tf.concat([linear_outputs, cin_outputs, dnn_outputs], axis=1)
        return xdeepfm_outputs


class FeatureImportanceBilinearInteractionNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, sen_pool_mode='Pooling:average', sen_reduction_factor=2, sen_activation='relu',
                 sen_l2=0.0, sen_initializer=None,
                 fbi_mode='Bilinear:field_interaction', fbi_unit=16, fbi_l2=0.0, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.sen_fn = SqueezeExcitationNetworkLayer(
            sen_pool_mode=sen_pool_mode,
            sen_reduction_factor=sen_reduction_factor,
            sen_activation=sen_activation,
            sen_l2=sen_l2,
            sen_initializer=sen_initializer,
            seed=seed,
        )
        self.fbi_fn_list = [
            AllFieldBinaryInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_l2=fbi_l2,
                fbi_initializer=fbi_initializer,
                fbi_hofm_order=fbi_hofm_order,
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(2)
        ]
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
        if len(input_shape) != 2:
            raise MLGBError
        if input_shape[0].rank != 2:
            raise MLGBError
        if input_shape[1].rank != 3:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_tensor, embed_3d_tensor = inputs

        fbi_tensor = self.fbi_fn_list[0](embed_3d_tensor)
        sen_tensor = self.sen_fn(embed_3d_tensor)
        fbi_sen_tensor = self.fbi_fn_list[1](sen_tensor)
        dnn_tensor = tf.concat([dense_tensor, fbi_tensor, fbi_sen_tensor], axis=1)

        fbn_outputs = self.dnn_fn(dnn_tensor)
        return fbn_outputs


class AutomaticFeatureInteractionLearningLayer(tf.keras.layers.Layer):
    def __init__(self, trm_layer_num=1, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_l2=0.0, trm_mha_initializer=None,
                 trm_residual_dropout=0.0, seed=None):
        super().__init__()
        self.trm_layer_num = trm_layer_num

        self.trm_fn_list = [
            InteractingLayer(
                trm_mha_head_num=trm_mha_head_num,
                trm_mha_head_dim=trm_mha_head_dim,
                trm_mha_if_mask=trm_mha_if_mask,
                trm_mha_l2=trm_mha_l2,
                trm_mha_initializer=trm_mha_initializer,
                trm_residual_dropout=trm_residual_dropout,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(trm_layer_num)
        ]
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == 3):
            raise MLGBError
        if input_shape[0] != input_shape[1]:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_q, x_k = inputs

        x = self.trm_fn_list[0]([x_q, x_k])
        if self.trm_layer_num > 1:
            for trm_fn in self.trm_fn_list[1:]:
                x = trm_fn([x, x])

        x = self.flatten_fn(x)
        return x






