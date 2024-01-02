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
from mlgb.tf.functions import (
    IdentityLayer,
    MaskLayer,
    FlattenAxesLayer,
)
from mlgb.tf.components.linears import (
    DeepNeuralNetworkLayer,
    DNN3dParallelLayer,
    FeedForwardNetworkLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'SharedBottomLayer',
    'EntireSpaceMultitaskModelLayer',
    'MultigateMixtureOfExpertLayer',
    'ProgressiveLayeredExtractionLayer',
    'ParameterEmbeddingPersonalizedNetworkLayer',
]


class SharedBottomHardLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list,
                 hard_bottom_dnn_hidden_units=(64, 32), hard_tower_dnn_hidden_units=(32,),
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.task_fn_list = task_fn_list
        self.task_num = len(task_fn_list)
        hard_tower_dnn_hidden_units = hard_tower_dnn_hidden_units + (1,)

        self.dnn_bottom_fn = DeepNeuralNetworkLayer(
            dnn_hidden_units=hard_bottom_dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            seed=seed,
        )
        self.dnn_tower_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=hard_tower_dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i + 1 if isinstance(seed, int) else seed,
            )
            for i in range(self.task_num)
        ]

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.dnn_bottom_fn(x)

        outputs = []
        for i in range(self.task_num):
            x_tower = self.dnn_tower_fn_list[i](x)
            x_tower = self.task_fn_list[i](x_tower)
            outputs.append(x_tower)
        return outputs


class SharedBottomSoftLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list,
                 soft_dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=True, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.task_fn_list = task_fn_list
        self.task_num = len(task_fn_list)
        soft_dnn_hidden_units = soft_dnn_hidden_units + (1,)

        self.dnn_parallel_fn = DNN3dParallelLayer(
            dnn_hidden_units=soft_dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            dnn_if_output2d=False,
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

        x = tf.stack([x] * self.task_num, axis=1)  # (b, task_num, f)
        x = self.dnn_parallel_fn(x)  # (b, task_num, dnn_unit)

        outputs = tf.unstack(x, num=self.task_num, axis=1)
        outputs = [self.task_fn_list[i](outputs[i]) for i in range(self.task_num)]
        return outputs


class SharedBottomLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list, share_mode='SB:hard',
                 hard_bottom_dnn_hidden_units=(64, 32), hard_tower_dnn_hidden_units=(32,), soft_dnn_hidden_units=(64, 32),
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if share_mode not in ('SB:hard', 'SB:soft'):
            raise MLGBError

        if share_mode == 'SB:hard':
            self.sb_fn = SharedBottomHardLayer(
                task_fn_list=task_fn_list,
                hard_bottom_dnn_hidden_units=hard_bottom_dnn_hidden_units,
                hard_tower_dnn_hidden_units=hard_tower_dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed,
            )
        else:
            self.sb_fn = SharedBottomSoftLayer(
                task_fn_list=task_fn_list,
                soft_dnn_hidden_units=soft_dnn_hidden_units,
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
        outputs = self.sb_fn(x)
        return outputs


class EntireSpaceMultitaskModelLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list, dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.task_fn_list = task_fn_list
        self.task_num = len(task_fn_list)
        self.dnn_hidden_units = dnn_hidden_units + (1,)

        self.dnn_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=self.dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(self.task_num)
        ]

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs

        x_ctr = self.dnn_fn_list[0](x)
        x_ctr = self.task_fn_list[0](x_ctr)
        x_cvr = self.dnn_fn_list[1](x)
        x_cvr = self.task_fn_list[1](x_cvr)

        x_ctcvr = x_ctr * x_cvr
        outputs = [x_ctr, x_ctcvr]
        return outputs


class MultigateMixtureOfExpertLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list, expert_num=8,
                 expert_dnn_hidden_units=(64, 32), gate_dnn_hidden_units=(32,), tower_dnn_hidden_units=(32,),
                 dnn_activation='selu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False, dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.expert_num = expert_num
        self.task_fn_list = task_fn_list
        self.task_num = len(task_fn_list)
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units + (1,)

        self.dnn_experts_fn = DNN3dParallelLayer(
            dnn_hidden_units=expert_dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_l2=dnn_l2,
            dnn_initializer=dnn_initializer,
            dnn_if_output2d=False,
            seed=seed,
        )
        if self.gate_dnn_hidden_units:
            self.dnn_gate_in_fn_list = [
                DeepNeuralNetworkLayer(
                    dnn_hidden_units=gate_dnn_hidden_units,
                    dnn_activation=dnn_activation,
                    dnn_dropout=dnn_dropout,
                    dnn_if_bn=dnn_if_bn,
                    dnn_if_ln=dnn_if_ln,
                    dnn_l2=dnn_l2,
                    dnn_initializer=dnn_initializer,
                    seed=seed + i if isinstance(seed, int) else seed,
                ) for i in range(self.task_num)
            ]
        self.dnn_gate_out_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=(expert_num,),
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(self.task_num)
        ]
        self.dnn_tower_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=self.tower_dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            ) for i in range(self.task_num)
        ]
        self.mask_fn = MaskLayer(att_if_mask=True)
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_experts = tf.stack([x] * self.expert_num, axis=1)
        x_experts = self.dnn_experts_fn(x_experts)  # (b, experts_num, unit)

        outputs = []
        for i in range(self.task_num):
            x_gate = self.dnn_gate_in_fn_list[i](x) if self.gate_dnn_hidden_units else x
            x_gate = self.dnn_gate_out_fn_list[i](x_gate)
            x_gate = tf.expand_dims(x_gate, axis=2)  # (b, experts_num, 1)
            x_gate = self.mask_fn(x_gate)
            x_gate = tf.nn.softmax(x_gate, axis=1)

            x_tower_in = x_experts * x_gate  #
            x_tower_in = self.flatten_fn(x_tower_in)
            x_tower_out = self.dnn_tower_fn_list[i](x_tower_in)
            x_tower_out = self.task_fn_list[i](x_tower_out)
            outputs.append(x_tower_out)
        return outputs


class ProgressiveLayeredExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list, tower_dnn_hidden_units=(64, 32),
                 expert_men_num=4, shared_men_num=4, men_hidden_units=(64, 32),
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        self.task_fn_list = task_fn_list
        self.task_num = len(task_fn_list)
        self.expert_men_num = expert_men_num
        self.shared_men_num = shared_men_num
        self.men_length = len(men_hidden_units)
        self.tower_dnn_hidden_units = tower_dnn_hidden_units + (1,)

        self.men_fn_list = [
            [
                DNN3dParallelLayer(
                    dnn_hidden_units=[men_hidden_units[i]],
                    dnn_activation=dnn_activation,
                    dnn_dropout=dnn_dropout,
                    dnn_if_bn=dnn_if_bn,
                    dnn_if_ln=dnn_if_ln,
                    dnn_l2=dnn_l2,
                    dnn_initializer=dnn_initializer,
                    seed=seed + i + j if isinstance(seed, int) else seed,
                )
                for i in range(self.men_length)
            ]
            for j in range(self.task_num + 1)  # e + s
        ]
        self.cgc_ffn_fn_list = [
            [
                FeedForwardNetworkLayer(
                    ffn_linear_if_twice=False,
                    ffn_if_bias=True,
                    ffn_activation=dnn_activation,
                    ffn_if_bn=dnn_if_bn,
                    ffn_if_ln=dnn_if_ln,
                    ffn_dropout=dnn_dropout,
                    ffn_l2=dnn_l2,
                    ffn_initializer=dnn_initializer,
                    seed=seed + i + j if isinstance(seed, int) else seed,
                )
                for i in range(self.men_length)
            ]
            for j in range(self.task_num + 1)  # e + s
        ]
        self.tower_dnn_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=self.tower_dnn_hidden_units,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(self.task_num)
        ]
        self.flatten_fn_list = [Flatten() for _ in range(self.task_num)]

    def build(self, input_shape):
        if input_shape.rank != 2:
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x = inputs
        x_s = tf.stack([x] * self.shared_men_num, axis=1)
        x_e_list = [tf.stack([x] * self.expert_men_num, axis=1)] * self.task_num

        for i in range(self.men_length):
            x_s = self.men_fn_list[-1][i](x_s)
            for j in range(self.task_num):
                x_e = self.men_fn_list[j][i](x_e_list[j])
                x_e_list[j] = self.cgc_e_fn(x_e, x_s, i, j)
            x_s = self.cgc_s_fn(x_e_list, i, -1)

        outputs = []
        for i in range(self.task_num):
            x_i = self.flatten_fn_list[i](x_e_list[i])
            x_i = self.tower_dnn_fn_list[i](x_i)
            x_i = self.task_fn_list[i](x_i)
            outputs.append(x_i)
        return outputs

    def cgc_e_fn(self, x_e, x_s, men_i, task_j):  # CustomizedGateControl
        x = tf.concat([x_e, x_s], axis=1)  # (b, e+s, f == men_unit)
        w = self.cgc_ffn_fn_list[task_j][men_i](x)
        w = tf.nn.softmax(w, axis=2)
        x = x * w
        return x

    def cgc_s_fn(self, x_e_list, men_i, task_j):
        x = tf.concat(x_e_list, axis=1)  # (b, e * task_num, f == men_unit), e = e + s
        w = self.cgc_ffn_fn_list[task_j][men_i](x)
        w = tf.nn.softmax(w, axis=2)
        x = x * w
        return x


class ParameterEmbeddingPersonalizedNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, task_fn_list, domain_num=1, tower_dnn_hidden_units=(64, 32),
                 gnu_factor=2.0, gnu_last_activation='sigmoid', gnu_if_concat_general_inputs=True,
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 dnn_l2=0.0, dnn_initializer=None,
                 seed=None):
        super().__init__()
        if not domain_num > 0:
            raise MLGBError
        if not gnu_factor > 0:
            raise MLGBError

        self.task_fn_list = task_fn_list
        self.task_num = len(task_fn_list)
        self.tower_dnn_hidden_units = tower_dnn_hidden_units + (domain_num,)
        self.tower_dnn_length = len(self.tower_dnn_hidden_units)
        self.domain_num = domain_num
        self.gnu_factor = gnu_factor
        self.gnu_if_concat_general_inputs = gnu_if_concat_general_inputs

        self.gnu_0_fn = FeedForwardNetworkLayer(
            ffn_linear_if_twice=True,
            ffn_activation=dnn_activation,
            ffn_last_activation=gnu_last_activation,
            ffn_last_factor=gnu_factor,
        )
        self.gnu_n1_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=[self.tower_dnn_hidden_units[i]],
                dnn_activation=dnn_activation,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_dropout=dnn_dropout,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(self.tower_dnn_length)
        ]
        self.gnu_n2_fn_list = [
            DeepNeuralNetworkLayer(
                dnn_hidden_units=[self.tower_dnn_hidden_units[i]],
                dnn_activation=gnu_last_activation,
                dnn_if_bn=dnn_if_bn,
                dnn_if_ln=dnn_if_ln,
                dnn_dropout=dnn_dropout,
                dnn_l2=dnn_l2,
                dnn_initializer=dnn_initializer,
                seed=seed + i if isinstance(seed, int) else seed,
            )
            for i in range(self.tower_dnn_length)
        ]
        self.tower_dnn_fn_list = [
            [
                DeepNeuralNetworkLayer(
                    dnn_hidden_units=[self.tower_dnn_hidden_units[i]],
                    dnn_activation=dnn_activation,
                    dnn_if_bn=dnn_if_bn,
                    dnn_if_ln=dnn_if_ln,
                    dnn_dropout=dnn_dropout,
                    dnn_l2=dnn_l2,
                    dnn_initializer=dnn_initializer,
                    seed=seed + i + j if isinstance(seed, int) else seed,
                )
                for i in range(self.tower_dnn_length)
            ]
            for j in range(self.task_num)
        ]
        if self.gnu_if_concat_general_inputs:
            self.stop_gradient_fn_list = [IdentityLayer(if_stop_gradient=True) for _ in range(2)]
        self.flatten_fn_list = [Flatten() for _ in range(self.task_num)]
        self.flatten_axes_fn = FlattenAxesLayer(axes=[1, 2])

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise MLGBError
        if not (input_shape[0].rank == input_shape[1].rank == input_shape[2].rank == 3):
            raise MLGBError
        if not (input_shape[0][2] == input_shape[1][2] == input_shape[2][2]):  # e
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        x_g, x_d, x_uia = inputs # (general_f, domain_f, user_item_author_f)

        if self.gnu_if_concat_general_inputs:
            x_g_ep = self.stop_gradient_fn_list[0](x_g)  # EPNet
            x_d = tf.concat([x_d, x_g_ep], axis=1)

        x_d = self.gnu_0_fn(x_d)
        x_g = tf.einsum('bie,bje->bije', x_g, x_d)
        x_g = self.flatten_axes_fn(x_g)

        if self.gnu_if_concat_general_inputs:
            x_g_pp = self.stop_gradient_fn_list[1](x_g)  # PPNet
            x_uia = tf.concat([x_uia, x_g_pp], axis=1)

        x_g = self.flatten_fn_list[0](x_g)
        x_uia = self.flatten_fn_list[1](x_uia)

        x_tower_list = [x_g] * self.task_num
        for i in range(self.tower_dnn_length):
            x_gnu = self.gnu_n_fn(x_uia, i)
            for j in range(self.task_num):
                x_tower = self.tower_dnn_fn_list[j][i](x_tower_list[j])
                x_tower_list[j] = x_tower * x_gnu

        outputs = []
        for i in range(self.task_num):
            if self.domain_num == 1:
                x_i = self.task_fn_list[i](x_tower_list[i])
                outputs.append(x_i)
            else:
                domain_outputs = []
                for j in range(self.domain_num):
                    x_i = self.task_fn_list[i][j](x_tower_list[i])
                    domain_outputs.append(x_i)
                outputs.append(domain_outputs)
        return outputs

    def gnu_n_fn(self, x, i):
        x = self.gnu_n1_fn_list[i](x)
        x = self.gnu_n2_fn_list[i](x)
        x = x * self.gnu_factor
        return x


























