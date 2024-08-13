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

from mlgb.torch.configs import (
    torch,
    MVPoolModeList,
)
from mlgb.torch.functions import RegularizationLayer
from mlgb.torch.inputs import InputsLayer
from mlgb.torch.components.linears import TaskLayer
from mlgb.torch.modules.ranking.multitask import (
    SharedBottomLayer,
    EntireSpaceMultitaskModelLayer,
    MultigateMixtureOfExpertLayer,
    ProgressiveLayeredExtractionLayer,
    ParameterEmbeddingPersonalizedNetworkLayer,
)
from mlgb.error import MLGBError

__all__ = [
    'SharedBottom', 'ESMM', 'MMoE', 'PLE', 'PEPNet',
]


class SharedBottom(torch.nn.Module):
    def __init__(self, feature_names, task=('binary', 'binary'), device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 share_mode='SB:hard',
                 soft_dnn_hidden_units=(64, 32),
                 hard_bottom_dnn_hidden_units=(64, 32), hard_tower_dnn_hidden_units=(32,),
                 dnn_activation='selu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: SharedBottom(SharedBottom)
        Paper Team: Sebastian Ruder(InsightCentre)
        Paper Year: 2017
        Paper Name: <An Overview of Multi-Task Learning in Deep Neural Networks>
        Paper Link: https://arxiv.org/pdf/1706.05098.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: Tuple[str], default ('binary', 'binary').
            :param device: str, default 'cpu'.
            :param inputs_if_multivalued: bool, default False.
            :param inputs_if_sequential: bool, default False.
            :param inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_2d_dim: Optional[int], default None. When None, each field has own embed_dim by feature_names.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task Effective Parameters:
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param share_mode: str, default 'SB:hard'. Union['SB:hard', 'SB:soft']
            :param soft_dnn_hidden_units: Tuple[int], default (64, 32).
            :param hard_bottom_dnn_hidden_units: Tuple[int], default (64, 32).
            :param hard_tower_dnn_hidden_units: Tuple[int], default (32,).
            :param dnn_activation: Optional[str], default 'selu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        task_list = task
        self.task_num = len(task_list)

        if isinstance(task_list, str) or self.task_num < 2:
            raise MLGBError
        for _task in task_list:
            if _task not in ('binary', 'regression'):
                raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if share_mode not in ('SB:hard', 'SB:soft'):
            raise MLGBError

        self.input_fn = InputsLayer(
            feature_names=feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.task_fn_list = torch.nn.ModuleList([
            TaskLayer(
                task=task_list[i],
                task_linear_if_identity=True,
                device=device,
            )
            for i in range(self.task_num)
        ])
        self.sb_fn = SharedBottomLayer(
            task_fn_list=self.task_fn_list,
            share_mode=share_mode,
            hard_bottom_dnn_hidden_units=hard_bottom_dnn_hidden_units,
            hard_tower_dnn_hidden_units=hard_tower_dnn_hidden_units,
            soft_dnn_hidden_units=soft_dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        outputs = self.sb_fn(dense_2d_tensor)
        return outputs


class ESMM(torch.nn.Module):
    def __init__(self, feature_names, task=('binary', 'binary'), device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='selu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: ESMM(EntireSpaceMultitaskModel)
        Paper Team: Alibaba
        Paper Year: 2018
        Paper Name: <Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate>
        Paper Link: https://arxiv.org/pdf/1804.07931.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: Tuple[str], default ('binary', 'binary').
            :param device: str, default 'cpu'.
            :param inputs_if_multivalued: bool, default False.
            :param inputs_if_sequential: bool, default False.
            :param inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_2d_dim: Optional[int], default None. When None, each field has own embed_dim by feature_names.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task Effective Parameters:
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'selu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        task_list = task
        self.task_num = len(task_list)

        if isinstance(task_list, str) or self.task_num < 2:
            raise MLGBError
        for _task in task_list:
            if _task != 'binary':
                raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')

        self.input_fn = InputsLayer(
            feature_names=feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.task_fn_list = torch.nn.ModuleList([
            TaskLayer(
                task=task_list[i],
                task_linear_if_identity=True,
                device=device,
            )
            for i in range(self.task_num)
        ])
        self.esmm_fn = EntireSpaceMultitaskModelLayer(
            task_fn_list=self.task_fn_list,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        outputs = self.esmm_fn(dense_2d_tensor)
        return outputs


class MMoE(torch.nn.Module):
    def __init__(self, feature_names, task=('binary', 'binary'), device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 expert_num=4,
                 expert_dnn_hidden_units=(64, 32), gate_dnn_hidden_units=(32,), tower_dnn_hidden_units=(32,),
                 dnn_activation='selu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: MMoE(MultigateMixtureOfExpert)
        Paper Team: Google(Alphabet)
        Paper Year: 2018
        Paper Name: <Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts>
        Paper Link: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: Tuple[str], default ('binary', 'binary').
            :param device: str, default 'cpu'.
            :param inputs_if_multivalued: bool, default False.
            :param inputs_if_sequential: bool, default False.
            :param inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_2d_dim: Optional[int], default None. When None, each field has own embed_dim by feature_names.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task Effective Parameters:
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param expert_num: int, default 4.
            :param expert_dnn_hidden_units: Tuple[int], default (64, 32).
            :param gate_dnn_hidden_units: Tuple[int], default (32,).
            :param tower_dnn_hidden_units: Tuple[int], default (32,).
            :param dnn_activation: Optional[str], default 'selu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        task_list = task
        self.task_num = len(task_list)

        if isinstance(task_list, str) or self.task_num < 2:
            raise MLGBError
        for _task in task_list:
            if _task not in ('binary', 'regression'):
                raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')

        self.input_fn = InputsLayer(
            feature_names=feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.task_fn_list = torch.nn.ModuleList([
            TaskLayer(
                task=task_list[i],
                task_linear_if_identity=True,
                device=device,
            )
            for i in range(self.task_num)
        ])
        self.mmoe_fn = MultigateMixtureOfExpertLayer(
            task_fn_list=self.task_fn_list,
            expert_num=expert_num,
            expert_dnn_hidden_units=expert_dnn_hidden_units,
            gate_dnn_hidden_units=gate_dnn_hidden_units,
            tower_dnn_hidden_units=tower_dnn_hidden_units,
            dnn_activation=dnn_activation,  # dead relu.
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        outputs = self.mmoe_fn(dense_2d_tensor)
        return outputs


class PLE(torch.nn.Module):
    def __init__(self, feature_names, task=('binary', 'binary'), device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 tower_dnn_hidden_units=(64, 32),
                 expert_men_num=4, shared_men_num=4, men_hidden_units=(64, 32),
                 dnn_activation='selu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False, dnn_initializer=None,
                 ):
        """
        Model Name: PLE(ProgressiveLayeredExtraction)
        Paper Team: Tencent
        Paper Year: 2020
        Paper Name: <Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations>
        Paper Link: https://www.sci-hub.se/10.1145/3383313.3412236

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: Tuple[str], default ('binary', 'binary').
            :param device: str, default 'cpu'.
            :param inputs_if_multivalued: bool, default False.
            :param inputs_if_sequential: bool, default False.
            :param inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_2d_dim: Optional[int], default None. When None, each field has own embed_dim by feature_names.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task Effective Parameters:
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param tower_dnn_hidden_units: Tuple[int], default (64, 32).
            :param expert_men_num: int, default 4.
            :param shared_men_num: int, default 4.
            :param men_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'selu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        task_list = task
        self.task_num = len(task_list)

        if isinstance(task_list, str) or self.task_num < 2:
            raise MLGBError
        for _task in task_list:
            if _task not in ('binary', 'regression'):
                raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')

        self.input_fn = InputsLayer(
            feature_names=feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.task_fn_list = torch.nn.ModuleList([
            TaskLayer(
                task=task_list[i],
                task_linear_if_identity=True,
                device=device,
            )
            for i in range(self.task_num)
        ])
        self.ple_fn = ProgressiveLayeredExtractionLayer(
            task_fn_list=self.task_fn_list,
            tower_dnn_hidden_units=tower_dnn_hidden_units,
            expert_men_num=expert_men_num,
            shared_men_num=shared_men_num,
            men_hidden_units=men_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_initializer=dnn_initializer,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        outputs = self.ple_fn(dense_2d_tensor)
        return outputs


class PEPNet(torch.nn.Module):
    def __init__(self, feature_names, task=('binary', 'binary'), device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=True,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 f_ep_id_list=(0,), f_pp_id_list=(1, 2, 3),
                 domain_num=1, tower_dnn_hidden_units=(64, 32),
                 gnu_factor=2.0, gnu_last_activation='sigmoid', gnu_if_concat_general_inputs=False,
                 dnn_activation='selu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: PEPNet(Parameter&EmbeddingPersonalizedNetwork)
        Paper Team: Kuaishou
        Paper Year: 2023
        Paper Name: <PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information>
        Paper Link: https://arxiv.org/pdf/2302.01115.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: Tuple[str], default ('binary', 'binary').
            :param device: str, default 'cpu'.
            :param inputs_if_multivalued: bool, default False.
            :param inputs_if_sequential: bool, default False.
            :param inputs_if_embed_dense: bool, default True. Must True.
            :param embed_dim: int, default 32.
            :param embed_2d_dim: Optional[int], default None. When None, each field has own embed_dim by feature_names.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                he_uniform end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task Effective Parameters:
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param f_ep_id_list: Tuple[str], default (0,). Feature Index. About domain.
            :param f_pp_id_list: Tuple[str], default (1, 2, 3). Feature Index. About user, item, author.
            :param domain_num: int, default 1.
            :param tower_dnn_hidden_units: Tuple[int], default (64, 32).
            :param gnu_factor: float, default 2.0.
            :param gnu_last_activation: Optional[str], default 'sigmoid'.
            :param gnu_if_concat_general_inputs: bool, default False.
            :param dnn_activation: Optional[str], default 'selu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        task_list = task
        self.task_num = len(task_list)

        if isinstance(task_list, str) or self.task_num < 2:
            raise MLGBError
        for _task in task_list:
            if _task not in ('binary', 'regression'):
                raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if not inputs_if_embed_dense:
            raise MLGBError('inputs_if_embed_dense')
        if not domain_num > 0:
            raise MLGBError('domain_num')
        if not gnu_factor > 0:
            raise MLGBError('gnu_factor')
        if not (min(f_ep_id_list) >= 0 and min(f_pp_id_list) >= 0):
            raise MLGBError('f_ep_id_list or f_pp_id_list')

        self.f_ep_id_list = list(f_ep_id_list)
        self.f_pp_id_list = list(f_pp_id_list)

        self.input_fn = InputsLayer(
            feature_names=feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,  #
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        if domain_num > 1:
            self.task_fn_list = torch.nn.ModuleList([
                torch.nn.ModuleList([
                    TaskLayer(
                        task=task_list[j],
                        task_linear_if_identity=True,
                        device=device,
                    )
                    for _ in range(domain_num)
                ])
                for j in range(self.task_num)
            ])
        else:
            self.task_fn_list = torch.nn.ModuleList([
                TaskLayer(
                    task=task_list[i],
                    task_linear_if_identity=True,
                    device=device,
                )
                for i in range(self.task_num)
            ])
        self.pep_net_fn = ParameterEmbeddingPersonalizedNetworkLayer(
            task_fn_list=self.task_fn_list,
            domain_num=domain_num,
            tower_dnn_hidden_units=tower_dnn_hidden_units,
            gnu_factor=gnu_factor,
            gnu_last_activation=gnu_last_activation,
            gnu_if_concat_general_inputs=gnu_if_concat_general_inputs,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        f_g_list = [i for i in range(embed_3d_tensor.shape[1]) if i not in set(self.f_ep_id_list + self.f_pp_id_list)]
        g_3d_tensor = torch.index_select(
            input=embed_3d_tensor,
            dim=1,
            index=torch.tensor(f_g_list, dtype=torch.int32, device=self.device),
        )
        ep_3d_tensor = torch.index_select(
            input=embed_3d_tensor,
            dim=1,
            index=torch.tensor(self.f_ep_id_list, dtype=torch.int32, device=self.device),
        )
        pp_3d_tensor = torch.index_select(
            input=embed_3d_tensor,
            dim=1,
            index=torch.tensor(self.f_pp_id_list, dtype=torch.int32, device=self.device),
        )

        outputs = self.pep_net_fn([g_3d_tensor, ep_3d_tensor, pp_3d_tensor])
        return outputs



























