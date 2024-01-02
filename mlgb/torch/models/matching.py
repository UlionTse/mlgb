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
    SampleModeList,
)
from mlgb.torch.functions import (
    RegularizationLayer,
)
from mlgb.torch.components.linears import (
    TaskLayer,
)
from mlgb.torch.components.trms import (
    LabelAttentionLayer,
)
from mlgb.torch.components.retrieval import (
    SampledSoftmaxLossLayer,
)
from mlgb.torch.modules.matching import (
    DeepStructuredSemanticModelUserEmbeddingLayer,
    DeepStructuredSemanticModelItemEmbeddingLayer,
    NeuralCollaborativeFilteringLayer,
    MultiInterestNetworkWithDynamicRoutingLayer,
    UserInputs_MultiInterestNetworkWithDynamicRoutingLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'NCF',
    'MatchFM', 'EBR', 'DSSM', 'YoutubeDNN', 'MIND',
]


class NCF(torch.nn.Module):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False,
                 item_inputs_if_multivalued=False,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 user_dnn_hidden_units=(), item_dnn_hidden_units=(),
                 dnn_hidden_units=(1024, 512), dnn_activation='tanh', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: NCF(NeuralCollaborativeFiltering)
        Paper Team: NUS(Xiangnan He(NUS), etc)
        Paper Year: 2017
        Paper Name: <Neural Collaborative Filtering>
        Paper Link: https://arxiv.org/pdf/1708.05031.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'multiclass:100'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: str, default 'cpu'.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default False.
            :param item_inputs_if_multivalued: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task SampledSoftmaxLoss Parameters:
            :param sample_mode: Optional[str], default 'Sample:batch'. Union['Sample:all', 'Sample:batch', 'Sample:uniform',
                                'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram']
            :param sample_num: Optional[int], default None. When None, equals multiclass_num. Must <= multiclass_num.
            :param sample_item_distribution_list: Optional[list], default None. When None,
                                equals batch_item_distribution like batch_normalization.
            :param sample_fixed_unigram_frequency_list: Optional[list], default None. When None, list of same frequency.
                                When sample_mode is 'Sample:fixed_unigram', it works.
            :param sample_fixed_unigram_distortion: float, default 1.0. When sample_mode is 'Sample:fixed_unigram', it works.

        Task Effective Parameters:
            :param model_embeds_if_l2_norm: bool, default True.
            :param model_result_temperature_ratio: Optional[float], default None, When None, equals 1.0.
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param user_dnn_hidden_units: tuple, default ().
            :param item_dnn_hidden_units: tuple, default ().
            :param dnn_hidden_units: tuple, default (1024, 512).
            :param dnn_activation: Optional[str], default 'tanh'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if task.startswith('multiclass:'):
            self.class_num = int(task.split(':')[1])
            self.user_dnn_hidden_units = user_dnn_hidden_units + (self.class_num,)
            self.item_dnn_hidden_units = item_dnn_hidden_units + (self.class_num,)
            self.dnn_hidden_units = dnn_hidden_units + (self.class_num,)

        if not (self.user_dnn_hidden_units[-1] == self.item_dnn_hidden_units[-1] == self.dnn_hidden_units[-1]):
            raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError
        if not embed_dim:
            raise MLGBError
        if model_result_temperature_ratio and not (1e-2 <= model_result_temperature_ratio <= 1.0):
            raise MLGBError
        if sample_mode and sample_mode not in SampleModeList:
            raise MLGBError

        user_feature_names, item_feature_names = feature_names

        self.user_dnn_embedding_fn = DeepStructuredSemanticModelUserEmbeddingLayer(
            user_feature_names=user_feature_names,
            user_inputs_if_multivalued=user_inputs_if_multivalued,
            user_inputs_if_sequential=user_inputs_if_sequential,
            user_inputs_if_embed_dense=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            user_dnn_hidden_units=self.user_dnn_hidden_units,
            dnn_activation=None,
            tower_embeds_flatten_mode='flatten',
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        self.item_dnn_embedding_fn = DeepStructuredSemanticModelItemEmbeddingLayer(
            item_feature_names=item_feature_names,
            item_inputs_if_multivalued=item_inputs_if_multivalued,
            item_inputs_if_sequential=False,
            item_inputs_if_embed_dense=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            item_dnn_hidden_units=self.item_dnn_hidden_units,
            dnn_activation=None,
            tower_embeds_flatten_mode='flatten',
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        self.user_mf_embedding_fn = DeepStructuredSemanticModelUserEmbeddingLayer(
            user_feature_names=user_feature_names,
            user_inputs_if_multivalued=user_inputs_if_multivalued,
            user_inputs_if_sequential=user_inputs_if_sequential,
            user_inputs_if_embed_dense=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            user_dnn_hidden_units=self.user_dnn_hidden_units,
            dnn_activation=None,
            tower_embeds_flatten_mode='sum',
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        self.item_mf_embedding_fn = DeepStructuredSemanticModelItemEmbeddingLayer(
            item_feature_names=item_feature_names,
            item_inputs_if_multivalued=item_inputs_if_multivalued,
            item_inputs_if_sequential=False,
            item_inputs_if_embed_dense=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            item_dnn_hidden_units=self.item_dnn_hidden_units,
            dnn_activation=None,
            tower_embeds_flatten_mode='sum',
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        self.ncf_fn = NeuralCollaborativeFilteringLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=False if sample_mode else True,
            task_multiclass_temperature_ratio=model_result_temperature_ratio,
            device=device,
        )
        if task.startswith('multiclass:') and sample_mode:
            self.sampled_softmax_loss = SampledSoftmaxLossLayer(
                sample_mode=sample_mode,
                sample_num=sample_num,
                sample_item_distribution_list=sample_item_distribution_list,
                sample_fixed_unigram_distortion=sample_fixed_unigram_distortion,
                sample_fixed_unigram_frequency_list=sample_fixed_unigram_frequency_list,
                device=device,
            )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs

        user_mf_embeddings = self.user_mf_embedding_fn(user_inputs)
        item_mf_embeddings = self.item_mf_embedding_fn(item_inputs)
        user_dnn_embeddings = self.user_dnn_embedding_fn(user_inputs)
        item_dnn_embeddings = self.item_dnn_embedding_fn(item_inputs)

        fm_outputs = self.ncf_fn([user_mf_embeddings, item_mf_embeddings, user_dnn_embeddings, item_dnn_embeddings])
        outputs = self.task_fn(fm_outputs)
        return outputs


class DSSMs(torch.nn.Module):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False, user_inputs_if_embed_dense=True,
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=True,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_flatten_mode='flatten', model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 user_dnn_hidden_units=(1024, 512), item_dnn_hidden_units=(1024, 512),
                 user_dnn_activation='tanh', item_dnn_activation='tanh',
                 dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        super().__init__()
        if task.startswith('multiclass:'):
            self.class_num = int(task.split(':')[1])
            self.user_dnn_hidden_units = user_dnn_hidden_units + (self.class_num,)
            self.item_dnn_hidden_units = item_dnn_hidden_units + (self.class_num,)

        if not (self.user_dnn_hidden_units[-1] == self.item_dnn_hidden_units[-1]):
            raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError
        if not embed_dim:
            raise MLGBError
        if model_embeds_flatten_mode not in ('flatten', 'sum'):
            raise MLGBError
        if model_result_temperature_ratio and not (1e-2 <= model_result_temperature_ratio <= 1.0):
            raise MLGBError
        if sample_mode and sample_mode not in SampleModeList:
            raise MLGBError

        user_feature_names, item_feature_names = feature_names

        self.user_embedding_fn = DeepStructuredSemanticModelUserEmbeddingLayer(
            user_feature_names=user_feature_names,
            user_inputs_if_multivalued=user_inputs_if_multivalued,
            user_inputs_if_sequential=user_inputs_if_sequential,
            user_inputs_if_embed_dense=user_inputs_if_embed_dense,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            user_dnn_hidden_units=self.user_dnn_hidden_units,
            dnn_activation=user_dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            tower_embeds_flatten_mode=model_embeds_flatten_mode,
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        self.item_embedding_fn = DeepStructuredSemanticModelItemEmbeddingLayer(
            item_feature_names=item_feature_names,
            item_inputs_if_multivalued=item_inputs_if_multivalued,
            item_inputs_if_sequential=False,
            item_inputs_if_embed_dense=item_inputs_if_embed_dense,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            item_dnn_hidden_units=self.item_dnn_hidden_units,
            dnn_activation=item_dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            tower_embeds_flatten_mode=model_embeds_flatten_mode,
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=False,
            task_linear_if_bias=False,
            task_multiclass_if_project=False,
            task_multiclass_if_softmax=False if sample_mode else True,
            task_multiclass_temperature_ratio=model_result_temperature_ratio,
            device=device,
        )
        if task.startswith('multiclass:') and sample_mode:
            self.sampled_softmax_loss = SampledSoftmaxLossLayer(
                sample_mode=sample_mode,
                sample_num=sample_num,
                sample_item_distribution_list=sample_item_distribution_list,
                sample_fixed_unigram_distortion=sample_fixed_unigram_distortion,
                sample_fixed_unigram_frequency_list=sample_fixed_unigram_frequency_list,
                device=device,
            )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs

        user_embeddings = self.user_embedding_fn(user_inputs)
        item_embeddings = self.item_embedding_fn(item_inputs)

        dssm_outputs = user_embeddings * item_embeddings
        outputs = self.task_fn(dssm_outputs)
        return outputs


class DSSM(DSSMs):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False,
                 item_inputs_if_multivalued=False,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_flatten_mode='flatten', model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 user_dnn_hidden_units=(1024, 512), item_dnn_hidden_units=(1024, 512),
                 user_dnn_activation='tanh', item_dnn_activation='tanh',
                 dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DSSM(DeepStructuredSemanticModel)
        Paper Team: Microsoft
        Paper Year: 2013
        Paper Name: <Learning deep structured semantic models for web search using clickthrough data>
        Paper Link: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'multiclass:100'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: str, default 'cpu'.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default False.
            :param item_inputs_if_multivalued: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task SampledSoftmaxLoss Parameters:
            :param sample_mode: Optional[str], default 'Sample:batch'. Union['Sample:all', 'Sample:batch', 'Sample:uniform',
                                'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram']
            :param sample_num: Optional[int], default None. When None, equals multiclass_num. Must <= multiclass_num.
            :param sample_item_distribution_list: Optional[list], default None. When None,
                                equals batch_item_distribution like batch_normalization.
            :param sample_fixed_unigram_frequency_list: Optional[list], default None. When None, list of same frequency.
                                When sample_mode is 'Sample:fixed_unigram', it works.
            :param sample_fixed_unigram_distortion: float, default 1.0. When sample_mode is 'Sample:fixed_unigram', it works.

        Task Effective Parameters:
            :param model_embeds_flatten_mode: str, default 'flatten', Union['flatten']
            :param model_embeds_if_l2_norm: bool, default True.
            :param model_result_temperature_ratio: Optional[float], default None, When None, equals 1.0.
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param user_dnn_hidden_units: tuple, default (1024, 512).
            :param item_dnn_hidden_units: tuple, default (1024, 512).
            :param user_dnn_activation: Optional[str], default 'tanh'.
            :param user_dnn_activation: Optional[str], default 'tanh'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__(
            feature_names, task, device,
            user_inputs_if_multivalued, user_inputs_if_sequential, True,
            item_inputs_if_multivalued, True,
            embed_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            sample_mode, sample_num, sample_item_distribution_list,
            sample_fixed_unigram_frequency_list, sample_fixed_unigram_distortion,
            model_embeds_flatten_mode, model_embeds_if_l2_norm, model_result_temperature_ratio,
            model_l1, model_l2,
            user_dnn_hidden_units, item_dnn_hidden_units,
            user_dnn_activation, item_dnn_activation,
            dnn_dropout, dnn_if_bn, dnn_if_ln,
        )
        if not (user_dnn_activation and item_dnn_activation):  # must both NN.
            raise MLGBError
        if model_embeds_flatten_mode != 'flatten':
            raise MLGBError


class EBR(DSSMs):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False,
                 item_inputs_if_multivalued=False,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_flatten_mode='flatten', model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 user_dnn_hidden_units=(100,), item_dnn_hidden_units=(100,),
                 user_dnn_activation=None, item_dnn_activation=None,  #
                 dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: EBR(EmbeddingBasedRetrieval)
        Paper Team: Facebook(Meta), etc.
        Paper Year: 2020
        Paper Name: <Embedding-based Retrieval in Facebook Search>
        Paper Link: https://browse.arxiv.org/pdf/2006.11632.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'multiclass:100'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: str, default 'cpu'.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default False.
            :param item_inputs_if_multivalued: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task SampledSoftmaxLoss Parameters:
            :param sample_mode: Optional[str], default 'Sample:batch'. Union['Sample:all', 'Sample:batch', 'Sample:uniform',
                                'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram']
            :param sample_num: Optional[int], default None. When None, equals multiclass_num. Must <= multiclass_num.
            :param sample_item_distribution_list: Optional[list], default None. When None,
                                equals batch_item_distribution like batch_normalization.
            :param sample_fixed_unigram_frequency_list: Optional[list], default None. When None, list of same frequency.
                                When sample_mode is 'Sample:fixed_unigram', it works.
            :param sample_fixed_unigram_distortion: float, default 1.0. When sample_mode is 'Sample:fixed_unigram', it works.

        Task Effective Parameters:
            :param model_embeds_flatten_mode: str, default 'flatten', Union['flatten']
            :param model_embeds_if_l2_norm: bool, default True.
            :param model_result_temperature_ratio: Optional[float], default None, When None, equals 1.0.
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param user_dnn_hidden_units: tuple, default ().
            :param item_dnn_hidden_units: tuple, default ().
            :param user_dnn_activation: Optional[str], default None.
            :param user_dnn_activation: Optional[str], default None.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__(
            feature_names, task, device,
            user_inputs_if_multivalued, user_inputs_if_sequential, True,
            item_inputs_if_multivalued, True,
            embed_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            sample_mode, sample_num, sample_item_distribution_list,
            sample_fixed_unigram_frequency_list, sample_fixed_unigram_distortion,
            model_embeds_flatten_mode, model_embeds_if_l2_norm, model_result_temperature_ratio,
            model_l1, model_l2,
            user_dnn_hidden_units, item_dnn_hidden_units,
            user_dnn_activation, item_dnn_activation,
            dnn_dropout, dnn_if_bn, dnn_if_ln,
        )
        if user_dnn_activation or item_dnn_activation:  # only projection, not NN.
            raise MLGBError
        if model_embeds_flatten_mode != 'flatten':
            raise MLGBError


class MatchFM(DSSMs):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False,
                 item_inputs_if_multivalued=False,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_flatten_mode='sum', model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 user_dnn_hidden_units=(), item_dnn_hidden_units=(),
                 user_dnn_activation=None, item_dnn_activation=None,  #
                 dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: FM(FactorizationMachine)
        Paper Team: Steffen Rendle(Google, 2013-Present)
        Paper Year: 2010
        Paper Name: <Factorization Machines>
        Paper Link: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf,
                    https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf,
                    https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-Steffen-Rendle-Osaka-University-2010.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'multiclass:100'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: str, default 'cpu'.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default False.
            :param item_inputs_if_multivalued: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task SampledSoftmaxLoss Parameters:
            :param sample_mode: Optional[str], default 'Sample:batch'. Union['Sample:all', 'Sample:batch', 'Sample:uniform',
                                'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram']
            :param sample_num: Optional[int], default None. When None, equals multiclass_num. Must <= multiclass_num.
            :param sample_item_distribution_list: Optional[list], default None. When None,
                                equals batch_item_distribution like batch_normalization.
            :param sample_fixed_unigram_frequency_list: Optional[list], default None. When None, list of same frequency.
                                When sample_mode is 'Sample:fixed_unigram', it works.
            :param sample_fixed_unigram_distortion: float, default 1.0. When sample_mode is 'Sample:fixed_unigram', it works.

        Task Effective Parameters:
            :param model_embeds_flatten_mode: str, default 'sum', Union['sum']
            :param model_embeds_if_l2_norm: bool, default True.
            :param model_result_temperature_ratio: Optional[float], default None, When None, equals 1.0.
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param user_dnn_hidden_units: tuple, default ().
            :param item_dnn_hidden_units: tuple, default ().
            :param user_dnn_activation: Optional[str], default None.
            :param user_dnn_activation: Optional[str], default None.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__(
            feature_names, task, device,
            user_inputs_if_multivalued, user_inputs_if_sequential, True,
            item_inputs_if_multivalued, True,
            embed_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            sample_mode, sample_num, sample_item_distribution_list,
            sample_fixed_unigram_frequency_list, sample_fixed_unigram_distortion,
            model_embeds_flatten_mode, model_embeds_if_l2_norm, model_result_temperature_ratio,
            model_l1, model_l2,
            user_dnn_hidden_units, item_dnn_hidden_units,
            user_dnn_activation, item_dnn_activation,
            dnn_dropout, dnn_if_bn, dnn_if_ln,
        )
        if user_dnn_activation or item_dnn_activation:  # only projection, not NN.
            raise MLGBError
        if model_embeds_flatten_mode != 'sum':
            raise MLGBError


class YoutubeDNN(DSSMs):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=False,
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_flatten_mode='flatten', model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 user_dnn_hidden_units=(1024, 512), item_dnn_hidden_units=(),  #
                 user_dnn_activation='tanh', item_dnn_activation=None,  #
                 dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: YoutubeDNN(DeepNeuralNetworkForYoutube)
        Paper Team: Google(Alphabet)
        Paper Year: 2016
        Paper Name: <Deep Neural Networks for YouTube Recommendations>
        Paper Link: https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf,
                    https://dl.acm.org/doi/pdf/10.1145/2959100.2959190

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'multiclass:100'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: str, default 'cpu'.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task SampledSoftmaxLoss Parameters:
            :param sample_mode: Optional[str], default 'Sample:batch'. Union['Sample:all', 'Sample:batch', 'Sample:uniform',
                                'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram']
            :param sample_num: Optional[int], default None. When None, equals multiclass_num. Must <= multiclass_num.
            :param sample_item_distribution_list: Optional[list], default None. When None,
                                equals batch_item_distribution like batch_normalization.
            :param sample_fixed_unigram_frequency_list: Optional[list], default None. When None, list of same frequency.
                                When sample_mode is 'Sample:fixed_unigram', it works.
            :param sample_fixed_unigram_distortion: float, default 1.0. When sample_mode is 'Sample:fixed_unigram', it works.

        Task Effective Parameters:
            :param model_embeds_flatten_mode: str, default 'flatten', Union['flatten']
            :param model_embeds_if_l2_norm: bool, default True.
            :param model_result_temperature_ratio: Optional[float], default None, When None, equals 1.0.
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param user_dnn_hidden_units: tuple, default (1024, 512).
            :param item_dnn_hidden_units: tuple, default ().
            :param user_dnn_activation: Optional[str], default 'tanh'.
            :param user_dnn_activation: Optional[str], default None.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__(
            feature_names, task, device,
            user_inputs_if_multivalued, user_inputs_if_sequential, True,
            False, False,  # Only need item_id embeddings.
            embed_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            sample_mode, sample_num, sample_item_distribution_list,
            sample_fixed_unigram_frequency_list, sample_fixed_unigram_distortion,
            model_embeds_flatten_mode, model_embeds_if_l2_norm, model_result_temperature_ratio,
            model_l1, model_l2,
            user_dnn_hidden_units, item_dnn_hidden_units,
            user_dnn_activation, item_dnn_activation,
            dnn_dropout, dnn_if_bn, dnn_if_ln,
        )
        if item_dnn_activation:  # only projection, not NN.
            raise MLGBError
        if model_embeds_flatten_mode != 'flatten':
            raise MLGBError


class MIND(torch.nn.Module):
    def __init__(self, feature_names, task='multiclass:100', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 embed_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 sample_mode='Sample:batch', sample_num=None, sample_item_distribution_list=None,
                 sample_fixed_unigram_frequency_list=None, sample_fixed_unigram_distortion=1.0,
                 model_embeds_if_l2_norm=True, model_result_temperature_ratio=None,
                 model_l1=0.0, model_l2=0.0,
                 mind_if_label_attention=False,
                 user_dnn_hidden_units=(1024, 512), item_dnn_hidden_units=(),  #
                 capsule_num=3, capsule_activation='squash', capsule_l2=0.0, capsule_initializer=None,
                 capsule_interest_num_if_dynamic=False, capsule_input_sequence_pad_mode='pre',
                 capsule_routing_initializer='random_normal',
                 dnn_activation='tanh', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: MIND(MultiInterestNetworkWithDynamicRouting)
        Paper Team: Alibaba
        Paper Year: 2019
        Paper Name: <Multi-Interest Network with Dynamic Routing for Recommendation at Tmall>
        Paper Link: https://arxiv.org/pdf/1904.08030.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'multiclass:100'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default False.
            :param embed_dim: int, default 32.
            :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
            :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
            :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
                                'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
            :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
                                xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.

        Task SampledSoftmaxLoss Parameters:
            :param sample_mode: Optional[str], default 'Sample:batch'. Union['Sample:all', 'Sample:batch', 'Sample:uniform',
                                'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram']
            :param sample_num: Optional[int], default None. When None, equals multiclass_num. Must <= multiclass_num.
            :param sample_item_distribution_list: Optional[list], default None. When None,
                                equals batch_item_distribution like batch_normalization.
            :param sample_fixed_unigram_frequency_list: Optional[list], default None. When None, list of same frequency.
                                When sample_mode is 'Sample:fixed_unigram', it works.
            :param sample_fixed_unigram_distortion: float, default 1.0. When sample_mode is 'Sample:fixed_unigram', it works.

        Task Effective Parameters:
            :param model_embeds_if_l2_norm: bool, default True.
            :param model_result_temperature_ratio: Optional[float], default None, When None, equals 1.0.
            :param model_l1: float, default 0.0.
            :param model_l2: float, default 0.0.

        Task Model Parameters:
            :param mind_if_label_attention: bool, default False.
            :param user_dnn_hidden_units: tuple, default (1024, 512).
            :param item_dnn_hidden_units: tuple, default ().
            :param capsule_num: int, default 3.
            :param capsule_activation: Optional[str], default 'squash'.
            :param capsule_l2: float, default 0.0.
            :param capsule_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
            :param capsule_interest_num_if_dynamic: bool, default False.
            :param capsule_input_sequence_pad_mode: str, default 'pre'.
            :param capsule_routing_initializer: Optional[str], default 'random_normal'. When None, activation judge first, xavier_normal end.
            :param dnn_activation: Optional[str], default 'tanh'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if task.startswith('multiclass:'):
            self.class_num = int(task.split(':')[1])
            self.user_dnn_hidden_units = user_dnn_hidden_units + (self.class_num,)
            self.item_dnn_hidden_units = item_dnn_hidden_units + (self.class_num,)

        if not (self.user_dnn_hidden_units[-1] == self.item_dnn_hidden_units[-1]):
            raise MLGBError
        if not user_inputs_if_sequential:
            raise MLGBError
        if capsule_input_sequence_pad_mode not in ('pre', 'post'):
            raise MLGBError
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError
        if not embed_dim:
            raise MLGBError
        if model_result_temperature_ratio and not (1e-2 <= model_result_temperature_ratio <= 1.0):
            raise MLGBError
        if sample_mode and sample_mode not in SampleModeList:
            raise MLGBError

        user_feature_names, item_feature_names = feature_names
        self.mind_if_label_attention = mind_if_label_attention

        self.user_embedding_fn = UserInputs_MultiInterestNetworkWithDynamicRoutingLayer(
            user_feature_names=user_feature_names,
            user_inputs_if_multivalued=user_inputs_if_multivalued,
            user_inputs_if_sequential=user_inputs_if_sequential,  #
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,  # (b, s, e)
            pool_seq_initializer=pool_seq_initializer,
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,  #
            capsule_num=capsule_num,
            capsule_activation=capsule_activation,
            capsule_l2=capsule_l2,
            capsule_initializer=capsule_initializer,
            capsule_interest_num_if_dynamic=capsule_interest_num_if_dynamic,
            capsule_input_sequence_pad_mode=capsule_input_sequence_pad_mode,
            capsule_routing_initializer=capsule_routing_initializer,
            dnn_hidden_units=self.user_dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            device=device,
        )
        self.item_embedding_fn = DeepStructuredSemanticModelItemEmbeddingLayer(
            item_feature_names=item_feature_names,
            item_inputs_if_multivalued=False,
            item_inputs_if_sequential=False,
            item_inputs_if_embed_dense=False,  # Only need item_id embeddings.
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            item_dnn_hidden_units=self.item_dnn_hidden_units,
            dnn_activation=None,
            tower_embeds_flatten_mode='flatten',
            tower_embeds_if_l2_norm=model_embeds_if_l2_norm,
            device=device,
        )
        if self.mind_if_label_attention:
            self.label_attention_fn = LabelAttentionLayer(
                softmax_axis=1,
                softmax_pre_temperature_ratio=model_result_temperature_ratio if model_result_temperature_ratio else 1.0,
                device=device,
            )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=False,
            task_linear_if_bias=False,
            task_multiclass_if_project=False,
            task_multiclass_if_softmax=False if sample_mode else True,  # None vs 'Sample:all'
            task_multiclass_temperature_ratio=model_result_temperature_ratio,
            device=device,
        )
        if task.startswith('multiclass:') and sample_mode:
            self.sampled_softmax_loss = SampledSoftmaxLossLayer(
                sample_mode=sample_mode,
                sample_num=sample_num,
                sample_item_distribution_list=sample_item_distribution_list,
                sample_fixed_unigram_distortion=sample_fixed_unigram_distortion,
                sample_fixed_unigram_frequency_list=sample_fixed_unigram_frequency_list,
                device=device,
            )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs

        user_embeddings = self.user_embedding_fn(user_inputs)
        item_embeddings = self.item_embedding_fn(item_inputs)

        if self.mind_if_label_attention:
            mind_outputs = self.label_attention_fn([item_embeddings, user_embeddings])  # q, k=v
        else:
            mind_outputs = user_embeddings * item_embeddings
        outputs = self.task_fn(mind_outputs)
        return outputs





















