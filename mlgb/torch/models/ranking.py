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
    PoolModeList,
    MVPoolModeList,
    EDCNModeList,
    FBIModeList,
    BiGRUModeList,
    SeqRecPointwiseModeList,
)
from mlgb.torch.functions import RegularizationLayer
from mlgb.torch.inputs import InputsLayer
from mlgb.torch.components.linears import TaskLayer
from mlgb.torch.modules.ranking.normal import (
    LinearOrLogisticRegressionLayer,
    MultiLayerPerceptronLayer,
    PiecewiseLinearModelLayer,
    DeepLearningRecommendationModelLayer,
    MaskNetLayer,

    DeepCrossingModelLayer,
    DeepCrossNetworkLayer,
    EnhancedDeepCrossNetworkLayer,
    AllFactorizationMachineLayer,
    FactorizationMachineLayer,
    FieldFactorizationMachineLayer,
    FieldWeightedFactorizationMachineLayer,
    FieldEmbeddedFactorizationMachineLayer,
    FieldVectorizedFactorizationMachineLayer,
    FieldMatrixedFactorizationMachineLayer,
    AttentionalFactorizationMachineLayer,
    LorentzFactorizationMachineLayer,
    InteractionMachineLayer,
    AllInputFactorizationMachineLayer,
    InputFactorizationMachineLayer,
    DualInputFactorizationMachineLayer,

    FactorizationMachineNeuralNetworkLayer,
    ProductNeuralNetworkLayer,
    ProductNetworkInNetworkLayer,
    OperationNeuralNetworkLayer,
    AdaptiveFactorizationNetworkLayer,

    NeuralFactorizationMachineLayer,
    WideDeepLearningLayer,
    DeepFactorizationMachineLayer,
    DeepFieldEmbeddedFactorizationMachineLayer,
    FieldLeveragedEmbeddingNetworkLayer,

    ConvolutionalClickPredictionModelLayer,
    FeatureGenerationByConvolutionalNeuralNetworkLayer,
    ExtremeDeepFactorizationMachineLayer,
    FeatureImportanceBilinearInteractionNetworkLayer,
    AutomaticFeatureInteractionLearningLayer,
)
from mlgb.torch.modules.ranking.sequential import (
    GatedRecurrentUnit4RecommendationLayer,
    ConvolutionalSequenceEmbeddingRecommendationLayer,
    SelfAttentiveSequentialRecommendationLayer,
    BidirectionalEncoderRepresentationTransformer4RecommendationLayer,
    BehaviorSequenceTransformerLayer,
    DeepInterestNetworkLayer,
    DeepInterestEvolutionNetworkLayer,
    DeepSessionInterestNetworkLayer,
)

from mlgb.error import MLGBError


__all__ = [
    'LR', 'PLM', 'MLP', 'DLRM', 'MaskNet',
    'DCM', 'DCN', 'EDCN',
    'FM', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FmFM', 'AFM', 'LFM', 'IM', 'IFM', 'DIFM',
    'FNN', 'PNN', 'PIN', 'ONN', 'AFN',
    'NFM', 'WDL', 'DeepFM', 'DeepFEFM', 'DeepIM', 'FLEN',
    'CCPM', 'FGCNN', 'XDeepFM', 'FiBiNet', 'AutoInt',

    'GRU4Rec', 'Caser', 'SASRec', 'BERT4Rec',
    'BST', 'DIN', 'DIEN', 'DSIN',
]


class LR(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu', 
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 linear_if_bias=True,
                 ):
        """
        Model Name: LR(LinearOrLogisticRegression)
        Paper Team: Microsoft
        Paper Year: 2007
        Paper Name: <Predicting Clicks: Estimating the Click-Through Rate for New Ads>
        Paper Link: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/predictingclicks.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression']
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
            :param linear_if_bias: bool, default True.
        """
        super().__init__()
        if task.startswith('multiclass'):
            raise MLGBError('task')
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
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.lr_fn = LinearOrLogisticRegressionLayer(
            linear_if_bias=linear_if_bias,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=True,  #
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        lr_outputs = self.lr_fn(dense_2d_tensor)
        outputs = self.task_fn(lr_outputs)
        return outputs


class MLP(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='tanh', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: MLP/DNN(MultiLayerPerceptron)
        Paper Team: Christopher M. Bishop(Microsoft, 1997-Present), Foreword by Geoffrey Hinton.
        Paper Year: 1995
        Paper Name: <Neural Networks for Pattern Recognition>
        Paper Link: http://diyhpl.us/~bryan/papers2/ai/ahuman-pdf-only/neural-networks/2005-Pattern%20Recognition.pdf,
                    http://people.sabanciuniv.edu/berrin/cs512/lectures/Book-Bishop-Neural%20Networks%20for%20Pattern%20Recognition.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param dnn_activation: Optional[str], default 'tanh'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
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
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dnn_fn = MultiLayerPerceptronLayer(
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_if_bias=True,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        dnn_outputs = self.dnn_fn(dense_2d_tensor)
        outputs = self.task_fn(dnn_outputs)
        return outputs


class PLM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 piece_inputs_if_multivalued=False, piece_inputs_if_sequential=False,
                 base_inputs_if_multivalued=False, base_inputs_if_sequential=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 plm_piece_num=2,
                 ):
        """
        Model Name: PLM/MLR(PiecewiseLinearModel)
        Paper Team: Alibaba
        Paper Year: 2017
        Paper Name: <Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction>
        Paper Link: https://arxiv.org/pdf/1704.05194.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression']
            :param device: str, default 'cpu'.
            :param piece_inputs_if_multivalued: bool, default False.
            :param piece_inputs_if_sequential: bool, default False.
            :param base_inputs_if_multivalued: bool, default False.
            :param base_inputs_if_sequential: bool, default False.
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
            :param plm_piece_num: int, default 2. Must >= 2.
        """
        super().__init__()
        if task.startswith('multiclass'):
            raise MLGBError('task')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if len(feature_names) != 2:
            raise MLGBError('feature_names')
        if plm_piece_num < 2:
            raise MLGBError('plm_piece_num')

        piece_feature_names, base_feature_names = feature_names

        self.input_piece_fn = InputsLayer(
            feature_names=piece_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=piece_inputs_if_multivalued,
            inputs_if_sequential=piece_inputs_if_sequential,
            inputs_if_embed_dense=False,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.input_base_fn = InputsLayer(
            feature_names=base_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=base_inputs_if_multivalued,
            inputs_if_sequential=base_inputs_if_sequential,
            inputs_if_embed_dense=False,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.plm_fn = PiecewiseLinearModelLayer(
            plm_task=task,
            plm_piece_num=plm_piece_num,
            device=device,
        )
        self.task_fn = TaskLayer(
            task='regression',  #
            task_linear_if_identity=True,  #
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        piece_inputs, base_inputs = inputs
        piece_dense_2d_tensor, _ = self.input_piece_fn(piece_inputs)
        base_dense_2d_tensor, _ = self.input_base_fn(base_inputs)

        plm_outputs = self.plm_fn([piece_dense_2d_tensor, base_dense_2d_tensor])
        outputs = self.task_fn(plm_outputs)
        return outputs


class DLRM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 dnn_bottom_hidden_units=(64, 32), dnn_top_hidden_units=(64, 32),
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DLRM(DeepLearningRecommendationModel)
        Paper Team: Facebook(Meta)
        Paper Year: 2019
        Paper Name: <Deep Learning Recommendation Model for Personalization and Recommendation Systems>
        Paper Link: https://arxiv.org/pdf/1906.00091.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param dnn_bottom_hidden_units: Tuple[int], default (64, 32).
            :param dnn_top_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
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
            outputs_dense_if_add_sparse=False,  #
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dlrm_fn = DeepLearningRecommendationModelLayer(
            dnn_bottom_hidden_units=dnn_bottom_hidden_units,
            dnn_top_hidden_units=dnn_top_hidden_units,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_2d_tensor = self.input_fn(inputs)

        dlrm_outputs = self.dlrm_fn([dense_2d_tensor, embed_2d_tensor])
        outputs = self.task_fn(dlrm_outputs)
        return outputs


class MaskNet(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 mask_mode='MaskNet:serial', mask_block_num=2,
                 block_activation='relu', block_if_bn=False, block_dropout=0.0, block_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu',
                 dnn_if_bn=False, dnn_if_ln=False, dnn_dropout=0.0, 
                 ):
        """
        Model Name: MaskNet(MaskNet)
        Paper Team: Weibo(Sina)
        Paper Year: 2021
        Paper Name: <MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask>
        Paper Link: https://arxiv.org/pdf/2102.07619.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param mask_mode: str, default 'MaskNet:serial'. Union['MaskNet:serial', 'MaskNet:parallel']
            :param mask_block_num: int, default 2. Must > 0.
            :param block_activation: Optional[str], default 'relu'.
            :param block_if_bn: bool, default False. Batch Normalization.
            :param block_dropout: float, default 0.0.
            :param block_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if mask_mode not in ('MaskNet:serial', 'MaskNet:parallel'):
            raise MLGBError('mask_mode')
        if not mask_block_num > 0:
            raise MLGBError('mask_block_num')

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
        self.mask_net_fn = MaskNetLayer(
            mask_mode=mask_mode,
            mask_block_num=mask_block_num,
            block_activation=block_activation,
            block_if_bn=block_if_bn,
            block_dropout=block_dropout,
            block_initializer=block_initializer,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_dropout=dnn_dropout,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        mask_net_outputs = self.mask_net_fn(embed_3d_tensor)
        outputs = self.task_fn(mask_net_outputs)
        return outputs


class DCM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DCM(DeepCrossingModel)
        Paper Team: Microsoft
        Paper Year: 2016
        Paper Name: <Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features>
        Paper Link: https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
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
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dcm_fn = DeepCrossingModelLayer(
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        dcm_outputs = self.dcm_fn(dense_2d_tensor)
        outputs = self.task_fn(dcm_outputs)
        return outputs


class DCN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 dcn_version='v2', dcn_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DCN(DeepCrossNetwork)
        Paper Team: Google(Alphabet)
        Paper Year: 2017, 2020
        Paper Name: <DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems>
        Paper Link: v1, https://arxiv.org/pdf/1708.05123.pdf; v2: https://arxiv.org/pdf/2008.13535.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param dcn_version: str, default 'v2'. Union['v1', 'v2']
            :param dcn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if dcn_version not in ('v1', 'v2'):
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
            embed_cate_if_output2d=True,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dcn_fn = DeepCrossNetworkLayer(
            dcn_version=dcn_version,
            dcn_initializer=dcn_initializer,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, _ = self.input_fn(inputs)

        dcn_outputs = self.dcn_fn(dense_2d_tensor)
        outputs = self.task_fn(dcn_outputs)
        return outputs


class EDCN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 edcn_layer_num=2,
                 bdg_mode='EDCN:attention_pooling', bdg_layer_num=1,
                 rgl_tau_ratio=1.0, rgl_initializer='ones',
                 dcn_version='v2', dcn_initializer=None,
                 dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: EDCN(EnhancedDeepCrossNetwork)
        Paper Team: Huawei
        Paper Year: 2021
        Paper Name: <Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models>
        Paper Link: https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param edcn_layer_num: int, default 2.
            :param bdg_mode: str, default 'EDCN:attention_pooling'. From Bridge Module. Union['EDCN:pointwise_addition',
                                'EDCN:hadamard_product', 'EDCN:concatenation', 'EDCN:attention_pooling']
            :param bdg_layer_num: int, default 1.
            :param rgl_tau_ratio: float, default 1.0. From Regulation Module.
            :param rgl_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dcn_version: str, default 'v2'. Union['v1', 'v2']
            :param dcn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if dcn_version not in ('v1', 'v2'):
            raise MLGBError('dcn_version')
        if bdg_mode not in EDCNModeList:
            raise MLGBError('bdg_mode')
        if not (edcn_layer_num >= 1 and bdg_layer_num >= 1):
            raise MLGBError('edcn_layer_num or bdg_layer_num')
        if not (0 < rgl_tau_ratio <= 1.0):
            raise MLGBError('rgl_tau_ratio')

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
        self.edcn_fn = EnhancedDeepCrossNetworkLayer(
            edcn_layer_num=edcn_layer_num,
            bdg_mode=bdg_mode,
            bdg_layer_num=bdg_layer_num,
            rgl_tau_ratio=rgl_tau_ratio,
            rgl_initializer=rgl_initializer,
            dcn_version=dcn_version,
            dcn_initializer=dcn_initializer,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        edcn_outputs = self.edcn_fn(embed_3d_tensor)
        outputs = self.task_fn(edcn_outputs)
        return outputs


class FMs(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 ):
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

        self.input_fn = InputsLayer(
            feature_names=feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_initializer=embed_initializer,
            embed_2d_dim=embed_2d_dim,
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.fm_fn = AllFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        fm_outputs = self.fm_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(fm_outputs)
        return outputs
    

class FM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM', fbi_unit=32, fbi_initializer=None,
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
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM'. Field Bi-Interaction Mode. Union['FM', 'FM3D']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            3, 'relu', 0.0,
        )
        if fbi_mode not in ('FM', 'FM3D'):
            raise MLGBError


class FFM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FFM', fbi_unit=32, fbi_initializer=None,
                 ):
        """
        Model Name: FFM(FieldFactorizationMachine)
        Paper Team: NTU
        Paper Year: 2016
        Paper Name: <Field-aware Factorization Machines for CTR Prediction>
        Paper Link: https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FFM'. Field Bi-Interaction Mode. Union['FFM',]
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            3, 'relu', 0.0,
        )
        if fbi_mode != 'FFM':
            raise MLGBError


class HOFM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='HOFM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3,
                 ):
        """
        Model Name: HOFM(HigherOrderFactorizationMachine)
        Paper Team: NTT
        Paper Year: 2016
        Paper Name: <Higher-Order Factorization Machines>
        Paper Link: https://arxiv.org/pdf/1607.07195v2.pdf,
                    https://dl.acm.org/doi/pdf/10.5555/3157382.3157473

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'HOFM'. Field Bi-Interaction Mode. Union['HOFM',]
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.

        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            fbi_hofm_order, 'relu', 0.0,
        )
        if fbi_mode != 'HOFM':
            raise MLGBError


class FwFM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FwFM', fbi_unit=32, fbi_initializer=None,
                 ):
        """
        Model Name: FwFM(FieldWeightedFactorizationMachine)
        Paper Team: Junwei Pan(Yahoo), etc.
        Paper Year: 2018, 2020
        Paper Name: <Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising>
        Paper Link: https://arxiv.org/pdf/1806.03514.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FwFM'. Field Bi-Interaction Mode. Union['FwFM',]
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            3, 'relu', 0.0,
        )
        if fbi_mode != 'FwFM':
            raise MLGBError


class FEFM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FEFM', fbi_unit=32, fbi_initializer=None,
                 ):
        """
        Model Name: FEFM(FieldEmbeddedFactorizationMachine)
        Paper Team: Harshit Pande(Adobe)
        Paper Year: 2020, 2021
        Paper Name: <FIELD-EMBEDDED FACTORIZATION MACHINES FOR CLICK-THROUGH RATE PREDICTION>
        Paper Link: https://arxiv.org/pdf/2009.09931v2.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FEFM'. Field Bi-Interaction Mode. Union['FEFM',]
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            3, 'relu', 0.0,
        )
        if fbi_mode != 'FEFM':
            raise MLGBError


class FmFM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FmFM', fbi_unit=32, fbi_initializer=None,
                 ):
        """
        Model Name: FmFM(FieldMatrixedFactorizationMachine)
        Paper Team: Yahoo
        Paper Year: 2021
        Paper Name: <FM^2: Field-matrixed Factorization Machines for Recommender Systems>
        Paper Link: https://arxiv.org/pdf/2102.12994v2.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FmFM'. Field Bi-Interaction Mode. Union['FvFM', 'FmFM']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            3, 'relu', 0.0,
        )
        if fbi_mode not in ('FvFM', 'FmFM'):
            raise MLGBError


class AFM(FMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='AFM', fbi_unit=32, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 ):
        """
        Model Name: AFM(AttentionalFactorizationMachine)
        Paper Team: ZJU&NUS(Jun Xiao(ZJU), Xiangnan He(NUS), etc.)
        Paper Year: 2017
        Paper Name: <Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks>
        Paper Link: https://arxiv.org/pdf/1708.04617.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'AFM'. Field Bi-Interaction Mode. Union['AFM',]
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            fbi_mode, fbi_unit, fbi_initializer,
            3, fbi_afm_activation, fbi_afm_dropout,
        )
        if fbi_mode != 'AFM':
            raise MLGBError


class LFM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 lfm_beta=1.0, fbi_initializer=None,
                 ):
        """
        Model Name: LFM(LorentzianFactorizationMachine)
        Paper Team: EBay
        Paper Year: 2019
        Paper Name: <Learning Feature Interactions with Lorentzian Factorization Machine>
        Paper Link: https://arxiv.org/pdf/1911.09821.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param lfm_beta: float, default 1.0.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__()
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
        self.lfm_fn = LorentzFactorizationMachineLayer(
            lfm_beta=lfm_beta,
            fbi_initializer=fbi_initializer,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        lfm_outputs = self.lfm_fn(embed_3d_tensor)
        outputs = self.task_fn(lfm_outputs)
        return outputs


class IMs(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 im_mode='IM', im_order=3, im_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if im_mode not in ('IM', 'DeepIM'):
            raise MLGBError('im_mode')
        if not (1 <= im_order <= 5):
            raise MLGBError('im_order')

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
        self.im_fn = InteractionMachineLayer(
            im_mode=im_mode,
            im_order=im_order,
            im_initializer=im_initializer,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        im_outputs = self.im_fn(embed_3d_tensor)
        outputs = self.task_fn(im_outputs)
        return outputs


class IM(IMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 im_order=3, im_initializer=None,
                 ):
        """
        Model Name: IM(InteractionMachine)
        Paper Team: Feng Yu(CASIA), RealAI, etc.
        Paper Year: 2020
        Paper Name: <Deep Interaction Machine: A Simple but Effective Model for High-order Feature Interactions>
        Paper Link: https://sci-hub.se/10.1145/3340531.3412077

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param im_order: int, default 3. Union[1, 2, 3, 4, 5]
            :param im_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            'IM', im_order, im_initializer,
            (64, 32), 'relu', 0.0, False, False,
        )


class DeepIM(IMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 im_order=3, im_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DeepIM(DeepInteractionMachine)
        Paper Team: Feng Yu(CASIA), RealAI, etc.
        Paper Year: 2020
        Paper Name: <Deep Interaction Machine: A Simple but Effective Model for High-order Feature Interactions>
        Paper Link: https://sci-hub.se/10.1145/3340531.3412077

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param im_order: int, default 3. Union[1, 2, 3, 4, 5]
            :param im_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__(
            feature_names, task, device,
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense,
            embed_dim, embed_2d_dim, embed_initializer,
            pool_mv_mode, pool_mv_axis, pool_mv_initializer,
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2,
            'DeepIM', im_order, im_initializer,
            dnn_hidden_units, dnn_activation, dnn_dropout, dnn_if_bn, dnn_if_ln,
        )


class IFMs(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 ifm_mode_if_dual=False,
                 fbi_mode='FM3D', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 fen_hidden_units=(64, 32), fen_activation='relu', fen_dropout=0.0,
                 fen_if_bn=False, fen_if_ln=False,
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=False, trm_mha_initializer=None, trm_residual_dropout=0.0,
                 ):
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode == 'FM':
            raise MLGBError
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.ifm_fn = AllInputFactorizationMachineLayer(
            ifm_mode_if_dual=ifm_mode_if_dual,
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            fen_hidden_units=fen_hidden_units, 
            fen_activation=fen_activation, 
            fen_dropout=fen_dropout, 
            fen_if_bn=fen_if_bn, 
            fen_if_ln=fen_if_ln, 
            trm_mha_head_num=trm_mha_head_num, 
            trm_mha_head_dim=trm_mha_head_dim, 
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer, 
            trm_residual_dropout=trm_residual_dropout, 
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        ifm_outputs = self.ifm_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(ifm_outputs)
        return outputs
    
    
class IFM(IFMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 fen_hidden_units=(64, 32), fen_activation='relu', fen_dropout=0.0,
                 fen_if_bn=False, fen_if_ln=False,
                 ):
        """
        Model Name: IFM(InputFactorizationMachine)
        Paper Team: THU
        Paper Year: 2019
        Paper Name: <An Input-aware Factorization Machine for Sparse Prediction>
        Paper Link: https://www.ijcai.org/proceedings/2019/0203.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM3D'. Field Bi-Interaction Mode. Union[
                                'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param fen_hidden_units: Tuple[int], default (64, 32).
            :param fen_activation: Optional[str], default 'tanh'.
            :param fen_dropout: float, default 0.0.
            :param fen_if_bn: bool, default False. Batch Normalization.
            :param fen_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__(
            feature_names, task, device, 
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense, 
            embed_dim, embed_2d_dim, embed_initializer, 
            pool_mv_mode, pool_mv_axis, pool_mv_initializer, 
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2, 
            False, 
            fbi_mode, fbi_unit, fbi_initializer, 
            fbi_hofm_order, fbi_afm_activation, fbi_afm_dropout, 
            fen_hidden_units, fen_activation, fen_dropout, 
            fen_if_bn, fen_if_ln, 
            8, 32,
            False, None, 0.0,
        )
        
        
class DIFM(IFMs):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 fen_hidden_units=(64, 32), fen_activation='relu', fen_dropout=0.0,
                 fen_if_bn=False, fen_if_ln=False,
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=False, trm_mha_initializer=None, trm_residual_dropout=0.0,
                 ):
        """
        Model Name: IFM(InputFactorizationMachine)
        Paper Team: THU
        Paper Year: 2020
        Paper Name: <A Dual Input-aware Factorization Machine for CTR Prediction>
        Paper Link: https://www.ijcai.org/proceedings/2020/0434.pdf,
                    https://dl.acm.org/doi/pdf/10.5555/3491440.3491874

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM3D'. Field Bi-Interaction Mode. Union[
                                'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.

            :param fen_hidden_units: Tuple[int], default (64, 32).
            :param fen_activation: Optional[str], default 'tanh'.
            :param fen_dropout: float, default 0.0.
            :param fen_if_bn: bool, default False. Batch Normalization.
            :param fen_if_ln: bool, default False. Layer Normalization.

            :param trm_mha_head_num: int, default 8. From MultiHeadAttention of Transformer.
            :param trm_mha_head_dim: int, default 32.
            :param trm_mha_if_mask: bool, default False.
            :param trm_mha_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_residual_dropout: float, default 0.0.
        """
        super().__init__(
            feature_names, task, device, 
            inputs_if_multivalued, inputs_if_sequential, inputs_if_embed_dense, 
            embed_dim, embed_2d_dim, embed_initializer, 
            pool_mv_mode, pool_mv_axis, pool_mv_initializer, 
            pool_seq_mode, pool_seq_axis, pool_seq_initializer,
            model_l1, model_l2, 
            True, 
            fbi_mode, fbi_unit, fbi_initializer, 
            fbi_hofm_order, fbi_afm_activation, fbi_afm_dropout,
            fen_hidden_units, fen_activation, fen_dropout, 
            fen_if_bn, fen_if_ln,
            trm_mha_head_num, trm_mha_head_dim, 
            trm_mha_if_mask, trm_mha_initializer, trm_residual_dropout
        )


class FNN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='tanh', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: FNN(FactorizationMachineNeuralNetwork)
        Paper Team: UCL(Weinan Zhang(UCL, SJTU), etc.)
        Paper Year: 2016
        Paper Name: <Deep Learning over Multi-field Categorical Data – A Case Study on User Response Prediction>
        Paper Link: https://arxiv.org/pdf/1601.02376.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM3D'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'tanh'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.fnn_fn = FactorizationMachineNeuralNetworkLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        fnn_outputs = self.fnn_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(fnn_outputs)
        return outputs


class PNN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='PNN:both', fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: PNN(ProductNeuralNetwork)
        Paper Team: SJTU&UCL(Yanru Qu(SJTU), Weinan Zhang(SJTU, UCL), etc.)
        Paper Year: 2016
        Paper Name: <Product-based Neural Networks for User Response>
        Paper Link: https://arxiv.org/pdf/1611.00144.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'PNN:both'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32). fbi_unit = dnn_hidden_units[0].
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')
        if len(dnn_hidden_units) < 2:
            raise MLGBError('dnn_hidden_units')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.pnn_fn = ProductNeuralNetworkLayer(
            fbi_mode=fbi_mode,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        pnn_outputs = self.pnn_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(pnn_outputs)
        return outputs


class PIN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 pin_parallel_num=4,
                 fbi_mode='PNN:both', fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=True,
                 ):
        """
        Model Name: PIN(ProductNetworkInNetwork)
        Paper Team: Huawei(Yanru Qu(Huawei(2017.3-2018.3), SJTU), Weinan Zhang(SJTU, UCL), etc.)
        Paper Year: 2018
        Paper Name: <Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data>
        Paper Link: https://arxiv.org/pdf/1807.00311.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param pin_parallel_num: int, default 4.
            :param fbi_mode: str, default 'PNN:both'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default True. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')
        if len(dnn_hidden_units) < 2:
            raise MLGBError('dnn_hidden_units')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.pin_fn = ProductNetworkInNetworkLayer(
            pin_parallel_num=pin_parallel_num,
            fbi_mode=fbi_mode,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,  #
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        pin_outputs = self.pin_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(pin_outputs)
        return outputs


class ONN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FFM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=True, dnn_if_ln=False,
                 ):
        """
        Model Name: ONN/NFFM(OperationNeuralNetwork)
        Paper Team: NJU
        Paper Year: 2019
        Paper Name: <Operation-aware Neural Networks for User Response Prediction>
        Paper Link: https://arxiv.org/pdf/1904.12579.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FFM'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode == 'FM':
            raise MLGBError
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.onn_fn = OperationNeuralNetworkLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            dnn_activation=dnn_activation,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        onn_outputs = self.onn_fn(embed_3d_tensor)
        outputs = self.task_fn(onn_outputs)
        return outputs


class AFN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 afn_mode_if_ensemble=True,
                 ltl_clip_min=1e-4, ltl_unit=32, ltl_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=True, dnn_if_ln=False, 
                 ensemble_dnn_hidden_units=(64, 32), ensemble_dnn_activation='relu', ensemble_dnn_dropout=0.0,
                 ensemble_dnn_if_bn=True, ensemble_dnn_if_ln=False,
                 ):
        """
        Model Name: AFN(AdaptiveFactorizationNetwork)
        Paper Team: SJTU
        Paper Year: 2019, 2020
        Paper Name: <Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions>
        Paper Link: https://arxiv.org/pdf/1909.03276v2.pdf,
                    https://ojs.aaai.org/index.php/AAAI/article/view/5768/5624

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param afn_mode_if_ensemble: bool, default True.
            :param ltl_clip_min: float, default 1e-4.
            :param ltl_unit: int, default 32.
            :param ltl_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param ensemble_dnn_hidden_units: Tuple[int], default (64, 32).
            :param ensemble_dnn_activation: Optional[str], default 'relu'.
            :param ensemble_dnn_dropout: float, default 0.0.
            :param ensemble_dnn_if_bn: bool, default False. Batch Normalization.
            :param ensemble_dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
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
        self.afn_fn = AdaptiveFactorizationNetworkLayer(
            afn_mode_if_ensemble=afn_mode_if_ensemble,
            ltl_clip_min=ltl_clip_min,
            ltl_unit=ltl_unit,
            ltl_initializer=ltl_initializer,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_bn=dnn_if_bn,
            dnn_if_ln=dnn_if_ln,
            ensemble_dnn_hidden_units=ensemble_dnn_hidden_units,
            ensemble_dnn_activation=ensemble_dnn_activation,
            ensemble_dnn_dropout=ensemble_dnn_dropout,
            ensemble_dnn_if_bn=ensemble_dnn_if_bn,
            ensemble_dnn_if_ln=ensemble_dnn_if_ln,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        afn_outputs = self.afn_fn(embed_3d_tensor)
        outputs = self.task_fn(afn_outputs)
        return outputs


class NFM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM3D', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: NFM(NeuralFactorizationMachine)
        Paper Team: NUS(Xiangnan He(NUS), Tat-Seng Chua(NUS))
        Paper Year: 2017
        Paper Name: <Neural Factorization Machines for Sparse Predictive Analytics>
        Paper Link: https://arxiv.org/pdf/1708.05027.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM3D'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode == 'FM':
            raise MLGBError('fbi_mode')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.nfm_fn = NeuralFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        nfm_outputs = self.nfm_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(nfm_outputs)
        return outputs


class WDL(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: WDL(WideDeepLearning)
        Paper Team: Google(Alphabet)
        Paper Year: 2016
        Paper Name: <Wide & Deep Learning for Recommender Systems>
        Paper Link: https://arxiv.org/pdf/1606.07792.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.wdl_fn = WideDeepLearningLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        wdl_outputs = self.wdl_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(wdl_outputs)
        return outputs


class DeepFM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DeepFM(DeepFactorizationMachine)
        Paper Team: Huawei
        Paper Year: 2017
        Paper Name: <DeepFM: A Factorization-Machine based Neural Network for CTR Prediction>
        Paper Link: https://arxiv.org/pdf/1703.04247.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FM'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dfm_fn = DeepFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        dfm_outputs = self.dfm_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(dfm_outputs)
        return outputs

    
class DeepFEFM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 fbi_mode='FEFM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DeepFEFM(DeepFieldEmbeddedFactorizationMachine)
        Paper Team: Harshit Pande(Adobe)
        Paper Year: 2020, 2021
        Paper Name: <FIELD-EMBEDDED FACTORIZATION MACHINES FOR CLICK-THROUGH RATE PREDICTION>
        Paper Link: https://arxiv.org/pdf/2009.09931v2.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param fbi_mode: str, default 'FEFM'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
            embed_cate_if_output2d=True if fbi_mode == 'FM' else False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.deepfefm_fn = DeepFieldEmbeddedFactorizationMachineLayer(
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        deepfefm_outputs = self.deepfefm_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(deepfefm_outputs)
        return outputs


class FLEN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 flen_group_indices=((0, 1), (2, 3), (4, 5, 6)),
                 fbi_fm_mode='FM3D', fbi_mf_mode='FwFM', fbi_unit=32, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: FLEN(FieldLeveragedEmbeddingNetwork)
        Paper Team: Meitu
        Paper Year: 2019, 2020
        Paper Name: <FLEN: Leveraging Field for Scalable CTR Prediction>
        Paper Link: https://arxiv.org/pdf/1911.04690v4.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param flen_group_indices: tuple(tuple), default ((0, 1), (2, 3), (4, 5, 6)). It means (user, item, context).
            :param fbi_fm_mode: str, default 'FM3D'. Field Bi-Interaction Mode. Union[
                                'FM3D', 'FFM', 'FwFM', 'FEFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_mf_mode: str, default 'FwFM'. Field Bi-Interaction Mode. Union[
                                'FM3D', 'FFM', 'FwFM', 'FEFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if (fbi_fm_mode in ('FM', 'HOFM')) or (fbi_mf_mode in ('FM', 'HOFM')):
            raise MLGBError('fbi_fm_mode')
        if (fbi_fm_mode not in FBIModeList) or (fbi_mf_mode not in FBIModeList):
            raise MLGBError('fbi_mf_mode')

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
        self.flen_fn = FieldLeveragedEmbeddingNetworkLayer(
            flen_group_indices=flen_group_indices,
            fbi_fm_mode=fbi_fm_mode,
            fbi_mf_mode=fbi_mf_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            dnn_dropout=dnn_dropout,
            dnn_if_ln=dnn_if_ln,
            dnn_if_bn=dnn_if_bn,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        flen_outputs = self.flen_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(flen_outputs)
        return outputs


class CCPM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 cnn_filter_nums=(64, 32), cnn_kernel_sizes=(64, 32), cnn_activation='tanh',
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: CCPM(ConvolutionalClickPredictionModel)
        Paper Team: CASIA
        Paper Year: 2015
        Paper Name: <A Convolutional Click Prediction Model>
        Paper Link: http://wnzhang.net/share/rtb-papers/cnn-ctr.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param cnn_filter_nums: Tuple[int], default (64, 32).
            :param cnn_kernel_sizes: Tuple[int], default (64, 32).
            :param cnn_activation: Optional[str], default 'tanh'.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
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
        self.ccpm_fn = ConvolutionalClickPredictionModelLayer(
            cnn_filter_nums=cnn_filter_nums,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_activation=cnn_activation,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)

        ccpm_outputs = self.ccpm_fn(embed_3d_tensor)
        outputs = self.task_fn(ccpm_outputs)
        return outputs


class FGCNN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 cnn_filter_nums=(64, 32), cnn_kernel_sizes=(64, 32), cnn_pool_sizes=(2, 2), cnn_activation='tanh',
                 recomb_dnn_hidden_units=(64, 32), recomb_dnn_activation='tanh',
                 fbi_mode='PNN:inner_product', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: FGCNN(FeatureGenerationByConvolutionalNeuralNetwork)
        Paper Team: Huawei
        Paper Year: 2019
        Paper Name: <Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction>
        Paper Link: https://arxiv.org/pdf/1904.04447.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param cnn_filter_nums: Tuple[int], default (64, 32).
            :param cnn_kernel_sizes: Tuple[int], default (64, 32).
            :param cnn_pool_sizes: Tuple[int], default (2, 2).
            :param cnn_activation: Optional[str], default 'tanh'.
            :param recomb_dnn_hidden_units: Tuple[int], default (64, 32).
            :param recomb_dnn_activation: Optional[str], default 'tanh'.
            :param fbi_mode: str, default 'PNN:inner_product'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode == 'FM':
            raise MLGBError
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')

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
        self.fgcnn_fn = FeatureGenerationByConvolutionalNeuralNetworkLayer(
            cnn_filter_nums=cnn_filter_nums,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_pool_sizes=cnn_pool_sizes,
            cnn_activation=cnn_activation,
            recomb_dnn_hidden_units=recomb_dnn_hidden_units,
            recomb_dnn_activation=recomb_dnn_activation,
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        fgcnn_outputs = self.fgcnn_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(fgcnn_outputs)
        return outputs


class XDeepFM(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=32, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 cin_interaction_num=4, cin_interaction_ratio=1.0,
                 cnn_filter_num=64, cnn_kernel_size=64, cnn_activation='relu',
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: XDeepFM(ExtremeDeepFactorizationMachine)
        Paper Team: Microsoft(Jianxun Lian(USTC, Microsoft(2018.7-Present)), etc.)
        Paper Year: 2018
        Paper Name: <xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems>
        Paper Link: https://arxiv.org/pdf/1803.05170v3.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param cin_interaction_num: int, default 4.
            :param cin_interaction_ratio: float, default 1.0. Must 0.5 <= cin_interaction_ratio <= 1.0.
            :param cnn_filter_num: int, default 64.
            :param cnn_kernel_size: int, default 64.
            :param cnn_activation: Optional[str], default 'relu'.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError
        if not (embed_dim and embed_2d_dim):
            raise MLGBError('embed_dim or embed_2d_dim')

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
        self.xdeepfm_fn = ExtremeDeepFactorizationMachineLayer(
            cin_interaction_num=cin_interaction_num,
            cin_interaction_ratio=cin_interaction_ratio,
            cnn_filter_num=cnn_filter_num,
            cnn_kernel_size=cnn_kernel_size,
            cnn_activation=cnn_activation,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        xdeepfm_outputs = self.xdeepfm_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(xdeepfm_outputs)
        return outputs


class FiBiNet(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 sen_pool_mode='Pooling:average', sen_reduction_factor=2, sen_activation='relu', sen_initializer=None,
                 fbi_mode='Bilinear:field_interaction', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: FiBiNet(FeatureImportanceBilinearInteractionNetwork)
        Paper Team: Weibo(Sina)
        Paper Year: 2019
        Paper Name: <FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction>
        Paper Link: https://arxiv.org/pdf/1905.09433.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param sen_pool_mode: str, default 'Pooling:average'. SqueezeExcitationNetwork. Union['Pooling:max',
                                'Pooling:average', 'Pooling:sum']
            :param sen_reduction_factor: int, default 2.
            :param sen_activation: Optional[str], default 'relu'.
            :param sen_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_mode: str, default 'Bilinear:field_interaction'. Field Bi-Interaction Mode. Union[
                                'FM', 'FM3D', 'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
                                'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
                                'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
            :param fbi_unit: int, default 32. length of latent_vector of fbi_weight.
            :param fbi_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param fbi_hofm_order: int, default 3. Must >= 3.
            :param fbi_afm_activation: Optional[str], default 'relu'.
            :param fbi_afm_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if fbi_mode == 'FM':
            raise MLGBError
        if fbi_mode not in FBIModeList:
            raise MLGBError('fbi_mode')
        if sen_pool_mode not in PoolModeList:
            raise MLGBError('sen_pool_mode')
        if not (isinstance(sen_reduction_factor, int) and sen_reduction_factor >= 1):
            raise MLGBError('sen_reduction_factor')

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
        self.fbn_fn = FeatureImportanceBilinearInteractionNetworkLayer(
            sen_pool_mode=sen_pool_mode,
            sen_reduction_factor=sen_reduction_factor,
            sen_activation=sen_activation,
            sen_initializer=sen_initializer,
            fbi_mode=fbi_mode,
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            fbi_hofm_order=fbi_hofm_order,
            fbi_afm_activation=fbi_afm_activation,
            fbi_afm_dropout=fbi_afm_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        dense_2d_tensor, embed_3d_tensor = self.input_fn(inputs)

        fbn_outputs = self.fbn_fn([dense_2d_tensor, embed_3d_tensor])
        outputs = self.task_fn(fbn_outputs)
        return outputs


class AutoInt(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=True,
                 embed_dim=32, embed_2d_dim=None, embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 trm_layer_num=1, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_residual_dropout=0.0,
                 ):
        """
        Model Name: AutoInt(AutomaticFeatureInteractionLearning)
        Paper Team: PKU
        Paper Year: 2018, 2019
        Paper Name: <AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks>
        Paper Link: https://arxiv.org/pdf/1810.11921v2.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
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
            :param trm_layer_num: int, default 1. From Transformer.
            :param trm_mha_head_num: int, default 8. From MultiHeadAttention of Transformer.
            :param trm_mha_head_dim: int, default 32.
            :param trm_mha_if_mask: bool, default True.
            :param trm_mha_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_residual_dropout: float, default 0.0.
        """
        super().__init__()
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if not (trm_mha_head_num > 0 and trm_mha_head_dim > 0):
            raise MLGBError('trm_mha_head_num or trm_mha_head_dim')

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
        self.autoint_fn = AutomaticFeatureInteractionLearningLayer(
            trm_layer_num=trm_layer_num,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_residual_dropout=trm_residual_dropout,
            device=device,
        )
        self.task_fn = TaskLayer(
            task=task,
            task_linear_if_identity=False,
            task_linear_if_weighted=True,
            task_linear_if_bias=True,
            task_multiclass_if_project=True,
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        _, embed_3d_tensor = self.input_fn(inputs)
        query_3d_tensor = key_3d_tensor = embed_3d_tensor

        autoint_ouputs = self.autoint_fn([query_3d_tensor, key_3d_tensor])
        outputs = self.task_fn(autoint_ouputs)
        return outputs

















class GRU4Rec(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 seq_rec_pointwise_mode='Add',
                 gru_hidden_units=(64, 32), gru_dropout=0.0, 
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: GRU4Rec(GatedRecurrentUnit4Recommendation)
        Paper Team: Telefonica
        Paper Year: 2015, 2016
        Paper Name: <Session-based Recommendations with Recurrent Neural Networks>
        Paper Link: https://arxiv.org/pdf/1511.06939.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param seq_rec_pointwise_mode: str, default 'Add'. Union['Add', 'LabelAttention', 'Add&LabelAttention']
            :param gru_hidden_units: Tuple[int], default (64, 32).
            :param gru_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        self.gru_hidden_units = gru_hidden_units + (embed_dim,)
        if self.gru_hidden_units[-1] != embed_dim:
            raise MLGBError
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError('seq_rec_pointwise_mode')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.gru4rec_fn = GatedRecurrentUnit4RecommendationLayer(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            gru_hidden_units=self.gru_hidden_units,
            gru_dropout=gru_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        gru4rec_outputs = self.gru4rec_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(gru4rec_outputs)
        return outputs


class Caser(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 seq_rec_pointwise_mode='Add',
                 cnn_filter_num=64, cnn_kernel_size=4, cnn_pool_size=2,
                 cnn_activation='tanh', cnn_l2=0.0, cnn_initializer=None,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: Caser(ConvolutionalSequenceEmbeddingRecommendation)
        Paper Team: SFU
        Paper Year: 2018
        Paper Name: <Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding>
        Paper Link: https://arxiv.org/pdf/1809.07426.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param seq_rec_pointwise_mode: str, default 'Add'. Union['Add', 'LabelAttention', 'Add&LabelAttention']
            :param cnn_filter_num: int, default 64.
            :param cnn_kernel_size: int, default 4.
            :param cnn_pool_size: int, default 2.
            :param cnn_activation: Optional[str], default 'tanh'.
            :param cnn_l2: float, default 0.0.
            :param cnn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError('seq_rec_pointwise_mode')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.caser_fn = ConvolutionalSequenceEmbeddingRecommendationLayer(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            cnn_filter_num=cnn_filter_num,
            cnn_kernel_size=cnn_kernel_size,
            cnn_pool_size=cnn_pool_size,
            cnn_activation=cnn_activation,
            cnn_l2=cnn_l2,
            cnn_initializer=cnn_initializer,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        caser_outputs = self.caser_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(caser_outputs)
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 seq_rec_pointwise_mode='Add',
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: SASRec(SelfAttentiveSequentialRecommendatio)
        Paper Team: UCSD
        Paper Year: 2018
        Paper Name: <Self-Attentive Sequential Recommendation>
        Paper Link: https://arxiv.org/pdf/1808.09781.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param seq_rec_pointwise_mode: str, default 'Add'. Union['Add', 'LabelAttention', 'Add&LabelAttention']
            :param trm_mha_head_num: int, default 4.
            :param trm_mha_head_dim: int, default 32.
            :param trm_mha_if_mask: bool, default True.
            :param trm_mha_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_if_ffn: bool, default True.
            :param trm_ffn_activation: Optional[str], default 'gelu'.
            :param trm_ffn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_residual_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError('seq_rec_pointwise_mode')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.sasrec_fn = SelfAttentiveSequentialRecommendationLayer(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        sasrec_outputs = self.sasrec_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(sasrec_outputs)
        return outputs


class BERT4Rec(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 seq_rec_pointwise_mode='Add',
                 trm_num=4, trm_if_pe=True, trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: BERT4Rec(BidirectionalEncoderRepresentationTransformer4Recommendation)
        Paper Team: Alibaba
        Paper Year: 2019
        Paper Name: <BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer>
        Paper Link: https://arxiv.org/pdf/1904.06690.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param seq_rec_pointwise_mode: str, default 'Add'. Union['Add', 'LabelAttention', 'Add&LabelAttention']
            :param trm_num: int, default 4.
            :param trm_if_pe: bool, default True.
            :param trm_mha_head_num: int, default 4.
            :param trm_mha_head_dim: int, default 32.
            :param trm_mha_if_mask: bool, default True.
            :param trm_mha_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_if_ffn: bool, default True.
            :param trm_ffn_activation: Optional[str], default 'gelu'.
            :param trm_ffn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_residual_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if seq_rec_pointwise_mode not in SeqRecPointwiseModeList:
            raise MLGBError('seq_rec_pointwise_mode')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.bert4rec_fn = BidirectionalEncoderRepresentationTransformer4RecommendationLayer(
            seq_rec_pointwise_mode=seq_rec_pointwise_mode,
            trm_num=trm_num,
            trm_if_pe=trm_if_pe,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        bert4rec_outputs = self.bert4rec_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(bert4rec_outputs)
        return outputs


class BST(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 trm_mha_head_num=8, trm_mha_head_dim=32,
                 trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None,
                 trm_residual_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='selu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: BST(BehaviorSequenceTransformer)
        Paper Team: Alibaba
        Paper Year: 2019
        Paper Name: <Behavior Sequence Transformer for E-commerce Recommendation in Alibaba>
        Paper Link: https://arxiv.org/pdf/1905.06874.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param trm_mha_head_num: int, default 8.
            :param trm_mha_head_dim: int, default 32.
            :param trm_mha_if_mask: bool, default True.
            :param trm_mha_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_if_ffn: bool, default True.
            :param trm_ffn_activation: Optional[str], default 'gelu'.
            :param trm_ffn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_residual_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'selu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.bst_fn = BehaviorSequenceTransformerLayer(
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        bst_outputs = self.bst_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(bst_outputs)
        return outputs


class DIN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 lau_version='v4', lau_hidden_units=(16,),
                 dnn_hidden_units=(64, 32), dnn_activation='dice', dnn_dropout=0.0, dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DIN(DeepInterestNetwork)
        Paper Team: Alibaba
        Paper Year: 2017, 2018
        Paper Name: <Deep Interest Network for Click-Through Rate Prediction>
        Paper Link: v1: https://arxiv.org/pdf/1706.06978v1.pdf; v4: https://arxiv.org/pdf/1706.06978v4.pdf.

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param lau_version: str, default 'v4'. From LocalActivationUnit. Union['v1', 'v4']
            :param lau_hidden_units: Tuple[int], default (16,).
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_dropout: float, default 0.0.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
        """
        super().__init__()
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if lau_version not in ('v1', 'v4'):
            raise MLGBError('lau_version')
        if not len(lau_hidden_units) > 0:
            raise MLGBError('lau_hidden_units')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.din_fn = DeepInterestNetworkLayer(
            lau_version=lau_version,
            lau_hidden_units=lau_hidden_units,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        key_3d_tensor = seq_3d_tensor
        query_3d_tensor = torch.sum(item_embed_3d_tensor, dim=1, keepdim=True)
        query_3d_tensor = torch.tile(query_3d_tensor, dims=[1, key_3d_tensor.shape[1], 1])

        din_outputs = self.din_fn([user_embed_3d_tensor, query_3d_tensor, key_3d_tensor])
        outputs = self.task_fn(din_outputs)
        return outputs


class DIEN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 gru_hidden_units=(64, 32), gru_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='dice', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DIEN(DeepInterestEvolutionNetwork)
        Paper Team: Alibaba
        Paper Year: 2018
        Paper Name: <Deep Interest Evolution Network for Click-Through Rate Prediction>
        Paper Link: https://arxiv.org/pdf/1809.03672.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param gru_hidden_units: Tuple[int], default (64, 32).
            :param gru_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dien_fn = DeepInterestEvolutionNetworkLayer(
            gru_hidden_units=gru_hidden_units,
            gru_dropout=gru_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        dien_outputs = self.dien_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(dien_outputs)
        return outputs


class DSIN(torch.nn.Module):
    def __init__(self, feature_names, task='binary', device='cpu',
                 user_inputs_if_multivalued=False, user_inputs_if_sequential=True,  #
                 item_inputs_if_multivalued=False, item_inputs_if_embed_dense=False,
                 embed_dim=32,  embed_initializer=None,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=1,  pool_seq_initializer=None,
                 model_l1=0.0, model_l2=0.0,
                 dsin_if_process_session=False, session_pool_mode='Pooling:average', session_size=4, session_stride=2,
                 bias_initializer='zeros',
                 trm_mha_head_num=4, trm_mha_head_dim=32, trm_mha_if_mask=True, trm_mha_initializer=None,
                 trm_if_ffn=True, trm_ffn_activation='gelu', trm_ffn_initializer=None, trm_residual_dropout=0.0,
                 gru_bi_mode='Frontward,Backward', gru_hidden_units=(64, 32), gru_dropout=0.0,
                 dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_dropout=0.0,
                 dnn_if_bn=False, dnn_if_ln=False,
                 ):
        """
        Model Name: DSIN(DeepSessionInterestNetwork)
        Paper Team: Alibaba
        Paper Year: 2019
        Paper Name: <Deep Session Interest Network for Click-Through Rate Prediction>
        Paper Link: https://arxiv.org/pdf/1905.06482.pdf

        Task Inputs Parameters:
            :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
            :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
            :param device: Optional[int], default None.
            :param user_inputs_if_multivalued: bool, default False.
            :param user_inputs_if_sequential: bool, default True.
            :param item_inputs_if_multivalued: bool, default False.
            :param item_inputs_if_embed_dense: bool, default False.
            :param embed_dim: int, default 32.
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
            :param dsin_if_process_session: bool, default False.
            :param session_pool_mode: str, default 'Pooling:average'. Union['Pooling:max', 'Pooling:average', 'Pooling:sum']
            :param session_size: int, default 4.
            :param session_stride: int, default 2.
            :param bias_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_mha_head_num: int, default 4.
            :param trm_mha_head_dim: int, default 32.
            :param trm_mha_if_mask: bool, default True.
            :param trm_mha_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_if_ffn: bool, default True.
            :param trm_ffn_activation: Optional[str], default 'gelu'.
            :param trm_ffn_initializer: Optional[str], default None. When None, activation judge first, he_uniform end.
            :param trm_residual_dropout: float, default 0.0.
            :param gru_bi_mode: str, default 'Frontward+Backward'. Union['Frontward', 'Backward', 'Frontward+Backward',
                            'Frontward-Backward', 'Frontward*Backward', 'Frontward,Backward']
            :param gru_hidden_units: Tuple[int], default (64, 32).
            :param gru_dropout: float, default 0.0.
            :param dnn_hidden_units: Tuple[int], default (64, 32).
            :param dnn_activation: Optional[str], default 'relu'.
            :param dnn_if_bn: bool, default False. Batch Normalization.
            :param dnn_if_ln: bool, default False. Layer Normalization.
            :param dnn_dropout: float, default 0.0.
        """
        super().__init__()
        self.gru_hidden_units = gru_hidden_units + (embed_dim,)
        if self.gru_hidden_units[-1] != embed_dim:
            raise MLGBError
        if not user_inputs_if_sequential:
            raise MLGBError('user_inputs_if_sequential')
        if pool_mv_mode not in MVPoolModeList:
            raise MLGBError('pool_mv_mode')
        if pool_seq_mode not in MVPoolModeList:
            raise MLGBError('pool_seq_mode')
        if not embed_dim:
            raise MLGBError('embed_dim')
        if session_pool_mode not in PoolModeList:
            raise MLGBError('session_pool_mode')
        if gru_bi_mode not in BiGRUModeList:
            raise MLGBError('gru_bi_mode')

        user_feature_names, item_feature_names = feature_names

        self.user_input_fn = InputsLayer(
            feature_names=user_feature_names,
            inputs_mode='Inputs:sequential',
            inputs_if_multivalued=user_inputs_if_multivalued,
            inputs_if_sequential=user_inputs_if_sequential,  #
            inputs_if_embed_dense=True,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.item_input_fn = InputsLayer(
            feature_names=item_feature_names,
            inputs_mode='Inputs:feature',
            inputs_if_multivalued=item_inputs_if_multivalued,
            inputs_if_sequential=False,
            inputs_if_embed_dense=item_inputs_if_embed_dense,
            outputs_dense_if_add_sparse=True,
            embed_dim=embed_dim,
            embed_initializer=embed_initializer,
            embed_cate_if_output2d=False,
            pool_mv_mode=pool_mv_mode,
            pool_mv_axis=pool_mv_axis,
            pool_mv_initializer=pool_mv_initializer,
            pool_seq_mode=pool_seq_mode,
            pool_seq_axis=pool_seq_axis,
            pool_seq_initializer=pool_seq_initializer,
            device=device,
        )
        self.dsin_fn = DeepSessionInterestNetworkLayer(
            dsin_if_process_session=dsin_if_process_session,
            session_pool_mode=session_pool_mode,
            session_size=session_size,
            session_stride=session_stride,
            bias_initializer=bias_initializer,
            trm_mha_head_num=trm_mha_head_num,
            trm_mha_head_dim=trm_mha_head_dim,
            trm_mha_if_mask=trm_mha_if_mask,
            trm_mha_initializer=trm_mha_initializer,
            trm_if_ffn=trm_if_ffn,
            trm_ffn_activation=trm_ffn_activation,
            trm_ffn_initializer=trm_ffn_initializer,
            trm_residual_dropout=trm_residual_dropout,
            gru_bi_mode=gru_bi_mode,
            gru_hidden_units=self.gru_hidden_units,
            gru_dropout=gru_dropout,
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
            task_multiclass_if_softmax=True,
            task_multiclass_temperature_ratio=None,
            device=device,
        )
        self.l1l2_loss = RegularizationLayer(model=self, l1=model_l1, l2=model_l2).get_l1l2_loss
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, inputs):
        user_inputs, item_inputs = inputs
        _, user_embed_3d_tensor, user_seq_3d_tensor = self.user_input_fn(user_inputs)
        _, item_embed_3d_tensor = self.item_input_fn(item_inputs)

        dsin_outputs = self.dsin_fn([user_embed_3d_tensor, user_seq_3d_tensor, item_embed_3d_tensor])
        outputs = self.task_fn(dsin_outputs)
        return outputs

















