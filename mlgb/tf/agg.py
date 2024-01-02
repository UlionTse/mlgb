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

from mlgb.tf.models.ranking import (
    LR, PLM, MLP, DLRM, MaskNet,
    DCM, DCN, EDCN,
    FM, FFM, HOFM, FwFM, FEFM, FmFM, AFM, LFM, IM, IFM, DIFM,
    FNN, PNN, PIN, ONN, AFN,
    NFM, WDL, DeepFM, DeepFEFM, DeepIM, FLEN,
    CCPM, FGCNN, XDeepFM, FiBiNet, AutoInt,

    GRU4Rec, Caser, SASRec, BERT4Rec,
    BST, DIN, DIEN, DSIN,
)
from mlgb.tf.models.mtl import (
    SharedBottom, ESMM, MMoE, PLE, PEPNet,
)
from mlgb.tf.models.matching import (
    NCF,
    DSSM, MatchFM, EBR, YoutubeDNN, MIND,
)


tf_ranking_models_map = {
    'LR': LR, 'MLP': MLP, 'PLM': PLM, 'DLRM': DLRM, 'MaskNet': MaskNet,
    'DCM': DCM, 'DCN': DCN, 'EDCN': EDCN,
    'FM': FM, 'FFM': FFM, 'HOFM': HOFM, 'FwFM': FwFM, 'FEFM': FEFM, 'FmFM': FmFM, 'AFM': AFM,
    'LFM': LFM, 'IM': IM, 'IFM': IFM, 'DIFM': DIFM,
    'FNN': FNN, 'PNN': PNN, 'PIN': PIN, 'ONN': ONN, 'AFN': AFN,
    'NFM': NFM, 'WDL': WDL, 'DeepFM': DeepFM, 'DeepFEFM': DeepFEFM, 'DeepIM': DeepIM, 'FLEN': FLEN,
    'CCPM': CCPM, 'FGCNN': FGCNN, 'XDeepFM': XDeepFM, 'FiBiNet': FiBiNet, 'AutoInt': AutoInt,
    'GRU4Rec': GRU4Rec, 'Caser': Caser, 'SASRec': SASRec, 'BERT4Rec': BERT4Rec,
    'BST': BST, 'DIN': DIN, 'DIEN': DIEN, 'DSIN': DSIN,

    # 'DNN': MLP, 'MLR': PLM, 'DeepCross': DCM, 'NFFM': ONN,
}
tf_mtl_models_map = {
    'SharedBottom': SharedBottom, 'ESMM': ESMM, 'MMoE': MMoE, 'PLE': PLE, 'PEPNet': PEPNet,
}
tf_matching_models_map = {
    'NCF': NCF,
    'DSSM': DSSM, 'MatchFM': MatchFM, 'EBR': EBR, 'YoutubeDNN': YoutubeDNN, 'MIND': MIND,
}

