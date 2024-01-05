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

import random

import numpy
import torch


InputsModeList = ['Inputs:feature', 'Inputs:sequential']
PoolModeList = ['Pooling:max', 'Pooling:average', 'Pooling:sum']
MVPoolModeList = ['Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
BiGRUModeList = ['Frontward', 'Backward', 'Frontward+Backward', 'Frontward-Backward', 'Frontward*Backward', 'Frontward,Backward']
SeqRecPointwiseModeList = ['Add', 'LabelAttention', 'Add&LabelAttention']

EDCNModeList = ['EDCN:pointwise_addition', 'EDCN:hadamard_product', 'EDCN:concatenation', 'EDCN:attention_pooling']
PNNModeList = ['PNN:inner_product', 'PNN:outer_product', 'PNN:both']
BLModeList = ['Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction']
FBIModeList = [
    'FM', 'FM3D',
    'FFM', 'HOFM', 'FwFM', 'FEFM', 'FvFM', 'FmFM', 'AFM',
    'PNN:inner_product', 'PNN:outer_product', 'PNN:both',
    'Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction',
]

SampleModeList = [
    'Sample:all', 'Sample:batch',
    # 'Sample:uniform', 'Sample:log_uniform', 'Sample:fixed_unigram', 'Sample:learned_unigram',
]










