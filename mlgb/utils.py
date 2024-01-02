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

import os

import numpy
import pandas
import tensorflow as tf
from mlgb.error import MLGBError


def check_filepath(*filepath):
    for fp in filepath:
        if not os.path.exists(fp):
            os.mkdir(fp)
    return


def get_dense_feature_name_dict(feature_name):
    return {'feature_name': feature_name}


def get_sparse_feature_name_dict(feature_name, feature_nunique, input_length, embed_dim=None, embed_alpha=6, mask_zero=True):
    # embed_dim: https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
    # embed_alpha=6: when embed_feature_nunique is in MinMax[4, 1e4], then embed_dim in MinMax[8, 60].
    name_dict = {
        'feature_name': feature_name,
        'input_length': input_length,
        'mask_zero': mask_zero,
        'feature_nunique': feature_nunique,
        'embed_feature_nunique': feature_nunique + 1 if mask_zero else feature_nunique,
        'embed_dim': embed_dim if embed_dim else int((feature_nunique + 1) ** 0.25 * embed_alpha),
    }
    return name_dict


def get_embed_width(sparse_feature_names, embed_dim):
    if embed_dim:
        return int(len(sparse_feature_names) * embed_dim)
    return int(sum([feat_dict['embed_dim'] for feat_dict in sparse_feature_names]))


def get_onehot_width(sparse_feature_names):
    return sum([feat_dict['feature_nunique'] for feat_dict in sparse_feature_names])


def get_padded_sequence(sequential_inputs, mode='numpy', sequence_max_length=8, sequence_pad_value=0.0):
    if mode not in ('numpy', 'pandas'):
        raise MLGBError

    _, n_sequences = sequential_inputs.shape

    sequence_pool = []
    for i in range(n_sequences):
        sequence_i = tf.keras.utils.pad_sequences(
            sequences=sequential_inputs[:, i] if mode == 'numpy' else sequential_inputs.iloc[:, i],
            maxlen=sequence_max_length,
            padding='pre',  # Union['pre', 'post']
            truncating='pre',
            value=sequence_pad_value,
        )
        sequence_pool.append(sequence_i)

    if mode == 'numpy':
        sequential_outputs = numpy.stack(sequence_pool, axis=1)  # (batch_size, features_width, sequence_length)
    else:
        sequential_outputs = pandas.DataFrame({
            sequential_inputs.columns[i]: sequence_pool[i].tolist() for i in range(n_sequences)
        })
    return sequential_outputs


