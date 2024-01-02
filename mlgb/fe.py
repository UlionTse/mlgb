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

import pandas
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, KBinsDiscretizer


def transform_dense_features_to_quantile(numerical_features, n_quantiles=100, quantile_distribution='uniform', seed=None):
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=quantile_distribution, random_state=seed)
    qt.fit(numerical_features)
    return pandas.DataFrame(qt.transform(numerical_features), columns=numerical_features.columns)


def transform_dense_features_to_discrete(numerical_features, n_bins=10, strategy='quantile', encode='onehot', seed=None):
    kbd = KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode=encode, random_state=seed)
    kbd.fit(numerical_features)
    if encode == 'onehot':
        kbd_out = kbd.transform(numerical_features).toarray()
    else:
        kbd_out = kbd.transform(numerical_features)
    names = [f'discrete:{col}' for col in kbd.get_feature_names_out()]
    return pandas.DataFrame(kbd_out, columns=names)


def transform_sparse_features_to_ordinal(categorical_features):
    oe = OrdinalEncoder()
    oe.fit(categorical_features)
    names = [f'ordinal:{col}' for col in oe.get_feature_names_out()]
    return pandas.DataFrame(oe.transform(categorical_features), columns=names)


def transform_sparse_features_to_onehot(categorical_features):
    ohe = OneHotEncoder()
    ohe.fit(categorical_features)
    return pandas.DataFrame(ohe.transform(categorical_features).toarray(), columns=ohe.get_feature_names_out())


def transform_sparse_features_to_interactive(categorical_features, if_need_onehot=True, name_prefix='interactive:'):
    ohe_features = transform_sparse_features_to_onehot(categorical_features) if if_need_onehot else categorical_features
    ohe_col_names = ohe_features.columns
    for col1 in ohe_col_names:
        for col2 in ohe_col_names:
            if col1.split('_')[0] < col2.split('_')[0]:
                ohe_features[f'{name_prefix}{col1}*{col2}'] = ohe_features.apply(lambda x: x[col1] * x[col2], axis=1)
    return ohe_features[[col for col in ohe_features.columns if col.startswith(name_prefix)]]

