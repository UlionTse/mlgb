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

import numpy
import pandas
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlgb.utils import (
    get_dense_feature_name_dict,
    get_sparse_feature_name_dict,
    get_padded_sequence,
)
from mlgb.fe import (
    transform_dense_features_to_quantile,
    transform_dense_features_to_discrete,
    transform_sparse_features_to_ordinal,
)
from mlgb.error import MLGBError


pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)


def make_sequence_data(
        n_samples=10,
        n_sequence_features=3,
        sequence_if_stable_length=False,
        sequence_max_length=10,
        sequence_choice_if_replace=False,
        vocabulary_size=None,
        ):
    if not (2 < n_sequence_features < 10):
        raise MLGBError

    def sequence_fn(x):
        sequence_min_length = max(1, sequence_max_length - n_sequence_features)
        sequence_size = numpy.random.randint(sequence_min_length, sequence_max_length + 1)
        x = numpy.random.choice(x, size=sequence_size, replace=sequence_choice_if_replace)
        return x

    def id_fn(x, i):  # (cate_id > shop_id > good_id)
        x = numpy.array(x)
        x = numpy.round(x // ((vocabulary_size ** (1 / n_sequence_features)) ** i))
        x = numpy.where(x >= 1, x, 1)
        x = x.astype(numpy.int32)
        return x

    vocabulary_size = vocabulary_size if vocabulary_size else int(1e3)
    vocabulary_size = vocabulary_size if vocabulary_size >= sequence_max_length else sequence_max_length

    sequence_id_pool = list(range(1, vocabulary_size))  # omit building dictionary
    df = pandas.DataFrame({'sequence': [sequence_id_pool] * n_samples})
    df['sequence'] = df['sequence'].apply(sequence_fn)
    for i in range(n_sequence_features):
        df[f'sequence_{i}'] = df['sequence'].apply(lambda x: id_fn(x, i))

    df = df.iloc[:, 1:]
    if sequence_if_stable_length:
        df = get_padded_sequence(df, mode='pandas', sequence_max_length=sequence_max_length)
    return df  # (b, f, s): (b, s, f) -> (b, f, s) -> (b, f, s, e) -> seq:(b, s, e) or mv:(b, f, e).


def make_fake_data(
        n_samples=1000,
        task='binary',
        n_classes=2,
        class_weights=(0.9,),
        n_numerical_features=30,
        n_categorical_features=5,
        n_sequential_features=5,
        sequence_if_stable_length=False,
        sequence_max_length=5,
        sequence_choice_if_replace=False,
        multiclass_features_if_add_item_id=False,
        multitask_cvr=0.1,
        vocabulary_size=None,
        shuffle=True,
        seed=None
        ):

    if isinstance(task, str) and task not in ('binary', 'regression'):
        raise MLGBError
    if isinstance(task, (tuple, list)) and tuple(task) != ('binary', 'binary',):
        raise MLGBError

    def regression_y_fn(y):
        betas = numpy.random.uniform(low=0.0, high=0.5, size=y.shape)
        y = numpy.where(y >= 0.5, y-betas, y+betas)
        return y

    def ctr_cvr_binary_y_fn(y, cvr=0.5):
        y_ctr = y.reshape(-1, 1)
        betas = numpy.random.choice([0, 1], p=(1 - cvr, cvr), size=y_ctr.shape, replace=True)
        y_cvr = numpy.where((y_ctr == 1) & (betas == 1), 1, 0)
        y_ctr_cvr = numpy.concatenate([y_ctr, y_cvr], axis=1)
        return y_ctr_cvr

    vocabulary_size = vocabulary_size if vocabulary_size else int(n_samples // 10)
    category_pool = [f'id_{i}' for i in range(1, vocabulary_size+1)]

    if n_classes <= 2:
        n_features = n_numerical_features
    else:
        n_features = max([n_numerical_features, n_classes + 3])

    numerical_features, y = make_classification(
        n_classes=n_classes,
        weights=class_weights,
        n_samples=n_samples,
        n_features=n_features,  # >= n_classes + 3 if n_classes > 2 else `any`.
        n_informative=n_classes,
        n_redundant=2,
        n_repeated=0,
        shuffle=shuffle,
        random_state=seed,
    )

    if n_features > n_numerical_features:
        numerical_features = numerical_features[:, :n_numerical_features]

    if task == 'regression':
        y = regression_y_fn(y)
    if isinstance(task, (tuple, list)) and tuple(task) == ('binary', 'binary',):
        y = ctr_cvr_binary_y_fn(y, multitask_cvr)

    numerical_features_names = [f'numerical:c{i + 1}' for i in range(n_numerical_features)]
    categorical_features_names = [f'categorical:c{i + 1}' for i in range(n_categorical_features)]
    sequential_features_names = [f'sequential:c{i + 1}' for i in range(n_sequential_features)]

    numerical_features = pandas.DataFrame(numerical_features, columns=numerical_features_names)
    categorical_features = numpy.random.choice(category_pool, size=[n_samples, n_categorical_features], replace=True)
    categorical_features = pandas.DataFrame(categorical_features, columns=categorical_features_names)
    if multiclass_features_if_add_item_id and n_classes > 2:
        categorical_features['y'] = y

    sequential_features = make_sequence_data(
        n_samples=n_samples,
        n_sequence_features=n_sequential_features,
        sequence_if_stable_length=sequence_if_stable_length,
        sequence_max_length=sequence_max_length,
        sequence_choice_if_replace=sequence_choice_if_replace,
        vocabulary_size=vocabulary_size,
    )
    sequential_features.columns = sequential_features_names

    data = {
        'x': pandas.concat([numerical_features, categorical_features, sequential_features], axis=1),
        'y': y,
        'is_multiclass': True if n_classes > 2 else False,
        'numerical_features_names': numerical_features_names,
        'categorical_features_names': categorical_features_names,
        'sequential_features_names': sequential_features_names,
        'sequence_max_length': sequence_max_length,
        # 'sequence_vocabulary_size': vocabulary_size,
    }
    return data


def make_input_data(data, task='binary', test_size=0.15, inputs_if_2_groups=False, multiclass_item_features_if_only_item_id=False,  seed=None):
    sparse_ord_features_from_numerical = transform_dense_features_to_discrete(
        numerical_features=data['x'][data['numerical_features_names']],
        n_bins=20,
        strategy='quantile',
        encode='ordinal',
        seed=seed
    )
    sparse_ord_features_from_categorical = transform_sparse_features_to_ordinal(
        categorical_features=data['x'][data['categorical_features_names']]
    )

    dense_features = transform_dense_features_to_quantile(
        numerical_features=data['x'][data['numerical_features_names']],
        n_quantiles=100,
        seed=seed
    )
    sparse_features = pandas.concat([sparse_ord_features_from_categorical, sparse_ord_features_from_numerical], axis=1)
    sequential_features = data['x'][data['sequential_features_names']]

    if tuple(task) == ('binary', 'binary',):
        label = data['y']
        x_data = pandas.concat([dense_features, sparse_features, sequential_features], axis=1)
        x_cols = list(x_data.columns)
        y_cols = ['y_ctr', 'y_cvr']
        x_data[y_cols] = label
        x_y_data = x_data
        train_data, test_data = train_test_split(x_y_data, test_size=test_size, random_state=seed)
        train_data, y_train = train_data[x_cols], train_data[y_cols].to_numpy()
        test_data, y_test = test_data[x_cols], test_data[y_cols].to_numpy()
    else:
        label = data['y']
        x_data = pandas.concat([dense_features, sparse_features, sequential_features], axis=1)
        train_data, test_data, y_train, y_test = train_test_split(x_data, label, stratify=label, test_size=test_size, random_state=seed)

    if inputs_if_2_groups:
        if multiclass_item_features_if_only_item_id and data['is_multiclass']:
            y_train_data = pandas.DataFrame({'y': y_train}).astype(numpy.int32)
            y_test_data = pandas.DataFrame({'y': y_test}).astype(numpy.int32)

            x1_train = tuple([
                train_data[dense_features.columns].to_numpy().astype(numpy.float32),
                train_data[sparse_features.columns].to_numpy().astype(numpy.int32),
                numpy.array(train_data[sequential_features.columns].to_numpy().tolist()).astype(numpy.int32),
            ])
            x1_test = tuple([
                test_data[dense_features.columns].to_numpy().astype(numpy.float32),
                test_data[sparse_features.columns].to_numpy().astype(numpy.int32),
                numpy.array(test_data[sequential_features.columns].to_numpy().tolist()).astype(numpy.int32),
            ])

            x2_train = tuple([
                y_train_data['y'].to_numpy().reshape(-1, 1).astype(numpy.float32),
                y_train_data['y'].to_numpy().reshape(-1, 1).astype(numpy.int32),
            ])
            x2_test = tuple([
                y_test_data['y'].to_numpy().reshape(-1, 1).astype(numpy.float32),
                y_test_data['y'].to_numpy().reshape(-1, 1).astype(numpy.int32),
            ])
            x_train = (x1_train, x2_train,)
            x_test = (x1_test, x2_test,)

            feature_1_names = tuple([
                tuple([
                    get_dense_feature_name_dict(name) for name in dense_features.columns
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=x_data[name].nunique(),
                        input_length=1,
                    ) for name in sparse_features.columns
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=len(set(it for item in x_data[name].to_numpy() for it in item)),
                        input_length=data['sequence_max_length'],
                    ) for name in sequential_features.columns
                ]),
            ])
            feature_2_names = tuple([
                tuple([
                    get_dense_feature_name_dict(name) for name in y_train_data.columns
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=y_train_data[name].nunique(),
                        input_length=1,
                    ) for name in y_train_data.columns
                ]),
            ])
            feature_names = (feature_1_names, feature_2_names,)
        else:
            id_dense_half = len(dense_features.columns) // 2
            id_sparse_half = len(sparse_features.columns) // 2
            id_sequential_half = len(sequential_features.columns) // 2

            x1_train = tuple([
                train_data[dense_features.columns[:id_dense_half]].to_numpy().astype(numpy.float32),
                train_data[sparse_features.columns[:id_sparse_half]].to_numpy().astype(numpy.int32),
                numpy.array(train_data[sequential_features.columns[:id_sequential_half]].to_numpy().tolist()).astype(numpy.int32),
            ])
            x1_test = tuple([
                test_data[dense_features.columns[:id_dense_half]].to_numpy().astype(numpy.float32),
                test_data[sparse_features.columns[:id_sparse_half]].to_numpy().astype(numpy.int32),
                numpy.array(test_data[sequential_features.columns[:id_sequential_half]].to_numpy().tolist()).astype(numpy.int32),
            ])

            x2_train = tuple([
                train_data[dense_features.columns[id_dense_half:]].to_numpy().astype(numpy.float32),
                train_data[sparse_features.columns[id_sparse_half:]].to_numpy().astype(numpy.int32),
                numpy.array(train_data[sequential_features.columns[id_sequential_half:]].to_numpy().tolist()).astype(numpy.int32),
            ])
            x2_test = tuple([
                test_data[dense_features.columns[id_dense_half:]].to_numpy().astype(numpy.float32),
                test_data[sparse_features.columns[id_sparse_half:]].to_numpy().astype(numpy.int32),
                numpy.array(test_data[sequential_features.columns[id_sequential_half:]].to_numpy().tolist()).astype(numpy.int32),
            ])
            x_train = (x1_train, x2_train,)
            x_test = (x1_test, x2_test,)

            feature_1_names = tuple([
                tuple([
                    get_dense_feature_name_dict(name) for name in dense_features.columns[:id_dense_half]
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=x_data[name].nunique(),
                        input_length=1,
                    ) for name in sparse_features.columns[:id_sparse_half]
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=len(set(it for item in x_data[name].to_numpy() for it in item)),
                        input_length=data['sequence_max_length'],
                    ) for name in sequential_features.columns[:id_sequential_half]
                ]),
            ])
            feature_2_names = tuple([
                tuple([
                    get_dense_feature_name_dict(name) for name in dense_features.columns[id_dense_half:]
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=x_data[name].nunique(),
                        input_length=1,
                    ) for name in sparse_features.columns[id_sparse_half:]
                ]),
                tuple([
                    get_sparse_feature_name_dict(
                        feature_name=name,
                        feature_nunique=len(set(it for item in x_data[name].to_numpy() for it in item)),
                        input_length=data['sequence_max_length'],
                    ) for name in sequential_features.columns[id_sequential_half:]
                ]),
            ])
            feature_names = (feature_1_names, feature_2_names,)
    else:
        x_train = tuple([
            train_data[dense_features.columns].to_numpy().astype(numpy.float32),
            train_data[sparse_features.columns].to_numpy().astype(numpy.int32),
            numpy.array(train_data[sequential_features.columns].to_numpy().tolist()).astype(numpy.int32),
        ])
        x_test = tuple([
            test_data[dense_features.columns].to_numpy().astype(numpy.float32),
            test_data[sparse_features.columns].to_numpy().astype(numpy.int32),
            numpy.array(test_data[sequential_features.columns].to_numpy().tolist()).astype(numpy.int32),
        ])
        feature_names = tuple([
            tuple([
                get_dense_feature_name_dict(name) for name in dense_features.columns
            ]),
            tuple([
                get_sparse_feature_name_dict(
                    feature_name=name,
                    feature_nunique=x_data[name].nunique(),
                    input_length=1,
                ) for name in sparse_features.columns
            ]),
            tuple([
                get_sparse_feature_name_dict(
                    feature_name=name,
                    feature_nunique=len(set(it for item in x_data[name].to_numpy() for it in item)),
                    input_length=data['sequence_max_length'],
                ) for name in sequential_features.columns
            ]),
        ])

    if task == 'binary' or tuple(task) == ('binary', 'binary',):
        y_train = y_train.astype(numpy.int32)
        y_test = y_test.astype(numpy.int32)
    else:
        y_train = y_train.astype(numpy.float32)
        y_test = y_test.astype(numpy.float32)
    return feature_names, (x_train, y_train), (x_test, y_test)


def get_label_data(n_samples=1000, task='binary', n_classes=2, class_weights=(0.9,), test_size=0.15, seed=None,
                   multitask_cvr=0.1, inputs_if_2_groups=False,
                   multiclass_item_features_if_only_item_id=False, multiclass_features_if_add_item_id=False,):
    if isinstance(task, str) and task not in ('binary', 'regression'):
        raise MLGBError
    if isinstance(task, (tuple, list)) and tuple(task) != ('binary', 'binary',):
        raise MLGBError
    if multiclass_features_if_add_item_id and multiclass_item_features_if_only_item_id and n_classes > 2:
        multiclass_features_if_add_item_id = False

    data = make_fake_data(
        n_samples=n_samples,
        task=task,
        n_classes=n_classes,
        class_weights=class_weights,
        n_numerical_features=50,
        n_categorical_features=10,
        n_sequential_features=8,
        sequence_if_stable_length=True,
        sequence_max_length=10,
        sequence_choice_if_replace=False,
        multiclass_features_if_add_item_id=multiclass_features_if_add_item_id,
        multitask_cvr=multitask_cvr,
        vocabulary_size=None,
        shuffle=True,
        seed=seed,
    )
    out = make_input_data(
        data=data,
        task=task,
        inputs_if_2_groups=inputs_if_2_groups,
        multiclass_item_features_if_only_item_id=multiclass_item_features_if_only_item_id,
        test_size=test_size,
        seed=seed,
    )
    return out


def get_regression_label_data(n_samples=1000, longtail_weight=0.9, test_size=0.15, inputs_if_2_groups=False, seed=None):
    data = get_label_data(
        n_samples=n_samples,
        inputs_if_2_groups=inputs_if_2_groups,
        task='regression',
        n_classes=2,
        class_weights=(longtail_weight,),
        test_size=test_size,
        seed=seed,
    )
    return data


def get_binary_label_data(n_samples=1000, negative_class_weight=0.9, test_size=0.15, inputs_if_2_groups=False, seed=None):
    data = get_label_data(
        n_samples=n_samples,
        inputs_if_2_groups=inputs_if_2_groups,
        task='binary',
        n_classes=2,
        class_weights=(negative_class_weight,),
        test_size=test_size,
        seed=seed,
    )
    return data


def get_multitask_label_data(n_samples=1000, negative_class_weight=0.9, multitask_cvr=0.5, test_size=0.15, seed=None):
    data = get_label_data(
        n_samples=n_samples,
        task=('binary', 'binary',),
        inputs_if_2_groups=False,
        multitask_cvr=multitask_cvr,
        n_classes=2,
        class_weights=(negative_class_weight,),
        test_size=test_size,
        seed=seed,
    )
    return data


def get_multiclass_label_data(n_samples=1000, n_classes=100, test_size=0.15, inputs_if_2_groups=False,
                              multiclass_item_features_if_only_item_id=False, seed=None):
    data = get_label_data(
        n_samples=n_samples,
        task='binary',
        n_classes=n_classes,
        class_weights=None,
        test_size=test_size,
        inputs_if_2_groups=inputs_if_2_groups,
        multiclass_features_if_add_item_id=False,
        multiclass_item_features_if_only_item_id=multiclass_item_features_if_only_item_id and inputs_if_2_groups and n_classes > 2,
        seed=seed,
    )
    return data


if __name__ == '__main__':
    # data = make_fake_data(n_samples=20)
    # binary_label_data = get_binary_label_data(50)
    seq_data = make_sequence_data(
        n_samples=10,
        n_sequence_features=3,
        sequence_if_stable_length=True,
        sequence_max_length=5,
        sequence_choice_if_replace=False,
    )
    # b_data = get_binary_label_data(1000)
    # r_data = get_regression_label_data(1000)
    # m_data = get_multiclass_label_data(1000, 10, inputs_if_2_groups=True, multiclass_item_features_if_only_item_id=True)

    # y_train = b_data[1][1]
    # y_test = b_data[2][1]
    # print(sum(y_train) / len(y_train))
    # print(sum(y_test) / len(y_test))

    # ctr_data = get_multitask_label_data(1000)
    # y_train = ctr_data[1][1]
    # y_test = ctr_data[2][1]
    # print(sum(y_train[:, 0]) / len(y_train))
    # print(sum(y_test[:, 0]) / len(y_test))
    # print(sum(y_train[:, 1]) / len(y_train))
    # print(sum(y_test[:, 1]) / len(y_test))

