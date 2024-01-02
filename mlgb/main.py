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


from mlgb.torch.configs import torch
from mlgb.torch.functions import SeedLayer
from mlgb.torch.agg import (
    torch_ranking_models_map,
    torch_matching_models_map,
    torch_mtl_models_map,
)
from mlgb.tf.agg import (
    tf_ranking_models_map,
    tf_matching_models_map,
    tf_mtl_models_map,
)
from mlgb.error import MLGBError, MLGBWarning


class Tse:
    def __init__(self):
        self.tf_ranking_models_map = tf_ranking_models_map
        self.tf_matching_models_map = tf_matching_models_map
        self.tf_mtl_models_map = tf_mtl_models_map
        self.torch_ranking_models_map = torch_ranking_models_map
        self.torch_matching_models_map = torch_matching_models_map
        self.torch_mtl_models_map = torch_mtl_models_map
        self.all_models_map = {
            'tf_ranking_models_map': self.tf_ranking_models_map,
            'tf_matching_models_map': self.tf_matching_models_map,
            'tf_mtl_models_map': self.tf_mtl_models_map,
            'torch_ranking_models_map': self.torch_ranking_models_map,
            'torch_matching_models_map': self.torch_matching_models_map,
            'torch_mtl_models_map': self.torch_mtl_models_map,
        }
        self.ranking_models = list(self.tf_ranking_models_map.keys())
        self.matching_models = list(self.tf_matching_models_map.keys())
        self.mtl_models = list(self.tf_mtl_models_map.keys())
        self.all_models = list(set(self.ranking_models + self.matching_models + self.mtl_models))

    def get_short_lang(self, lang='TensorFlow'):
        if lang not in ('TensorFlow', 'PyTorch', 'tf', 'torch'):
            raise MLGBError('lang')
        return 'tf' if lang in ('TensorFlow', 'tf') else 'torch'

    def find_model(self, model_name='LR', aim='ranking', lang='TensorFlow'):
        if aim not in ('ranking', 'matching', 'mtl'):
            raise MLGBError('aim')
        if aim == 'ranking' and model_name not in self.ranking_models:
            raise MLGBError('model_name & aim')
        if aim == 'matching' and model_name not in self.matching_models:
            raise MLGBError('model_name & aim')
        if aim == 'mtl' and model_name not in self.mtl_models:
            raise MLGBError('model_name & aim')

        lang = self.get_short_lang(lang)
        model = self.all_models_map[f'{lang}_{aim}_models_map'][model_name]
        return model

    def get_model_help(self, model_name='LR', aim='ranking', lang='TensorFlow'):
        model = self.find_model(model_name=model_name, aim=aim, lang=lang)
        help(model)
        return

    def get_model(self, feature_names, model_name='LR', task='binary', aim='ranking', lang='TensorFlow',
                  device=None, seed=None, **kwargs):
        """
        :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
        :param model_name: str, default 'LR'. Union[`mlgb.ranking_models`, `mlgb.matching_models`, `mlgb.mtl_models`]
        :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
        :param aim: str, default 'ranking'. Union['ranking', 'matching', 'mtl']
        :param lang: str, default 'TensorFlow'. Union['TensorFlow', 'PyTorch', 'tf', 'torch']
        :param device: Optional[str, int], default None. Only for PyTorch.
        :param seed: Optional[int], default None.
        :param **kwargs: more model parameters by `mlgb.get_model_help(model_name)`.
        """

        model = self.find_model(model_name=model_name, aim=aim, lang=lang)
        if self.get_short_lang(lang) == 'tf':
            if device:
                MLGBWarning(f'tf_device:"{device}" needs to be set in global.')
            kwargs.update({'task': task, 'seed': seed})
        else:
            _ = SeedLayer(seed=seed).reset()
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            kwargs.update({'task': task, 'device': device})
        return model(feature_names, **kwargs)


tse = Tse()
get_model = tse.get_model
get_model_help = tse.get_model_help
mtl_models = tse.mtl_models
ranking_models = tse.ranking_models
matching_models = tse.matching_models

