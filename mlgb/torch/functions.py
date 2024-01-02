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
    random,
    numpy,
    torch,
)
from mlgb.error import MLGBError


__all__ = [
    'SeedLayer',
    'IdentityLayer',
    'FlattenLayer',
    'FlattenAxesLayer',
    'TransposeLayer',
    'RegularizationLayer',
    'BatchNormalizationLayer',
    'LayerNormalizationLayer',
    'ActivationLayer',
    'InitializerLayer',
    'OneHotLayer',
    'SparseEmbeddingLayer',
    'DenseEmbeddingLayer',
    'PositionalEncoding',
    'BiasEncoding',
    'MaskLayer',
    'SimplePoolingLayer',
    'MultiValuedPoolingLayer',
    'KMaxPoolingLayer',
]


class SeedLayer:
    def __init__(self, seed=None):
        self.seed = seed

    def reset(self):
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        return


class RegularizationLayer:
    def __init__(self, model, l1=0.0, l2=0.0):
        if not (l1 >= 0 and l2 >= 0):
            raise MLGBError

        self.model = model
        self.l1 = l1
        self.l2 = l2

    def get_l1l2_loss(self):
        x = torch.concat([w.detach().view(-1) for w in self.model.parameters()], dim=0)  # no_grad

        l1l2_loss = 0.0
        if self.l1 > 0:
            l1l2_loss += self.p_norm(x, p=1, if_root=False)
        if self.l2 > 0:
            l1l2_loss += self.p_norm(x, p=2, if_root=False)
        return l1l2_loss

    def p_norm(self, x, p=2, if_root=True):  # torch.norm()
        if not p >= 1:
            raise MLGBError

        lp = torch.sum(torch.abs(x) ** p, dim=None, keepdim=False) ** ((1 / p) if if_root else 1)
        return lp


class IdentityLayer(torch.nn.Module):
    def __init__(self, if_stop_gradient=False, device='cpu'):
        super().__init__()
        self.if_stop_gradient = if_stop_gradient
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x
        x = x.detach() if self.if_stop_gradient else x  # torch.tensor(x, requires_grad=False)
        return x


class TransposeLayer(torch.nn.Module):
    def __init__(self, perm=(1, 2), device='cpu'):
        super().__init__()
        if len(perm) != 2:
            raise MLGBError

        self.perm = perm
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.transpose(x, *self.perm)
        return x


class FlattenLayer(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.reshape(x, shape=[x.shape[0], -1])
        return x


class FlattenAxesLayer(torch.nn.Module):
    def __init__(self, axes=(1, 2), device='cpu'):
        super().__init__()
        if len(axes) != 2:
            raise MLGBError
        if 0 in axes:
            raise MLGBError

        self.axes = axes
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 4:
                raise MLGBError

            _, self.p1, self.p2, self.p3 = x.shape
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if list(self.axes) == [1, 2]:
            x = torch.reshape(x, shape=[-1, self.p1 * self.p2, self.p3])
        elif list(self.axes) == [2, 3]:
            x = torch.reshape(x, shape=[-1, self.p1, self.p2 * self.p3])
        elif list(self.axes) == [1, 3]:
            x = torch.reshape(x, shape=[-1, self.p1 * self.p3, self.p2])
        else:
            raise MLGBError
        return x


class BatchNormalizationLayer(torch.nn.Module):
    def __init__(self, axis=1, eps=1e-3, momentum=0.99, if_affine=True, device='cpu'):
        super().__init__()
        # tf_moving_average_decay = 1 - tf_momentum = torch_momentum = torch_moving_average_factor.
        self.eps = eps
        self.momentum = 1.0 - momentum  # use tf_momentum.
        self.if_affine = if_affine
        self.dims = list(range(3))
        self.dim = self.dims[axis]
        self.perm = [self.dim, 1]  #
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.transpose(x, *self.perm)
        x = torch.nn.BatchNorm1d(
            num_features=x.shape[1],
            eps=self.eps,
            momentum=self.momentum,
            affine=self.if_affine,
            device=self.device,
        )(x)
        x = torch.transpose(x, *self.perm)
        return x


class LayerNormalizationLayer(torch.nn.Module):
    def __init__(self, axis=1, eps=1e-3, if_affine=True, device='cpu'):
        super().__init__()
        self.eps = eps
        self.if_affine = if_affine
        self.dims = list(range(3))
        self.dim = self.dims[axis]
        self.perm = [self.dim, -1]  #
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.transpose(x, *self.perm)
        x = torch.nn.LayerNorm(
            normalized_shape=[x.shape[-1]],
            eps=self.eps,
            elementwise_affine=self.if_affine,
            bias=self.if_affine,
            device=self.device,
        )(x)
        x = torch.transpose(x, *self.perm)
        return x


class DiceLayer(torch.nn.Module):
    def __init__(self, dice_if_official_bn=True, device='cpu'):
        super().__init__()
        self.epsilon = 1e-8
        self.alpha = torch.nn.init.ones_(torch.nn.parameter.Parameter(
            data=torch.empty(
                size=[],  # scalar is freedom.
                # device=self.device,
            ),
            requires_grad=True,
        ))

        self.official_bn_fn = BatchNormalizationLayer(
            axis=1,
            eps=self.epsilon,
            if_affine=False,
            device=device,
        )
        self.bn_fn = self.official_bn_fn if dice_if_official_bn else self.unofficial_bn_fn
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        # swish(x) = silu(x) = sigmoid(x) * x
        # prelu(x) = tf.where(x > 0, x, self.alpha * x)
        # dice(x) = tf.where(x > 0, sigmoid(x) * x, (1 - sigmoid(x)) * self.alpha * x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        p = self.bn_fn(x)
        p = torch.sigmoid(p)
        x = p * x + (1 - p) * self.alpha * x
        return x

    def unofficial_bn_fn(self, x):
        diff = x - torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        x = diff / torch.sqrt(var + self.epsilon)
        return x


class SquashLayer(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.alpha = 1.0
        self.beta = 1e-8
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x
        l1 = self.p_norm(x, p=1, if_root=True)
        l2 = self.p_norm(x, p=2, if_root=True)
        x = x * (l2 / (l2 + self.alpha)) * (1 / (l1 + self.beta))
        return x

    def p_norm(self, x, p=2, if_root=True):  # torch.norm()
        lp = torch.sum(torch.abs(x) ** p, dim=None, keepdim=False) ** ((1 / p) if if_root else 1)
        return lp


class ActivationLayer(torch.nn.Module):
    def __init__(self, activation=None, device='cpu'):
        super().__init__()
        self.activation = activation
        self.activation_map = {
            'relu': torch.nn.ReLU(),
            'gelu': torch.nn.GELU(),
            'silu': torch.nn.SiLU(),
            'prelu': torch.nn.PReLU(),
            'selu': torch.nn.SELU(),
            'tanh': torch.nn.Tanh(),
            'sigmoid': torch.nn.Sigmoid(),
            'dice': DiceLayer(dice_if_official_bn=True, device=device),
            'squash': SquashLayer(device=device),
        }
        if self.activation is None:
            self.activation_fn = IdentityLayer(device=device)
        elif isinstance(self.activation, torch.nn.Module):
            self.activation_fn = self.activation
        else:
            self.activation_fn = self.activation_map[self.activation]
        
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x
        x = self.activation_fn(x)
        return x


class InitializerLayer:  # Not Layer
    def __init__(self, initializer=None, activation=None):
        self.initializer = initializer
        self.activation = activation
        self.initializer_map = {
            'glorot_normal': torch.nn.init.xavier_normal_,
            'glorot_uniform': torch.nn.init.xavier_uniform_,
            'xavier_normal': torch.nn.init.xavier_normal_,
            'xavier_uniform': torch.nn.init.xavier_uniform_,
            'he_normal': torch.nn.init.kaiming_normal_,  # he_scale=2, lecun_scale=1.
            'he_uniform': torch.nn.init.kaiming_uniform_,
            'kaiming_normal': torch.nn.init.kaiming_normal_,
            'kaiming_uniform': torch.nn.init.kaiming_uniform_,
            'random_normal': torch.nn.init.normal_,
            'random_uniform': torch.nn.init.uniform_,
            'truncated_normal': torch.nn.init.trunc_normal_,
            'orthogonal': torch.nn.init.orthogonal_,
            'zeros': torch.nn.init.zeros_,
            'ones': torch.nn.init.ones_,
        }
        self.activation_initializer_map = {
            'relu': self.initializer_map['he_normal'],
            'gelu': self.initializer_map['he_normal'],
            'silu': self.initializer_map['he_normal'],
            'prelu': self.initializer_map['he_normal'],
            'selu': self.initializer_map['he_normal'],  # lecun_normal
            'tanh': self.initializer_map['glorot_normal'],
            'sigmoid': self.initializer_map['glorot_normal'],
            'dice': self.initializer_map['glorot_normal'],
            'squash': self.initializer_map['random_normal'],
        }

    def get(self):
        if callable(self.initializer):  # types.FunctionType
            initializer_fn = self.initializer
        elif not (self.initializer in self.initializer_map or self.activation in self.activation_initializer_map):
            initializer_fn = self.initializer_map['glorot_normal']
        elif self.initializer:
            initializer_fn = self.initializer_map[self.initializer]
        else:
            initializer_fn = self.activation_initializer_map[self.activation]
        return initializer_fn


class OneHotLayer(torch.nn.Module):
    def __init__(self, sparse_feature_names, onehot_dim=None, device='cpu'):
        super().__init__()
        self.sparse_feature_names = sparse_feature_names
        self.onehot_dim = onehot_dim
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError

            self.fields_width = x.shape[1]
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x
        x = x.to(dtype=torch.int64) if x.dtype != torch.int64 else x  # Must LongTensor.

        sparse_tensor = x
        onehot_tensor_list = []
        for i in range(self.fields_width):
            depth = self.onehot_dim if self.onehot_dim else self.sparse_feature_names[i]['feature_nunique']
            sparse_i_tensor = sparse_tensor[:, i]
            onehot_i_tensor = torch.nn.functional.one_hot(sparse_i_tensor, num_classes=depth)
            onehot_tensor_list.append(onehot_i_tensor)
        onehot_tensor = torch.concat(onehot_tensor_list, dim=1)
        return onehot_tensor


class SparseEmbeddingLayer(torch.nn.Module):
    def __init__(self, sparse_feature_names, embed_dim=None, embed_if_output2d=False, device='cpu'):
        super().__init__()
        self.sparse_feature_names = sparse_feature_names
        self.embed_dim = embed_dim
        self.embed_if_output2d = embed_if_output2d
        self.embed_fn_map = {
            f'{i}': torch.nn.Embedding(
                num_embeddings=feat_dict['embed_feature_nunique'],
                embedding_dim=feat_dict['embed_dim'] if not self.embed_dim else self.embed_dim,
                padding_idx=0,
                device=device,
            )
            for i, feat_dict in enumerate(self.sparse_feature_names)
        }
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (2, 3):
                raise MLGBError

            self.fields_width = x.shape[1]
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x
        x = x.to(dtype=torch.int32) if x.dtype not in (torch.int32, torch.int64) else x  # Must LongTensor or IntTensor.

        sparse_tensor = x
        embed_tensor_list = [self.embed_fn_map[f'{i}'](sparse_tensor[:, i, ...]) for i in range(self.fields_width)]
        d23_tensor = torch.concat(embed_tensor_list, dim=1)

        if self.embed_if_output2d:
            return d23_tensor, d23_tensor

        if not self.embed_dim:
            raise MLGBError

        d34_tensor = torch.stack(embed_tensor_list, dim=1)
        return d23_tensor, d34_tensor


class DenseEmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_dim=32, embed_initializer=None, device='cpu'):
        super().__init__()
        if not embed_dim:
            raise MLGBError

        self.embed_dim = embed_dim
        self.initializer_fn = InitializerLayer(
            initializer=embed_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 2:
                raise MLGBError

            self.embed_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[x.shape[1], self.embed_dim],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.einsum('bf,fe->bfe', x, self.embed_weight)
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, att_if_pe=True, device='cpu'):
        super().__init__()
        self.att_if_pe = att_if_pe
        self.wave_length = 1e4  # 1e4 * pi
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            _, f, e = x.shape

            pos_f = numpy.arange(f).reshape(-1, 1)
            pos_e = numpy.arange(e) // 2 * 2
            pos_x = pos_f / numpy.power(self.wave_length, pos_e / e)  # (f, e)

            pos_x[:, 0::2] = numpy.sin(pos_x[:, 0::2])
            pos_x[:, 1::2] = numpy.cos(pos_x[:, 1::2])

            self.pe = torch.as_tensor(pos_x.reshape(1, f, e), dtype=torch.float32, device=self.device)
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if self.att_if_pe:
            x += self.pe
        return x


class BiasEncoding(torch.nn.Module):
    def __init__(self, if_bias=True, bias_initializer='zeros', device='cpu'):
        super().__init__()
        self.if_bias = if_bias
        self.initializer_fn = InitializerLayer(
            initializer=bias_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim < 2:
                raise MLGBError

            if self.if_bias:
                self.bias = self.initializer_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=x.shape[1:],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))

            self.built = True
        return

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if self.if_bias:
            x += self.bias
        return x


class MaskLayer(torch.nn.Module):
    def __init__(self, att_if_mask=True, device='cpu'):
        super().__init__()
        self.min_inf = -4294967295  # float32: -2 ** 32 + 1
        self.att_if_mask = att_if_mask
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def forward(self, x):
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if self.att_if_mask:
            x = torch.where(torch.not_equal(x, 0), x, self.min_inf)
        return x
        

class SimplePoolingLayer(torch.nn.Module):
    def __init__(self, pool_mode='Pooling:average', pool_axis=-1, pool_axis_if_keep=False, device='cpu'):
        super().__init__()
        if pool_mode not in ('Pooling:max', 'Pooling:average', 'Pooling:sum'):
            raise MLGBError

        self.pool_mode = pool_mode
        self.pool_axis = pool_axis
        self.pool_axis_if_keep = pool_axis_if_keep
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (2, 3, 4):
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = self.pool_fn(x, self.pool_mode, self.pool_axis, self.pool_axis_if_keep)
        return x

    def pool_fn(self, x, pool_mode, pool_axis, pool_axis_if_keep):
        pool_fn_map = {
            'Pooling:average': lambda x: torch.mean(x, dim=pool_axis, keepdim=pool_axis_if_keep),
            'Pooling:sum': lambda x: torch.sum(x, dim=pool_axis, keepdim=pool_axis_if_keep),
            'Pooling:max': lambda x: torch.max(x, dim=pool_axis, keepdim=pool_axis_if_keep),
        }
        x = pool_fn_map[pool_mode](x)
        return x


class MultiValuedPoolingLayer(torch.nn.Module):
    def __init__(self, pool_mode='Attention', pool_axis=2, pool_if_output2d=False, pool_initializer=None, device='cpu'):
        super().__init__()
        if pool_mode not in ('Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum'):
            raise MLGBError
        if pool_axis not in (1, 2, 3, -1, -2, -3):
            raise MLGBError

        self.pool_mode = pool_mode
        self.pool_axis = pool_axis
        self.pool_if_output2d = pool_if_output2d
        self.initializer_fn = InitializerLayer(
            initializer=pool_initializer,
            activation=None,
        ).get()
        if self.pool_mode.startswith('Pooling:'):
            self.pool_fn = SimplePoolingLayer(
                pool_mode=pool_mode,
                pool_axis=pool_axis,
                device=device,
            )
        self.flatten_fn = FlattenLayer(device=device)
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 4:
                raise MLGBError

            _, self.fields_width, self.sequence_length, _ = x.shape
            if self.pool_mode in ('Weighted', 'Attention'):
                self.pool_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                    data=torch.empty(
                        size=[self.fields_width, self.sequence_length],
                        device=self.device,
                    ),
                    requires_grad=True,
                ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)  # (batch_size, fields_width, sequence_length, embed_dim)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        if self.pool_mode.startswith('Pooling:'):
            x = self.pool_fn(x)
        elif self.pool_mode in ('Weighted', 'Attention'):
            x_w = torch.einsum('ijkl,jk->ijkl', x, self.pool_weight)
            if self.pool_mode == 'Attention':
                x_w = MaskLayer(att_if_mask=True)(x_w)
                x_w = x * torch.softmax(x_w, dim=self.pool_axis)
            x = torch.sum(x_w, dim=self.pool_axis, keepdim=False)
        else:
            raise MLGBError

        if self.pool_if_output2d:
            x = self.flatten_fn(x)
        return x  # (b, f, e) or (b, f*e)


class KMaxPoolingLayer(torch.nn.Module):
    def __init__(self, pool_axis=-1, device='cpu'):
        super().__init__()
        if pool_axis == 0:
            raise MLGBError

        self.axis = pool_axis
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (3, 4):
                raise MLGBError

            self.built = True
        return

    def forward(self, x, k):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = torch.topk(x, k=k, dim=self.axis, largest=True, sorted=True)[0]
        return x









