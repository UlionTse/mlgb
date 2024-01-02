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
    numpy, 
    torch,
)
from mlgb.torch.functions import (
    ActivationLayer, 
    InitializerLayer,
)
from mlgb.error import MLGBError


__all__ = [
    'AllFieldBinaryInteractionLayer',
    'AttentionalFieldBinaryInteractionLayer',
    'TwoAllFieldBinaryInteractionLayer',
    'GroupedAllFieldWiseBinaryInteractionLayer',
]


class BinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_initializer=None, device='cpu'):
        super().__init__()
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (2, 3):
                raise MLGBError

            self.fbi_if_keepdim = True if x.ndim == 2 else False
            self.fbi_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1] + list(x.shape[1:]),  # torch: init must > 2D.
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return


    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = x * self.fbi_weight
        square_of_sum = torch.square(torch.sum(x, dim=1, keepdim=self.fbi_if_keepdim))
        sum_of_square = torch.sum(torch.square(x), dim=1, keepdim=self.fbi_if_keepdim)
        x = 0.5 * torch.subtract(square_of_sum, sum_of_square)  # (batch_size, embed_dim) or (batch_size, 1)
        return x


class HigherOrderBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_hofm_order=3, fbi_initializer=None, device='cpu'):
        super().__init__()
        if not (fbi_hofm_order >= 3):
            raise MLGBError

        self.fbi_hofm_order = fbi_hofm_order
        self.fm_fn = BinaryInteractionLayer(
            fbi_initializer=fbi_initializer,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError
            if self.fbi_hofm_order > x.shape[1]:
                raise MLGBError

            self.fields_width = x.shape[1]
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_fm = self.fm_fn(x)
        x = x * self.fm_fn.fbi_weight
        x_hofm = self.get_anova_kernel(x, self.fbi_hofm_order)
        x = torch.concat([x_fm, x_hofm], dim=1)
        return x

    def get_anova_kernel(self, x, hofm_order):
        a = torch.ones_like(x)
        a = torch.index_select(
            input=a,
            dim=1,
            index=torch.tensor([0] * (self.fields_width + 1), dtype=torch.int32, device=self.device),
        )  # (b, f+1, e)

        ak_list = []
        for i in range(hofm_order):
            a_i_0 = torch.zeros_like(x)[:, :i+1, :]
            a_i = x[:, i:, :] * a[:, i:-1, :]
            a_i = torch.concat([a_i_0, a_i], dim=1)
            a = torch.cumsum(a_i, dim=1)  # (b, f+1, e)

            if i >= 2:
                ak_list.append(a[:, -1, :])  # (b, e)

        ak = torch.concat(ak_list, dim=1)  # (b, (order-2)*e)
        return ak


class FieldBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_initializer=None, device='cpu'):
        super().__init__()
        if fbi_mode not in ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.fbi_unit = fbi_unit  # fbi_weight__length_of_latent_vector
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError
    
            _, self.fields_width, self.embed_dim = x.shape
            self.product_width = int(self.fields_width * (self.fields_width - 1) // 2)
            self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T
            # self.ij_ids = numpy.array(list(itertools.combinations(range(self.fields_width), 2)))
    
            self.weight_shape_map = {
                'FwFM': [1, self.product_width, 1],  # torch: init must > 2D.
                'FFM': [self.fields_width, self.embed_dim, self.fbi_unit],
                'PNN:inner_product': [self.product_width, self.embed_dim, self.fbi_unit],
                'PNN:outer_product': [self.product_width, self.embed_dim, self.embed_dim, self.fbi_unit],
            }
            self.fbi_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=self.weight_shape_map[self.fbi_mode],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_i = torch.index_select(
            input=x,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )

        if self.fbi_mode == 'FFM':
            w_i = torch.index_select(
                input=self.fbi_weight,
                dim=0,
                index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
            )
            w_j = torch.index_select(
                input=self.fbi_weight,
                dim=0,
                index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
            )
            w_ij = w_i * w_j
        else:
            w_ij = self.fbi_weight

        if self.fbi_mode == 'FwFM':
            x_ij = x_i * x_j
            x = torch.einsum('bfe,ofo->be', x_ij, w_ij)  # (batch_size, embed_dim)
        elif self.fbi_mode in ('FFM', 'PNN:inner_product'):
            x_ij = x_i * x_j
            x = torch.einsum('bfe,feu->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        elif self.fbi_mode == 'PNN:outer_product':
            x_i = torch.unsqueeze(x_i, dim=3)
            x_j = torch.unsqueeze(x_j, dim=2)
            x_ij = x_i @ x_j  # (bfe1,bf1e->bfee)
            x = torch.einsum('bfij,fiju->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        return x


class FieldProductBothInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='PNN:both', fbi_unit=32, fbi_initializer=None, device='cpu'):
        super().__init__()
        if fbi_mode != 'PNN:both':
            raise MLGBError

        self.ip_fn = FieldBinaryInteractionLayer(
            fbi_mode='PNN:inner_product',
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            device=device,
        )
        self.op_fn = FieldBinaryInteractionLayer(
            fbi_mode='PNN:outer_product',
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            self.built = True
        return


    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_ip = self.ip_fn(x)
        x_op = self.op_fn(x)
        x = torch.concat([x_ip, x_op], dim=1)  # (batch_size, fbi_unit * 2)
        return x


class BilinearInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='Bilinear:field_interaction', fbi_initializer=None, device='cpu'):
        super().__init__()
        if fbi_mode not in ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            _, self.fields_width, self.embed_dim = x.shape
            self.product_width = int(self.fields_width * (self.fields_width - 1) // 2)
            self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T

            self.weight_num_map = {
                'Bilinear:field_all': 1,
                'Bilinear:field_each': self.fields_width,
                'Bilinear:field_interaction': self.product_width,
                'FEFM': self.product_width,
                'FvFM': self.product_width,
                'FmFM': self.product_width,
            }

            if self.fbi_mode == 'FvFM':
                self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim]
            else:
                self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim, self.embed_dim]

            self.fbi_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=self.fbi_weight_shape,
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return


    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_i = torch.index_select(
            input=x,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )

        if self.fbi_mode == 'Bilinear:field_all':
            w = torch.concat([self.fbi_weight] * self.product_width, dim=0)
        elif self.fbi_mode == 'Bilinear:field_each':
            w = torch.index_select(
                input=self.fbi_weight,
                dim=0,
                index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
            )
        elif self.fbi_mode in ('Bilinear:field_interaction', 'FmFM', 'FvFM'):
            w = self.fbi_weight
        elif self.fbi_mode == 'FEFM':
            w1 = self.fbi_weight
            w2 = torch.transpose(self.fbi_weight, *[1, 2])
            w = (w1 + w2) * 0.5  # symmetric matrix
        else:
            raise MLGBError

        if self.fbi_mode == 'FvFM':
            x_i_mid = x_i * w  # (bfe,fe->bfe)
        else:
            x_i_mid = torch.einsum('bfe,fee->bfe', x_i, w)

        if self.fbi_mode == 'FEFM':
            x = torch.einsum('bme,bne->bmn', x_i_mid, x_j)  # bmm, (batch_size, product_width, product_width)
        else:
            x = x_i_mid * x_j  # (batch_size, product_width, embed_dim)

        x = torch.sum(x, dim=2, keepdim=False)  # (batch_size, product_width)
        return x


class AttentionalFieldBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_afm_activation='relu', fbi_afm_dropout=0.0, fbi_unit=32, fbi_initializer=None, device='cpu'):
        super().__init__()
        self.fbi_unit = fbi_unit
        self.drop_fn = torch.nn.Dropout(p=fbi_afm_dropout)
        self.activation_fn = ActivationLayer(
            activation=fbi_afm_activation,
            device=device,
        )
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=fbi_afm_activation,
        ).get()
        self.initializer_zeros_fn = InitializerLayer(
            initializer='zeros',
            activation=fbi_afm_activation,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            _, self.fields_width, self.embed_dim = x.shape
            self.ij_ids = numpy.array(numpy.triu_indices(n=self.fields_width, k=1)).T

            self.att_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.embed_dim, self.fbi_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.att_bias = self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1, 1, self.fbi_unit],  # torch: init must > 2D.
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.proj_h = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.fbi_unit, 1],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.proj_p = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.embed_dim, 1],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return


    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x_i = torch.index_select(
            input=x,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )
        x = x_i * x_j  # (batch_size, product_width, embed_dim)

        w = x @ self.att_weight + self.att_bias  # (batch_size, product_width, att_unit)
        w = self.activation_fn(w)
        w = w @ self.proj_h  # (batch_size, product_width, 1)
        w = torch.softmax(w, dim=1)

        x = x * w  # (batch_size, product_width, embed_dim)
        x = self.drop_fn(x)
        x = x @ self.proj_p  # (batch_size, product_width, 1)
        x = torch.squeeze(x, dim=2)  # (batch_size, product_width)
        return x


class AllFieldBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_initializer=None,
                 fbi_hofm_order=3, fbi_afm_activation='relu', fbi_afm_dropout=0.0, device='cpu'):
        super().__init__()
        self.bi_mode_list = ('FM', 'FM3D',)
        self.ho_model_list = ('HOFM',)
        self.at_model_list = ('AFM',)
        self.pb_mode_list = ('PNN:both',)
        self.fb_mode_list = ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product')
        self.bl_mode_list = ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM')
        self.fbi_mode_list = self.bi_mode_list + self.ho_model_list + self.at_model_list + self.pb_mode_list + self.fb_mode_list + self.bl_mode_list
        if fbi_mode not in self.fbi_mode_list:
            raise MLGBError

        self.fbi_mode = fbi_mode

        if fbi_mode in self.pb_mode_list:
            self.fbi_fn = FieldProductBothInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.fb_mode_list:
            self.fbi_fn = FieldBinaryInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.bl_mode_list:
            self.fbi_fn = BilinearInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.at_model_list:
            self.fbi_fn = AttentionalFieldBinaryInteractionLayer(
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.ho_model_list:
            self.fbi_fn = HigherOrderBinaryInteractionLayer(
                fbi_hofm_order=fbi_hofm_order,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        else:
            self.fbi_fn = BinaryInteractionLayer(
                fbi_initializer=fbi_initializer,
                device=device,
            )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim not in (2, 3):
                raise MLGBError
            if x.ndim == 2 and self.fbi_mode != 'FM':
                raise MLGBError
            if x.ndim != 2 and self.fbi_mode == 'FM':
                raise MLGBError

        self.built = True
        return


    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        x = self.fbi_fn(x)
        return x


class TwoBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_initializer=None, device='cpu'):
        super().__init__()
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim not in (2, 3):
                raise MLGBError
            if x[0].ndim != x[1].ndim:
                raise MLGBError
            if x[0].ndim == 3 and (x[0].shape[2] != x[1].shape[2]):  # embed_dim
                raise MLGBError

            self.fbi_if_keepdim = True if x[0].ndim == 2 else False

            self.fields_i_width, self.fields_j_width = x[0].shape[1], x[1].shape[1]
            self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)
            # self.ij_ids = numpy.array([[i, j] for i in range(self.fields_i_width) for j in range(self.fields_j_width)])

            self.fbi_i_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1] + list(x[0].shape[1:]),  # torch: must > 2D.
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.fbi_j_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1] + list(x[1].shape[1:]),  # torch: must > 2D.
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_i, x_j = x

        x_i = x_i * self.fbi_j_weight
        x_j = x_j * self.fbi_j_weight

        x_i = torch.index_select(
            input=x_i,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x_j,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )

        x = x_i * x_j
        x = torch.sum(x, dim=1, keepdim=self.fbi_if_keepdim)  # (batch_size, embed_dim) or (batch_size, 1)
        return x


class TwoFieldBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_initializer=None, device='cpu'):
        super().__init__()
        if fbi_mode not in ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.fbi_unit = fbi_unit  # fbi_weight__length_of_latent_vector
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim != 3:
                raise MLGBError
            if x[0].ndim != x[1].ndim:
                raise MLGBError
            if x[0].ndim == 3 and (x[0].shape[2] != x[1].shape[2]):  # embed_dim
                raise MLGBError

            self.embed_dim = x[0].shape[2]
            self.fields_i_width, self.fields_j_width = x[0].shape[1], x[1].shape[1]
            self.product_width = int(self.fields_i_width * self.fields_j_width)
            self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)

            self.weight_shape_map = {
                'FwFM': [1, self.product_width, 1],  # torch: init must > 2D.
                'FFM': [self.fields_i_width + self.fields_j_width, self.embed_dim, self.fbi_unit],
                'PNN:inner_product': [self.product_width, self.embed_dim, self.fbi_unit],
                'PNN:outer_product': [self.product_width, self.embed_dim, self.embed_dim, self.fbi_unit],
            }
            self.fbi_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=self.weight_shape_map[self.fbi_mode],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_i, x_j = x

        x_i = torch.index_select(
            input=x_i,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x_j,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )

        if self.fbi_mode == 'FFM':
            w_i = torch.index_select(
                input=self.fbi_weight,
                dim=0,
                index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
            )
            w_j = torch.index_select(
                input=self.fbi_weight,
                dim=0,
                index=torch.tensor(self.ij_ids[:, 1] + self.fields_i_width, dtype=torch.int32, device=self.device),  # shift
            )
            w_ij = w_i * w_j
        else:
            w_ij = self.fbi_weight

        if self.fbi_mode == 'FwFM':
            x_ij = x_i * x_j
            x = torch.einsum('bfe,ofo->be', x_ij, w_ij)  # (batch_size, embed_dim)
        elif self.fbi_mode in ('FFM', 'PNN:inner_product'):
            x_ij = x_i * x_j
            x = torch.einsum('bfe,feu->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        elif self.fbi_mode == 'PNN:outer_product':
            x_i = torch.unsqueeze(x_i, dim=3)
            x_j = torch.unsqueeze(x_j, dim=2)
            x_ij = x_i @ x_j  # (bfe1,bf1e->bfee)
            x = torch.einsum('bfij,fiju->bu', x_ij, w_ij)  # (batch_size, self.fbi_unit)
        else:
            raise MLGBError
        return x


class TwoFieldProductBothInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='PNN:both', fbi_unit=32, fbi_initializer=None, device='cpu'):
        super().__init__()
        if fbi_mode != 'PNN:both':
            raise MLGBError

        self.ip_fn = TwoFieldBinaryInteractionLayer(
            fbi_mode='PNN:inner_product',
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            device=device,
        )
        self.op_fn = TwoFieldBinaryInteractionLayer(
            fbi_mode='PNN:outer_product',
            fbi_unit=fbi_unit,
            fbi_initializer=fbi_initializer,
            device=device,
        )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim != 3:
                raise MLGBError
            if x[0].ndim != x[1].ndim:
                raise MLGBError
            if x[0].ndim == 3 and (x[0].shape[2] != x[1].shape[2]):  # embed_dim
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_i, x_j = x

        x_ip = self.ip_fn([x_i, x_j])
        x_op = self.op_fn([x_i, x_j])
        x = torch.concat([x_ip, x_op], dim=1)  # (batch_size, fbi_unit * 2)
        return x


class TwoBilinearInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='Bilinear:field_interaction', fbi_initializer=None, device='cpu'):
        super().__init__()
        if fbi_mode not in ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM'):
            raise MLGBError

        self.fbi_mode = fbi_mode
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=None,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim != 3:
                raise MLGBError
            if x[0].ndim != x[1].ndim:
                raise MLGBError
            if x[0].ndim == 3 and (x[0].shape[2] != x[1].shape[2]):  # embed_dim
                raise MLGBError

            self.embed_dim = x[0].shape[2]
            self.fields_i_width, self.fields_j_width = x[0].shape[1], x[1].shape[1]
            self.product_width = int(self.fields_i_width * self.fields_j_width)
            self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)

            self.weight_num_map = {
                'Bilinear:field_all': 1,
                'Bilinear:field_each': self.fields_width,
                'Bilinear:field_interaction': self.product_width,
                'FEFM': self.product_width,
                'FvFM': self.product_width,
                'FmFM': self.product_width,
            }

            if self.fbi_mode == 'FvFM':
                self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim]
            else:
                self.fbi_weight_shape = [self.weight_num_map[self.fbi_mode], self.embed_dim, self.embed_dim]

            self.fbi_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=self.fbi_weight_shape,
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_i, x_j = x

        x_i = torch.index_select(
            input=x_i,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x_j,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )

        if self.fbi_mode == 'Bilinear:field_all':
            w = torch.concat([self.fbi_weight] * self.product_width, dim=0)
        elif self.fbi_mode == 'Bilinear:field_each':
            w = torch.index_select(
                input=self.fbi_weight,
                dim=0,
                index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
            )
        elif self.fbi_mode in ('Bilinear:field_interaction', 'FmFM'):
            w = self.fbi_weight
        elif self.fbi_mode == 'FEFM':
            w1 = self.fbi_weight
            w2 = torch.transpose(self.fbi_weight, *[1, 2])
            w = (w1 + w2) * 0.5  # symmetric matrix
        else:
            raise MLGBError

        if self.fbi_mode == 'FvFM':
            x_i_mid = x_i * w  # (bfe,fe->bfe)
        else:
            x_i_mid = torch.einsum('bfe,fee->bfe', x_i, w)

        if self.fbi_mode == 'FEFM':
            x = torch.einsum('bme,bne->bmn', x_i_mid, x_j)  # bmm, (batch_size, product_width, product_width)
        else:
            x = x_i_mid * x_j  # (batch_size, product_width, embed_dim)

        x = torch.sum(x, dim=2, keepdim=False)  # (batch_size, product_width)
        return x


class TwoAttentionalFieldBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_afm_activation='relu', fbi_afm_dropout=0.0, fbi_unit=32, fbi_initializer=None, device='cpu'):
        super().__init__()
        self.fbi_unit = fbi_unit
        self.drop_fn = torch.nn.Dropout(p=fbi_afm_dropout)
        self.activation_fn = ActivationLayer(
            activation=fbi_afm_activation,
            device=device,
        )
        self.initializer_fn = InitializerLayer(
            initializer=fbi_initializer,
            activation=fbi_afm_activation,
        ).get()
        self.initializer_zeros_fn = InitializerLayer(
            initializer='zeros',
            activation=fbi_afm_activation,
        ).get()
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim != 3:
                raise MLGBError
            if x[0].ndim != x[1].ndim:
                raise MLGBError
            if x[0].ndim == 3 and (x[0].shape[2] != x[1].shape[2]):  # embed_dim
                raise MLGBError

            self.embed_dim = x[0].shape[2]
            self.fields_i_width, self.fields_j_width = x[0].shape[1], x[1].shape[1]
            self.product_width = int(self.fields_i_width * self.fields_j_width)
            self.ij_ids = numpy.array(numpy.meshgrid(range(self.fields_i_width), range(self.fields_j_width))).T.reshape(-1, 2)

            self.att_weight = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.embed_dim, self.fbi_unit],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.att_bias = self.initializer_zeros_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[1, 1, self.fbi_unit],  # torch: init must > 2D.
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.proj_h = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.fbi_unit, 1],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.proj_p = self.initializer_fn(torch.nn.parameter.Parameter(
                data=torch.empty(
                    size=[self.embed_dim, 1],
                    device=self.device,
                ),
                requires_grad=True,
            ))
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_i, x_j = x

        x_i = torch.index_select(
            input=x_i,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 0], dtype=torch.int32, device=self.device),
        )
        x_j = torch.index_select(
            input=x_j,
            dim=1,
            index=torch.tensor(self.ij_ids[:, 1], dtype=torch.int32, device=self.device),
        )
        x = x_i * x_j  # (batch_size, product_width, embed_dim)

        w = x @ self.att_weight + self.att_bias  # (batch_size, product_width, att_unit)
        w = self.activation_fn(w)
        w = w @ self.proj_h  # (batch_size, product_width, 1)
        w = torch.softmax(w, dim=1)

        x = x * w  # (batch_size, product_width, embed_dim)
        x = self.drop_fn(x)
        x = x @ self.proj_p  # (batch_size, product_width, 1)
        x = torch.squeeze(x, dim=2)  # (batch_size, product_width)
        return x


class TwoAllFieldBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, fbi_mode='FFM', fbi_unit=32, fbi_initializer=None, 
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0, 
                 device='cpu'):
        super().__init__()
        self.bi_mode_list = ('FM', 'FM3D',)
        self.at_mode_list = ('AFM',)
        self.pb_mode_list = ('PNN:both',)
        self.fb_mode_list = ('FFM', 'FwFM', 'PNN:inner_product', 'PNN:outer_product')
        self.bl_mode_list = ('Bilinear:field_all', 'Bilinear:field_each', 'Bilinear:field_interaction', 'FEFM', 'FvFM', 'FmFM')
        self.fbi_mode_list = self.bi_mode_list + self.at_mode_list + self.pb_mode_list + self.fb_mode_list + self.bl_mode_list
        if fbi_mode not in self.fbi_mode_list:
            raise MLGBError

        self.fbi_mode = fbi_mode
        if fbi_mode in self.pb_mode_list:
            self.two_fbi_fn = TwoFieldProductBothInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.fb_mode_list:
            self.two_fbi_fn = TwoFieldBinaryInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_unit=fbi_unit,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.bl_mode_list:
            self.two_fbi_fn = TwoBilinearInteractionLayer(
                fbi_mode=fbi_mode,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        elif fbi_mode in self.at_model_list:
            self.two_fbi_fn = TwoAttentionalFieldBinaryInteractionLayer(
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                fbi_initializer=fbi_initializer,
                device=device,
            )
        else:
            self.two_fbi_fn = TwoBinaryInteractionLayer(
                fbi_initializer=fbi_initializer,
                device=device,
            )
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if len(x) != 2:
                raise MLGBError
            if x[0].ndim not in (2, 3):
                raise MLGBError
            if x[0].ndim != x[1].ndim:
                raise MLGBError
            if x[0].ndim == 3 and (x[0].shape[2] != x[1].shape[2]):  # embed_dim
                raise MLGBError
            if x[0].ndim == 2 and self.fbi_mode != 'FM':
                raise MLGBError
            if x[0].ndim != 2 and self.fbi_mode == 'FM':
                raise MLGBError
            
            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = [(t.to(device=self.device) if t.device.type != self.device.type else t) for t in x]
        x_i, x_j = x

        x = self.two_fbi_fn([x_i, x_j])
        return x


class GroupedAllFieldWiseBinaryInteractionLayer(torch.nn.Module):
    def __init__(self, group_indices=((0, 1), (2, 3), (4, 5, 6)),
                 fbi_fm_mode='FM3D', fbi_mf_mode='FwFM', fbi_unit=32, fbi_initializer=None,
                 fbi_afm_activation='relu', fbi_afm_dropout=0.0,
                 device='cpu'):
        super().__init__()
        if group_indices and len(group_indices) < 2:
            raise MLGBError

        self.group_indices = group_indices  # (user, item, context)
        self.group_width = len(self.group_indices)
        self.product_width = int(self.group_width * (self.group_width - 1) // 2)
        self.ij_ids = numpy.array(numpy.triu_indices(n=self.group_width, k=1)).T

        self.fbi_fm_fn_list = torch.nn.ModuleList([
            AllFieldBinaryInteractionLayer(
                fbi_mode=fbi_fm_mode,
                fbi_unit=fbi_unit,
                fbi_initializer=fbi_initializer,
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                device=device,
            )
            for _ in range(self.group_width)
        ])
        self.fbi_mf_fn_list = torch.nn.ModuleList([
            TwoAllFieldBinaryInteractionLayer(
                fbi_mode=fbi_mf_mode,
                fbi_unit=fbi_unit,
                fbi_initializer=fbi_initializer,
                fbi_afm_activation=fbi_afm_activation,
                fbi_afm_dropout=fbi_afm_dropout,
                device=device,
            )
            for _ in range(self.product_width)
        ])
        self.built = False
        self.device = torch.device(device=device)
        self.to(device=self.device)

    def build(self, x):
        if not self.built:
            if x.ndim != 3:
                raise MLGBError

            self.built = True
        return

    def forward(self, x):
        self.build(x)
        x = x.to(device=self.device) if x.device.type != self.device.type else x

        fbi_fm_pool, fbi_mf_pool = [], []
        for k in range(self.group_width):
            x_k = torch.index_select(
                input=x,
                dim=1,
                index=torch.tensor(self.group_indices[k], dtype=torch.int32, device=self.device),
            )
            fbi_fm_k_outputs = self.fbi_fm_fn_list[k](x_k)
            fbi_fm_pool.append(fbi_fm_k_outputs)

        for k, (i, j) in enumerate(self.ij_ids):
            x_i = torch.index_select(
                input=x,
                dim=1,
                index=torch.tensor(self.group_indices[i], dtype=torch.int32, device=self.device),
            )
            x_j = torch.index_select(
                input=x,
                dim=1,
                index=torch.tensor(self.group_indices[j], dtype=torch.int32, device=self.device),
            )
            fbi_mf_k_ouputs = self.fbi_mf_fn_list[k]([x_i, x_j])
            fbi_mf_pool.append(fbi_mf_k_ouputs)

        fbi_pool = fbi_fm_pool + fbi_mf_pool
        x = torch.concat(fbi_pool, dim=1)
        return x























