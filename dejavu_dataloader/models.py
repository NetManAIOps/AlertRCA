import torch as th
from torch import nn as nn
from einops import rearrange
from functools import reduce
from math import floor
from typing import Union, Tuple, List, Type, Callable, Dict
import logging as logger

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def conv_2d_output_shape(
        h_w: Tuple, kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: int = 1, pad: int = 0, dilation: int = 1
) -> Tuple[int, int]:
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[-2] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[-1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w

def conv_1d_output_shape(
        length: int, kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: int = 1, pad: int = 0, dilation: int = 1
) -> int:
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    length = floor(((length + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    return length

class SequentialModelBuilder:
    def __init__(self, input_shape: Union[th.Size, Tuple[int, ...]], debug=False):
        self._layer_cls: List[Union[Type[nn.Module], Callable[..., nn.Module]]] = []
        self._layer_kwargs: List[Dict] = []
        self._layer_args: List[List] = []
        self._output_shape: List[Tuple[int, ...]] = [input_shape]
        self._debug = debug

    @property
    def output_shapes(self):
        return self._output_shape

    @property
    def last_shape(self):
        return self._output_shape[-1]

    def build(self) -> nn.Module:
        layers = []
        if self._debug:
            logger.debug(f"=============================build layers==================================")
        for _cls, args, kwargs, shape in zip(
                self._layer_cls, self._layer_args, self._layer_kwargs, self._output_shape[1:]
        ):
            if self._debug:
                logger.debug(f"{_cls.__name__} {args=} {kwargs=} {shape=}")
            # noinspection PyArgumentList
            layers.append(_cls(*args, **kwargs))
        return nn.Sequential(
            *layers
        )

    def add_activation(self, activation='gelu') -> 'SequentialModelBuilder':
        if activation.lower() == 'relu':
            self._layer_cls.append(nn.ReLU)
        elif activation.lower() == 'gelu':
            self._layer_cls.append(nn.GELU)
        elif activation.lower() == 'softplus':
            self._layer_cls.append(nn.Softplus)
        else:
            raise RuntimeError(f"{activation=} unknown")
        self._layer_kwargs.append({})
        self._layer_args.append([])
        self._output_shape.append(self._output_shape[-1])
        return self

    def add_linear(
            self, out_features: int, bias: bool = True,
            device=None, dtype=None
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Linear)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(in_features=in_shape[-1], out_features=out_features, bias=bias,
                 device=device, dtype=dtype)
        )
        self._layer_args.append([])
        self._output_shape.append(in_shape[:-1] + (out_features,))
        return self

    def add_reshape(self, *args) -> 'SequentialModelBuilder':
        self._layer_cls.append(Reshape)
        self._layer_kwargs.append({})
        self._layer_args.append(list(args))
        self._output_shape.append(tuple(args))
        return self

    def add_flatten(self, start_dim: int = 1, end_dim: int = -1) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Flatten)
        self._layer_kwargs.append(dict(start_dim=start_dim, end_dim=end_dim))
        self._layer_args.append(list())
        input_shape = self.last_shape
        if end_dim != -1:
            self._output_shape.append(
                input_shape[:start_dim] + (
                    reduce(lambda a, b: a * b, input_shape[start_dim:end_dim + 1], 1),
                ) + input_shape[end_dim + 1:]
            )
        else:
            self._output_shape.append(
                input_shape[:start_dim] + (
                    reduce(lambda a, b: a * b, input_shape[start_dim:], 1),
                )
            )
        return self

    def add_max_pool_1d(self, kernel_size, stride=None, padding=0, dilation=1) -> 'SequentialModelBuilder':
        if stride is None:
            stride = kernel_size
        self._layer_cls.append(nn.MaxPool1d)
        self._layer_kwargs.append(dict(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-1] + (
                floor((self.last_shape[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1),
            )
        )
        return self

    def add_conv_1d(
            self,
            out_channels: int,
            kernel_size: Tuple[int],
            stride=1,
            padding=0,
            bias: bool = True,
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Conv1d)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(
                in_channels=in_shape[-2], out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        )
        self._layer_args.append([])
        after_conv_size = conv_1d_output_shape(in_shape[-1], kernel_size, stride=stride, pad=padding)
        self._output_shape.append(in_shape[:-2] + (out_channels, after_conv_size))
        return self

    def add_conv_2d(
            self,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride=1,
            padding=0,
            bias: bool = True,
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Conv2d)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(
                in_channels=in_shape[-3], out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        )
        self._layer_args.append([])
        after_conv_size = conv_2d_output_shape(in_shape[-2:], kernel_size, stride=stride, pad=padding)
        self._output_shape.append(in_shape[:-3] + (out_channels,) + after_conv_size)
        return self

    def add_dropout(self, p: float = 0.5):
        self._layer_cls.append(nn.Dropout)
        self._layer_kwargs.append({})
        self._layer_args.append([p])
        self._output_shape.append(self.last_shape)
        return self

    def add_conv_transpose_1d(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0
    ):
        self._layer_cls.append(nn.ConvTranspose1d)
        self._layer_kwargs.append(dict(
            in_channels=self.last_shape[-2],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        ))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-2] + (out_channels,) + (
                (self.last_shape[-1] - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1,
            )
        )
        return self

class GRUFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_layers: int = 1):
        super().__init__()
        self.x_dim = input_size[-2]
        self.n_ts = input_size[-1]
        self.z_dim = embedding_size
        self.num_layers = num_layers

        self.encoder = th.nn.GRU(
            input_size=self.x_dim, hidden_size=self.z_dim, num_layers=num_layers,
        )

        self.unify_mapper = SequentialModelBuilder(
            (self.z_dim, self.n_ts), debug=False,
        ).add_reshape(
            1, self.z_dim, self.n_ts,
        ).add_conv_1d(
            out_channels=10, kernel_size=(3,)
        ).add_activation().add_flatten(start_dim=-2).add_linear(embedding_size).add_reshape(
            embedding_size
        ).build()

    def forward(self, x):
        z = self.encode(x)
        embedding = self.unify_mapper(z)
        return embedding

    def encode(self, input_x: th.Tensor) -> th.Tensor:
        """
        :param input_x: (n_metrics, n_ts)
        :return: (self.z_dim, n_ts)
        """
        _, n_ts = input_x.size()
        x = rearrange(input_x, "m t -> t 1 m", m=self.x_dim, t=n_ts)
        assert x.size() == (n_ts, 1, self.x_dim)
        z, _ = self.encoder(x)
        return rearrange(z, "t 1 z -> z t", z=self.z_dim, t=n_ts)