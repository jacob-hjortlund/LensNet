import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from equinox import Module
from typing import Callable


class DepthSeparableConv(Module):

    _depthwise: eqx.nn.Conv
    _pointwise: eqx.nn.Conv
    activation: Callable

    def __init__(
        self,
        rng_key: jax.Array,
        n_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        activation: Callable = jax.nn.leaky_relu,
    ):

        self.activation = activation

        self._depthwise = eqx.nn.Conv(
            num_spatial_dims=n_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            key=rng_key,
        )

        self._pointwise = eqx.nn.Conv(
            num_spatial_dims=n_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            key=rng_key,
        )

    def depthwise(self, x):

        return self._depthwise(x)

    def pointwise(self, x):

        return self._pointwise(x)

    def __call__(self, x):

        return self.pointwise(self.depthwise(x))


class ResConvBlock(Module):

    conv_layer_1: DepthSeparableConv
    conv_layer_2: DepthSeparableConv
    norm_1: eqx.nn.BatchNorm
    norm_2: eqx.nn.BatchNorm
    activation: Callable

    def __init__(
        self,
        rng_key,
        n_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        activation: Callable = jax.nn.leaky_relu,
    ):

        padding = kernel_size // 2

        self.activation = activation
        self.norm_1 = eqx.nn.BatchNorm(input_size=in_channels, axis_name="batch")
        self.norm_2 = eqx.nn.BatchNorm(input_size=in_channels, axis_name="batch")

        layer_keys = jr.split(rng_key, 2)
        self.conv_layer_1 = DepthSeparableConv(
            layer_keys[0],
            n_dim=n_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
        )

        self.conv_layer_2 = DepthSeparableConv(
            layer_keys[1],
            n_dim=n_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
        )

    def __call__(self, x, state):

        # First Convolution
        out, state = self.norm_1(x, state)
        out = self.activation(out)
        out = self.conv_layer_1(out)

        # Second convolution w. residual connection
        out, state = self.norm_2(out, state)
        out = self.activation(out)
        out = self.conv_layer_2.depthwise(out) + out
        out = self.conv_layer_2.pointwise(out)

        return out, state


class CNN(Module):

    blocks: list
    activation: Callable

    def __init__(
        self,
        rng_key,
        n_dim: int,
        n_blocks: int,
        n_channels: list,
        kernel_sizes: list,
        activation: Callable = jax.nn.leaky_relu,
    ):

        if type(n_channels) == int:
            n_channels = [n_channels] * (n_blocks + 1)

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * n_blocks

        if len(n_channels) != n_blocks + 1:
            raise ValueError("n_channels must have length n_blocks + 1")

        self.activation = activation

        self.blocks = []
        for i in range(1, len(n_channels)):
            self.blocks.append(
                ResConvBlock(
                    rng_key,
                    n_dim=n_dim,
                    in_channels=n_channels[i - 1],
                    out_channels=n_channels[i],
                    kernel_size=kernel_sizes[i - 1],
                    activation=activation,
                )
            )

    def __call__(self, x, state):

        for block in self.blocks:
            x, state = block(x, state)
        x = self.activation(x)

        return x, state
