import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from equinox import Module
from typing import Callable


class RBFInterpolator(Module):

    softplus_length_scale: float

    def __init__(self, scale):

        # TODO: Initialize the length scale of the RBF
        self.softplus_length_scale = scale

    def pairwise_compute(self, kernel, x, xp):

        kernel = jax.vmap(kernel, in_axes=(0, None))
        kernel = jax.vmap(kernel, in_axes=(None, 0))

        return kernel(x, xp)

    def log_rbf_kernel(self, x, xp):

        scale = 1e-5 + jax.nn.softplus(self.softplus_length_scale)
        return -jnp.power(jnp.linalg.norm(x - xp, keepdims=True) / scale, 2)

    def __call__(self, x, xp):
        """_summary_

        Args:
            x (_type_): Input features with shape (n_x, x_dim)
            xp (_type_): Induced features with shape (n_xp, x_dim)

        Returns:
            weights (_type_): Normalized RBF kernel weights with shape (n_xp, n_x, x_dim)
            density (_type_): Density of the RBF kernel with shape (n_xp, x_dim)
        """

        log_phi = self.pairwise_compute(self.log_rbf_kernel, x, xp)
        weights = jax.nn.softmax(log_phi, axis=-2)
        density = jnp.sum(jnp.exp(log_phi), axis=-2)

        return weights, density


class SetConv(Module):

    interpolator: Module
    resizer: eqx.nn.Linear

    def __init__(
        self,
        rng_key: jax.Array,
        in_channels: int,
        out_channels: int,
        interpolator: Module,
    ):

        self.interpolator = interpolator
        self.resizer = eqx.nn.Linear(
            in_features=in_channels + 1,
            out_features=out_channels,
            key=rng_key,
        )

    def __call__(self, x_context, x_induced, y_context):
        """_summary_

        Args:
            x_context (_type_): Context features with shape (n_context, x_dim)
            x_induced (_type_): Induced features with shape (n_induced, x_dim)
            y_context (_type_): Context targets with shape (n_context, y_dim)

        Returns:
            output (_type_): Output targets with shape (out_channels, n_induced)
        """

        # (n_induced, n_context, x_dim)
        weights, density = self.interpolator(x_context, x_induced)

        # Compute the target channel of the interpolation
        y_induced = jnp.sum(weights * y_context, axis=-2)

        targets = jnp.concatenate([y_induced, density], axis=-1)

        # Resize to number of output channels
        output = jax.vmap(self.resizer)(targets)
        output = jnp.transpose(output)

        return output
