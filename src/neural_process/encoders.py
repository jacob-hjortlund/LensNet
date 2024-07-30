import jax
import equinox as eqx
import jax.numpy as jnp

from equinox import Module
from typing import Callable
from LensNet.src.nn import CNN
from LensNet.src.nn.setconv import RBFInterpolator, SetConv


class XYEncoder(Module):

    XT_mlp: eqx.nn.MLP
    XTY_mlp: eqx.nn.MLP

    def __init__(
        self,
        x_dim,
        y_dim,
        rng_key,
        XT_depth=1,
        XT_width=128,
        XT_output_dim=128,
        XTY_depth=2,
        XTY_width=256,
        XTY_output_dim=128,
        activation=jax.nn.leaky_relu,
    ):
        """Encoder for context features and values. First transforms context features
        using an MLP with XT_depth layers and XT_width units. Then concatenates the
        transformed features with context values and encodes them using an MLP with
        XTY_depth layers and XTY_width units.

        Args:
            x_dim (int): Dimension of context features.
            y_dim (int): Dimension of context values.
            rng_key (jax.random.PRNGKey): Random key.
            XT_depth (int): Depth of context feature transformation MLP.
            XT_width (int): Width of context feature transformation MLP hidden layers.
            XT_output_dim (int): Dimension of context feature transformation output.
            XTY_depth (int): Depth of encoding MLP.
            XTY_width (int): Width of encoding MLP hidden layers.
            XTY_output_dim (int): Dimension of encoding MLP output.
            activation (Callable): Activation function.
        """

        key1, key2 = jax.random.split(rng_key)
        self.XT_mlp = eqx.nn.MLP(
            in_size=x_dim,
            out_size=XT_output_dim,
            depth=XT_depth,
            width_size=XT_width,
            activation=activation,
            key=key1,
        )
        self.XTY_mlp = eqx.nn.MLP(
            in_size=XT_output_dim + y_dim,
            out_size=XTY_output_dim,
            depth=XTY_depth,
            width_size=XTY_width,
            activation=activation,
            key=key2,
        )

    def single_call(self, x, y):
        """Encode a single pair of context features and values.

        Args:
            x (jax.Array): (x_dim,) Context features.
            y (jax.Array): (y_dim,) Context values.

        Returns:
            E (jax.Array): (R_dim,) Global representation for the pair.
        """

        # print("Encoder Single Call X: ", x.shape)
        # print("Encoder Single Call Y: ", y.shape, "\n")
        xT = self.XT_mlp(x)
        xTy = jnp.concatenate([xT, y], axis=-1)
        E = self.XTY_mlp(xTy)

        return E

    def aggregate(self, E, X):
        """Aggregate global representations using the mean.

        Args:
            E (jax.Array): (n_context, R_dim) Global representations.
            X (jax.Array): (n_context, x_dim) Context features.

        Returns:
            R (jax.Array): (R_dim,) Aggregated global representation.
        """

        where_observed = jnp.all(X != -9999.0, axis=-1).reshape(-1, 1)
        where_observed = jnp.repeat(where_observed, E.shape[-1], axis=1)
        R = jnp.mean(E, axis=0, where=where_observed)
        return R

    def __call__(self, X, Y):
        """Encode context set to a global representation.

        Args:
            X (jax.Array): (n_context, x_dim) Set of all context features.
            Y (jax.Array): (n_context, y_dim) Set of all context values.

        Returns:
            R (jax.Array): (R_dim,) Global representation.
        """

        # print("\nEncoder X: ", X.shape)
        # print("Encoder Y", Y.shape)
        E = jax.vmap(self.single_call)(X, Y)
        R = self.aggregate(E, X)

        return R


class SetConvolutionEncoder(Module):

    context_setconv = SetConv
    target_setconv = SetConv
    cnn = CNN

    def __init__(
        self,
        x_dim,
        y_dim,
        rng_key,
        n_resnet_blocks=5,
        setconv_channels=128,
        cnn_channels=128,
        cnn_kernel_sizes=5,
        activation=jax.nn.leaky_relu,
    ):

        if type(cnn_channels) == int:
            cnn_channels = [setconv_channels] + [cnn_channels] * n_resnet_blocks
        if cnn_channels[0] != setconv_channels:
            raise ValueError("First CNN channel must match SetConv channel size")

        if cnn_kernel_sizes == int:
            cnn_kernel_sizes = [cnn_kernel_sizes] * n_resnet_blocks

        (
            context_interp_key,
            context_setconv_key,
            target_interp_key,
            target_setconv_key,
            cnn_key,
        ) = jax.random.split(rng_key, 5)
        context_interpolator = RBFInterpolator(context_interp_key)
        self.context_setconv = SetConv(
            context_setconv_key,
            in_channels=y_dim,
            out_channels=setconv_channels,
            interpolator=context_interpolator,
        )

        target_interpolator = RBFInterpolator(target_interp_key)
        self.target_setconv = SetConv(
            target_setconv_key,
            in_channels=setconv_channels,
            out_channels=setconv_channels,
            interpolator=target_interpolator,
        )

        self.cnn = CNN(
            cnn_key,
            n_dim=y_dim,
            n_blocks=n_resnet_blocks,
            n_channels=cnn_channels,
            kernel_size=cnn_kernel_sizes,
            activation=activation,
        )

    def encode(self, X, Y):

        X_induced = jnp.linspace(-1, 1, 100)
        set_convolution = self.context_setconv(X, X_induced, Y)
        R_global = self.cnn(set_convolution)

        return R_global

    def target_dependent_encoding(self, X, Y):

        X_induced = jnp.linspace(-1, 1, 100)
        R_global = self.encode(X, Y)
        R_target = self.target_setconv(X, X_induced, R_global)

        return R_target

    def __call__(self, X, Y):

        return self.target_dependent_encoding(X, Y)
