import jax
import equinox as eqx
import jax.numpy as jnp

from equinox import Module


class GaussianDecoder(Module):

    mean_mlp: eqx.nn.MLP
    logvar_mlp: eqx.nn.MLP

    def __init__(
        self,
        x_dim,
        y_dim,
        depth,
        width,
        R_dim,
        rng_key,
        activation=jax.nn.relu,
    ):
        """Decoder for global representation. Decodes global representation and target
        feature to target value mean and variance.

        Args:
            x_dim (int): Dimension of target features.
            y_dim (int): Dimension of target values.
            depth (int): Depth of MLP.
            width (int): Width of MLP.
            R_dim (int): Dimension of global representation.
            rng_key (jax.random.PRNGKey): Random key.
            activation (Callable): Activation function.
        """

        mean_key, var_key = jax.random.split(rng_key)

        self.mean_mlp = eqx.nn.MLP(
            in_size=x_dim + R_dim,
            out_size=y_dim,
            depth=depth,
            width_size=width,
            activation=activation,
            key=mean_key,
        )

        self.logvar_mlp = eqx.nn.MLP(
            in_size=x_dim + R_dim,
            out_size=y_dim,
            depth=depth,
            width_size=width,
            activation=activation,
            key=var_key,
        )

    def single_call(self, R_target):
        """Decode global representation at a target feature.

        Args:
            R_target (jax.Array): (R_dim) Target dependent representation

        Returns:
            y_pred (jax.Array): (y_dim,) Predicted target value.
            y_logvar (jax.Array): (y_dim,) Predicted target value log-variance.
        """

        y_pred = self.mean_mlp(R_target)
        y_logvar = self.logvar_mlp(R_target)

        # print("Decoder Single Call X: ", x_target.shape)
        # print("Decoder Single Call R: ", R.shape)
        # print("Decoder Single Call Y_pred: ", y_pred.shape)
        # print("Decoder Single Call Y_logvar: ", y_logvar.shape)

        return y_pred, y_logvar

    def __call__(self, R_target):
        """Decode global representation for target set X_target.

        Args:
            R_target (jax.Array): (n_target, R_dim) Target dependent representations.

        Returns:
            Y_mu (jax.Array): (n_target, y_dim) Predicted target value means.
            Y_logvar (jax.Array): (n_target, y_dim) Predicted target value log variances.
        """

        # print("\nDecoder X_trgt: ", X_target.shape)
        # print("Decoder R: ", R.shape, "\n")

        Y_mu, Y_logvar = jax.vmap(
            self.single_call,
        )(R_target)

        Y_mu = jnp.squeeze(Y_mu)
        Y_logvar = jnp.squeeze(Y_logvar)

        # print("Decoder Y_mu: ", Y_mu.shape)
        # print("Decoder Y_logvar: ", Y_logvar.shape, "\n")

        return Y_mu, Y_logvar
