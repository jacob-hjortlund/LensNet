import abc
import jax
import equinox as eqx
import jax.numpy as jnp

from equinox import Module


class AbstractConditionalNeuralProcess(Module):

    @abc.abstractmethod
    def encode_globally(self, X_context, Y_context):
        """Encode context set to a global representation.

        Args:
            X_context: (batch_size, *n_context, x_dim) Set of all context featuxes.
            Y_context: (batch_size, *n_context, y_dim) Set of all context values.

        Returns:
            R: (batch_size, global_dim) Global representation.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, X_target, R):
        """Decode global representation to target set.

        Args:
            X_target: (batch_size, *n_target, x_dim) Set of all target featuxes.
            R: (batch_size, global_dim) Global representation.

        Returns:
            mu_Y: (batch_size, *n_target, y_dim) Mean of target values.
            var_Y: (batch_size, *n_target, y_dim) Variance of target values.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, X_context, Y_context, X_target):
        """Forward pass of the model.

        Args:
            X_context: (batch_size, *n_context, x_dim) Set of all context featuxes.
            Y_context: (batch_size, *n_context, y_dim) Set of all context values.
            X_target: (batch_size, *n_target, x_dim) Set of all target featuxes.

        Returns:
            mu_Y: (batch_size, *n_target, y_dim) Mean of target values.
            var_Y: (batch_size, *n_target, y_dim) Variance of target values.
        """

        raise NotImplementedError
