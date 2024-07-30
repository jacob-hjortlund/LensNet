import jax
import equinox as eqx
import jax.numpy as jnp

from LensNet.src.neural_process import cnp
from equinox import Module


def gaussian_nll(mu, logvar, y):

    return 0.5 * jnp.sum(jnp.exp(-logvar) * (y - mu) ** 2 + logvar, axis=-1)


def CNPLoss(
    model: cnp.ConditionalNeuralProcess,
    x_context: jnp.ndarray,
    y_context: jnp.ndarray,
    x_target: jnp.ndarray,
    y_target: jnp.ndarray,
):

    mu_y, logvar_y = jax.vmap(model)(x_context, y_context, x_target)
    nlls = gaussian_nll(mu_y, logvar_y, y_target)
    batch_loss = jnp.mean(nlls)

    # print("\nCNPLoss x_trgt: ", x_target.shape)
    # print("CNPLoss y_trgt: ", y_target.shape)
    # print("nlls: ", nlls.shape)
    # print("batch_loss: ", batch_loss.shape, "\n")

    return batch_loss
