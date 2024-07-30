import jax
import encoders
import decoders
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from equinox import Module
from LensNet.src.neural_process.base import AbstractConditionalNeuralProcess


class ConditionalNeuralProcess(AbstractConditionalNeuralProcess):

    encoder: encoders.XYEncoder
    decoder: decoders.GaussianDecoder

    def __init__(
        self,
        X_dim,
        Y_dim,
        rng_key,
        R_dim=128,
        encoder_X_depth=1,
        encoder_X_width=128,
        encoder_R_depth=2,
        encoder_R_width=256,
        decoder_depth=4,
        decoder_width=128,
        activation=jax.nn.leaky_relu,
    ):

        encoder_key, decoder_key = jr.split(rng_key)
        self.encoder = encoders.XYEncoder(
            x_dim=X_dim,
            y_dim=Y_dim,
            XT_depth=encoder_X_depth,
            XT_width=encoder_X_width,
            XT_output_dim=R_dim,
            XTY_depth=encoder_R_depth,
            XTY_width=encoder_R_width,
            XTY_output_dim=R_dim,
            activation=activation,
            rng_key=encoder_key,
        )

        self.decoder = decoders.GaussianDecoder(
            x_dim=X_dim,
            y_dim=Y_dim,
            R_dim=R_dim,
            depth=decoder_depth,
            width=decoder_width,
            activation=activation,
            rng_key=decoder_key,
        )

    def encode_globally(self, X_context, Y_context):

        return self.encoder(X_context, Y_context)

    def decode(self, X_target, R):

        return self.decoder(X_target, R)

    def __call__(self, X_context, Y_context, X_target):

        R = self.encode_globally(X_context, Y_context)
        Y_mu, Y_logvar = self.decode(X_target, R)

        return Y_mu, Y_logvar


class ConvCNP(AbstractConditionalNeuralProcess):

    encoder: Module
    decoder: Module
    n_induced: int

    def encode_globally(self, X_context, Y_context):

        pass

    def decode(self, X_target, R):

        pass

    def __call__(self, X_context, Y_context, X_target):

        pass
