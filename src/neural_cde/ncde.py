import jax
import diffrax

import jax.nn as nn
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        key,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=nn.softplus,
            final_activation=nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):

        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):

    initial: eqx.nn.MLP
    vector_field: VectorField
    linear: eqx.nn.Linear
    control_fn: diffrax.AbstractPath

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        control_fn: diffrax.AbstractPath,
        *,
        key,
        **kwargs
    ):

        super().__init__(**kwargs)
        ikey, vkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(
            in_size=data_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=ikey,
        )
        self.vector_field = VectorField(
            data_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=vkey,
        )
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)
        self.control_fn = control_fn

    def __call__(self, ts, coeffs):

        control_path = self.control_fn(ts, coeffs)
        term = diffrax.ControlTerm(self.vector_field, control_path).to_ode()
        solver = diffrax.Tsit5()
        dt0 = None
        y0 = self.initial(control_path.evaluate(ts[0]))
        saveat = diffrax.SaveAt(ts=ts)
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )

        prediction = jax.vmap(lambda y: nn.sigmoid(self.linear(y))[0])(solution.ys)

        return prediction
