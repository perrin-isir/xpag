"""
This is the SAC agent from JAXRL (https://github.com/ikostrikov/jaxrl),
put in a single file (without the buffer).
It implements the version of Soft-Actor-Critic described in
https://arxiv.org/abs/1812.05905.

The main modifications are:
    - the save() and load() methods of the Model class,
    - the removal of the dependency to tqdm,
    - the split of _sample_actions() into _sample_actions_deterministic() and
      _sample_actions_probabilistic(),
    - the inclusion of the parameter 'distribution' in the method sample_actions()
      of the class SACLearner.

To understand the code, the best is to check it on https://github.com/ikostrikov/jaxrl,
where there is a coherent structure and separated files.
The purpose here is just to have access to the SAC agent from JAXRL without
requiring all the JAXRL dependencies.

Here is the JAXRL License:
MIT License

Copyright (c) 2021 Ilya Kostrikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import flax
import optax
import collections
import functools
from typing import Union, Any, Callable, Dict, Optional, Sequence, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)

        params = variables.pop("params")

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1, apply_fn=model_def.apply, params=params, tx=tx, opt_state=opt_state
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def apply_gradient(
        self,
        loss_fn: Optional[Callable[[Params], Any]] = None,
        grads: Optional[Any] = None,
        has_aux: bool = True,
    ) -> Union[Tuple["Model", Any], "Model"]:
        assert (
            loss_fn is not None or grads is not None
        ), "Either a loss function or grads must be specified."
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert has_aux, "When grads are provided, expects no aux outputs."

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(
            step=self.step + 1, params=new_params, opt_state=new_opt_state
        )
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        os.makedirs(
            os.path.dirname(os.path.join(save_path, "step.flax")), exist_ok=True
        )
        os.makedirs(
            os.path.dirname(os.path.join(save_path, "params.flax")), exist_ok=True
        )
        os.makedirs(
            os.path.dirname(os.path.join(save_path, "opt_state.flax")), exist_ok=True
        )
        with open(os.path.join(save_path, "step.flax"), "wb") as f1:
            f1.write(flax.serialization.to_bytes(self.step))
        with open(os.path.join(save_path, "params.flax"), "wb") as f2:
            f2.write(flax.serialization.to_bytes(self.params))
        with open(os.path.join(save_path, "opt_state.flax"), "wb") as f3:
            f3.write(flax.serialization.to_bytes(self.opt_state))

    def load(self, load_path: str) -> "Model":
        with open(os.path.join(load_path, "step.flax"), "rb") as f1:
            step = flax.serialization.from_bytes(self.step, f1.read())
        with open(os.path.join(load_path, "params.flax"), "rb") as f2:
            params = flax.serialization.from_bytes(self.params, f2.read())
        with open(os.path.join(load_path, "opt_state.flax"), "rb") as f3:
            opt_state = flax.serialization.from_bytes(self.opt_state, f3.read())
        return self.replace(
            step=jax.tree_util.tree_map(jnp.array, step),
            params=jax.tree_util.tree_map(jnp.array, params),
            opt_state=jax.tree_util.tree_map(jnp.array, opt_state),
        )


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


def update_temperature(
    temp: Model, entropy: float, target_entropy: float
) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({"params": temp_params})
        temp_loss = (temperature * (entropy - target_entropy)).mean()
        return temp_loss, {"temperature": temperature, "temp_loss": temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info


Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


def update_actor(
    key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch
) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


def update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    backup_entropy: bool,
) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn(
            {"params": critic_params}, batch.observations, batch.actions
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = vmap_critic(self.hidden_dims, activations=self.activations)(
            states, actions
        )
        return qs


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


@functools.partial(jax.jit, static_argnames="actor_apply_fn")
def _sample_actions_deterministic(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations, temperature)
    return rng, dist.bijector(dist.distribution.loc)


@functools.partial(jax.jit, static_argnames="actor_apply_fn")
def _sample_actions_probabilistic(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: jnp.ndarray,
    temperature: float = 1.0,
    distribution: str = "log_prob",
) -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == "det":
        return _sample_actions_deterministic(
            rng, actor_apply_fn, actor_params, observations, temperature
        )
    else:
        return _sample_actions_probabilistic(
            rng, actor_apply_fn, actor_params, observations, temperature
        )


@functools.partial(jax.jit, static_argnames=("backup_entropy", "update_target"))
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    update_target: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class SACLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float,
        critic_lr: float,
        temp_lr: float,
        hidden_dims: Sequence[int],
        discount: float,
        tau: float,
        target_update_period: int,
        target_entropy: Optional[float],
        backup_entropy: bool,
        init_temperature: float,
        init_mean: Optional[jnp.ndarray],
        policy_final_fc_init_scale: float,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in
        https://arxiv.org/abs/1812.05905
        """
        tfd = tfp.distributions
        tfb = tfp.bijectors

        class NormalTanhPolicy(nn.Module):
            hidden_dims: Sequence[int]
            action_dim: int
            state_dependent_std: bool = True
            dropout_rate: Optional[float] = None
            final_fc_init_scale: float = 1.0
            log_std_min: Optional[float] = None
            log_std_max: Optional[float] = None
            tanh_squash_distribution: bool = True
            init_mean: Optional[jnp.ndarray] = None

            @nn.compact
            def __call__(
                self,
                observations: jnp.ndarray,
                temperature: float = 1.0,
                training: bool = False,
            ) -> tfd.Distribution:
                outputs = MLP(
                    self.hidden_dims,
                    activate_final=True,
                    dropout_rate=self.dropout_rate,
                )(observations, training=training)

                means = nn.Dense(
                    self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
                )(outputs)
                if self.init_mean is not None:
                    means += self.init_mean

                if self.state_dependent_std:
                    log_stds = nn.Dense(
                        self.action_dim,
                        kernel_init=default_init(self.final_fc_init_scale),
                    )(outputs)
                else:
                    log_stds = self.param(
                        "log_stds", nn.initializers.zeros, (self.action_dim,)
                    )

                log_std_min = self.log_std_min or LOG_STD_MIN
                log_std_max = self.log_std_max or LOG_STD_MAX
                log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

                if not self.tanh_squash_distribution:
                    means = nn.tanh(means)

                base_dist = tfd.MultivariateNormalDiag(
                    loc=means, scale_diag=jnp.exp(log_stds) * temperature
                )
                if self.tanh_squash_distribution:
                    return tfd.TransformedDistribution(
                        distribution=base_dist, bijector=tfb.Tanh()
                    )
                else:
                    return base_dist

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale,
        )
        actor = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = DoubleCritic(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )

        temp = Model.create(
            Temperature(init_temperature),
            inputs=[temp_key],
            tx=optax.adam(learning_rate=temp_lr),
        )

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 1

    def sample_actions(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        distribution: str = "log_prob",
    ) -> jnp.ndarray:
        rng, actions = sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature,
            distribution,
        )
        self.rng = rng
        actions = jnp.asarray(actions)
        return jnp.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.step % self.target_update_period == 0,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
