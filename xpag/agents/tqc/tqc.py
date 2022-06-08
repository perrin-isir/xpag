# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.
"""
This is an implementation of a TQC agent (see https://arxiv.org/abs/2005.04269).
Some functions and classes are taken from the implementation of TQC in
RLJAX (https://github.com/ku2482/rljax).
Here is the License of RLJAX:
"""
# MIT License
#
# Copyright (c) 2020 Toshiki Watanabe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC
import numpy as np
import os
from xpag.agents.agent import Agent
from xpag.agents.sac.sac_from_jaxrl import (
    PRNGKey,
    InfoDict,
    Params,
    Batch,
    SACLearner,
    Model,
    MLP,
    update_temperature,
    target_update,
)
from xpag.tools.utils import squeeze
import functools
from typing import Callable, Any, Tuple, Sequence, Optional
import flax
import flax.linen as nn
import jax
import optax
import jax.numpy as jnp

# from IPython import embed


@functools.partial(jax.jit, static_argnames="critic_apply_fn")
def _qvalue(
    critic_apply_fn: Callable[..., Any],
    critic_params: flax.core.FrozenDict[str, Any],
    observations: np.ndarray,
    actions: np.ndarray,
) -> Tuple[jnp.ndarray]:
    return jnp.minimum(
        *critic_apply_fn({"params": critic_params}, observations, actions)
    )
    # c1, _ = critic_apply_fn({"params": critic_params}, observations, actions)
    # return c1


class QuantileCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_quantiles: int = 25

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, self.num_quantiles), activations=self.activations
        )(inputs)
        return jnp.squeeze(critic)


class MultiQuantileCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 5
    num_quantiles: int = 25

    @nn.compact
    def __call__(self, states, actions):
        vmap_critic = nn.vmap(
            QuantileCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = vmap_critic(
            self.hidden_dims,
            activations=self.activations,
            # num_quantiles=self.num_quantiles
        )(states, actions)
        return qs


@jax.jit
def huber(td: jnp.ndarray) -> jnp.ndarray:
    """Huber function."""
    abs_td = jnp.abs(td)
    return jnp.where(abs_td <= 1.0, jnp.square(td), abs_td)


@functools.partial(jax.jit, static_argnums=2)
def quantile_loss(
    td: jnp.ndarray,
    cum_p: jnp.ndarray,
    # weight: jnp.ndarray,
    loss_type: str,
) -> jnp.ndarray:
    """
    Calculate quantile loss.
    """
    if loss_type == "l2":
        element_wise_loss = jnp.square(td)
    elif loss_type == "huber":
        element_wise_loss = huber(td)
    else:
        NotImplementedError
    element_wise_loss *= jax.lax.stop_gradient(jnp.abs(cum_p[..., None] - (td < 0)))
    batch_loss = element_wise_loss.sum(axis=1).mean(axis=1, keepdims=True)
    # return (batch_loss * weight).mean()
    return batch_loss.mean()

    # @partial(jax.jit, static_argnums=0)
    # def _calculate_value(
    #     self,
    #     params_critic: hk.Params,
    #     state: np.ndarray,
    #     action: np.ndarray,
    # ) -> jnp.ndarray:
    #     return jnp.concatenate(self._calculate_value_list(params_critic, state,
    #                                                       action), axis=1)
    #
    # @partial(jax.jit, static_argnums=0)
    # def _calculate_target(
    #     self,
    #     params_critic_target: hk.Params,
    #     log_alpha: jnp.ndarray,
    #     reward: np.ndarray,
    #     done: np.ndarray,
    #     next_state: np.ndarray,
    #     next_action: jnp.ndarray,
    #     next_log_pi: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     next_quantile = self._calculate_value(params_critic_target, next_state,
    #                                           next_action)
    #     next_quantile = jnp.sort(next_quantile)[:, : self.num_quantiles_target]
    #     next_quantile -= jnp.exp(log_alpha) * self._calculate_log_pi(next_action,
    #                                                                  next_log_pi)
    #     return jax.lax.stop_gradient(
    #         reward + (1.0 - done) * self.discount * next_quantile)
    #
    # @partial(jax.jit, static_argnums=0)
    # def _calculate_loss_critic_and_abs_td(
    #     self,
    #     quantile_list: List[jnp.ndarray],
    #     target: jnp.ndarray,
    #     weight: np.ndarray,
    # ) -> jnp.ndarray:
    #     loss_critic = 0.0
    #     for quantile in quantile_list:
    #         loss_critic += quantile_loss(target[:, None, :] - quantile[:, :, None],
    #                                      self.cum_p_prime, weight, "huber")
    #     loss_critic /= self.num_critics * self.num_quantiles
    #     abs_td = jnp.abs(
    #         target[:, None, :] - quantile_list[0][:, :, None]
    #     ).mean(axis=1).mean(axis=1, keepdims=True)
    #     return loss_critic, jax.lax.stop_gradient(abs_td)


def update_actor(
    key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch
) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({"params": actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        # embed()
        # q1, q2 = critic(batch.observations, actions)
        # q = jnp.minimum(q1, q2)
        # actor_loss = (log_probs * temp() - q).mean()
        qs = jnp.mean(critic(batch.observations, actions), axis=(0, 2))
        actor_loss = (log_probs * temp() - qs).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    num_quantiles_target: int,
    cum_p_prime: jnp.ndarray,
    backup_entropy: bool,
) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    # next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    # next_q = jnp.minimum(next_q1, next_q2)
    # target_q = batch.rewards + discount * batch.masks * next_q

    next_quantile = jnp.concatenate(
        target_critic(batch.next_observations, next_actions), axis=1
    )
    next_quantile = jnp.sort(next_quantile)[:, :num_quantiles_target]
    target_q = (
        jnp.expand_dims(batch.rewards, axis=-1)
        + discount * jnp.expand_dims(batch.masks, axis=-1) * next_quantile
    )

    if backup_entropy:
        target_q -= jnp.expand_dims(
            discount * batch.masks * temp() * next_log_probs, axis=-1
        )

    #     self,
    #     quantile_list: List[jnp.ndarray],
    #     target: jnp.ndarray,
    #     weight: np.ndarray,
    # ) -> jnp.ndarray:
    #     loss_critic = 0.0
    #     for quantile in quantile_list:
    #         loss_critic += quantile_loss(target[:, None, :] - quantile[:, :, None],
    #                                      self.cum_p_prime, weight, "huber")
    #     loss_critic /= self.num_critics * self.num_quantiles
    #     abs_td = jnp.abs(
    #         target[:, None, :] - quantile_list[0][:, :, None]
    #     ).mean(axis=1).mean(axis=1, keepdims=True)
    #     return loss_critic, jax.lax.stop_gradient(abs_td)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # q1, q2 = critic.apply_fn(
        #     {"params": critic_params}, batch.observations, batch.actions
        # )
        qs = critic.apply_fn(
            {"params": critic_params}, batch.observations, batch.actions
        )
        critic_loss = jnp.array(0.0)
        for q in qs:
            critic_loss += quantile_loss(
                target_q[:, None, :] - q[:, :, None], cum_p_prime, "huber"
            )
        # loss_critic /= self.num_critics * self.num_quantiles
        critic_loss /= qs.shape[0] * qs.shape[2]
        # abs_td = jnp.abs(
        #             target[:, None, :] - quantile_list[0][:, :, None]
        #         ).mean(axis=1).mean(axis=1, keepdims=True)
        # critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            # "q1": q1.mean(),
            # "q2": q2.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


@functools.partial(
    jax.jit, static_argnames=("backup_entropy", "update_target", "num_quantiles_target")
)
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
    num_quantiles_target: int,
    cum_p_prime: jnp.ndarray,
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
        num_quantiles_target,
        cum_p_prime,
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


class TQCLearner(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims_actor: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        init_mean: Optional[np.ndarray] = None,
        policy_final_fc_init_scale: float = 1.0,
        # hidden_dims_critic: Sequence[int] = (512, 512, 512),
        hidden_dims_critic: Sequence[int] = (256, 256),
        num_critics=5,
        num_quantiles=25,
        num_quantiles_to_drop=2,
    ):
        super().__init__(
            seed,
            observations,
            actions,
            actor_lr,
            critic_lr,
            temp_lr,
            hidden_dims_actor,
            discount,
            tau,
            target_update_period,
            target_entropy,
            backup_entropy,
            init_temperature,
            init_mean,
            policy_final_fc_init_scale,
        )
        self.rng, critic_key = jax.random.split(self.rng)
        # critic_def = DoubleCritic(hidden_dims_critic)
        critic_def = MultiQuantileCritic(
            hidden_dims_critic, nn.relu, num_critics, num_quantiles
        )
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )

        self.critic = critic
        self.target_critic = target_critic

        self.cum_p_prime = jnp.expand_dims(
            (jnp.arange(0, num_quantiles, dtype=jnp.float32) + 0.5) / num_quantiles, 0
        )
        self.num_quantiles = num_quantiles
        self.num_quantiles_target = (
            num_quantiles - num_quantiles_to_drop
        ) * num_critics
        # embed()

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
            self.num_quantiles_target,
            self.cum_p_prime,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info


class TQC(Agent, ABC):
    def __init__(self, observation_dim, action_dim, params=None):
        """
        Interface to TQC agent
        """

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, params)

        if "seed" in self.params:
            start_seed = self.params["seed"]
        else:
            start_seed = 42

        self.tqclearner_params = {
            "actor_lr": 0.0003,
            "backup_entropy": True,
            "critic_lr": 0.0003,
            "discount": 0.99,
            "hidden_dims_actor": (256, 256),
            "init_temperature": 1.0,
            "target_entropy": None,
            "target_update_period": 1,
            "tau": 0.005,
            "temp_lr": 0.0003,
        }

        for key in self.tqclearner_params:
            if key in self.params:
                self.tqclearner_params[key] = self.params[key]

        self.tqc = TQCLearner(
            start_seed,
            np.zeros((1, 1, observation_dim)),
            np.zeros((1, 1, action_dim)),
            **self.tqclearner_params
        )

    def value(self, observation, action):
        return np.asarray(
            _qvalue(
                self.tqc.critic.apply_fn, self.tqc.critic.params, observation, action
            )
        )

    def select_action(self, observation, eval_mode=False):
        # return self.sac.sample_actions(observation)
        return self.tqc.sample_actions(
            observation, distribution="det" if eval_mode else "log_prob"
        )

    def train_on_batch(self, batch):
        jaxrl_batch = Batch(
            observations=batch["observation"],
            actions=batch["action"],
            rewards=squeeze(batch["reward"]),
            masks=squeeze(1 - batch["done"] * (1 - batch["truncation"])),
            next_observations=batch["next_observation"],
        )

        return self.tqc.update(jaxrl_batch)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "step.npy"), self.tqc.step)
        self.tqc.actor.save(os.path.join(directory, "actor"))
        self.tqc.critic.save(os.path.join(directory, "critic"))
        self.tqc.target_critic.save(os.path.join(directory, "target_critic"))
        self.tqc.temp.save(os.path.join(directory, "temp"))

    def load(self, directory):
        self.tqc.step = np.load(os.path.join(directory, "step.npy")).item()
        self.tqc.actor = self.tqc.actor.load(os.path.join(directory, "actor"))
        self.tqc.critic = self.tqc.critic.load(os.path.join(directory, "critic"))
        self.tqc.target_critic = self.tqc.target_critic.load(
            os.path.join(directory, "target_critic")
        )
        self.tqc.temp = self.tqc.temp.load(os.path.join(directory, "temp"))

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)
