# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC
from typing import Any, Tuple, Sequence, Callable
import dataclasses
import numpy as np
import torch
import flax
from flax import linen
import jax
import jax.numpy as jnp
import optax
from xpag.agents.agent import Agent
import os
import joblib

Params = Any
PRNGKey = jnp.ndarray


@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    target_policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    key: PRNGKey
    steps: jnp.ndarray


class TD3(Agent, ABC):
    def __init__(
        self,
        observation_dim,
        action_dim,
        params=None,
    ):
        """
        Jax implementation of TD3 (https://arxiv.org/abs/1802.09477).
        This version assumes that the actions are between -1 and 1 (for all
        dimensions).
        """

        discount = 0.99 if "discount" not in params else params["discount"]
        reward_scale = 1.0 if "reward_scale" not in params else params["reward_scale"]
        policy_lr = 3e-4 if "policy_lr" not in params else params["policy_lr"]
        critic_lr = 3e-4 if "critic_lr" not in params else params["critic_lr"]
        soft_target_tau = (
            0.005 if "soft_target_tau" not in params else params["soft_target_tau"]
        )
        self.backend = None if "backend" not in params else params["backend"]

        class CustomMLP(linen.Module):
            """MLP module."""

            layer_sizes: Sequence[int]
            activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
            kernel_init_hidden_layer: Callable[
                ..., Any
            ] = jax.nn.initializers.lecun_uniform()
            kernel_init_last_layer: Callable[
                ..., Any
            ] = jax.nn.initializers.lecun_uniform()
            bias_init_hidden_layer: Callable[
                ..., Any
            ] = jax.nn.initializers.lecun_uniform()
            bias_init_last_layer: Callable[
                ..., Any
            ] = jax.nn.initializers.lecun_uniform()
            activate_final: bool = False
            bias: bool = True

            @linen.compact
            def __call__(self, data: jnp.ndarray):
                hidden = data
                for i, hidden_size in enumerate(self.layer_sizes):
                    hidden = linen.Dense(
                        hidden_size,
                        name=f"hidden_{i}",
                        kernel_init=self.kernel_init_hidden_layer
                        if (i != len(self.layer_sizes) - 1)
                        else self.kernel_init_last_layer,
                        bias_init=self.bias_init_hidden_layer
                        if (i != len(self.layer_sizes) - 1)
                        else self.bias_init_last_layer,
                        use_bias=self.bias,
                    )(hidden)
                    if i != len(self.layer_sizes) - 1 or self.activate_final:
                        hidden = self.activation(hidden)
                return hidden

        def kernel_init_hidden_layer(key_, shape, dtype=jnp.float_):
            # len(shape) should be 2
            dtype = jax.dtypes.canonicalize_dtype(dtype)
            mval = 1.0 / jnp.sqrt(jnp.maximum(shape[0], shape[1]))
            return jax.random.uniform(key_, shape, dtype, -mval, mval)

        def bias_init_hidden_layer(key_, shape, dtype=jnp.float_):
            return 0.1 * jnp.ones(shape, jax.dtypes.canonicalize_dtype(dtype))

        def init_last_layer(key_, shape, dtype=jnp.float_):
            dtype = jax.dtypes.canonicalize_dtype(dtype)
            mval = 1e-3
            return jax.random.uniform(key_, shape, dtype, -mval, mval)

        def make_td3_networks(
            param_size: int,
            obs_size: int,
            action_size: int,
            hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        ) -> Tuple[FeedForwardModel, FeedForwardModel]:
            """Creates a policy and value networks for TD3."""
            policy_module = CustomMLP(
                layer_sizes=hidden_layer_sizes + (param_size,),
                activation=linen.relu,
                kernel_init_hidden_layer=kernel_init_hidden_layer,
                kernel_init_last_layer=init_last_layer,
                bias_init_hidden_layer=bias_init_hidden_layer,
                bias_init_last_layer=init_last_layer,
            )

            class QModule(linen.Module):
                """Q Module."""

                n_critics: int = 2

                @linen.compact
                def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
                    hidden = jnp.concatenate([obs, actions], axis=-1)
                    res = []
                    for _ in range(self.n_critics):
                        q = CustomMLP(
                            layer_sizes=hidden_layer_sizes + (1,),
                            activation=linen.relu,
                            kernel_init_hidden_layer=kernel_init_hidden_layer,
                            kernel_init_last_layer=init_last_layer,
                            bias_init_hidden_layer=bias_init_hidden_layer,
                            bias_init_last_layer=init_last_layer,
                        )(hidden)
                        res.append(q)
                    return jnp.concatenate(res, axis=-1)

            q_module = QModule()

            dummy_obs = jnp.zeros((1, obs_size))
            dummy_action = jnp.zeros((1, action_size))
            policy = FeedForwardModel(
                init=lambda key_: policy_module.init(key_, dummy_obs),
                apply=policy_module.apply,
            )
            value = FeedForwardModel(
                init=lambda key_: q_module.init(key_, dummy_obs, dummy_action),
                apply=q_module.apply,
            )
            return policy, value

        self._config_string = str(list(locals().items())[1:])
        super().__init__("TD3", observation_dim, action_dim, params)

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau

        if "seed" in self.params:
            start_seed = self.params["seed"]
        else:
            start_seed = 42

        self.key, local_key, key_models = jax.random.split(
            jax.random.PRNGKey(start_seed), 3
        )

        self.policy_model, self.value_model = make_td3_networks(
            action_dim, observation_dim, action_dim
        )

        self.policy_optimizer = optax.adam(learning_rate=1.0 * policy_lr)
        self.q_optimizer = optax.adam(learning_rate=1.0 * critic_lr)

        key_policy, key_q = jax.random.split(key_models)
        self.policy_params = self.policy_model.init(key_policy)
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.q_params = self.value_model.init(key_q)
        self.q_optimizer_state = self.q_optimizer.init(self.q_params)

        def postprocess(x):
            return jnp.tanh(x)

        self.postprocess = postprocess

        def actor_loss(
            policy_params: Params, q_params: Params, observations
        ) -> jnp.ndarray:
            p_actions = self.policy_model.apply(policy_params, observations)
            p_actions = self.postprocess(p_actions)
            q_action = self.value_model.apply(q_params, observations, p_actions)
            min_q = jnp.min(q_action, axis=-1)
            return -jnp.mean(min_q)

        def critic_loss(
            q_params: Params,
            target_policy_params: Params,
            target_q_params: Params,
            observations,
            actions,
            new_observations,
            rewards,
            mask,
            key_,
        ) -> jnp.ndarray:
            next_pre_actions = self.policy_model.apply(
                target_policy_params, new_observations
            )
            new_next_actions = self.postprocess(next_pre_actions)
            policy_noise = 0.2
            noise_clip = 0.5
            new_next_actions = jnp.clip(
                new_next_actions
                + jnp.clip(
                    policy_noise
                    * jax.random.normal(key_, shape=new_next_actions.shape),
                    -noise_clip,
                    noise_clip,
                ),
                -1.0,
                1.0,
            )

            q_old_action = self.value_model.apply(q_params, observations, actions)
            next_q = self.value_model.apply(
                target_q_params, new_observations, new_next_actions
            )
            next_v = jnp.min(next_q, axis=-1)
            target_q = jax.lax.stop_gradient(
                rewards * self.reward_scale
                + mask * discount * jnp.expand_dims(next_v, -1)
            )
            q_error = q_old_action - target_q
            q_loss = 2.0 * jnp.mean(jnp.square(q_error))
            return q_loss

        self.critic_grad = jax.value_and_grad(critic_loss)
        self.actor_grad = jax.value_and_grad(actor_loss)

        def update_step(
            state: TrainingState, observations, actions, rewards, new_observations, mask
        ) -> (TrainingState, dict):

            key, key_critic = jax.random.split(state.key, 2)

            actor_l, actor_grads = self.actor_grad(
                state.policy_params, state.target_q_params, observations
            )

            policy_params_update, policy_optimizer_state = self.policy_optimizer.update(
                actor_grads, state.policy_optimizer_state
            )

            policy_params = optax.apply_updates(
                state.policy_params, policy_params_update
            )

            critic_l, critic_grads = self.critic_grad(
                state.q_params,
                state.target_policy_params,
                state.target_q_params,
                observations,
                actions,
                new_observations,
                rewards,
                mask,
                key_critic,
            )

            q_params_update, q_optimizer_state = self.q_optimizer.update(
                critic_grads, state.q_optimizer_state
            )
            q_params = optax.apply_updates(state.q_params, q_params_update)

            new_target_q_params = jax.tree_multimap(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_q_params,
                q_params,
            )

            new_target_policy_params = jax.tree_multimap(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_policy_params,
                policy_params,
            )

            new_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                target_policy_params=new_target_policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                key=key,
                steps=state.steps + 1,
            )

            metrics = {}

            return new_state, metrics

        self.update_step = jax.jit(update_step, backend=self.backend)

        def select_action_probabilistic(observation, policy_params, key_):
            pre_action = self.policy_model.apply(policy_params, observation)
            pre_action = self.postprocess(pre_action)
            expl_noise = 0.1
            return jnp.clip(
                pre_action
                + expl_noise * jax.random.normal(key_, shape=pre_action.shape),
                -1.0,
                1.0,
            )

        def select_action_deterministic(observation, policy_params, key_=None):
            pre_action = self.policy_model.apply(policy_params, observation)
            return self.postprocess(pre_action)

        self.select_action_probabilistic = jax.jit(
            select_action_probabilistic, backend=self.backend
        )
        self.select_action_deterministic = jax.jit(
            select_action_deterministic, backend=self.backend
        )

        def q_value(observation, action, q_params):
            q_action = self.value_model.apply(q_params, observation, action)
            min_q = jnp.min(q_action, axis=-1)
            return min_q

        self.q_value = jax.jit(q_value, backend=self.backend)

        self.training_state = TrainingState(
            policy_optimizer_state=self.policy_optimizer_state,
            policy_params=self.policy_params,
            target_policy_params=self.policy_params,
            q_optimizer_state=self.q_optimizer_state,
            q_params=self.q_params,
            target_q_params=self.q_params,
            key=local_key,
            steps=jnp.zeros((1,)),
        )

    def value(self, observation, action):
        if torch.is_tensor(observation):
            version = "torch"
        else:
            version = "numpy"
        if version == "numpy":
            return self.q_value(
                observation,
                action,
                # self.training_state.target_q_params
                self.training_state.q_params,
            )
        else:
            return self.q_value(
                observation.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                # self.training_state.target_q_params
                self.training_state.q_params,
            )

    def select_action(self, observation, deterministic=True):
        self.key, key_sample = jax.random.split(self.key)
        if deterministic:
            apply_func = self.select_action_deterministic
        else:
            apply_func = self.select_action_probabilistic
        if torch.is_tensor(observation):
            version = "torch"
        else:
            version = "numpy"
        if version == "numpy":
            action = apply_func(
                observation, self.training_state.policy_params, key_sample
            )
        else:
            action = apply_func(
                observation.detach().cpu().numpy(),
                self.training_state.policy_params,
                key_sample,
            )
        if len(action.shape) == 1:
            return np.asarray(jnp.expand_dims(action, axis=0))
        else:
            return np.asarray(action)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for filename in self.training_state.__dict__.keys():
            with open(os.path.join(directory, filename + ".joblib"), "wb") as f_:
                joblib.dump(self.training_state.__dict__[filename], f_)

    def load(self, directory):
        load_all = {}
        for filename in self.training_state.__dict__.keys():
            load_all[filename] = jax.tree_util.tree_multimap(
                jnp.array, joblib.load(os.path.join(directory, filename + ".joblib"))
            )
        self.training_state = TrainingState(
            policy_optimizer_state=load_all["policy_optimizer_state"],
            policy_params=load_all["policy_params"],
            target_policy_params=load_all["target_policy_params"],
            q_optimizer_state=load_all["q_optimizer_state"],
            q_params=load_all["q_params"],
            target_q_params=load_all["target_q_params"],
            key=load_all["key"],
            steps=load_all["steps"],
        )

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)

    def train_on_batch(self, batch):
        if torch.is_tensor(batch["reward"]):
            version = "torch"
        else:
            version = "numpy or jax"
        if version == "numpy or jax":
            observations = batch["observation"]
            actions = batch["action"]
            rewards = batch["reward"]
            new_observations = batch["next_observation"]
            mask = 1 - batch["done"] * (1 - batch["truncation"])
        else:
            observations = batch["observation"].detach().cpu().numpy()
            actions = batch["action"].detach().cpu().numpy()
            rewards = batch["reward"].detach().cpu().numpy()
            new_observations = batch["next_observation"].detach().cpu().numpy()
            mask = 1 - batch["done"].detach().cpu().numpy() * (
                1 - batch["truncation"].detach().cpu().numpy()
            )

        self.training_state, metrics = self.update_step(
            self.training_state, observations, actions, rewards, new_observations, mask
        )

        return metrics
