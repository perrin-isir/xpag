# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.
#
#
# This file is an implementation of the SAC (Soft-Actor Critic) algorithm.
# It is partially derived from the implementation of SAC in brax
# [https://github.com/google/brax/blob/main/brax/training/sac.py]
# which contains the following copyright notice:
#
# Copyright 2022 The Brax Authors.
# Licensed under the Apache License, Version 2.0.

from abc import ABC
from typing import Any, Tuple, Sequence, Callable
import dataclasses
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
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    key: PRNGKey
    steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params


class SAC(Agent, ABC):
    def __init__(
            self,
            observation_dim,
            action_dim,
            params=None,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=3e-4,
            soft_target_tau=0.005
    ):

        class CustomMLP(linen.Module):
            """MLP module."""
            layer_sizes: Sequence[int]
            activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
            kernel_init_hidden_layer: Callable[
                ..., Any] = jax.nn.initializers.lecun_uniform()
            kernel_init_last_layer: Callable[
                ..., Any] = jax.nn.initializers.lecun_uniform()
            bias_init_hidden_layer: Callable[
                ..., Any] = jax.nn.initializers.lecun_uniform()
            bias_init_last_layer: Callable[
                ..., Any] = jax.nn.initializers.lecun_uniform()
            activate_final: bool = False
            bias: bool = True

            @linen.compact
            def __call__(self, data: jnp.ndarray):
                hidden = data
                for i, hidden_size in enumerate(self.layer_sizes):
                    hidden = linen.Dense(
                        hidden_size,
                        name=f'hidden_{i}',
                        kernel_init=self.kernel_init_hidden_layer if (
                                i != len(self.layer_sizes) - 1
                        ) else self.kernel_init_last_layer,
                        bias_init=self.bias_init_hidden_layer if (
                                i != len(self.layer_sizes) - 1
                        ) else self.bias_init_last_layer,
                        use_bias=self.bias)(
                        hidden)
                    if i != len(self.layer_sizes) - 1 or self.activate_final:
                        hidden = self.activation(hidden)
                return hidden

        def kernel_init_hidden_layer(key_, shape, dtype=jnp.float_):
            # len(shape) should be 2
            dtype = jax.dtypes.canonicalize_dtype(dtype)
            mval = 1. / jnp.sqrt(jnp.maximum(shape[0], shape[1]))
            return jax.random.uniform(key_, shape, dtype, -mval, mval)

        def bias_init_hidden_layer(key_, shape, dtype=jnp.float_):
            return 0.1 * jnp.ones(shape, jax.dtypes.canonicalize_dtype(dtype))

        def init_last_layer(key_, shape, dtype=jnp.float_):
            dtype = jax.dtypes.canonicalize_dtype(dtype)
            mval = 1e-3
            return jax.random.uniform(key_, shape, dtype, -mval, mval)

        def make_sac_networks(
                param_size: int,
                obs_size: int,
                action_size: int,
                hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        ) -> Tuple[FeedForwardModel, FeedForwardModel]:
            """Creates a policy and value networks for SAC."""
            policy_module = CustomMLP(
                layer_sizes=hidden_layer_sizes + (param_size,),
                activation=linen.relu,
                kernel_init_hidden_layer=kernel_init_hidden_layer,
                kernel_init_last_layer=init_last_layer,
                bias_init_hidden_layer=bias_init_hidden_layer,
                bias_init_last_layer=init_last_layer
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
                            bias_init_last_layer=init_last_layer
                        )(hidden)
                        res.append(q)
                    return jnp.concatenate(res, axis=-1)

            q_module = QModule()

            dummy_obs = jnp.zeros((1, obs_size))
            dummy_action = jnp.zeros((1, action_size))
            policy = FeedForwardModel(
                init=lambda key_: policy_module.init(key_, dummy_obs),
                apply=policy_module.apply)
            value = FeedForwardModel(
                init=lambda key_: q_module.init(key_, dummy_obs, dummy_action),
                apply=q_module.apply)
            return policy, value

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, params)

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau

        if 'seed' in self.params:
            start_seed = self.params['seed']
        else:
            start_seed = 42

        self.key, local_key, key_models = jax.random.split(
            jax.random.PRNGKey(start_seed), 3)

        self.policy_model, self.value_model = make_sac_networks(
            2 * action_dim, observation_dim, action_dim)

        self.log_alpha = jnp.asarray(0., dtype=jnp.float32)
        self.alpha_optimizer = optax.adam(learning_rate=alpha_lr)
        self.alpha_optimizer_state = self.alpha_optimizer.init(self.log_alpha)

        self.policy_optimizer = optax.adam(learning_rate=1. * policy_lr)
        self.q_optimizer = optax.adam(learning_rate=1. * critic_lr)

        key_policy, key_q = jax.random.split(key_models)
        self.policy_params = self.policy_model.init(key_policy)
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.q_params = self.value_model.init(key_q)
        self.q_optimizer_state = self.q_optimizer.init(self.q_params)

        class NormalDistribution:
            def __init__(self, loc, scale):
                self.loc = loc
                self.scale = scale

            def sample(self, seed):
                return jax.random.normal(
                    seed, shape=self.loc.shape) * self.scale + self.loc

        def create_dist(logits):
            loc, log_scale = jnp.split(logits, 2, axis=-1)
            log_sig_max = 2.
            log_sig_min = -20.
            log_scale_clip = jnp.clip(log_scale, log_sig_min, log_sig_max)
            scale = jnp.exp(log_scale_clip)
            dist = NormalDistribution(loc=loc, scale=scale)
            return dist

        def sample_no_postprocessing(logits, seed_):
            return create_dist(logits).sample(seed=seed_)

        def postprocess(x):
            return jnp.tanh(x)

        def dist_log_prob(logits, pre_tanh_action):
            action = jnp.tanh(pre_tanh_action)
            loc, log_scale = jnp.split(logits, 2, axis=-1)
            log_sig_max = 2.
            log_sig_min = -20.
            epsilon = 1e-6
            log_scale_clip = jnp.clip(log_scale, log_sig_min, log_sig_max)
            scale = jnp.exp(log_scale_clip)
            var = scale ** 2
            normal_log_prob = -((pre_tanh_action - loc) ** 2) / (2 * var) - log_scale \
                              - jnp.log(jnp.sqrt(2 * jnp.pi))
            log_prob = normal_log_prob - jnp.log(1 - action * action + epsilon)
            return jnp.sum(log_prob, axis=-1)

        self.sample_no_postprocessing = sample_no_postprocessing
        self.postprocess = postprocess
        self.dist_log_prob = dist_log_prob
        target_entropy = -1. * action_dim

        def alpha_loss(log_alpha: jnp.ndarray, log_pi: jnp.ndarray) -> jnp.ndarray:
            alpha_l = log_alpha * jax.lax.stop_gradient(-log_pi - target_entropy)
            return jnp.mean(alpha_l)

        def actor_loss(policy_params: Params, q_params: Params, alpha: jnp.ndarray,
                       observations, key_) -> jnp.ndarray:
            dist_params = self.policy_model.apply(policy_params, observations)
            p_actions = self.sample_no_postprocessing(dist_params, key_)
            log_p = self.dist_log_prob(dist_params, p_actions)
            p_actions = self.postprocess(p_actions)
            q_action = self.value_model.apply(q_params, observations, p_actions)
            min_q = jnp.min(q_action, axis=-1)
            actor_l = alpha * log_p - min_q
            return jnp.mean(actor_l)

        def critic_loss(q_params: Params,
                        policy_params: Params,
                        target_q_params: Params,
                        alpha: jnp.ndarray,
                        observations,
                        actions,
                        new_observations,
                        rewards,
                        done,
                        key_) -> jnp.ndarray:
            next_dist_params = self.policy_model.apply(policy_params,
                                                       new_observations)
            next_pre_actions = self.sample_no_postprocessing(next_dist_params,
                                                             key_)
            new_log_pi = self.dist_log_prob(next_dist_params, next_pre_actions)
            new_next_actions = self.postprocess(next_pre_actions)
            q_old_action = self.value_model.apply(q_params, observations, actions)
            next_q = self.value_model.apply(target_q_params, new_observations,
                                            new_next_actions)
            next_v = jnp.min(next_q, axis=-1) - alpha * new_log_pi
            target_q = jax.lax.stop_gradient(
                rewards * self.reward_scale +
                done * discount * jnp.expand_dims(next_v, -1)
            )
            q_error = q_old_action - target_q
            q_loss = 2. * jnp.mean(jnp.square(q_error))
            return q_loss

        self.alpha_grad = jax.value_and_grad(alpha_loss)
        self.critic_grad = jax.value_and_grad(critic_loss)
        self.actor_grad = jax.value_and_grad(actor_loss)

        def update_step(
                state: TrainingState,
                observations,
                actions,
                rewards,
                new_observations,
                done
        ) -> TrainingState:

            key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)

            dist_params = self.policy_model.apply(state.policy_params, observations)
            pre_actions = self.sample_no_postprocessing(dist_params, key_alpha)
            log_pi = self.dist_log_prob(dist_params, pre_actions)

            alpha_l, alpha_grads = self.alpha_grad(state.alpha_params,
                                                   log_pi)
            alpha_params_update, alpha_optimizer_state = self.alpha_optimizer.update(
                alpha_grads, state.alpha_optimizer_state)
            alpha_params = optax.apply_updates(state.alpha_params, alpha_params_update)
            alpha = jnp.exp(alpha_params)

            actor_l, actor_grads = self.actor_grad(state.policy_params,
                                                   state.target_q_params,
                                                   alpha,
                                                   observations,
                                                   key_actor)

            policy_params_update, policy_optimizer_state = self.policy_optimizer.update(
                actor_grads, state.policy_optimizer_state)
            policy_params = optax.apply_updates(state.policy_params,
                                                policy_params_update)

            critic_l, critic_grads = self.critic_grad(state.q_params,
                                                      state.policy_params,
                                                      state.target_q_params,
                                                      alpha,
                                                      observations,
                                                      actions,
                                                      new_observations,
                                                      rewards,
                                                      done,
                                                      key_critic)

            q_params_update, q_optimizer_state = self.q_optimizer.update(
                critic_grads, state.q_optimizer_state)
            q_params = optax.apply_updates(state.q_params, q_params_update)

            new_target_q_params = jax.tree_multimap(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_q_params, q_params)

            new_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                key=key,
                steps=state.steps + 1,
                alpha_optimizer_state=alpha_optimizer_state,
                alpha_params=alpha_params,
            )

            return new_state

        self.update_step = jax.jit(update_step)

        def select_action_probabilistic(observation, policy_params, key_):
            logits = self.policy_model.apply(policy_params, observation)
            actions = self.sample_no_postprocessing(logits, key_)
            return self.postprocess(actions)

        def select_action_deterministic(observation, policy_params, key_=None):
            logits = self.policy_model.apply(policy_params, observation)
            loc, _ = jnp.split(logits, 2, axis=-1)
            return self.postprocess(loc)

        self.select_action_probabilistic = jax.jit(select_action_probabilistic)
        self.select_action_deterministic = jax.jit(select_action_deterministic)

        def q_value(observation, action, q_params):
            q_action = self.value_model.apply(q_params, observation, action)
            min_q = jnp.min(q_action, axis=-1)
            return min_q

        self.q_value = jax.jit(q_value)

        self.training_state = TrainingState(
            policy_optimizer_state=self.policy_optimizer_state,
            policy_params=self.policy_params,
            q_optimizer_state=self.q_optimizer_state,
            q_params=self.q_params,
            target_q_params=self.q_params,
            key=local_key,
            steps=jnp.zeros((1,)),
            alpha_optimizer_state=self.alpha_optimizer_state,
            alpha_params=self.log_alpha
        )

    def value(self, observation, action):
        if torch.is_tensor(observation):
            version = 'torch'
        else:
            version = 'numpy'
        if version == 'numpy':
            return self.q_value(
                observation,
                action,
                self.training_state.target_q_params
                # self.training_state.q_params
            )
        else:
            return self.q_value(
                observation.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                self.training_state.target_q_params
                # self.training_state.q_params
            )

    def select_action(self, observation, deterministic=True):
        self.key, key_sample = jax.random.split(self.key)
        if deterministic:
            apply_func = self.select_action_deterministic
        else:
            apply_func = self.select_action_probabilistic
        if torch.is_tensor(observation):
            version = 'torch'
        else:
            version = 'numpy'
        if version == 'numpy':
            action = apply_func(
                observation,
                self.training_state.policy_params,
                key_sample)
        else:
            action = apply_func(
                observation.detach().cpu().numpy(),
                self.training_state.policy_params,
                key_sample)
        if len(action.shape) == 1:
            return jnp.expand_dims(action, axis=0)
        else:
            return action

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for filename in self.training_state.__dict__.keys():
            with open(os.path.join(directory, filename + '.joblib'), 'wb') as f_:
                joblib.dump(self.training_state.__dict__[filename], f_)

    def load(self, directory):
        load_all = {}
        for filename in self.training_state.__dict__.keys():
            load_all[filename] = jax.tree_util.tree_multimap(
                jnp.array, joblib.load(os.path.join(directory, filename + '.joblib')))
        return TrainingState(
            policy_optimizer_state=load_all['policy_optimizer_state'],
            policy_params=load_all['policy_params'],
            q_optimizer_state=load_all['q_optimizer_state'],
            q_params=load_all['q_params'],
            target_q_params=load_all['target_q_params'],
            key=load_all['key'],
            steps=load_all['steps'],
            alpha_optimizer_state=load_all['alpha_optimizer_state'],
            alpha_params=load_all['alpha_params'],
        )

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)

    def train(self, pre_sample, sampler, batch_size):
        batch = sampler.sample(pre_sample, batch_size)
        self.train_on_batch(batch)

    def train_on_batch(self, batch):
        if torch.is_tensor(batch['r']):
            version = 'torch'
        else:
            version = 'numpy'
        if version == 'numpy':
            observations = jnp.array(batch['obs'])
            actions = jnp.array(batch['actions'])
            rewards = jnp.array(batch['r'])
            new_observations = jnp.array(batch['obs_next'])
            done = jnp.array(1.0 - batch['terminals'])
        else:
            observations = jnp.array(batch['obs'].detach().cpu().numpy())
            actions = jnp.array(batch['actions'].detach().cpu().numpy())
            rewards = jnp.array(batch['r'].detach().cpu().numpy())
            new_observations = jnp.array(batch['obs_next'].detach().cpu().numpy())
            done = jnp.array(1.0 - batch['terminals'].detach().cpu().numpy())

        self.training_state = self.update_step(
            self.training_state,
            observations,
            actions,
            rewards,
            new_observations,
            done
        )
