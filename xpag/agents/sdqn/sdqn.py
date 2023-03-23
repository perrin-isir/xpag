# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

from typing import Any, Tuple, Sequence, Callable
import dataclasses
import flax
from flax import linen
import jax
import jax.numpy as jnp
import optax
from xpag.agents.agent import Agent
from xpag.setters.setter import Setter

# import os
# import joblib

Params = Any
PRNGKey = jnp.ndarray


@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    critic_up_optimizer_state: optax.OptState
    critic_up_params: Params
    target_critic_up_params: Params
    critic_low_optimizer_state: optax.OptState
    critic_low_params: Params
    target_critic_low_params: Params
    key: PRNGKey
    steps: jnp.ndarray


class SDQNSetter(Setter):
    def __init__(self):
        super().__init__("SDQNSetter")

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):
        return (
            observation,
            action_info["onehot_action"],
            new_observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class SDQN(Agent):
    def __init__(
        self,
        observation_dim,
        action_dim,
        params=None,
    ):
        """
        Jax implementation of SDQN ().
        """

        discount = 0.99 if "discount" not in params else params["discount"]
        reward_scale = 1.0 if "reward_scale" not in params else params["reward_scale"]
        actor_lr = 3e-3 if "actor_lr" not in params else params["actor_lr"]
        critic_lr = 3e-3 if "critic_lr" not in params else params["critic_lr"]
        soft_target_tau = 5e-2 if "tau" not in params else params["tau"]
        hidden_dims = (
            (256, 256) if "hidden_dims" not in params else params["hidden_dims"]
        )
        start_seed = 0 if "seed" not in params else params["seed"]
        action_bins = 5 if "action_bins" not in params else params["action_bins"]
        # By default, actions are assumed to be between -1 and 1 across all
        # dimensions
        max_action = 1.0
        action_array = (
            jnp.tile(
                jnp.arange(
                    -max_action + max_action / action_bins,
                    max_action,
                    2.0 * max_action / action_bins,
                ),
                (action_dim, 1),
            )
            if "action_array" not in params
            else jnp.array(params["action_array"])
        )
        # cpu, gpu or tpu backend
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

        def make_sdqn_networks(
            obs_size: int,
            action_size: int,
            action_bins: int,
            hidden_layer_sizes: Tuple[int, ...],
        ) -> Tuple[FeedForwardModel, Sequence[FeedForwardModel]]:
            """
            Create Q-value networks for SDQN.
            """

            class QModule(linen.Module):
                """Q Module."""

                n_critics: int = 2

                @linen.compact
                def __call__(
                    self, obs: jnp.ndarray, actions: jnp.ndarray, output_size: int
                ):
                    hidden = jnp.concatenate([obs, actions], axis=-1)
                    res = []
                    for _ in range(self.n_critics):
                        q = CustomMLP(
                            layer_sizes=hidden_layer_sizes + (output_size,),
                            activation=linen.relu,
                            kernel_init_hidden_layer=kernel_init_hidden_layer,
                            kernel_init_last_layer=init_last_layer,
                            bias_init_hidden_layer=bias_init_hidden_layer,
                            bias_init_last_layer=init_last_layer,
                        )(hidden)
                        res.append(q)
                    # return jnp.concatenate(res, axis=-1)
                    return res

            critic_up = FeedForwardModel(
                init=lambda key_: QModule().init(
                    key_,
                    jnp.zeros((1, obs_size)),
                    jnp.zeros((1, action_size * action_bins)),
                    1,
                ),
                apply=QModule().apply,
            )

            critics_low = []
            for i in range(action_size):
                critics_low.append(
                    FeedForwardModel(
                        init=lambda key_, i_: QModule().init(
                            key_,
                            jnp.zeros((1, obs_size)),
                            jnp.zeros((1, i_ * action_bins)),
                            action_bins,
                        ),
                        apply=QModule().apply,
                    )
                )

            return critic_up, critics_low

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SDQN", observation_dim, action_dim, params)

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.action_bins = action_bins
        self.action_array = action_array

        self.key, local_key, key_models = jax.random.split(
            jax.random.PRNGKey(start_seed), 3
        )

        key_models, key_q = jax.random.split(key_models)
        self.critic_up, self.critic_low = make_sdqn_networks(
            observation_dim, action_dim, self.action_bins, hidden_dims
        )
        self.critic_up_params = self.critic_up.init(key_q)
        self.critic_up_optimizer = optax.adam(learning_rate=1.0 * critic_lr)
        self.critic_up_optimizer_state = self.critic_up_optimizer.init(
            self.critic_up_params
        )
        list_critic_low_params = []
        # self.critic_low_optimizers = []
        # self.critic_low_optimizer_states = []
        for i, qmod in enumerate(self.critic_low):
            key_models, key_q = jax.random.split(key_models)
            qparams = qmod.init(key_q, i)
            list_critic_low_params.append(qparams)
            # optimizer = optax.adam(learning_rate=1.0 * critic_lr)
            # self.critic_low_optimizers.append(optimizer)
            # self.critic_low_optimizer_states.append(optimizer.init(qparams))

        self.critic_low_params = flax.core.frozen_dict.FrozenDict(
            {str(i): list_critic_low_params[i] for i in range(len(self.critic_low))}
        )
        self.critic_low_optimizer = optax.adam(learning_rate=1.0 * critic_lr)
        self.critic_low_optimizer_state = self.critic_low_optimizer.init(
            self.critic_low_params
        )

        # self.critic_low_params = tuple(self.critic_low_params)
        # self.critic_low_optimizers = tuple(self.critic_low_optimizers)
        # self.critic_low_optimizer_states = tuple(self.critic_low_optimizer_states)

        self.training_state = TrainingState(
            critic_up_optimizer_state=self.critic_up_optimizer_state,
            critic_up_params=self.critic_up_params,
            target_critic_up_params=self.critic_up_params,
            critic_low_optimizer_state=self.critic_low_optimizer_state,
            critic_low_params=self.critic_low_params,
            target_critic_low_params=self.critic_low_params,
            key=local_key,
            steps=jnp.zeros((1,)),
        )

        self.dummy_obs = jnp.zeros((2, self.observation_dim))
        self.dummy_action = jnp.zeros((2, self.action_dim * self.action_bins))

        def greedy_actions(
            action_bins: int, action_dim: int, critic_low_params: Params, obs
        ):
            obs_shape = obs.shape[0]
            carry = jnp.zeros((obs_shape, 0 * action_bins))

            def action_progress(carry_action, i_):
                res1, res2 = self.critic_low[i_].apply(
                    critic_low_params[str(i_)], obs, carry_action, action_bins
                )
                action_choice_array = jnp.argmax(res1, axis=-1)
                a_progress = (
                    jnp.zeros((obs_shape, action_bins))
                    .at[jnp.arange(obs_shape), action_choice_array]
                    .set(1)
                )
                return jnp.column_stack((carry_action, a_progress))

            for k in range(action_dim):
                carry = action_progress(carry, k)

            return carry

        self.greedy_actions = jax.jit(
            greedy_actions,
            static_argnames=["action_bins", "action_dim"],
            backend=self.backend,
        )

        def critic_up_loss(
            critic_up_params: Params,
            target_critic_up_params: Params,
            critic_low_params: Params,
            observations,
            actions,
            new_observations,
            rewards,
            mask,
        ) -> jnp.ndarray:
            next_actions = self.greedy_actions(
                self.action_bins, self.action_dim, critic_low_params, new_observations
            )

            # Compute the target Q value
            target_q1, target_q2 = self.critic_up.apply(
                target_critic_up_params, new_observations, next_actions, 1
            )
            next_v = jnp.min(jnp.array([target_q1, target_q2]), axis=0)
            target_q = jax.lax.stop_gradient(
                rewards * self.reward_scale + mask * self.discount * next_v
            )
            current_q1, current_q2 = self.critic_up.apply(
                critic_up_params, observations, actions, 1
            )
            c_up_loss = jnp.mean(
                jnp.square(current_q1 - target_q) + jnp.square(current_q2 - target_q)
            )
            return c_up_loss

        self.critic_up_grad = jax.value_and_grad(critic_up_loss)

        def critic_low_equality_loss(
            critic_low_params: Params,
            critic_up_params: Params,
            observations,
            actions,
        ) -> jnp.ndarray:
            obs_shape = observations.shape[0]
            current_q1, current_q2 = self.critic_up.apply(
                critic_up_params, observations, actions, 1
            )
            actions_first_part, actions_second_part = jnp.split(
                actions, [(self.action_dim - 1) * self.action_bins], axis=1
            )
            idxs = jnp.argmax(actions_second_part, axis=1).reshape((obs_shape, 1))
            res1, res2 = self.critic_low[self.action_dim - 1].apply(
                critic_low_params[str(self.action_dim - 1)],
                observations,
                actions_first_part,
                self.action_bins,
            )
            current_q1_low = jnp.take(res1, idxs)
            current_q2_low = jnp.take(res2, idxs)
            equality_loss = jnp.mean(
                jnp.square(current_q1 - current_q1_low)
                + jnp.square(current_q2 - current_q2_low)
            )
            return equality_loss

        self.critic_low_equality_loss_grad = jax.value_and_grad(
            critic_low_equality_loss
        )
        #
        # self.critic_up = jax.jit(critic_up_loss, static_argnames=["action_bins"])

        # carry = action_progress(carry_action_init, 0)
        # result = action_progress(carry, 1)
        # final, result = jax.lax.scan(action_progress, carry_action_init, indices)
        # return final, result

        #
        #     nr_obs = observation.size()[0]
        #     chosen_actions = torch.zeros([nr_obs, 0])
        #     for j in range(self.action_dim):
        #         value = self.critics_low_target.critics[j](observation,
        #         chosen_actions)[
        #             0].detach()
        #         i = torch.argmax(value, dim=1).reshape([nr_obs, 1])
        #         ones = torch.ones([nr_obs, 1])
        #         actions = torch.zeros([nr_obs, self.bins_per_action]).scatter_(1, i,
        #                                                                        ones)
        #         chosen_actions = torch.cat([chosen_actions, actions], 1)
        #     return chosen_actions

        # def critic_up_loss(
        #     q_up_params: Params,
        #     target_q_up_params: Params,
        #     observations,
        #     actions,
        #     new_observations,
        #     rewards,
        #     mask,
        #     key_,
        # ) -> jnp.ndarray:
        #     next_pre_actions = self.policy_model.apply(
        #         target_policy_params, new_observations
        #     )
        #     new_next_actions = self.postprocess(next_pre_actions)
        #     policy_noise = 0.2
        #     noise_clip = 0.5
        #     new_next_actions = jnp.clip(
        #         new_next_actions
        #         + jnp.clip(
        #             policy_noise
        #             * jax.random.normal(key_, shape=new_next_actions.shape),
        #             -noise_clip,
        #             noise_clip,
        #         ),
        #         -1.0,
        #         1.0,
        #     )
        #
        #     q_old_action = self.value_model.apply(q_params, observations, actions)
        #     next_q = self.value_model.apply(
        #         target_q_params, new_observations, new_next_actions
        #     )
        #     next_v = jnp.min(next_q, axis=-1)
        #     target_q = jax.lax.stop_gradient(
        #         rewards * self.reward_scale
        #         + mask * discount * jnp.expand_dims(next_v, -1)
        #     )
        #     q_error = q_old_action - target_q
        #     q_loss = 2.0 * jnp.mean(jnp.square(q_error))
        #     return q_loss
        #
        # self.critic_grad = jax.value_and_grad(critic_loss)
        # self.actor_grad = jax.value_and_grad(actor_loss)
        #

        def update_step(
            action_bins: int,
            action_dim: int,
            state: TrainingState,
            observations,
            actions,
            rewards,
            new_observations,
            mask,
        ) -> Tuple[TrainingState, dict]:

            key, key_critic = jax.random.split(state.key, 2)

            critic_up_l, critic_up_grads = self.critic_up_grad(
                state.critic_up_params,
                state.target_critic_up_params,
                state.critic_low_params,
                observations,
                actions,
                new_observations,
                rewards,
                mask,
            )

            (
                critic_up_params_update,
                critic_up_optimizer_state,
            ) = self.critic_up_optimizer.update(
                critic_up_grads, state.critic_up_optimizer_state
            )

            critic_up_params = optax.apply_updates(
                state.critic_up_params, critic_up_params_update
            )

            target_critic_up_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_critic_up_params,
                critic_up_params,
            )

            (
                critic_low_equality_l,
                critic_low_equality_grads,
            ) = self.critic_low_equality_loss_grad(
                state.critic_low_params,
                critic_up_params,
                observations,
                actions,
            )

            (
                critic_low_params_update,
                critic_low_optimizer_state,
            ) = self.critic_low_optimizer.update(
                critic_low_equality_grads, state.critic_low_optimizer_state
            )

            critic_low_params = optax.apply_updates(
                state.critic_low_params, critic_low_params_update
            )

            target_critic_low_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_critic_low_params,
                critic_low_params,
            )

            # for k in range(action_dim):
            #     j = action_dim - 1 - k

            new_state = TrainingState(
                critic_up_optimizer_state=critic_up_optimizer_state,
                critic_up_params=critic_up_params,
                target_critic_up_params=target_critic_up_params,
                critic_low_optimizer_state=critic_low_optimizer_state,
                critic_low_params=critic_low_params,
                target_critic_low_params=target_critic_low_params,
                key=key,
                steps=state.steps + 1,
            )

            metrics = {}
            return new_state, metrics

        self.update_step = jax.jit(
            update_step,
            static_argnames=["action_bins", "action_dim"],
            backend=self.backend,
        )
        # self.update_step = update_step

        # def update_step(
        #     state:
        #     TrainingState, observations, actions, rewards, new_observations, mask
        # ) -> Tuple[TrainingState, dict]:
        #
        #     key, key_critic = jax.random.split(state.key, 2)
        #
        #     actor_l, actor_grads = self.actor_grad(
        #         state.policy_params, state.target_q_params, observations
        #     )
        #
        #     policy_params_update, policy_optimizer_state =
        #     self.policy_optimizer.update(
        #         actor_grads, state.policy_optimizer_state
        #     )
        #
        #     policy_params = optax.apply_updates(
        #         state.policy_params, policy_params_update
        #     )
        #
        #     critic_l, critic_grads = self.critic_grad(
        #         state.q_params,
        #         state.target_policy_params,
        #         state.target_q_params,
        #         observations,
        #         actions,
        #         new_observations,
        #         rewards,
        #         mask,
        #         key_critic,
        #     )
        #
        #     q_params_update, q_optimizer_state = self.q_optimizer.update(
        #         critic_grads, state.q_optimizer_state
        #     )
        #     q_params = optax.apply_updates(state.q_params, q_params_update)
        #
        #     new_target_q_params = jax.tree_util.tree_map(
        #         lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
        #         state.target_q_params,
        #         q_params,
        #     )
        #
        #     new_target_policy_params = jax.tree_util.tree_map(
        #         lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
        #         state.target_policy_params,
        #         policy_params,
        #     )
        #
        #     new_state = TrainingState(
        #         policy_optimizer_state=policy_optimizer_state,
        #         policy_params=policy_params,
        #         target_policy_params=new_target_policy_params,
        #         q_optimizer_state=q_optimizer_state,
        #         q_params=q_params,
        #         target_q_params=new_target_q_params,
        #         key=key,
        #         steps=state.steps + 1,
        #     )
        #
        #     metrics = {}
        #
        #     return new_state, metrics
        #
        # self.update_step = jax.jit(update_step, backend=self.backend)
        #
        # def select_action_probabilistic(observation, policy_params, key_):
        #     pre_action = self.policy_model.apply(policy_params, observation)
        #     pre_action = self.postprocess(pre_action)
        #     expl_noise = 0.1
        #     return jnp.clip(
        #         pre_action
        #         + expl_noise * jax.random.normal(key_, shape=pre_action.shape),
        #         -1.0,
        #         1.0,
        #     )
        #
        # def select_action_deterministic(observation, policy_params, key_=None):
        #     pre_action = self.policy_model.apply(policy_params, observation)
        #     return self.postprocess(pre_action)
        #
        # self.select_action_probabilistic = jax.jit(
        #     select_action_probabilistic, backend=self.backend
        # )
        # self.select_action_deterministic = jax.jit(
        #     select_action_deterministic, backend=self.backend
        # )
        #
        # def q_value(observation, action, q_params):
        #     q_action = self.value_model.apply(q_params, observation, action)
        #     min_q = jnp.min(q_action, axis=-1)
        #     return min_q
        #
        # self.q_value = jax.jit(q_value, backend=self.backend)
        #
        # self.training_state = TrainingState(
        #     policy_optimizer_state=self.policy_optimizer_state,
        #     policy_params=self.policy_params,
        #     target_policy_params=self.policy_params,
        #     q_optimizer_state=self.q_optimizer_state,
        #     q_params=self.q_params,
        #     target_q_params=self.q_params,
        #     key=local_key,
        #     steps=jnp.zeros((1,)),
        # )

    # def value(self, observation, action):
    #     return self.q_value(
    #         observation,
    #         action,
    #         self.training_state.q_params,
    #     )

    def select_action(self, observation, eval_mode=False):
        onehot_action = self.greedy_actions(
            self.action_bins, self.action_dim, self.critic_low_params, observation
        )
        where_ones = jnp.where(onehot_action == 1)[1].reshape(
            (observation.shape[0], self.action_dim)
        )
        act = jnp.take(self.action_array, where_ones)
        return act, {"onehot_action": onehot_action}

    #     self.key, key_sample = jax.random.split(self.key)
    #     if eval_mode:
    #         apply_func = self.select_action_deterministic
    #     else:
    #         apply_func = self.select_action_probabilistic
    #     action = apply_func(observation,
    #     self.training_state.policy_params, key_sample)
    #     if len(action.shape) == 1:
    #         return jnp.expand_dims(action, axis=0)
    #     else:
    #         return action

    def save(self, directory):
        pass

    #     os.makedirs(directory, exist_ok=True)
    #     for filename in self.training_state.__dict__.keys():
    #         with open(os.path.join(directory, filename + ".joblib"), "wb") as f_:
    #             joblib.dump(self.training_state.__dict__[filename], f_)
    #
    def load(self, directory):
        pass

    #     load_all = {}
    #     for filename in self.training_state.__dict__.keys():
    #         load_all[filename] = jax.tree_util.tree_map(
    #             jnp.array, joblib.load(os.path.join(directory, filename + ".joblib"))
    #         )
    #     self.training_state = TrainingState(
    #         policy_optimizer_state=load_all["policy_optimizer_state"],
    #         policy_params=load_all["policy_params"],
    #         target_policy_params=load_all["target_policy_params"],
    #         q_optimizer_state=load_all["q_optimizer_state"],
    #         q_params=load_all["q_params"],
    #         target_q_params=load_all["target_q_params"],
    #         key=load_all["key"],
    #         steps=load_all["steps"],
    #     )

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)

    def train_on_batch(self, batch):
        observations = batch["observation"]
        actions = batch["action"]
        rewards = batch["reward"]
        new_observations = batch["next_observation"]
        mask = 1 - batch["terminated"]

        self.training_state, metrics = self.update_step(
            self.action_bins,
            self.action_dim,
            self.training_state,
            observations,
            actions,
            rewards,
            new_observations,
            mask,
        )
        return metrics
