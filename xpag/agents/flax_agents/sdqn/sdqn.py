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
import numpy as np

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


class FlaxSDQN(Agent):
    def __init__(
        self,
        observation_dim,
        action_dim,
        params=None,
    ):
        """
        Jax implementation of SDQN (https://arxiv.org/pdf/1705.05035.pdf).
        """

        discount = 0.99 if "discount" not in params else params["discount"]
        reward_scale = 1.0 if "reward_scale" not in params else params["reward_scale"]
        critic_lr = 3e-3 if "critic_lr" not in params else params["critic_lr"]
        critic_up_lr = (
            critic_lr if "critic_up_lr" not in params else params["critic_up_lr"]
        )
        critic_low_lr = (
            0.1 * critic_lr
            if "critic_low_lr" not in params
            else params["critic_low_lr"]
        )
        soft_target_tau = 5e-2 if "tau" not in params else params["tau"]
        hidden_dims = (
            (256, 256) if "hidden_dims" not in params else params["hidden_dims"]
        )
        start_seed = np.random.randint(1e9) if "seed" not in params else params["seed"]
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
        # action_array = (
        #     jnp.tile(
        #         jnp.arange(
        #             -max_action,
        #             max_action + 1e-9,
        #             2.0 * max_action / (action_bins-1),
        #         ),
        #         (action_dim, 1),
        #     )
        #     if "action_array" not in params
        #     else jnp.array(params["action_array"])
        # )

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
            for _ in range(action_size):
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
        self.critic_up_optimizer = optax.adam(learning_rate=critic_up_lr)
        self.critic_up_optimizer_state = self.critic_up_optimizer.init(
            self.critic_up_params
        )
        list_critic_low_params = []
        for i, qmod in enumerate(self.critic_low):
            key_models, key_q = jax.random.split(key_models)
            qparams = qmod.init(key_q, i)
            list_critic_low_params.append(qparams)

        self.critic_low_params = flax.core.frozen_dict.FrozenDict(
            {str(i): list_critic_low_params[i] for i in range(len(self.critic_low))}
        )
        self.critic_low_optimizer = optax.adam(learning_rate=critic_low_lr)
        self.critic_low_optimizer_state = self.critic_low_optimizer.init(
            self.critic_low_params
        )

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

            # Get current Q estimates
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
            current_q1, current_q2 = jax.lax.stop_gradient(
                self.critic_up.apply(critic_up_params, observations, actions, 1)
            )
            actions_first_part, actions_second_part = jnp.split(
                actions, [(self.action_dim - 1) * self.action_bins], axis=1
            )
            idxs = jnp.argmax(actions_second_part, axis=1).reshape((obs_shape, 1))

            # Current estimates with critics_low
            res1, res2 = self.critic_low[self.action_dim - 1].apply(
                critic_low_params[str(self.action_dim - 1)],
                observations,
                actions_first_part,
                self.action_bins,
            )
            current_q1_low = jnp.take_along_axis(res1, idxs, axis=1)
            current_q2_low = jnp.take_along_axis(res2, idxs, axis=1)
            equality_loss = jnp.mean(
                jnp.square(current_q1 - current_q1_low)
                + jnp.square(current_q2 - current_q2_low)
            )
            return equality_loss

        self.critic_low_equality_loss_grad = jax.value_and_grad(
            critic_low_equality_loss
        )

        def critic_low_inner_loss(
            critic_low_params: Params,
            target_critic_low_params: Params,
            observations,
            actions,
            j: int,
        ) -> jnp.ndarray:
            obs_shape = observations.shape[0]
            actions_first_part, actions_second_part = jnp.split(
                actions, [j * self.action_bins], axis=1
            )
            # Compute target
            res1, res2 = jax.lax.stop_gradient(
                self.critic_low[j].apply(
                    target_critic_low_params[str(j)],
                    observations,
                    actions_first_part,
                    self.action_bins,
                )
            )
            res1_max = jnp.max(res1, axis=1).reshape((obs_shape, 1))
            res2_max = jnp.max(res2, axis=1).reshape((obs_shape, 1))
            target_val = jnp.min(
                jnp.column_stack((res1_max, res2_max)), axis=1
            ).reshape((obs_shape, 1))

            # Compute estimates
            actions_first_part, actions_second_part = jnp.split(
                actions_first_part, [(j - 1) * self.action_bins], axis=1
            )
            idxs = jnp.argmax(actions_second_part, axis=1).reshape((obs_shape, 1))
            res1, res2 = self.critic_low[j - 1].apply(
                critic_low_params[str(j - 1)],
                observations,
                actions_first_part,
                self.action_bins,
            )
            current_q1_low = jnp.take_along_axis(res1, idxs, axis=1)
            current_q2_low = jnp.take_along_axis(res2, idxs, axis=1)
            inner_loss = jnp.mean(
                jnp.square(target_val - current_q1_low)
                + jnp.square(target_val - current_q2_low)
            )
            return inner_loss

        def critic_low_inner_loss_grad(j):
            return jax.value_and_grad(
                lambda clp, tclp, o, a: critic_low_inner_loss(clp, tclp, o, a, j)
            )

        self.critic_low_inner_loss_grad = critic_low_inner_loss_grad

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

            for k in range(action_dim - 1):
                j = action_dim - 1 - k

                (
                    critic_low_inner_l,
                    critic_low_inner_grads,
                ) = self.critic_low_inner_loss_grad(j)(
                    critic_low_params,
                    state.target_critic_low_params,
                    observations,
                    actions,
                )

                (
                    critic_low_params_update,
                    critic_low_optimizer_state,
                ) = self.critic_low_optimizer.update(
                    critic_low_inner_grads, critic_low_optimizer_state
                )

                critic_low_params = optax.apply_updates(
                    critic_low_params, critic_low_params_update
                )

            target_critic_low_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_critic_low_params,
                critic_low_params,
            )

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

            # metrics = {"critic_up_loss": critic_up_l}
            metrics = {}
            return new_state, metrics

        # self.update_step = update_step
        self.update_step = jax.jit(
            update_step,
            static_argnames=["action_bins", "action_dim"],
            backend=self.backend,
        )

    def select_action(self, observation, eval_mode=False):
        onehot_action = self.greedy_actions(
            self.action_bins,
            self.action_dim,
            self.training_state.critic_low_params,
            observation,
        )
        where_ones = jnp.where(onehot_action == 1)[1].reshape(
            (observation.shape[0], self.action_dim)
        )
        act = jnp.take(self.action_array, where_ones)
        return act, {"onehot_action": onehot_action}

    def save(self, directory):
        pass

    def load(self, directory):
        pass

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


class FlaxSDQNSetter(Setter):
    def __init__(self, sdqn_agent: FlaxSDQN):
        super().__init__("SDQNSetter")
        self.agent = sdqn_agent

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
        if "onehot_action" not in action_info:
            action_shape = action.shape[0]
            action_compare_bins = jnp.abs(
                jnp.repeat(action, self.agent.action_bins, axis=1)
                - self.agent.action_array.flatten()
            ).reshape((action_shape, self.agent.action_dim, self.agent.action_bins))
            action_bin_choices = jnp.argmin(action_compare_bins, axis=-1)
            onehot_action = (
                jnp.zeros(
                    (action_shape, self.agent.action_bins * self.agent.action_dim)
                )
                .at[
                    jnp.repeat(
                        jnp.arange(action_shape).reshape((action_shape, 1)),
                        self.agent.action_dim,
                        axis=1,
                    ),
                    action_bin_choices
                    + jnp.arange(self.agent.action_dim) * self.agent.action_bins,
                ]
                .set(1)
            )
        else:
            onehot_action = action_info["onehot_action"]
        return (
            observation,
            onehot_action,
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
