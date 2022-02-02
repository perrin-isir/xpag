from abc import ABC
from typing import Any, Dict, Mapping, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp
import optax
from xpag.agents.agent import Agent
from IPython import embed

Metrics = Mapping[str, jnp.ndarray]


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
    normalizer_params: Params
    # The is passed to the rewarder to update the reward.
    rewarder_state: Any


def make_sac_networks(
        param_size: int,
        obs_size: int,
        action_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> Tuple[networks.FeedForwardModel, networks.FeedForwardModel]:
    """Creates a policy and a value networks for SAC."""
    policy_module = networks.MLP(
        layer_sizes=hidden_layer_sizes + (param_size,),
        activation=linen.relu,
        kernel_init=jax.nn.initializers.lecun_uniform())

    class QModule(linen.Module):
        """Q Module."""
        n_critics: int = 2

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = networks.MLP(
                    layer_sizes=hidden_layer_sizes + (1,),
                    activation=linen.relu,
                    kernel_init=jax.nn.initializers.lecun_uniform())(
                    hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule()

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    policy = networks.FeedForwardModel(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=policy_module.apply)
    value = networks.FeedForwardModel(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action),
        apply=q_module.apply)
    return policy, value


class SACJAX(Agent, ABC):
    def __init__(
            self,
            observation_dim,
            action_dim,
            device,
            params=None,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=3e-4,
            soft_target_tau=0.005,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, device, params)
        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau

        process_count = jax.process_count()
        process_id = jax.process_index()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count

        seed = 0
        key = jax.random.PRNGKey(seed)
        global_key, local_key = jax.random.split(key)
        del key
        local_key = jax.random.fold_in(local_key, process_id)
        key_models, key_rewarder = jax.random.split(global_key, 2)
        local_key, key_env, key_eval = jax.random.split(local_key, 3)

        self.parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_dim)

        self.policy_model, self.value_model = make_sac_networks(
            self.parametric_action_distribution.param_size, observation_dim, action_dim)

        self.log_alpha = jnp.asarray(0., dtype=jnp.float32)
        self.alpha_optimizer = optax.adam(learning_rate=alpha_lr)
        self.alpha_optimizer_state = self.alpha_optimizer.init(self.log_alpha)

        self.policy_optimizer = optax.adam(learning_rate=policy_lr)
        self.q_optimizer = optax.adam(learning_rate=critic_lr)
        key_policy, key_q = jax.random.split(key_models)
        self.policy_params = self.policy_model.init(key_policy)
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.q_params = self.value_model.init(key_q)
        self.q_optimizer_state = self.q_optimizer.init(self.q_params)

        normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
            normalization.create_observation_normalizer(
                observation_dim,
                normalize_observations=False,
                pmap_to_devices=local_devices_to_use))

        rewarder_state = None
        compute_reward = None

        target_entropy = -1. * action_dim

        def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params,
                       observations, key: PRNGKey) -> jnp.ndarray:
            """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
            dist_params = self.policy_model.apply(policy_params, observations)
            action = self.parametric_action_distribution.sample_no_postprocessing(
                dist_params, key)
            log_prob = self.parametric_action_distribution.log_prob(dist_params, action)
            alpha = jnp.exp(log_alpha)
            alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
            return jnp.mean(alpha_loss)

        def critic_loss(q_params: Params, policy_params: Params,
                        target_q_params: Params, alpha: jnp.ndarray,
                        observations,
                        actions,
                        new_observations,
                        rewards,
                        done,
                        key: PRNGKey) -> jnp.ndarray:
            q_old_action = self.value_model.apply(q_params, observations,
                                                  actions)
            next_dist_params = self.policy_model.apply(policy_params, new_observations)
            next_action = self.parametric_action_distribution.sample_no_postprocessing(
                next_dist_params, key)
            next_log_prob = self.parametric_action_distribution.log_prob(
                next_dist_params, next_action)
            next_action = self.parametric_action_distribution.postprocess(next_action)
            next_q = self.value_model.apply(target_q_params, new_observations,
                                            next_action)
            next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
            target_q = jax.lax.stop_gradient(
                rewards * self.reward_scale + discount * next_v)
            # transitions.d_t * discounting * next_v)
            q_error = q_old_action - jnp.expand_dims(target_q, -1)

            # Better bootstrapping for truncated episodes.
            # q_error *= jnp.expand_dims(1 - transitions.truncation_t, -1)
            q_error *= jnp.expand_dims(1 - done, -1)

            # q_loss = 0.5 * jnp.mean(jnp.square(q_error))
            q_loss = jnp.mean(jnp.square(q_error))
            return q_loss

        def actor_loss(policy_params: Params, q_params: Params, alpha: jnp.ndarray,
                       observations, key: PRNGKey) -> jnp.ndarray:
            dist_params = self.policy_model.apply(policy_params, observations)
            action = self.parametric_action_distribution.sample_no_postprocessing(
                dist_params, key)
            log_prob = self.parametric_action_distribution.log_prob(dist_params, action)
            action = self.parametric_action_distribution.postprocess(action)
            q_action = self.value_model.apply(q_params, observations, action)
            min_q = jnp.min(q_action, axis=-1)
            actor_loss = alpha * log_prob - min_q
            return jnp.mean(actor_loss)

        self.alpha_grad = jax.jit(jax.value_and_grad(alpha_loss))
        self.critic_grad = jax.jit(jax.value_and_grad(critic_loss))
        self.actor_grad = jax.jit(jax.value_and_grad(actor_loss))

        self.key = jax.random.PRNGKey(seed)

        def select_action_probabilistic(observation, policy_params, key_):
            logits = self.policy_model.apply(policy_params, observation)
            actions = self.parametric_action_distribution.sample_no_postprocessing(
                logits, key_)
            return actions

        self.select_action_probabilistic = jax.jit(select_action_probabilistic)

        def update_step(
                state: TrainingState,
                observations,
                actions,
                rewards,
                new_observations,
                done
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            (key, key_alpha, key_critic, key_actor,
             key_rewarder) = jax.random.split(state.key, 5)

            new_rewarder_state = state.rewarder_state
            rewarder_metrics = {}

            alpha_loss, alpha_grads = self.alpha_grad(state.alpha_params,
                                                      state.policy_params,
                                                      observations, key_alpha)
            alpha = jnp.exp(state.alpha_params)
            critic_loss, critic_grads = self.critic_grad(state.q_params,
                                                         state.policy_params,
                                                         state.target_q_params, alpha,
                                                         observations,
                                                         actions,
                                                         new_observations,
                                                         rewards,
                                                         done, key_critic)
            actor_loss, actor_grads = self.actor_grad(state.policy_params,
                                                      state.q_params,
                                                      alpha, observations,
                                                      key_actor)

            policy_params_update, policy_optimizer_state = self.policy_optimizer.update(
                actor_grads, state.policy_optimizer_state)
            policy_params = optax.apply_updates(state.policy_params,
                                                policy_params_update)
            q_params_update, q_optimizer_state = self.q_optimizer.update(
                critic_grads, state.q_optimizer_state)
            q_params = optax.apply_updates(state.q_params, q_params_update)
            alpha_params_update, alpha_optimizer_state = self.alpha_optimizer.update(
                alpha_grads, state.alpha_optimizer_state)
            alpha_params = optax.apply_updates(state.alpha_params, alpha_params_update)
            new_target_q_params = jax.tree_multimap(
                lambda x, y: x * (1 - soft_target_tau) + y * soft_target_tau,
                state.target_q_params, q_params)

            metrics = {
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': jnp.exp(alpha_params),
                **rewarder_metrics
            }

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
                normalizer_params=state.normalizer_params,
                rewarder_state=new_rewarder_state)
            return new_state, metrics

        self.update_step = jax.jit(update_step)

        self.training_state = TrainingState(
            policy_optimizer_state=self.policy_optimizer_state,
            policy_params=self.policy_params,
            q_optimizer_state=self.q_optimizer_state,
            q_params=self.q_params,
            target_q_params=self.q_params,
            # key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
            key=local_key,
            steps=jnp.zeros((local_devices_to_use,)),
            alpha_optimizer_state=self.alpha_optimizer_state,
            alpha_params=self.log_alpha,
            normalizer_params=normalizer_params,
            rewarder_state=rewarder_state)

    def select_action(self, observation, deterministic=True):
        self.key, key_sample = jax.random.split(self.key)
        return self.select_action_probabilistic(observation,
                                                self.training_state.policy_params,
                                                key_sample)

    def save(self, directory):
        pass

    def load(self, directory):
        pass

    def train(self, pre_sample, sampler, batch_size):
        batch = sampler.sample(pre_sample, batch_size)
        self.train_on_batch(batch)

    def train_on_batch(self, batch):
        self.training_state, _ = self.update_step(
            self.training_state,
            jax.numpy.array(batch['obs']),
            jax.numpy.array(batch['actions']),
            jax.numpy.array(batch['r']),
            jax.numpy.array(batch['obs_next']),
            jax.numpy.array(batch['terminals']))
