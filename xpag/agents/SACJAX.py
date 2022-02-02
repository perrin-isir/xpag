from abc import ABC
from typing import Any, Dict, Mapping, Tuple, Sequence, Callable

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
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Distribution, Normal

from IPython import embed

Metrics = Mapping[str, jnp.ndarray]

########################################################################################
def fanin_init(tensor: torch.Tensor) -> torch.Tensor:
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_hidden_layer(layer: torch.nn.Module, b_init_value: float = 0.1):
    torch.manual_seed(0)
    fanin_init(layer.weight)
    layer.bias.data.fill_(b_init_value)


def initialize_last_layer(layer: torch.nn.Module, init_w: float = 1e-3):
    torch.manual_seed(0)
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

class TanhNormal(Distribution, ABC):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean: torch.Tensor, normal_std: torch.Tensor,
                 epsilon: float = 1e-6, device: str = 'cpu'):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        super().__init__(validate_args=False)
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.device = device

    def sample_n(self, n: int, return_pre_tanh_value: bool = False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value: torch.Tensor,
                 pre_tanh_value: bool = None) -> torch.Tensor:
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value: bool = False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value: bool = False):
        """
        Sampling in the reparameterization case.
        """
        z = (
                self.normal_mean
                + self.normal_std
                * Normal(
            torch.zeros(self.normal_mean.size()).to(self.device),
            torch.ones(self.normal_std.size()).to(self.device),
        ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, device: str = 'cpu'):
        super().__init__()

        # hidden layers definition
        self.l1 = nn.Linear(observation_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_mean = nn.Linear(256, action_dim)

        # std layer definition
        self.l3_log_std = nn.Linear(256, action_dim)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3_mean)
        initialize_last_layer(self.l3_log_std)

        self.device = device
        # self.max_action = torch.tensor(max_action, device=self.device)

    def forward(self, x: torch.Tensor, deterministic: bool = False,
                return_log_prob: bool = False):
        # forward pass
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean, log_std = self.l3_mean(x), self.l3_log_std(x)

        # compute std
        log_sig_max = 2
        log_sig_min = -20
        log_std = torch.clamp(log_std, log_sig_min, log_sig_max)
        std = torch.exp(log_std)

        # compute other relevant quantities
        log_prob, entropy, mean_action_log_prob, pre_tanh_value = None, None, None, None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std, device=self.device)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.rsample()

        # action = action * self.max_action
        return (
            action,
            mean,
            log_std,
            log_prob,
            entropy,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(observation_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(observation_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3)

        initialize_hidden_layer(self.l4)
        initialize_hidden_layer(self.l5)
        initialize_last_layer(self.l6)

        # self.max_action = torch.tensor(max_action, device=self.device)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        xu = torch.cat([x, u], 1)
        xu = xu.float()

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x: torch.Tensor, u: torch.Tensor):
        xu = torch.cat([x, u], 1)
        xu = xu.float()

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class LogAlpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(1, requires_grad=True))


cnt_weight = -1
cnt_bias = -1

#######################################################################################


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
    # normalizer_params: Params
    # The is passed to the rewarder to update the reward.
    # rewarder_state: Any


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

        ################################################################################
        self.device = 'cuda'
        self.actor = Actor(observation_dim, action_dim, self.device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)

        self.critic = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        class CustomMLP(linen.Module):
            """MLP module."""
            layer_sizes: Sequence[int]
            activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
            kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
            bias_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
            activate_final: bool = False
            bias: bool = True

            @linen.compact
            def __call__(self, data: jnp.ndarray):
                hidden = data
                for i, hidden_size in enumerate(self.layer_sizes):
                    hidden = linen.Dense(
                        hidden_size,
                        name=f'hidden_{i}',
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        use_bias=self.bias)(
                        hidden)
                    if i != len(self.layer_sizes) - 1 or self.activate_final:
                        hidden = self.activation(hidden)
                return hidden

        weight_init_list = [
            torch.transpose(self.actor.l1.weight, 0, 1),
            torch.transpose(self.actor.l2.weight, 0, 1),
            torch.hstack((torch.transpose(self.actor.l3_mean.weight, 0, 1),
                          torch.transpose(self.actor.l3_log_std.weight, 0, 1))),
            torch.transpose(self.critic.l1.weight, 0, 1),
            torch.transpose(self.critic.l2.weight, 0, 1),
            torch.transpose(self.critic.l3.weight, 0, 1),
            torch.transpose(self.critic_target.l1.weight, 0, 1),
            torch.transpose(self.critic_target.l2.weight, 0, 1),
            torch.transpose(self.critic_target.l3.weight, 0, 1)
        ]

        bias_init_list = [
            self.actor.l1.bias,
            self.actor.l2.bias,
            torch.hstack((self.actor.l3_mean.bias, self.actor.l3_log_std.bias)),
            self.critic.l1.bias,
            self.critic.l2.bias,
            self.critic.l3.bias,
            self.critic_target.l1.bias,
            self.critic_target.l2.bias,
            self.critic_target.l3.bias,
        ]

        def custom_weight_init(key, shape, dtype=jnp.float_):
            global cnt_weight
            print("weight", shape)
            cnt_weight += 1
            # return jax.random.uniform(key, shape, dtype, -1)
            return jax.numpy.array(weight_init_list[cnt_weight].detach().cpu().numpy())

        def custom_bias_init(key, shape, dtype=jnp.float_):
            global cnt_bias
            print("bias", shape)
            cnt_bias += 1
            return jax.numpy.array(bias_init_list[cnt_bias].detach().cpu().numpy())

        ################################################################################

        def make_sac_networks(
                param_size: int,
                obs_size: int,
                action_size: int,
                hidden_layer_sizes: Tuple[int, ...] = (256, 256),
        ) -> Tuple[networks.FeedForwardModel, networks.FeedForwardModel]:
            """Creates a policy and a value networks for SAC."""
            # policy_module = networks.MLP(
            policy_module = CustomMLP(
                layer_sizes=hidden_layer_sizes + (param_size,),
                activation=linen.relu,
                # kernel_init=jax.nn.initializers.lecun_uniform()
                kernel_init=custom_weight_init,
                bias_init=custom_bias_init
            )

            class QModule(linen.Module):
                """Q Module."""
                n_critics: int = 2

                @linen.compact
                def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
                    hidden = jnp.concatenate([obs, actions], axis=-1)
                    res = []
                    for _ in range(self.n_critics):
                        # q = networks.MLP(
                        q = CustomMLP(
                            layer_sizes=hidden_layer_sizes + (1,),
                            activation=linen.relu,
                            # kernel_init=jax.nn.initializers.lecun_uniform()
                            kernel_init=custom_weight_init,
                            bias_init=custom_bias_init
                        )(hidden)
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

        # normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
        #     normalization.create_observation_normalizer(
        #         observation_dim,
        #         normalize_observations=False,
        #         pmap_to_devices=local_devices_to_use))

        # rewarder_state = None
        # compute_reward = None

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

            # new_rewarder_state = state.rewarder_state
            # rewarder_metrics = {}

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
                # **rewarder_metrics
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
                # normalizer_params=state.normalizer_params,
                # rewarder_state=new_rewarder_state
            )
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
            # normalizer_params=normalizer_params,
            # rewarder_state=rewarder_state
        )

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
