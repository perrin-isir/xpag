import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from acme.jax import networks as networks_lib
from acme.jax.utils import process_multiple_batches
from acme.types import Transition as AcmeTransition


@dataclass
class ACMESACParameters:
    tau: float = 0.005
    reward_scale: float = 1.0
    discount: float = 0.99
    entropy_coefficient: Optional[float] = None
    target_entropy: float = 0.0

#
# def make_initial_state(
#     networks: sac_networks.SACNetworks,
#     policy_optimizer: optax.GradientTransformation,
#     q_optimizer: optax.GradientTransformation,
#     random_key: networks_lib.PRNGKey,
#     acme_sac_params: ACMESACParameters,
# ) -> TrainingState:
#     """Initialises the training state (parameters and optimiser state)."""
#     adaptive_entropy_coefficient = acme_sac_params.entropy_coefficient is None
#
#     key_policy, key_q, random_key = jax.random.split(random_key, 3)
#
#     policy_params = networks.policy_network.init(key_policy)
#     policy_optimizer_state = policy_optimizer.init(policy_params)
#
#     q_params = networks.q_network.init(key_q)
#     q_optimizer_state = q_optimizer.init(q_params)
#
#     state = TrainingState(
#         policy_optimizer_state=policy_optimizer_state,
#         q_optimizer_state=q_optimizer_state,
#         policy_params=policy_params,
#         q_params=q_params,
#         target_q_params=q_params,
#         key=random_key,
#     )
#
#     if adaptive_entropy_coefficient:
#         # The alpha optimizer is hard-coded deep inside
#         # acme's SAC implementation and there is no way
#         # to access it so we have to hard-code it identically
#         # here
#         alpha_optimizer = optax.adam(learning_rate=3e-4)
#         log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
#         alpha_optimizer_state = alpha_optimizer.init(log_alpha)
#         state = state._replace(
#             alpha_optimizer_state=alpha_optimizer_state, alpha_params=log_alpha
#         )
#     return state


def gen_multi_step_update_step_func(
    networks: sac_networks.SACNetworks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    acme_sac_params: ACMESACParameters,
):

    update_step_func = SACLearner(
        networks=networks,
        rng=jax.random.PRNGKey(0),  # not used
        iterator=[],  # not used
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        tau=acme_sac_params.tau,
        reward_scale=acme_sac_params.reward_scale,
        discount=acme_sac_params.discount,
        entropy_coefficient=acme_sac_params.entropy_coefficient,
        target_entropy=acme_sac_params.target_entropy,
        logger=logging.getLogger(f"{__name__}.ACMESAC"),
        num_sgd_steps_per_step=1,
    )._update_step

    def multi_step_update_step_func(
        state: TrainingState, transitions: AcmeTransition, nb_steps: int
    ):
        return process_multiple_batches(update_step_func, nb_steps)(state, transitions)

    return multi_step_update_step_func


def gen_select_action(networks: sac_networks.SACNetworks):
    def select_action(
        policy_params: networks_lib.Params,
        observation: networks_lib.Observation,
        rng: jax.random.PRNGKey,
        exploration: bool,
    ):
        if isinstance(observation, dict):
            observation = observation["observation"]

        rng1, rng2 = jax.random.split(rng)

        dist_params = networks.policy_network.apply(policy_params, observation)
        if exploration:
            action = dist_params.sample(seed=rng1)
        else:
            action = dist_params.mode()
        return action, rng2

    return select_action


class ACMESAC(DiversityAgent):
    """Wrapper for the jax implementation of SAC from Deepmind's acme."""

    def __init__(
        self,
        networks: sac_networks.SACNetworks,
        policy_optimizer: optax.GradientTransformation,
        q_optimizer: optax.GradientTransformation,
        acme_sac_params: ACMESACParameters,
        initial_training_state: TrainingState,
        actor_random_key,
    ):
        self._logger = logging.getLogger(f"{__name__}.ACMESAC")

        multi_step_update_step_func = gen_multi_step_update_step_func(
            networks,
            policy_optimizer,
            q_optimizer,
            acme_sac_params,
        )

        self._update_step_func = jax.jit(multi_step_update_step_func, static_argnums=2)

        select_action = gen_select_action(networks)

        self._select_action = jax.jit(
            select_action,
            static_argnums=3,
        )

        self._jax_cpu_device = jax.devices("cpu")[0]
        self._jax_params_holder = JaxParamsHolder(params=initial_training_state)
        self._random_key = actor_random_key
        self._discount_vector = None

    def name(self) -> str:
        return "ACME_SAC"

    def select_action(
        self,
        observation: Observation,
        extra: Optional[Extra] = None,
        exploration: bool = False,
    ) -> Tuple[Action, Extra]:
        action, self._random_key = self._select_action(
            self._jax_params_holder.params.policy_params,
            observation,
            self._random_key,
            exploration,
        )

        return (action.block_until_ready(), {})

    def observe(self, step: Step) -> Extra:
        return {}  # Nothing to do here for SAC

    def save(self, directory: str, checkpoint_nb: int = 0) -> Tuple[bool, str]:
        raise NotImplementedError()

    def load(
        self, directory: str, checkpoint_nb: Optional[int] = None
    ) -> Tuple[bool, str]:
        raise NotImplementedError()

    def train_step(self, data_batch: Transition, nb_steps: int):
        if self._discount_vector is None:
            self._discount_vector = jax.device_put(
                jnp.ones(data_batch.reward.shape), data_batch.reward.device()
            )

        data_batch = to_acme_transition(data_batch, self._discount_vector)

        (training_state, all_metrics) = self._update_step_func(
            self._jax_params_holder.params, data_batch, nb_steps
        )

        self._jax_params_holder.params = jax.tree_map(
            lambda x: x.block_until_ready(), training_state
        )
        all_metrics = jax.tree_map(
            lambda x: jax.device_put(x.block_until_ready(), self._jax_cpu_device),
            all_metrics,
        )
        for metric_name, metric_value in all_metrics.items():
            metric = Metric(
                name=metric_name,
                value=int(metric_value)
                if "int" in str(metric_value.dtype)
                else float(metric_value),
                computed_by_learner=True,
                needs_aggregation=True,
                aggregation_key_type=MetricXAxisType.BATCH,
            )
            self._logger.info(f"{metric}")

    def get_shared_params(self, for_actor_only: bool = False) -> SharedParameters:
        return SharedParameters(
            networks=[],
            variables=[],
            jax_parameters=[self._jax_params_holder],
        )