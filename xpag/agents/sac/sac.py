# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import os
from xpag.agents.agent import Agent
from xpag.agents.sac.sac_from_jaxrl import Batch, SACLearner
from xpag.tools.utils import squeeze
import functools
from typing import Callable, Any, Tuple
import flax
import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames="critic_apply_fn")
def _qvalue(
    critic_apply_fn: Callable[..., Any],
    critic_params: flax.core.FrozenDict[str, Any],
    observations: jnp.ndarray,
    actions: jnp.ndarray,
) -> Tuple[jnp.ndarray]:
    return jnp.minimum(
        *critic_apply_fn({"params": critic_params}, observations, actions)
    )


class SAC(Agent):
    """
    Interface to the SAC agent from JAXRL (https://github.com/ikostrikov/jaxrl)

    Methods:
    - :meth:`value` - computes Q-values given a batch of observations and a batch of
        actions.
    - :meth:`select_action` - selects actions given a batch of observations ; there are
        two modes: one that includes stochasticity for exploration (eval_mode==False),
        and one that deterministically returns the best possible action
        (eval_mode==True).
    - :meth:`train_on_batch` - trains the agent on a batch of transitions (one gradient
        step).
    - :meth:`save` - saves the agent to the disk.
    - :meth:`load` - loads a saved agent.
    - :meth:`write_config` - writes the configuration of the agent (mainly its
        non-default parameters) in a file.

    Attributes:
    - :attr:`_config_string` - the configuration of the agent (mainly its non-default
        parameters)
    - :attr:`saclearner_params` - the SAC parameters in a dict :
        "actor_lr" (default=3e-3): the actor learning rate
        "critic_lr" (default=3e-3): the critic learning rate
        "temp_lr" (default=3e-3): the temperature learning rate
        "backup_entropy" (default=True): if True, activates the entropy-regularization
            of the critic loss
        "discount" (default=0.99): the discount factor
        "hidden_dims" (default=(256,256)): the hidden layer dimensions for the actor
            and critic networks
        "init_temperature" (default=1.): the initial temperature
        "target_entropy": the target entropy; if None, it will be set
            to -action_dim / 2
        "target_update_period" (default=1): defines how often a soft update of the
            target critic is performed
        "tau" (default=5e-2): the soft update coefficient
        "policy_final_fc_init_scale" (default=1.): scale parameter for the
            initialization of the final fully connected layers of the actor network
    - :attr:`sac` - the SACLearner object that contains and trains the agent and critic
        networks
    """

    def __init__(self, observation_dim, action_dim, params=None):

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, params)

        start_seed = 0 if "seed" not in params else params["seed"]

        self.saclearner_params = {
            "actor_lr": 3e-3,
            "critic_lr": 3e-3,
            "temp_lr": 3e-3,
            "backup_entropy": True,
            "discount": 0.99,
            "hidden_dims": (256, 256),
            "init_temperature": 1.0,
            "init_mean": None,
            "target_entropy": None,
            "target_update_period": 1,
            "tau": 5e-2,
            "policy_final_fc_init_scale": 1.0,
        }

        for key in self.saclearner_params:
            if key in self.params:
                self.saclearner_params[key] = self.params[key]

        self.sac = SACLearner(
            start_seed,
            jnp.zeros((1, 1, observation_dim)),
            jnp.zeros((1, 1, action_dim)),
            **self.saclearner_params
        )

    def value(self, observation, action):
        return jnp.asarray(
            _qvalue(
                self.sac.critic.apply_fn, self.sac.critic.params, observation, action
            )
        )

    def select_action(self, observation, eval_mode=False):
        return self.sac.sample_actions(
            observation, distribution="det" if eval_mode else "log_prob"
        )

    def train_on_batch(self, batch):
        saclearner_batch = Batch(
            observations=batch["observation"],
            actions=batch["action"],
            rewards=squeeze(batch["reward"]),
            masks=squeeze(1 - batch["terminated"]),
            next_observations=batch["next_observation"],
        )

        return self.sac.update(saclearner_batch)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        jnp.save(os.path.join(directory, "step.npy"), self.sac.step)
        self.sac.actor.save(os.path.join(directory, "actor"))
        self.sac.critic.save(os.path.join(directory, "critic"))
        self.sac.target_critic.save(os.path.join(directory, "target_critic"))
        self.sac.temp.save(os.path.join(directory, "temp"))

    def load(self, directory):
        self.sac.step = jnp.load(os.path.join(directory, "step.npy")).item()
        self.sac.actor = self.sac.actor.load(os.path.join(directory, "actor"))
        self.sac.critic = self.sac.critic.load(os.path.join(directory, "critic"))
        self.sac.target_critic = self.sac.target_critic.load(
            os.path.join(directory, "target_critic")
        )
        self.sac.temp = self.sac.temp.load(os.path.join(directory, "temp"))

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)
