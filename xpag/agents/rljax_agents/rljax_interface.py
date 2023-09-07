# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

from xpag.agents.agent import Agent
from xpag.agents.rljax_agents.algorithm import SAC
import numpy as np


class DummyBuffer:
    def __init__(self):
        self.next_batch = None

    def set_next_batch(self, batch):
        self.next_batch = batch

    def sample(self, batch_size):
        return 1.0, self.next_batch


class RljaxSAC(Agent):
    """
    Interface to the SAC agent from RLJAX (https://github.com/toshikwa/rljax)

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
    - :attr:`sac_params` - the SAC parameters in a dict :
        "actor_lr" (default=3e-3): the actor learning rate
        "critic_lr" (default=3e-3): the critic learning rate
        "temp_lr" (default=3e-3): the temperature learning rate
        "discount" (default=0.99): the discount factor
        "hidden_dims" (default=(256,256)): the hidden layer dimensions for the actor
        and critic networks
        "init_temperature" (default=1.): the initial temperature
        "target_update_period" (default=1): defines how often a soft update of the
        target critic is performed
        "tau" (default=5e-2): the soft update coefficient
    - :attr:`sac` - the SAC algorithm as implemented in the RLJAX library
    """

    def __init__(self, observation_dim, action_dim, params=None):

        self._config_string = str(list(locals().items())[1:])
        super().__init__("SAC", observation_dim, action_dim, params)

        start_seed = np.random.randint(1e9) if "seed" not in params else params["seed"]

        self.sac_params = {
            "actor_lr": 3e-4,
            "critic_lr": 3e-3,
            "temp_lr": 3e-4,
            "discount": 0.99,
            "hidden_dims": (256, 256),
            "init_temperature": 1.0,
            "target_update_period": 1,
            "tau": 5e-2,
        }

        for key in self.sac_params:
            if key in self.params:
                self.sac_params[key] = self.params[key]

        self.sac = SAC(
            np.inf,
            observation_dim,
            action_dim,
            start_seed,
            max_grad_norm=None,
            gamma=self.sac_params["discount"],
            nstep=1,
            num_critics=2,
            buffer_size=None,
            use_per=False,
            batch_size=None,
            start_steps=None,
            update_interval=self.sac_params["target_update_period"],
            tau=self.sac_params["tau"],
            fn_actor=None,
            fn_critic=None,
            lr_actor=self.sac_params["actor_lr"],
            lr_critic=self.sac_params["critic_lr"],
            lr_alpha=self.sac_params["temp_lr"],
            units_actor=self.sac_params["hidden_dims"],
            units_critic=self.sac_params["hidden_dims"],
            log_std_min=-20.0,
            log_std_max=2.0,
            d2rl=False,
            init_alpha=self.sac_params["init_temperature"],
            adam_b1_alpha=0.9,
        )

        self.sac.buffer = DummyBuffer()

    def value(self, observation, action):
        return self.sac.calculate_value(self.sac.params_critic, observation, action)

    def select_action(self, observation, eval_mode=False):
        if eval_mode:
            return self.sac.select_action(observation)
        else:
            return self.sac.explore(observation)

    def train_on_batch(self, batch):
        obs = batch["observation"]
        act = batch["action"]
        reward = batch["reward"]
        mask = batch["terminated"]
        next_obs = batch["next_observation"]
        batch = (obs, act, reward, mask, next_obs)
        self.sac.buffer.set_next_batch(batch)
        self.sac.update()
        return None

    def save(self, directory):
        self.sac.save_params(directory)

    def load(self, directory):
        self.sac.load_params(directory)

    def write_config(self, output_file: str):
        print(self._config_string, file=output_file)
