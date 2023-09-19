from xpag.agents.agent import Agent
from xpag.agents.flax_agents.sac.sac import FlaxSAC
from xpag.agents.flax_agents.td3.td3 import FlaxTD3
from xpag.agents.flax_agents.tqc.tqc import FlaxTQC
from xpag.agents.flax_agents.sdqn.sdqn import FlaxSDQN, FlaxSDQNSetter
from xpag.agents.rljax_agents.rljax_interface import RljaxSAC


def agent_factory(name, haiku_agent_class, flax_agent_class):
    class AgentClass(Agent):
        def __init__(
            self, observation_dim, action_dim, params=None, haiku_or_flax="flax"
        ):
            super().__init__(name, observation_dim, action_dim, params)
            assert haiku_or_flax in ["haiku", "flax"], (
                "haiku_or_flax argument must be" " 'haiku' or 'flax'"
            )
            if haiku_or_flax == "haiku":
                self.agent = haiku_agent_class(observation_dim, action_dim, params)
            else:
                self.agent = flax_agent_class(observation_dim, action_dim, params)

        def value(self, observation, action):
            return self.agent.value(observation, action)

        def select_action(self, observation, eval_mode=False):
            return self.agent.select_action(observation, eval_mode)

        def train_on_batch(self, batch):
            return self.agent.train_on_batch(batch)

        def write_config(self, output_file: str):
            self.agent.write_config(output_file)

        def save(self, directory: str):
            self.agent.save(directory)

        def load(self, directory: str):
            self.agent.load(directory)

    return AgentClass


SAC = agent_factory("SAC", RljaxSAC, FlaxSAC)
TD3 = FlaxTD3
TQC = FlaxTQC
SDQN = FlaxSDQN
SDQNSetter = FlaxSDQNSetter
