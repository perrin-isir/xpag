from abc import ABC, abstractmethod
import torch
import os
import numpy as np


class RLAgent(ABC):
    def __init__(self, name: str, observation_dim: int, action_dim: int, device: str,
                 params: dict):
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.params = params

    @staticmethod
    def soft_update(source, target, tau):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @abstractmethod
    def train(self, buffer, sampler, batch_size):
        pass

    @abstractmethod
    def select_action(self, observation, deterministic=True):
        pass

    def write_config(self, output_file):
        pass

    @abstractmethod
    def save(self, directory):
        pass

    @abstractmethod
    def load(self, directory):
        pass

    def save_nets(self, nets, directory):
        sdir = directory + "/" + self.__class__.__name__
        for i, net in enumerate(nets):
            filename = "net_" + str(i + 1)
            os.makedirs(sdir, exist_ok=True)
            torch.save(net.state_dict(), "%s/%s.pth" % (sdir, filename))

    def load_nets(self, nets, directory):
        sdir = directory + "/" + self.__class__.__name__
        for i, net in enumerate(nets):
            filename = "net_" + str(i + 1)
            net.load_state_dict(
                torch.load("%s/%s.pth" % (sdir, filename), map_location=self.device)
            )

