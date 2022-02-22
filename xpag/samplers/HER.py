# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
import torch
from xpag.tools.utils import DataType
from xpag.samplers.sampler import Sampler


class HER(Sampler):
    def __init__(
        self,
        compute_reward,
        replay_strategy: str = "future",
        datatype: DataType = DataType.TORCH,
    ):
        super().__init__(datatype)
        self.replay_strategy = replay_strategy
        self.replay_k = 4.0
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + self.replay_k))
        else:
            self.future_p = 0
        self.reward_func = compute_reward

    def sample(self, buffers, batch_size_in_transitions):
        rollout_batch_size = buffers["episode_length"].shape[0]
        batch_size = batch_size_in_transitions
        # select rollouts and time steps
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()

        if self.datatype == DataType.TORCH:
            t_samples = (torch.rand_like(t_max_episodes) * t_max_episodes).long()
        else:
            t_samples = np.random.randint(t_max_episodes)

        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        # HER indexes
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        if self.datatype == DataType.TORCH:
            future_offset = (
                torch.rand_like(t_max_episodes) * (t_max_episodes - t_samples)
            ).long()
        else:
            future_offset = np.random.uniform(size=batch_size) * (
                t_max_episodes - t_samples
            )
            future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        # replace desired goal with achieved goal
        future_ag = buffers["ag_next"][episode_idxs[her_indexes], future_t]
        transitions["g"][her_indexes] = future_ag
        # recomputing rewards
        if self.datatype == DataType.TORCH:
            transitions["r"] = torch.unsqueeze(
                self.reward_func(transitions["ag_next"], transitions["g"], {}), -1
            )
        else:
            transitions["r"] = np.expand_dims(
                self.reward_func(transitions["ag_next"], transitions["g"], {}), 1
            )
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        if self.datatype == DataType.TORCH:
            transitions["obs"] = torch.hstack((transitions["obs"], transitions["g"]))
            transitions["obs_next"] = torch.hstack(
                (transitions["obs_next"], transitions["g"])
            )
        else:
            transitions["obs"] = np.concatenate(
                [transitions["obs"], transitions["g"]], axis=1
            )
            transitions["obs_next"] = np.concatenate(
                [transitions["obs_next"], transitions["g"]], axis=1
            )
        return transitions
