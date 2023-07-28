import numpy as np


def fake_state(observation_dim):
    return np.zeros((1, observation_dim)).astype(np.float32)
    # state = state_space.sample()[None, ...]
    # if len(state_space.shape) == 1:
    #     state = state.astype(np.float32)
    # return state


def fake_action(action_dim):
    return np.zeros((1, action_dim)).astype(np.float32)
    # action = action_space.sample().astype(np.float32)[None, ...]
    # return action
