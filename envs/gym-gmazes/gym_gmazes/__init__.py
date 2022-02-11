import os
import gym
from gym.envs.registration import register


def envpath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


print("gym-gmazes: ")
print("|    gym version and path:", gym.__version__, gym.__path__)

print("|    REGISTERING GMazeDubins-v0 from", envpath())
register(
    id="GMazeDubins-v0",
    entry_point="gym_gmazes.envs:GMazeDubins",
)

print("|    REGISTERING GMazeGoalDubins-v0 from", envpath())
register(
    id="GMazeGoalDubins-v0",
    entry_point="gym_gmazes.envs:GMazeGoalDubins",
)
