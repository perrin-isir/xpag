import os
import gym
from gym.envs.registration import register


def envpath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


print("gym-gmazes: ")
print("|    gym version and path:", gym.__version__, gym.__path__)

print("|    REGISTERING GMazeSimple-v0 from", envpath())
register(
    id="GMazeSimple-v0",
    entry_point="gym_gmazes.envs:GMazeSimple",
)

print("|    REGISTERING GMazeGoalSimple-v0 from", envpath())
register(
    id="GMazeGoalSimple-v0",
    entry_point="gym_gmazes.envs:GMazeGoalSimple",
)
