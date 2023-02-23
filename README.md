# ![alt text](logo.png "logo")

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*xpag* ("e**xp**loring **ag**ents") is a modular reinforcement learning library with JAX agents, currently in beta version.

-----
## Install

<details><summary>JAX</summary>
<p>

The *xpag* agents are written in JAX and Flax,
and some functionalities specific to Brax environments require it to be 
installed too.

**The library JAX is required by *xpag*, but its automatic installation may not work properly (e.g. not GPU-compatible).**  
We recommend to install it properly before installing *xpag*, following these guidelines:  

[https://github.com/google/jax#installation](https://github.com/google/jax#installation)  

*Remark:* to verify that the installation went well, check the backend used by JAX with the following command (in a python console and with `jax` imported and configured):
```
print(jax.lib.xla_bridge.get_backend().platform)
```
It will print "cpu", "gpu" or "tpu" depending on the platform that JAX is using.

Flax and Brax should be installed automatically as dependencies of xpag, but if necessary you can follow these guidelines to install it:
    
[https://github.com/google/flax#quick-install](https://github.com/google/flax#quick-install)  
[https://github.com/google/brax#using-brax-locally](https://github.com/google/brax#using-brax-locally)
</p>
</details>

<details><summary>Option 1: pip</summary>
<p>

    pip install git+https://github.com/perrin-isir/xpag

</p>
</details>

<details><summary>Option 2: conda</summary>
<p>

    git clone https://github.com/perrin-isir/xpag.git
    cd xpag

Choose a conda environmnent name, for instance `xpagenv`.  
The following command creates the `xpagenv` environment with the requirements listed in [environment.yaml](environment.yaml):

    conda create --name xpagenv --file environment.yaml

If you prefer to update an existing environment (`existing_env`):

    conda update --name existing_env --file environment.yaml

To activate the `xpagenv` environment:

    conda activate xpagenv

Finally, to install the *xpag* library in the activated virtual environment:

    pip install -e .

</p>
</details>


<details><summary>Tutorials</summary>
<p>

The following libraries, not required by *xpag*, are required for the [tutorials](https://github.com/perrin-isir/xpag-tutorials):
  - MuJoCo (`pip install mujoco`): see [https://github.com/deepmind/mujoco](https://github.com/deepmind/mujoco)
  - imageio (`pip install imageio`): see [https://github.com/imageio/imageio](https://github.com/imageio/imageio)
</p>
</details>

-----
## Tutorials

The *xpag-tutorials* repository contains a list of tutorials (colab notebooks) for *xpag*:  
[https://github.com/perrin-isir/xpag-tutorials](https://github.com/perrin-isir/xpag-tutorials)


-----
## Short documentation
<details><summary><B><I>xpag</I>: a platform for goal-conditioned RL</B></summary>

*xpag* allows standard reinforcement learning, but it has been designed with
goal-conditioned reinforcement learning (GCRL) in mind (check out the [train_gmazes.ipynb](https://colab.research.google.com/github/perrin-isir/xpag-tutorials/blob/main/train_gmazes.ipynb)
tutorial for a simple example of GCRL). 

In GCRL, agents have a goal, which is part of the input they take, and the reward mainly depends on 
the degree of achievement of that goal. Beyond the usual modules in 
RL platforms (environment, agent, buffer/sampler), *xpag* introduces a 
module called "setter" which, among other things, can help to set and manage
goals (for example modifying the goal several times in a single episode).
Although the setter is largely similar to an environment wrapper, it 
is separated from the environment because in some cases it should be considered as 
an independent entity (e.g. a teacher), or as a part of the agent itself.

*xpag* relies on a single reinforcement learning loop (the `learn()`
function in [xpag/tools/learn.py](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py))
in which the environment, the agent, the buffer and the setter interact (see below). 
The `learn()` function  has the following first 3 arguments (returned by [gym_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/gym_vec_env.py) and 
[brax_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/brax_vec_env.py)):
  * `env`: the training environment, which runs 1 or more rollouts in parallel.
  * `eval_env`: the evaluation environment, identical to `env` except that it runs 
  a single rollout.
  * `env_info`: a dictionary containing information about the environment:
    * `env_info["env_type"]`: the type of environment; for the moment *xpag* 
  differentiates 3 types of environments: "Brax" environments, "Mujoco" environments, and
  "Gym" environments. This information is used to adapt the way episodes are saved and replayed.
    * `env_info["name"]`: the name of the environment.
    * `env_info["is_goalenv"]`: whether the environment is a goal-based environment or 
  not.
    * `env_info["num_envs"]`: the number of parallel rollouts in `env`
    * `env_info["max_episode_steps"]`: the maximum number of steps in episodes (*xpag* 
  does not allow potentially infinite episodes).
    * `env_info["action_space"]`: the action space (of type [gym.spaces.Space](https://github.com/openai/gym/blob/master/gym/spaces/space.py)) that takes into account parallel rollouts. It can be useful to sample random actions.
    * `env_info["single_action_space"]`: the action space (of type [gym.spaces.Space](https://github.com/openai/gym/blob/master/gym/spaces/space.py)) for single rollouts.  
  
  `learn()` also takes in input the agent, the buffer and the setter and various parameters. Detailed information about the arguments of `learn()` can be
  found in the code documentation (check [xpag/tools/learn.py](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py)).

The components that interact during learning are:
<details><summary><B>the environment (env)</B></summary>

In *xpag*, environments must allow parallel rollouts, and *xpag* keeps the same API even in the case of a single rollout,
i.e. when the number of "parallel environments" is 1. Basically, all environments are 
"vector environments".

* `env.reset(seed: Optional[Union[int, List[int]]], options: Optional[dict])` -> `observation: Union[np.array, jax.numpy.array], info: dict`  
Following the gym Vector API
(see [https://www.gymlibrary.dev/api/vector/#vectorenv](https://www.gymlibrary.dev/api/vector/#vectorenv)), environments have 
a `reset()` function that returns an `observation` (which is actually a batch of observations for all the 
parallel rollouts) and an optional dictionary `info` (see [https://www.gymlibrary.dev/api/vector/#reset](https://www.gymlibrary.dev/api/vector/#reset)).  
We expect `observation` to be a numpy array, or a jax.numpy array, and its first dimension 
selects between parallel rollouts, which means that `observation[i]` is the observation in
the i-th rollout. In the case of a single rollout, `observation[0]` is the observation
in this rollout.


* `env.step(action: Union[np.array, jax.numpy.array])` -> `observation, reward, terminated, truncated, info`  
Again, following the gym Vector API, environments have a `step()` function that takes
in input an action (which is actually a batch of actions, one per rollout) and returns:
`observation`, `reward`, `terminated`, `truncated`, `info` (cf. [https://www.gymlibrary.dev/api/vector/#step](https://www.gymlibrary.dev/api/vector/#step)).
There are slight differences with the gym Vector API. First, in *xpag* this API also covers the case
of a single rollout. Second, *xpag* assumes that `reward` and `done` have shape `(n, 1)`, not
`(n,)` (where n is the number of parallel rollouts). More broadly, whether they are due to a single rollout or to
unidimensional elements, single-dimensional entries are not squeezed in *xpag*.
Third, in *xpag*, `info` is a dictionary, not a tuple of dictionaries
(however its entries may be tuples). 


* `env.reset_done(done, seed: Optional[Union[int, List[int]]], options: Optional[dict])` -> `observation, info`   
The most significant difference with the gym Vector API is that *xpag* requires a `reset_done()` function which takes a `done` array of Booleans in input and performs a reset for
the i-th rollout if and only if `done[i]` is evaluated to True. Besides `done`, the arguments of `reset_done()` are the same as the ones of `reset()`: `seed` and `options`, and its outputs are also the same: `observation`, `info`.
For rollouts that are not reset, the returned observation is the same as the observation returned by the last
`step()`. `reset()` must be called once for the initial reset, and afterwards only `reset_done()` should be used. Auto-resets (automatic resets after terminal transitions) are not allowed in *xpag*. 
The main reason to prefer `reset_done()` to auto-resets
is that with auto-resets, terminal transitions must be special and contain additional
information. With `reset_done()`, this is no longer necessary. Furthermore,
by modifying the `done` array returned by a step of the environment, it becomes possible 
to easily force the termination of an episode, or to force an episode to continue despite 
reaching a terminal transition (but this must be done with caution).


* `gym_vec_env(env_name: str, num_envs: int, wrap_function: Callable = None)` -> `env, eval_env, env_info: dict`  
`brax_vec_env(env_name: str, num_envs: int, wrap_function: Callable = None, *, force_cpu_backend : bool = False)` -> `env, eval_env, env_info: dict`  
The [gym_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/gym_vec_env.py) and 
[brax_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/brax_vec_env.py) functions (see [tutorials](https://github.com/perrin-isir/xpag-tutorials))
call wrappers that automatically add the `reset_done()` function to Gym and Brax 
environments, and make the wrapped environments fit the *xpag* API.


* *Goal-based environments:*  
Goal-based environments (for GCRL) must have a similar interface to the one defined in 
the [Gym-Robotics](https://github.com/Farama-Foundation/gym-robotics) library
(see `GoalEnv` in [core.py](https://github.com/Farama-Foundation/Gym-Robotics/blob/main/gym_robotics/core.py)), with minor differences.
Their observation spaces are of type [gym.spaces.Dict](https://github.com/openai/gym/blob/master/gym/spaces/dict.py), with the following keys 
in the `observation` dictionaries: `"observation"`, `"achieved_goal"`, and `"desired_goal"`.
Goal-based environments must also have in attribute a `compute_reward()` function that computes rewards.
In *xpag*, the inputs of `compute_reward()` can be different from the ones considered in 
the original `GoalEnv` class. For example, in the
[GoalEnvWrapper](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/goalenv_wrapper.py) class,
which can be used to turn standard environments into goal-based environments, the
arguments of `compute_reward()` are assumed to be `achieved_goal` (the goal achieved *after* `step()`),
`desired_goal` (the desired goal *before* `step()`), `action`, `observation` (the observation *after* `step()`),
`reward` (the reward of the base environment), `terminated`, `truncated` and `info` (the outputs of the
`step()` function). In the version of [HER](https://github.com/perrin-isir/xpag/blob/main/xpag/samplers/HER.py)
  (cf. [https://arxiv.org/pdf/1707.01495.pdf](https://arxiv.org/pdf/1707.01495.pdf)) in *xpag*,
it is assumed that `compute_reward()` depends only on  `achieved_goal`, `desired_goal`, `action` and `observation`.  
In goal-based environments, the multiple observations from parallel rollouts are concatenated as in the gym function `concatenate()`
(cf. [https://github.com/openai/gym/blob/master/gym/vector/utils/numpy_utils.py](https://github.com/openai/gym/blob/master/gym/vector/utils/numpy_utils.py)), 
which means that the batched observations are always single dictionaries in which the 
entries `"observation"`, `"achieved_goal"` and `"desired_goal"` are arrays of observations,
achieved goals and desired goals.


* `info`  
*xpag* assumes that, in goal-based environments, the `info` dictionary returned by `step()`
always contains `info["is_success"]`, an array of Booleans (one per rollout)
that are `True` if the corresponding transition is a successfull achievement of the
desired goal, and `False` otherwise (*remark:* this does not need to coincide
with episode termination).

</details>

<details><summary><B>the agent (agent)</B></summary>

*xpag* only considers off-policy agents. (TODO) 

</details>

<details><summary><B>the buffer (buffer)</B></summary> TODO </details>
<details><summary><B>the sampler (sampler)</B></summary> TODO </details>
<details><summary><B>the setter (setter)</B></summary> TODO </details>

The figure below summarizes the RL loop and the interactions between the components:
(TODO)
</details>

-----
## Acknowledgements

* Maintainer and main contributor:
  - Nicolas Perrin-Gilbert (CNRS, ISIR)

  Other people who contributed to *xpag*:
  - Olivier Serris (ISIR)
  - Alexandre Chenu (ISIR)

* The [SAC agent](https://github.com/perrin-isir/xpag/blob/main/xpag/agents/sac) is based on the implementation of SAC in [JAXRL](https://github.com/ikostrikov/jaxrl), and some elements of the [TQC agent](https://github.com/perrin-isir/xpag/blob/main/xpag/agents/tqc) come from the implementation of TQC in [RLJAX](https://github.com/ku2482/rljax).

-----
## Citing the project
To cite this repository in publications:

```bibtex
@misc{xpag,
  author = {Perrin-Gilbert, Nicolas},
  title = {xpag: a modular reinforcement learning library with JAX agents},
  year = {2022},
  url = {https://github.com/perrin-isir/xpag}
}
```
