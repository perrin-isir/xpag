# ![alt text](logo.png "logo")

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*xpag* ("e**xp**loring **ag**ents") is a modular reinforcement learning library with JAX agents, currently in beta version.

-----
## Install

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

    conda env create --name xpagenv --file environment.yaml

If you prefer to update an existing environment (`existing_env`):

    conda env update --name existing_env --file environment.yaml

To activate the `xpagenv` environment:

    conda activate xpagenv

Finally, to install the *xpag* library in the activated virtual environment:

    pip install -e .

</p>
</details>


<details><summary>JAX, Flax, Brax, etc.</summary>
<p>

The *xpag* agents are written in JAX and Flax,
and some functionalities specific to Brax environments require it to be 
installed.

**The modules JAX, Flax and Brax are required but are NOT automatically installed as dependencies of *xpag*.**

- To install JAX, follow these guidelines:  

    [https://github.com/google/jax#installation](https://github.com/google/jax#installation)  

    *Remark:* to verify that the installation went well, check the backend used by JAX with the following command (in a python console and with `jax` imported and configured):
    ```
    print(jax.lib.xla_bridge.get_backend().platform)
    ```
    It will print "cpu", "gpu" or "tpu" depending on the platform that JAX is using.

- Once JAX is installed, `pip install flax` and `pip install brax` should install recent versions of Flax and Brax. If there are issues, follow these guidelines:

  [https://github.com/google/flax#quick-install](https://github.com/google/flax#quick-install) and [https://github.com/google/brax#using-brax-locally](https://github.com/google/brax#using-brax-locally) 

- *xpag* works without the following libraries, but they are required for the [tutorials](https://github.com/perrin-isir/xpag-tutorials):
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

In GCRL, agents have a goal, and the reward mainly depends on 
the degree of achievement of that goal. *xpag* introduces a 
module called "setter" which, among other things, can help to set and manage
goals. Although the setter is largely similar to an environment wrapper, it 
is separated from the environment because in some cases it should be considered as 
an independent entity (e.g. a teacher), or as a part of the agent itself.

*xpag* relies on a single reinforcement learning loop (the `learn()`
function in [xpag/tools/learn.py](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py))
in which the following components interact:

<details><summary><B>the environment</B></summary>

In *xpag*, environments must allow parallel rollouts, and *xpag* keeps the same API even in the case of a single rollout (`num_envs == 1`).

* `reset()`  
Following the gym Vector API
(see [https://www.gymlibrary.ml/content/vector_api](https://www.gymlibrary.ml/content/vector_api)), environments have 
a `reset()` function that returns an observation (which is a batch of observations for all 
parallel rollouts) and an optional dictionary `info` (when the `return_info` argument is
True, see [https://www.gymlibrary.ml/content/vector_api/#reset](https://www.gymlibrary.ml/content/vector_api/#reset)).


* `step()`
Again, following the gym Vector API, environments have a `step()` function that takes
in input an action (which is actually a batch of actions, one per rollout) and returns:
`observation`, `reward`, `done`, `info` (cf. [https://www.gymlibrary.ml/content/api/#stepping](https://www.gymlibrary.ml/content/api/#stepping)).
There are differences with the gym Vector API. First, this a detail but we name the
ouputs `observation`, `reward`, ... (singular) instead of `observations`, `rewards`, ... (plural) because in *xpag* this also covers the case `num_envs == 1`.
Second, *xpag* assumes that `reward` and `done` have the shape `(num_envs, 1)`, not
`(num_envs,)`. More broadly, whether they are due to `num_envs == 1` or to
unidimensional elements, single-dimensional entries are not squeezed in *xpag*.
Third, in *xpag*, `info` is a single dictionary, not a tuple of dictionaries
(but its entries may be tuples). 


* `reset_done()`:  
The most significant difference with the gym Vector API is that *xpag* requires a `reset_done()` function which takes the `done` array in input and performs a reset for
the i-th rollout if and only if `done[i]` is evaluated to True. Besides `done`, the arguments of `reset_done()` are the same as the ones of `reset()`: `seed`, `return_info` and `options`, and its outputs are also the same: either `observation`, or `observation`, `info` if `return_info` is True.
The [gym_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/gym_vec_env.py) and 
[brax_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/brax_vec_env.py) functions (see [tutorials](https://github.com/perrin-isir/xpag-tutorials))
call wrappers that automatically add the `reset_done()` function to Gym and Brax 
environments, and make the wrapped environments fit the *xpag* API. `reset()` must be called once for the initial reset, and after that only `reset_done()` should be used. Auto-resets (automatic resets after terminal transitions) are not allowed in *xpag*. 
The main reason to prefer `reset_done()` to auto-resets
is that with auto-resets, terminal transitions must be special and contain additional
information. With `reset_done()`, this is no longer necessary. Furthermore,
by modifying the `done` array returned by a step of the environment, it becomes possible 
to force the termination of an episode, or to force an episode to continue despite 
reaching a terminal transition (but this must be done with caution).


* *Goal-based environments:*  
Goal-based environments (for GCRL) must follow a similar interface to the one defined in 
the [Gym-Robotics](https://github.com/Farama-Foundation/gym-robotics) library
(see [core.py](https://github.com/Farama-Foundation/Gym-Robotics/blob/main/gym_robotics/core.py)):
their observation spaces are of type [gym.spaces.Dict](https://github.com/openai/gym/blob/master/gym/spaces/dict.py), with the following keys 
in the `observation` dictionaries: "observation", "achieved_goal", "desired_goal".
They must also have a `compute_reward()` function that computes rewards from transitions.
Multiple rollouts are concatenated in the same way as the gym function `concatenate()`
(cf. [https://github.com/openai/gym/blob/master/gym/vector/utils/numpy_utils.py](https://github.com/openai/gym/blob/master/gym/vector/utils/numpy_utils.py)), 
which means that the batched observations are always single dictionaries in which the 
entries "observation", "achieved_goal" and "desired_goal" are arrays of observations,
achieved goals and desired goals.


* `info`  
The `info` dictionary returned by the environment steps must always contain
`info["truncation"]`, an array of Booleans (one per rollout). `info["truncation"][i]`
is True if the i-th rollout has been terminated without reaching a terminal state 
(for example because the episode reached maximum length). *Remark:* in gym, the
conventional name is `info["TimeLimit.truncated"]`, but this is automatically 
modified in the wrapper applied by the 
[gym_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/gym_vec_env.py)
function.  
*xpag* also assumes that, for goal-based environments, the `info` dictionary 
always contains `info["is_success"]`, an array of Booleans (one per rollout)
that are `True` if the corresponding transition is a successfull achievement of the
desired goal, and `False` otherwise (*remark:* this does not need to coincide
with episode termination).  


* `env_info`:  
The [learn()](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py) function 
is the functions that runs the training loop in *xpag*. Its three first arguments are:
  * `env`: the training environment, which runs `num_envs` rollouts in parallel.
  * `eval_env`: the evaluation environment, identical to `env` except that it runs 
  single rollouts.
  * `env_info`: a dictionary containing information about the environment:
    * `env_info["env_type"]`: the type of environment; for the moment *xpag* 
  differentiates 3 types of environments: "Brax" environments, "Mujoco" environments, and
  "Gym" environments. This information is used to adapt the way episodes are saved.
    * `env_info["name"]`: the name of the environment.
    * `env_info["is_goalenv"]`: whether the environment is a goal-based environment or 
  not.
    * `env_info["num_envs"]`: the number of parallel rollouts in `env`
    * `env_info["max_episode_steps"]`: the maximum number of steps in episodes (*xpag* 
  does not allow potentially infinite episodes).
  *Remark:* 
    * `env_info["action_space"]`: the action space (of type [gym.spaces.Space](https://github.com/openai/gym/blob/master/gym/spaces/space.py)) that takes into account parallel rollouts.
  This can be useful to sample random actions.
    * `env_info["single_action_space"]`: the action space for single rollouts.

</details>

<details><summary><B>the agent</B></summary>

*xpag* only considers the case of a unique off-policy agent training on parallel rollouts. 

</details>

<details><summary><B>the buffer</B></summary> TODO </details>
<details><summary><B>the sampler</B></summary> TODO </details>
<details><summary><B>the setter</B></summary> TODO </details>

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
