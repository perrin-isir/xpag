# ![alt text](logo.png "logo")

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*xpag* ("e**xp**loring **ag**ents") is a modular reinforcement learning platform, currently in beta version.

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

    conda env update --name existing_env --file environment.yml

To activate the `xpagenv` environment:

    conda activate xpagenv

Finally, to install the *xpag* library in the activated virtual environment:

    pip install -e .

</p>
</details>


<details><summary>JAX and Brax installation</summary>
<p>

The *xpag* agents are written in JAX,
and some functionalities specific to Brax environments require it to be 
installed.

*The modules JAX and Brax are not automatically installed as dependencies of xpag.*

- To install JAX, follow these guidelines:  

    [https://github.com/google/jax#Installation](https://github.com/google/jax#Installation)  

    *Remark:* to verify that the installation went well, check the backend used by JAX with the following command (in a python console and with `jax` imported and configured):
    ```
    print(jax.lib.xla_bridge.get_backend().platform)
    ```
    It will print "cpu", "gpu" or "tpu" depending on the platform that JAX is using.

- Once JAX is installed, `pip install brax` should install Brax. Otherwise, follow these guidelines:

  [https://github.com/google/brax#readme](https://github.com/google/brax#readme) 

</p>
</details>

-----
## Tutorials

The *xpag-tutorials* repository contains a list of tutorials (colab notebooks) for *xpag*:  
[https://github.com/perrin-isir/xpag-tutorials](https://github.com/perrin-isir/xpag-tutorials)


-----
## Structure
<details><summary><B><I>A platform for goal-conditioned RL</I></B></summary>

*xpag* allows standard reinforcement learning, but it has been designed with
goal-conditioned reinforcement learning (GCRL) in mind (check out the [train_gmazes.ipynb](https://colab.research.google.com/github/perrin-isir/xpag-tutorials/blob/main/train_gmazes.ipynb)
tutorial for a simple example of GCRL). 

In GCRL, agents follow a goal, and the reward depends on 
the degree of achievement of that goal. 
In some cases, goals are defined by the environment, but in others, they are defined by
the agent itself, and they can possibly be changed several times during an episode. 
For this reason, *xpag* introduces a dedicated module called 
"goal-setter", which can be considered either as a part of the environment, or as 
a part of the agent.

*xpag* relies on a single reinforcement learning loop (the `learn()`
function in [xpag/tools/learn.py](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py)),
with the following components that can be independently modified:

<details><summary><B>the environment</B></summary>

Environments must allow parallel rollouts. Following the gym interface
(see [gym.openai.com/docs/#environments](https://gym.openai.com/docs/#environments)), they have 
a `reset()` function that returns an observation (which is actually a concatenation of the 
observations for all parallel rollouts), and a `step()` function that takes in input 
an action (which is, again, a concatenation of actions) and returns:
`observation`, `reward`, `done`, `info` (cf. [gym.openai.com/docs/#observations](https://gym.openai.com/docs/#observations)).

Additionally, *xpag* requires a `reset_done()` function which performs a reset for
the i-th rollout if and only if its previous step was terminal (`done[i] == True`).
The [gym_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/gym_vec_env.py) and 
[brax_vec_env()](https://github.com/perrin-isir/xpag/blob/main/xpag/wrappers/brax_vec_env.py) functions (see [tutorials](https://github.com/perrin-isir/xpag-tutorials))
call wrappers that automatically add the `reset_done()` function to Gym and Brax 
environments. `reset_done()` must be the only way to perform resets after epsiode terminations,
therefore auto-resets (automatic resets after terminal transitions) are not allowed. 
The main reason to prefer `reset_done()` instead of auto-resets
is that with auto-resets, terminal transitions must be special and contain additional
information. With `reset_done()`, this is no longer necessary.

Goal-based environments (for GCRL) must follow the interface defined in 
the [Gym-Robotics](https://github.com/Farama-Foundation/gym-robotics) library
(see [core.py](https://github.com/Farama-Foundation/Gym-Robotics/blob/main/gym_robotics/core.py)):
their observation spaces are of type [gym.spaces.Dict](https://github.com/openai/gym/blob/master/gym/spaces/dict.py), with the following keys 
in the observation dictionaries: "observation", "achieved_goal", "desired_goal".
They must also have a `compute_reward()` function that computes rewards based on 
achieved and desired goals. *xpag* also assumes that, for goal-based
environments, the `info` dictionary returned by the step function contains 
`info["is_success"]`, an array of Booleans (one per rollout) that are `True` if the corresponding
transition is a successfull achievement of the desired goal, and `False` otherwise.

The three first arguments of the 
[learn()](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py) function 
are:
* `env`: the training environment, which runs `num_envs` rollouts in parallel.
* `eval_env`: the evaluation environment, identical to `env` except that it runs 
single rollouts.
* `env_info`: a dictionary containing information about the environment:
  * `env_info["env_type"]`: the type of environment; for the moment *xpag* 
differentiates 3 types of environments: "Brax" environments, "Mujoco" environments, and
"Gym" environments. This information is used to adapt the way episodes are saved.
  * `env_info["name"]`: a name for the environment.
  * `env_info["is_goalenv"]`: whether the environment is a goal-based environment or 
not.
  * `env_info["num_envs"]`: the number of parallel rollouts in `env`
  * `env_info["max_episode_steps"]`: the maximum number of steps in episodes (*xpag* 
does not allow potentially infinite episodes). *xpag* assumes that the `info` dictionary
returned by the step function contains `info["truncation"]`, an array of Booleans (one 
per rollout). `info["truncation"][i]` is True if and only if the i-th rollout has 
been terminated because it reached maximum length. Remark: if maximum length is reached,
but the transition would anyway be terminal for other reasons, then `info["truncation"][i]`
should be False.
  * `env_info["action_space"]`: the action space (of type [gym.spaces.Space](https://github.com/openai/gym/blob/master/gym/spaces/space.py)) that takes into account parallel rollouts.
This can be useful to sample random actions.
  * `env_info["single_action_space"]`: : the action space for single rollouts.

</details>

<details><summary><B>the agent</B></summary>

*xpag* only considers the case of a unique off-policy agent. 

</details>

<details><summary><B>the buffer</B></summary></details>
<details><summary><B>the sampler</B></summary></details>
<details><summary><B>the goal-setter</B></summary></details>

The figure below summarizes the RL loop and the interactions between the components:
(TODO)
</details>

-----
## Citing the project
To cite this repository in publications:

```bibtex
@misc{xpag,
  author = {Perrin-Gilbert, Nicolas},
  title = {xpag},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/perrin-isir/xpag}},
}
```
