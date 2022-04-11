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
<details><summary><B><I>A reinforcement learning platform for goal-conditioned reinforcement learning</I></B></summary>

*xpag* allows standard reinforcement learning, but it has been designed with
goal-conditioned reinforcement learning (GCRL) in mind (check out the [train_gmazes.ipynb](https://colab.research.google.com/github/perrin-isir/xpag-tutorials/blob/main/train_gmazes.ipynb)
tutorial for a simple example of GCRL). 

In GCRL, agents follow a goal, and the reward depends on how well this goal is being achieved. 
In some cases, goals are defined by the environment, but in others, they are defined by
the agent itself, and can possibly be changed several times during an episode. 
For this reason, *xpag* introduces a dedicated module called 
"goal-setter", which can either be considered as a part of the environment, or as 
a part of the agent.

Overall, *xpag* relies on a fixed reinforcement learning loop (the `learn()`
function in [xpag/tools/learn.py](https://github.com/perrin-isir/xpag/blob/main/xpag/tools/learn.py)), but with components that are 
easy to modify. They are:

- the environment
- the agent
- the buffer
- the sampler
- the goal-setter

The figure below summarizes the RL loop and the interactions between the components:

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
