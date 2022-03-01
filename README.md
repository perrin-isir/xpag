# xpag
*xpag* ("e**xp**loring **ag**ents") is a modular reinforcement learning platform, currently in beta version.

## Installation

<details><summary>Option 1: pip</summary>
<p>

    pip install git+https://github.com/perrin-isir/xpag#egg=xpag

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

Recommended installation steps (with conda): 
```
git clone https://github.com/perrin-isir/xpag.git
cd xpag
```
Choose a conda environmnent name, for instance `xpagenv`.  
The following command creates the `xpagenv` environment with the requirements listed in [environment.yaml](environment.yaml):
```
conda env create --name xpagenv --file environment.yaml
```
If you prefer to update an existing environment (`existing_env`), use the command:
```
conda env update --name existing_env --file environment.yml
```
To activate the `xpagenv` environment:
```
conda activate xpagenv
```
Use the following command to install the *xpag* library in the activated virtual environment:
```
pip install -e .
```

Two more steps:
* You need to properly install `jax` and `brax` in the xpagenv environment.  
Follow these guidelines:  
[https://github.com/google/jax#Installation](https://github.com/google/jax#Installation)  
[https://github.com/google/brax#readme](https://github.com/google/brax#readme)  
If you have installed `jax` on GPU and want to verify that it is working, try (in a python console):
```python
import jax
print(jax.lib.xla_bridge.get_backend().platform)
```
It will print "cpu" or "gpu" depending on the platform that jax is using.

* *xpag* also uses `mujoco_py`, which requires mujoco.  
You can download it here: [https://mujoco.org/download](https://mujoco.org/download).
`mujoco_py` may require mujoco to be put in a specific folder, for instance
`~/.mujoco/mujoco210/`, with the following line added to your `~/.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```
To test mujoco, run (from the mujoco folder):
```
./bin/testspeed ./model/humanoid.xml 2000
```
Remark: on Ubuntu, `mujoco_py` may also require the installation of `libglew-dev` and `patchelf`.
