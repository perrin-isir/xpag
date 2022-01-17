# xpag
*xpag* ("e**xp**loring **ag**ents") is a modular reinforcement learning platform.

## Installation

Recommended installation steps (with conda): 
```
git clone https://github.com/perrin-isir/xpag.git
cd xpag
```
Choose a conda environmnent name, for instance `xpagenv`.  
The following command creates the `xpagenv` environment with the requirements listed in [environment.yaml](xomx/environment.yaml):
```
conda env create --name xpagenv --file environment.yaml
```
If you prefer to update an existing environment (`existing_env`), use the command:
```
conda env update --name existing_env --file environment.yml
```
To activate the environment:
```
conda activate xpagenv
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
You can download it here: [https://mujoco.org/download](https://mujoco.org/download)  
`mujoco_py` may require mujoco to be put in a specific folder, for instance
`~/.mujoco/mujoco210/`, with the following line added to your `~/.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/perrin/.mujoco/mujoco200/bin
```
To test mujoco, run (from the mujoco folder):
```
./bin/testspeed ./model/humanoid.xml 2000
```
