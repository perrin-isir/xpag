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

#### JAX installation

The *xpag* agents are written in JAX, which is not automatically installed as a dependency.


To install JAX, follow these guidelines:  
[https://github.com/google/jax#Installation](https://github.com/google/jax#Installation)  

*Remark:* to verify that the installation went well, check the backend used by JAX
with the following command (in a python console and with `jax` imported and configured):

    print(jax.lib.xla_bridge.get_backend().platform)

It will print "cpu", "gpu" or "tpu" depending on the platform that JAX is using.