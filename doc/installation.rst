************
Installation
************

Option 1: conda (preferred option)
==================================

This option is preferred because it relies mainly on conda-forge packages (which among other things simplifies the installation of JAX).

.. code:: console

    git clone https://github.com/perrin-isir/xpag.git
    cd xpag
    conda update conda

Install micromamba if you don't already have it (you can also simply use conda, by replacing below `micromamba create`, `micromamba update` and `micromamba activate` respectively by `conda env create`, `conda env update` and `conda activate`, but this will lead to a significantly slower installation):

.. code:: console

    conda install -c conda-forge micromamba

Choose a conda environmnent name, for instance `xpagenv`. The following command creates the `xpagenv` environment with the requirements listed in `environment.yaml <https://github.com/perrin-isir/xpag/blob/main/environment.yaml>`__:

.. code:: console

    micromamba create --name xpagenv --file environment.yaml

If you prefer to update an existing environment (`existing_env`):

.. code:: console

    micromamba update --name existing_env --file environment.yaml

Then, activate the `xpagenv` environment:

.. code:: console

    micromamba activate xpagenv

Finally, install the *xpag* library in the activated environment:

.. code:: console

    pip install -e .

Option 2: pip
=============

For the pip install, you need to properly install JAX yourself. Otherwise, if JAX is installed automatically as a pip dependency of *xpag*, it will probably not work as desired (e.g. it will not be GPU-compatible). So you should install it beforehand, following these guidelines: 

`https://github.com/google/jax#installation <https://github.com/google/jax#installation>`__

Then, install *xpag* with:

.. code:: console

    pip install git+https://github.com/perrin-isir/xpag

JAX
===

To verify that the JAX installation went well, check the backend used by JAX with the following command:

.. code:: console

    python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform)"

It will print "cpu", "gpu" or "tpu" depending on the platform JAX is using.

Tutorials
=========


The following libraries, not required by *xpag*, are required for the `tutorials <https://github.com/perrin-isir/xpag-tutorials>`__:

- MuJoCo (``pip install mujoco``): see `https://github.com/deepmind/mujoco <https://github.com/deepmind/mujoco>`__
- imageio (``pip install imageio``): see `https://github.com/imageio/imageio <https://github.com/imageio/imageio>`__
