from setuptools import setup, find_packages

# Install with 'pip install -e .'

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="xpag",
    version="0.2.9",  # -version-
    author="Nicolas Perrin-Gilbert",
    description="xpag: Exploring Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/perrin-isir/xpag",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.8.0",
        "numpy>=1.21.5",
        "matplotlib>=3.1.3",
        "joblib>=1.1.0",
        "gymnasium>=0.28.1",
        "Pillow>=9.0.1",
        "ipywidgets>=7.6.5",
        "jax>=0.4.8",
        "optax>=0.1.2",
        "flax>=0.7.2",
        "dm-haiku>=0.0.10",
        "brax>=0.9.1",
        "tensorflow-probability>=0.15.0",
        "mediapy>=1.1.4",
    ],
    license="LICENSE",
)
