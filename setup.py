from setuptools import setup, find_packages

# Install with 'pip install -e .'

setup(
    name="xpag",
    version="0.1.0",
    author="Nicolas Perrin-Gilbert",
    description="xpag: Exploring Agents",
    url="https://github.com/perrin-isir/xpag",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.8.0",
        "numpy>=1.21.5",
        "matplotlib>=3.1.3",
        "joblib>=1.1.0",
        "gym>=0.22.0",
        "torch>=1.10.0",
        "Pillow>=9.0.1",
        "ipywidgets>=7.6.5",
        "tensorflow-probability>=0.15.0",
    ],
    license="LICENSE",
)
