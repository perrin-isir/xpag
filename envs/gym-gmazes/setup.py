from setuptools import setup, find_packages

# Install with 'pip install -e .'

setup(
    name="gym_gmazes",
    version="0.1.0",
    author="Nicolas Perrin-Gilbert",
    description="gym maze environments",
    url="https://github.com/perrin-isir/xpag/tree/main/envs/gym-gmazes",
    packages=find_packages(
        include=[
            "gym-gmazes",
            "gym-gmazes.*",
            "gym-gmazes.env.*",
        ]
    ),
    install_requires=[
        "gym>=0.22.0",
        "torch>=1.10.1",
    ],
    license="LICENSE",
)
