from setuptools import setup, find_packages

requirements = [
    "torch>=1.7",
    "gpytorch>=1.3",
    "botorch>=0.3.3",
    "scipy>=1.6.0",
    "jupyter",
    "matplotlib",
]

dev_requires = [
    "black",
    "flake8",
    "pytest",
    "coverage",
]

setup(
    name="contextual_rs",
    version="1.0",
    description="Contextual Ranking & Selection with Generalized PCS",
    author="Sait Cakmak",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requires},
)
