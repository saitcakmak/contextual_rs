from setuptools import setup, find_packages

requirements = [
    "torch>=1.7.1",
    "gpytorch>=1.4",
    "botorch>=0.4",
    "scipy>=1.6.0",
    "jupyter",
    "matplotlib",
    "pandas",
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
