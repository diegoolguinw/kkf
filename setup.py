from setuptools import setup, find_packages

setup(
    name="KKKF",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "scikit-learn"
    ],
    author="Diego Olguin-Wende",
    description="KKKF: a library for Python implementation of Kernel-Koopman-Kalman Filter.",
)
