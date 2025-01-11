from setuptools import setup, find_packages

setup(
    name="kkf",
    version="0.11",
    packages=find_packages(),
    install_requires=[
        "numpy", 
        "scipy", 
        "scikit-learn"
    ],
    author="Diego Olguin-Wende",
    author_email="dolguin@dim.uchile.cl",  
    description="kkf: a library for Python implementation of Kernel-Koopman-Kalman Filter.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/diegoolguinw/kkf",  
    classifiers=[                      
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)