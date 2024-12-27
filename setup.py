from setuptools import setup, find_packages

setup(
    name="KKKF",
    version="0.5",
    packages=find_packages(),
    install_requires=[
        "numpy", 
        "scipy", 
        "scikit-learn"
    ],
    author="Diego Olguin-Wende",
    author_email="dolguin@dim.uchile.cl",  
    description="KKKF: a library for Python implementation of Kernel-Koopman-Kalman Filter.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/diegoolguinw/KKKF",  
    classifiers=[                      
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)