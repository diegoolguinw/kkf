"""Kernel Koopman Kalman Filter (KKF) - A library for nonlinear state estimation.

This package provides a complete implementation of the Koopman-Kalman Filter,
combining Koopman operator theory with Kalman filtering for accurate state
estimation in nonlinear dynamical systems.

Main Components
---------------
- systems : Dynamical system definitions
- koopman : Koopman operator approximation via kEDMD
- covariances : Covariance matrix computations
- solution : Filter solution data structure
- filter : Main filtering algorithm

Quick Start
-----------
>>> from KKF import DynamicalSystem, KoopmanOperator, apply_koopman_kalman_filter
>>> # or
>>> from KKF.filter import apply_koopman_kalman_filter
"""

# Core classes and utilities
from .covariances import (
    compute_initial_covariance,
    compute_dynamics_covariance,
    compute_observation_covariance,
)
from .systems import DynamicalSystem, create_additive_system
from .koopman import KoopmanOperator
from .solution import KoopmanKalmanFilterSolution

# Main filtering function
from .filter import apply_koopman_kalman_filter

__all__ = [
    # Classes
    "DynamicalSystem",
    "KoopmanOperator",
    "KoopmanKalmanFilterSolution",
    # Functions
    "create_additive_system",
    "apply_koopman_kalman_filter",
    "compute_initial_covariance",
    "compute_dynamics_covariance",
    "compute_observation_covariance",
]

__version__ = "0.2.0"