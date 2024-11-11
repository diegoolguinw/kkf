from dataclasses import dataclass
import numpy as np
from numpy.linalg import pinv, cholesky
from numpy.typing import NDArray

from typing import Any, Tuple, Optional, Callable

from .covariances import compute_initial_covariance, compute_dynamics_covariance, compute_observation_covariance
from .KKKFsol import KoopmanKalmanFilterSolution

def apply_koopman_kalman_filter(
    koopman_operator: Any,
    observations: NDArray[np.float64],
    initial_distribution: Any,
    n_features: int,
    noise_samples: int = 100
) -> 'KoopmanKalmanFilterSolution':
    """
    Implementation of the Koopman-Kalman Filter algorithm.
    
    This function combines Koopman operator theory with Kalman filtering to perform
    state estimation for nonlinear dynamical systems. It operates by lifting the
    state space to a higher-dimensional feature space where the dynamics are
    approximately linear.
    
    Parameters
    ----------
    koopman_operator : object
        Object containing Koopman operator methods and attributes:
        - kEDMD(n_features): Extended Dynamic Mode Decomposition method
        - dyn_sys: Dynamical system object
        - U, B, C: Matrices for Koopman approximation
        - phi: Feature map function
    observations : np.ndarray
        Array of shape (n_timesteps, n_outputs) containing system measurements.
    initial_distribution : object
        Distribution object for initial state with methods:
        - mean: Returns mean of initial state
        - rvs: Random sampling method
    n_features : int
        Number of features in the lifted space.
    noise_samples : int, optional
        Number of samples for noise covariance estimation. Default is 100.
        
    Returns
    -------
    KoopmanKalmanFilterSolution
        Object containing all filter estimates and covariances.
        
    Notes
    -----
    The algorithm consists of three main phases:
    1. Initialization:
       - Sets up initial states and covariances
       - Computes Koopman approximation
    
    2. Prediction Step:
       - Projects state forward using system dynamics
       - Updates covariances using Koopman operator
    
    3. Update Step:
       - Incorporates new measurements
       - Updates state and covariance estimates
    
    The filter operates in both the original state space (x) and the lifted
    feature space (z), maintaining estimates and covariances in both spaces.
    """
    # Compute Koopman approximation
    koopman_operator.compute_edmd(n_features)
    dynamical_system = koopman_operator.dynamical_system

    # Extract system and Koopman components
    state_dynamics = dynamical_system.f
    measurement_model = dynamical_system.g
    U, B, C = koopman_operator.U, koopman_operator.B, koopman_operator.C
    phi = koopman_operator.phi

    ### Initialization ###
    x0 = initial_distribution.mean  # Initial state estimate
    z0 = phi(x0)  # Initial state in feature space

    # Get problem dimensions
    n_timesteps, n_states = observations.shape[0], len(x0)
    n_outputs = observations.shape[1]

    # Initialize state arrays
    x_minus, x_plus = _initialize_state_arrays(n_timesteps, n_states)
    z_minus, z_plus = _initialize_state_arrays(n_timesteps, n_features)
    
    # Initialize covariance arrays
    Px_minus, Px_plus = _initialize_covariance_arrays(n_timesteps, n_states)
    Pz_minus, Pz_plus = _initialize_covariance_arrays(n_timesteps, n_features)
    
    # Initialize filter matrices
    S = np.zeros((n_timesteps, n_outputs, n_outputs))  # Innovation covariance
    K = np.zeros((n_timesteps, n_features, n_outputs))  # Kalman gain

    # Set initial conditions
    x_minus[0, :], x_plus[0, :] = x0, x0
    z_minus[0, :], z_plus[0, :] = z0, z0

    # Initialize covariances
    initial_covariance = compute_initial_covariance(
        x_minus[0, :], n_features, initial_distribution, 
        koopman_operator, noise_samples
    )
    Pz_minus[0, :, :], Pz_plus[0, :, :] = initial_covariance, initial_covariance

    ### Main Filter Loop ###
    for t in range(1, n_timesteps):
        # Prediction Step
        x_minus[t], z_minus[t], Pz_minus[t] = _prediction_step(
            x_plus[t-1], z_plus[t-1], Pz_plus[t-1],
            state_dynamics, phi, U, dynamical_system,
            koopman_operator, n_features, noise_samples
        )

        # Update Step
        x_plus[t], z_plus[t], Pz_plus[t], S[t], K[t] = _update_step(
            x_minus[t], z_minus[t], Pz_minus[t],
            observations[t], measurement_model,
            C, B, dynamical_system, noise_samples
        )

        # Update state-space covariances
        Px_minus[t], Px_plus[t] = _update_state_covariances(
            Pz_minus[t], Pz_plus[t], B
        )

    return KoopmanKalmanFilterSolution(
        x_plus, x_minus, Pz_plus, Pz_minus, Px_plus, Px_minus, S, K
    )

def _initialize_state_arrays(
    n_timesteps: int, 
    n_dim: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Initialize arrays for state estimates."""
    return (np.zeros((n_timesteps, n_dim)), 
            np.zeros((n_timesteps, n_dim)))

def _initialize_covariance_arrays(
    n_timesteps: int, 
    n_dim: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Initialize arrays for covariance matrices."""
    return (np.zeros((n_timesteps, n_dim, n_dim)), 
            np.zeros((n_timesteps, n_dim, n_dim)))

def _prediction_step(
    x_prev: NDArray[np.float64],
    z_prev: NDArray[np.float64],
    Pz_prev: NDArray[np.float64],
    state_dynamics: Callable,
    phi: Callable,
    U: NDArray[np.float64],
    dynamical_system: Any,
    koopman_operator: Any,
    n_features: int,
    noise_samples: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform the prediction step of the filter."""
    # Compute dynamics covariance
    Qz = compute_dynamics_covariance(
        x_prev, n_features, dynamical_system, 
        koopman_operator, noise_samples
    )
    
    # Predict states
    x_pred = state_dynamics(x_prev, )
    z_pred = phi(x_pred)
    
    # Predict covariance
    Pz_pred = U @ Pz_prev @ U.T + Qz
    
    return x_pred, z_pred, Pz_pred

def _update_step(
    x_minus: NDArray[np.float64],
    z_minus: NDArray[np.float64],
    Pz_minus: NDArray[np.float64],
    observation: NDArray[np.float64],
    measurement_model: Callable,
    C: NDArray[np.float64],
    B: NDArray[np.float64],
    dynamical_system: Any,
    noise_samples: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], 
           NDArray[np.float64], NDArray[np.float64]]:
    """Perform the update step of the filter."""
    # Compute measurement covariance
    Rz = compute_observation_covariance(
        x_minus, len(observation), dynamical_system, noise_samples
    )
    
    # Compute innovation
    innovation = observation - measurement_model(x_minus)
    
    # Compute innovation covariance
    S = C @ Pz_minus @ C.T + Rz
    
    # Compute Kalman gain
    K = Pz_minus @ C.T @ pinv(cholesky(S))
    
    # Update states
    z_plus = z_minus + K @ innovation
    x_plus = B @ z_plus
    
    # Update covariance
    Pz_plus = (np.eye(len(z_minus)) - K @ C) @ Pz_minus
    
    return x_plus, z_plus, Pz_plus, S, K

def _update_state_covariances(
    Pz_minus: NDArray[np.float64],
    Pz_plus: NDArray[np.float64],
    B: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update covariances in the state space."""
    Px_minus = B @ Pz_minus @ B.T
    Px_plus = B @ Pz_plus @ B.T
    return Px_minus, Px_plus