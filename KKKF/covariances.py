import numpy as np
from typing import Callable, Any, Union
from scipy.stats import rv_continuous
from numpy.typing import NDArray

def compute_initial_covariance(
    x: NDArray[np.float64],
    n_features: int,
    initial_distribution: rv_continuous,
    koopman_operator: Any,
    n_samples: int
) -> NDArray[np.float64]:
    """
    Compute the covariance matrix for the initial distribution in feature space.
    
    This function samples from an initial distribution and computes the covariance
    matrix of the transformed samples using a Koopman operator's feature map.
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector or reference point.
    n_features : int
        Number of features in the transformed space.
    initial_distribution : scipy.stats.rv_continuous
        Initial probability distribution to sample from.
    koopman_operator : object
        Object containing the feature map phi method for state transformation.
        Must have a method phi(x) that maps states to feature space.
    n_samples : int
        Number of samples to use for covariance estimation.
        
    Returns
    -------
    np.ndarray
        Covariance matrix of size (n_features, n_features) in the transformed space.
        
    Notes
    -----
    The function performs the following steps:
    1. Samples from the initial distribution
    2. Applies the Koopman operator's feature map to each sample
    3. Computes the covariance matrix of the transformed samples
    """
    # Sample from initial distribution
    samples = initial_distribution.rvs(size=n_samples).reshape((n_samples, len(x)))
    
    # Initialize array for transformed samples
    transformed_samples = np.zeros((n_samples, n_features))
    
    # Transform each sample using the feature map
    for i in range(n_samples):
        transformed_samples[i, :] = koopman_operator.phi(samples[i,:])
    
    # Compute and return covariance matrix
    return np.cov(transformed_samples, rowvar=False)

def compute_dynamics_covariance(
    x: NDArray[np.float64],
    n_features: int,
    dynamics: Any,
    koopman_operator: Any,
    n_samples: int
) -> NDArray[np.float64]:
    """
    Compute the covariance matrix for the system dynamics in feature space.
    
    This function samples from the dynamics distribution, applies the system
    dynamics, and computes the covariance matrix of the transformed results.
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector or reference point.
    n_features : int
        Number of features in the transformed space.
    dynamics : object
        Dynamical system object containing:
        - dist_dyn: scipy.stats.rv_continuous distribution for dynamics noise
        - dynamics(x, w): method implementing the system dynamics
    koopman_operator : object
        Object containing the feature map phi method for state transformation.
        Must have a method phi(x) that maps states to feature space.
    n_samples : int
        Number of samples to use for covariance estimation.
        
    Returns
    -------
    np.ndarray
        Covariance matrix of size (n_features, n_features) in the transformed space.
        
    Notes
    -----
    The function performs the following steps:
    1. Samples from the dynamics noise distribution
    2. Applies the system dynamics to each sample
    3. Transforms the results using the Koopman operator's feature map
    4. Computes the covariance matrix of the transformed samples
    """
    # Sample from dynamics distribution
    noise_samples = dynamics.dist_dyn.rvs(size=n_samples).reshape((n_samples, len(x)))
    
    # Initialize array for transformed samples
    transformed_samples = np.zeros((n_samples, n_features))
    
    # Apply dynamics and transform each sample
    for i in range(n_samples):
        state_evolution = dynamics.dynamics(x, noise_samples[i,:])
        transformed_samples[i, :] = koopman_operator.phi(state_evolution)
    
    # Compute and return covariance matrix
    return np.cov(transformed_samples, rowvar=False)

def compute_observation_covariance(
    x: NDArray[np.float64],
    n_outputs: int,
    dynamics: Any,
    n_samples: int
) -> NDArray[np.float64]:
    """
    Compute the covariance matrix for the observation/measurement process.
    
    This function samples from the measurement noise distribution and computes
    the covariance matrix of the measurement process.
    
    Parameters
    ----------
    x : np.ndarray
        Current state vector or reference point.
    n_outputs : int
        Number of measurement outputs.
    dynamics : object
        Dynamical system object containing:
        - dist_obs: scipy.stats.rv_continuous distribution for measurement noise
        - measurements(x, w): method implementing the measurement process
    n_samples : int
        Number of samples to use for covariance estimation.
        
    Returns
    -------
    np.ndarray
        Covariance matrix of size (n_outputs, n_outputs) for the measurement process.
        
    Notes
    -----
    The function performs the following steps:
    1. Samples from the measurement noise distribution
    2. Applies the measurement function to each sample
    3. Computes the covariance matrix of the measurements
    """
    # Sample from measurement distribution
    noise_samples = dynamics.dist_obs.rvs(size=n_samples).reshape((n_samples, n_outputs))
    
    # Initialize array for measurements
    measurement_samples = np.zeros((n_samples, n_outputs))
    
    # Apply measurement function to each sample
    for i in range(n_samples):
        measurement_samples[i, :] = dynamics.measurements(x, noise_samples[i,:])
    
    # Compute and return covariance matrix
    return np.cov(measurement_samples, rowvar=False)