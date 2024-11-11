from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Optional

import numpy as np
from numpy.linalg import pinv, cholesky
from typing import Any, Tuple, Optional, Callable
from numpy.typing import NDArray

from .covariances import compute_initial_covariance, compute_dynamics_covariance, compute_observation_covariance

@dataclass
class KoopmanKalmanFilterSolution:
    """
    A class to store the solution of a Koopman-Kalman Filter iteration.
    
    This class maintains the state estimates, covariance matrices, and filter gains
    for both the prior (minus) and posterior (plus) estimates in both state and 
    feature spaces.
    
    Attributes
    ----------
    x_plus : np.ndarray
        Posterior state estimate after measurement update.
    x_minus : np.ndarray
        Prior state estimate from prediction step.
    Pz_plus : np.ndarray
        Posterior covariance matrix in feature space after measurement update.
    Pz_minus : np.ndarray
        Prior covariance matrix in feature space from prediction step.
    Px_plus : np.ndarray
        Posterior covariance matrix in state space after measurement update.
    Px_minus : np.ndarray
        Prior covariance matrix in state space from prediction step.
    S : np.ndarray
        Innovation (residual) covariance matrix.
    K : np.ndarray
        Kalman gain matrix.
        
    Notes
    -----
    The class uses the common Kalman filter notation where:
    - (-) denotes prior estimates before measurement update
    - (+) denotes posterior estimates after measurement update
    - Pz refers to covariances in the feature/transformed space
    - Px refers to covariances in the original state space
    
    Examples
    --------
    >>> solution = KoopmanKalmanFilterSolution(
    ...     x_plus=np.array([1.0, 2.0]),
    ...     x_minus=np.array([0.9, 1.9]),
    ...     Pz_plus=np.eye(2),
    ...     Pz_minus=np.eye(2) * 1.1,
    ...     Px_plus=np.eye(2) * 0.9,
    ...     Px_minus=np.eye(2),
    ...     S=np.eye(2) * 0.5,
    ...     K=np.array([[0.1, 0], [0, 0.1]])
    ... )
    """
    
    x_plus: NDArray[np.float64]
    x_minus: NDArray[np.float64]
    Pz_plus: NDArray[np.float64]
    Pz_minus: NDArray[np.float64]
    Px_plus: NDArray[np.float64]
    Px_minus: NDArray[np.float64]
    S: NDArray[np.float64]
    K: NDArray[np.float64]
    
    def __post_init__(self):
        """Validate the dimensions of the input arrays."""
        # Ensure all inputs are numpy arrays
        for attr in ['x_plus', 'x_minus', 'Pz_plus', 'Pz_minus', 
                    'Px_plus', 'Px_minus', 'S', 'K']:
            value = getattr(self, attr)
            if not isinstance(value, np.ndarray):
                setattr(self, attr, np.array(value))
    
    def get_state_dimension(self) -> int:
        """
        Get the dimension of the state vector.
        
        Returns
        -------
        int
            The dimension of the state vector.
        """
        return len(self.x_plus)
    
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of the feature space.
        
        Returns
        -------
        int
            The dimension of the feature space.
        """
        return self.Pz_plus.shape[0]
    
    def get_estimation_error(self) -> NDArray[np.float64]:
        """
        Calculate the difference between prior and posterior estimates.
        
        Returns
        -------
        np.ndarray
            The difference between posterior and prior state estimates.
        """
        return self.x_plus - self.x_minus
    
    def get_trace_reduction(self) -> float:
        """
        Calculate the reduction in uncertainty as measured by trace of covariance.
        
        Returns
        -------
        float
            The relative reduction in trace of the state covariance matrix.
        """
        trace_minus = np.trace(self.Px_minus)
        trace_plus = np.trace(self.Px_plus)
        return (trace_minus - trace_plus) / trace_minus if trace_minus != 0 else 0.0
    
    def to_dict(self) -> dict:
        """
        Convert the solution to a dictionary format.
        
        Returns
        -------
        dict
            Dictionary containing all solution components.
        """
        return {
            'x_plus': self.x_plus,
            'x_minus': self.x_minus,
            'Pz_plus': self.Pz_plus,
            'Pz_minus': self.Pz_minus,
            'Px_plus': self.Px_plus,
            'Px_minus': self.Px_minus,
            'S': self.S,
            'K': self.K
        }