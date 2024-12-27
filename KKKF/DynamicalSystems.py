from typing import Callable, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_continuous

class DynamicalSystem:
    """
    A class representing a general dynamical system with state and measurement equations.
    
    This class encapsulates the dynamics, measurements, and associated probability
    distributions for both state and measurement noise. Can be considered as:

    Discrete time: 
        Dynamics: x_{k+1} = f(x_{k}) + w_{k}
        Measurements: y_{k} = g(x_{k}) + v_{k}

    Continous time:
        Dynamics: x'(t) = f(x(t)) + w(t)
        Measurements: y(t) = g(x(t)) + v(t)
    
    Attributes
    ----------
    nx : int
        Dimension of the state space.
    ny : int
        Dimension of the measurement/output space.
    f : Callable
        State transition function (dynamics).
    g : Callable
        Measurement/output function.
    dist_X : rv_continuous
        Probability distribution for initial states.
    dist_dyn : rv_continuous
        Probability distribution for dynamics noise.
    dist_obs : rv_continuous
        Probability distribution for measurement noise.
    discrete_time : bool
        Indicates if the system is in discrete time, if False the system is in continous time.
        
    Notes
    -----
    The functions f and g should have the following signatures:
    - f(x: ndarray, w: ndarray) -> ndarray
    - g(x: ndarray, v: ndarray) -> ndarray
    where:
    - x is the state vector
    - w is the process noise
    - v is the measurement noise
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        g: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        dist_X: rv_continuous,
        dist_dyn: rv_continuous,
        dist_obs: rv_continuous,
        discrete_time: bool
    ):
        self.nx = nx
        self.ny = ny
        self.f = f
        self.g = g
        self.dist_X = dist_X
        self.dist_dyn = dist_dyn
        self.dist_obs = dist_obs
        self.discrete_time = discrete_time
        
    def dynamics(self, x: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply the state transition function.
        
        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        w : np.ndarray
            Process noise vector.
            
        Returns
        -------
        np.ndarray
            Next state vector.
        """
        return self.f(x) + w
    
    def measurements(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply the measurement function.
        
        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        v : np.ndarray
            Measurement noise vector.
            
        Returns
        -------
        np.ndarray
            Measurement/output vector.
        """
        return self.g(x) + v
    
    def sample_state(self, size: int = 1) -> NDArray[np.float64]:
        """
        Sample from the state distribution.
        
        Parameters
        ----------
        size : int, optional
            Number of samples to draw. Default is 1.
            
        Returns
        -------
        np.ndarray
            Sampled state(s).
        """
        return self.dist_X.rvs(size=size).reshape((size, self.nx))

def create_additive_system(
    nx: int,
    ny: int,
    f: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    g: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    dist_X: rv_continuous,
    dist_dyn: rv_continuous,
    dist_obs: rv_continuous,
    N_samples: int
) -> DynamicalSystem:
    """
    Create an additive dynamical system where noise is added to the state and observation functions.
    
    Parameters
    ----------
    nx : int
        Dimension of the state space.
    ny : int
        Dimension of the measurement space.
    f : Callable
        State transition function (with noise).
    g : Callable
        Measurement function (with noise).
    dist_X : rv_continuous
        Initial state distribution.
    dist_dyn : rv_continuous
        Dynamics noise distribution.
    dist_obs : rv_continuous
        Measurement noise distribution.
    N_samples: int
        Number of samples to generate empirical mean of dynamics and observation
        
    Returns
    -------
    DynamicalSystem
        New dynamical system instance with additive noise.
        
    Notes
    -----
    The resulting system has the form:
    x[k+1] = f(x[k]) + w[k]
    y[k] = g(x[k]) + v[k]
    where w and v are noise terms.
    """
    new_f = lambda x: np.mean([f(x, w) for w in dist_dyn.rvs(N_samples)])
    new_g = lambda x: np.mean([g(x, v) for v in dist_obs.rvs(N_samples)])
    return DynamicalSystem(nx, ny, new_f, new_g, dist_X, dist_dyn, dist_obs)