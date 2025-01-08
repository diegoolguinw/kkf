from typing import Callable, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_continuous
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor
from .DynamicalSystems import DynamicalSystem, create_additive_system

class KoopmanOperator:
    """
    Implementation of Koopman operator approximation using kernel-based Extended 
    Dynamic Mode Decomposition (kEDMD).
    
    This class provides methods to compute finite-dimensional approximations of the
    Koopman operator for nonlinear dynamical systems.
    
    Attributes
    ----------
    kernel_function : Callable
        Kernel function for computing feature space mappings.
    dynamical_system : DynamicalSystem
        The underlying dynamical system.
    X : Optional[np.ndarray]
        Dictionary of states used for kernel computations.
    phi : Optional[Callable]
        Feature map function.
    U : Optional[np.ndarray]
        Koopman operator matrix.
    G : Optional[np.ndarray]
        Gram matrix.
    C : Optional[np.ndarray]
        Output matrix.
    B : Optional[np.ndarray]
        State-to-feature space transformation matrix.
        
    Notes
    -----
    The Koopman operator framework lifts nonlinear dynamics to a linear setting
    in a higher-dimensional feature space. This implementation uses kernel methods
    to compute the necessary feature spaces and operators.
    """
    
    def __init__(
        self,
        kernel_function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        dynamical_system: DynamicalSystem
    ):
        self.kernel_function = kernel_function
        self.dynamical_system = dynamical_system
        self.X: Optional[NDArray[np.float64]] = None
        self.phi: Optional[Callable] = None
        self.U: Optional[NDArray[np.float64]] = None
        self.G: Optional[NDArray[np.float64]] = None
        self.C: Optional[NDArray[np.float64]] = None
        self.B: Optional[NDArray[np.float64]] = None
        
    def compute_edmd(self, n_features: int, optimize: bool = True, n_restarts_optimizer: int = 10) -> None:
        """
        Compute the kernel-based Extended Dynamic Mode Decomposition (kEDMD).
        
        This method constructs finite-dimensional approximations of the Koopman
        operator and associated matrices using kernel methods.
        
        Parameters
        ----------
        n_features : int
            Number of features to use in the approximation.

        optimize : bool
            Whether to optimize the kernel function. If True, the method will
            optimize the kernel function using Gaussian Process Regression. If False, 
            the provided kernel function will be used without optimization. Default is True.

        n_restarts_optimizer : int
            Number of restarts for the optimizer. If optimize is False, will be ignored. Default is 10.
            
        Notes
        -----
        The method performs the following steps:
        1. Generates dictionary points using the state distribution
        2. Constructs the feature map using the kernel function
        3. Computes the Gram matrix and its inverse
        4. Constructs the Koopman operator approximation
        5. Computes output and state transformation matrices
        """
        # Extract system components
        f, g = self.dynamical_system.f, self.dynamical_system.g
        
        # Generate dictionary points
        self.X = self.dynamical_system.sample_state(n_features)

        # Optimize kernel function
        if optimize:
            self.opt_kernel(X=self.X, n_restarts_optimizer=n_restarts_optimizer)
        
        # Define feature map
        self.phi = lambda x: self.kernel_function(x, self.X)[0]
        
        # Compute Gram matrix
        self.G = self.kernel_function(self.X, self.X)
        G_inv = np.linalg.inv(self.G)
        
        # Compute Koopman operator approximation
        next_states = f(self.X.T).T
        self.U = self.kernel_function(self.X, next_states) @ G_inv
        
        # Compute output and state transformation matrices
        self.C = g(self.X.T) @ G_inv
        self.B = self.X.T @ G_inv
        
    def get_feature_dimension(self) -> Optional[int]:
        """
        Get the dimension of the feature space.
        
        Returns
        -------
        Optional[int]
            Dimension of the feature space, or None if EDMD hasn't been computed.
        """
        return self.X.shape[0] if self.X is not None else None

    def opt_kernel(self, X: NDArray[np.float64], n_restarts_optimizer: int = 10) -> None:
        """
        Optimal kernel function for the Koopman operator.
        
        Parameters
        ----------
        X : np.ndarray
            Set of points.

        n_restarts_optimizer : int
            Number of restarts for the optimizer. Default is 10.
            
        Notes
        -----
        This method uses Gaussian Process Regression to optimize the kernel function.
        """
        # Compute the output of the dynamical system
        y = self.dynamical_system.f(X.T).T

        # Fit the Gaussian process
        gp = GaussianProcessRegressor(kernel=self.kernel_function, n_restarts_optimizer=n_restarts_optimizer)
        gp.fit(X.T, y.T)

        # Return the optimal kernel
        self.kernel_function = gp.kernel_
        