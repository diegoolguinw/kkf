from typing import Callable, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_continuous
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
        
    def compute_edmd(self, n_features: int) -> None:
        """
        Compute the kernel-based Extended Dynamic Mode Decomposition (kEDMD).
        
        This method constructs finite-dimensional approximations of the Koopman
        operator and associated matrices using kernel methods.
        
        Parameters
        ----------
        n_features : int
            Number of features to use in the approximation.
            
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
        nx, ny = self.dynamical_system.nx, self.dynamical_system.ny
        
        # Generate dictionary points
        self.X = self.dynamical_system.sample_state(n_features)
        
        # Define feature map
        self.phi = lambda x: self.kernel_function(x, self.X).reshape((n_features,))
        
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