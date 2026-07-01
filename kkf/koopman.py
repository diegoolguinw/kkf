"""Koopman operator approximation module.

Implements kernel-based Extended Dynamic Mode Decomposition (kEDMD)
for computing Koopman operator approximations.
"""

import warnings
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor

from .systems import DynamicalSystem


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
        dynamical_system: DynamicalSystem,
    ) -> None:
        self.kernel_function = kernel_function
        self.dynamical_system = dynamical_system
        self.X: Optional[NDArray[np.float64]] = None
        self.phi: Optional[Callable] = None
        self.U: Optional[NDArray[np.float64]] = None
        self.G: Optional[NDArray[np.float64]] = None
        self.C: Optional[NDArray[np.float64]] = None
        self.B: Optional[NDArray[np.float64]] = None

    def compute_edmd(
        self,
        n_features: int,
        optimize: bool = False,
        n_restarts_optimizer: int = 10,
        reg: float = 1e-10,
    ) -> None:
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
            the provided kernel function will be used without optimization. Default is False
            (fast; enable it explicitly to fit kernel hyperparameters).
        n_restarts_optimizer : int
            Number of restarts for the optimizer. If optimize is False, will be ignored. Default is 10.
        reg : float
            Tikhonov (jitter) regularization added to the diagonal of the Gram matrix
            before inversion, scaled by its mean diagonal. Kernel Gram matrices are often
            ill-conditioned; a small positive value keeps the inversion numerically stable.
            Increase it if you see unstable estimates. Default is 1e-10.

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
            self.optimize_kernel(X=self.X, n_restarts_optimizer=n_restarts_optimizer)

        # Define feature map
        self.phi = lambda x: self.kernel_function(x, self.X)[0]

        # Compute Gram matrix, regularizing the diagonal to keep the inversion stable
        self.G = self.kernel_function(self.X, self.X)
        jitter = reg * np.mean(np.diag(self.G))
        G_inv = np.linalg.inv(self.G + jitter * np.eye(self.G.shape[0]))

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

    def optimize_kernel(self, X: NDArray[np.float64], n_restarts_optimizer: int = 10) -> None:
        """
        Optimize the kernel function for the Koopman operator.

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
        gp = GaussianProcessRegressor(
            kernel=self.kernel_function, n_restarts_optimizer=n_restarts_optimizer
        )
        gp.fit(X.T, y.T)

        # Update with optimal kernel
        self.kernel_function = gp.kernel_

    def opt_kernel(self, X: NDArray[np.float64], n_restarts_optimizer: int = 10) -> None:
        """Deprecated alias for :meth:`optimize_kernel`.

        .. deprecated::
            Use :meth:`optimize_kernel` instead. This alias will be removed in a
            future major release.
        """
        warnings.warn(
            "KoopmanOperator.opt_kernel is deprecated; use optimize_kernel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.optimize_kernel(X=X, n_restarts_optimizer=n_restarts_optimizer)

    # Descriptive, discoverable aliases for the single-letter matrices computed by
    # ``compute_edmd``. The short names (U, G, C, B, X, phi) are kept for backward
    # compatibility and mathematical convention.
    @property
    def koopman_matrix(self) -> Optional[NDArray[np.float64]]:
        """Koopman operator approximation (alias of ``U``)."""
        return self.U

    @property
    def gram_matrix(self) -> Optional[NDArray[np.float64]]:
        """Kernel Gram matrix (alias of ``G``)."""
        return self.G

    @property
    def output_matrix(self) -> Optional[NDArray[np.float64]]:
        """Feature-to-output matrix (alias of ``C``)."""
        return self.C

    @property
    def state_matrix(self) -> Optional[NDArray[np.float64]]:
        """Feature-to-state matrix (alias of ``B``)."""
        return self.B

    @property
    def dictionary(self) -> Optional[NDArray[np.float64]]:
        """Dictionary of sampled states (alias of ``X``)."""
        return self.X

    @property
    def feature_map(self) -> Optional[Callable]:
        """Feature map function (alias of ``phi``)."""
        return self.phi
