"""Comprehensive test suite for KKF library core functionality."""

import numpy as np
import pytest
from scipy import stats
from sklearn.gaussian_process.kernels import Matern, RBF

from KKF import (
    DynamicalSystem,
    create_additive_system,
    KoopmanOperator,
    KoopmanKalmanFilterSolution,
    compute_initial_covariance,
    compute_dynamics_covariance,
    compute_observation_covariance,
)
from KKF.applyKKF import apply_koopman_kalman_filter


class TestDynamicalSystem:
    """Tests for DynamicalSystem class."""

    @pytest.fixture
    def simple_system(self) -> DynamicalSystem:
        """Create a simple linear dynamical system for testing."""
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]
        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        return DynamicalSystem(
            nx, ny, f, g, dist_X, dist_dyn, dist_obs, discrete_time=True
        )

    def test_dynamical_system_initialization(self, simple_system: DynamicalSystem) -> None:
        """Test DynamicalSystem initialization."""
        assert simple_system.nx == 2
        assert simple_system.ny == 1
        assert simple_system.discrete_time is True

    def test_dynamics_application(self, simple_system: DynamicalSystem) -> None:
        """Test dynamics method."""
        x = np.array([1.0, 2.0])
        w = np.array([0.1, 0.2])
        result = simple_system.dynamics(x, w)

        expected = np.array([0.9 * 1.0 + 0.1, 0.9 * 2.0 + 0.2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_measurements_application(self, simple_system: DynamicalSystem) -> None:
        """Test measurements method."""
        x = np.array([1.0, 2.0])
        v = np.array([0.05])
        result = simple_system.measurements(x, v)

        expected = np.array([1.0 + 0.05])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sample_state_shape(self, simple_system: DynamicalSystem) -> None:
        """Test sample_state returns correct shape."""
        samples = simple_system.sample_state(size=100)
        assert samples.shape == (100, 2)

    def test_sample_state_single(self, simple_system: DynamicalSystem) -> None:
        """Test sample_state with single sample."""
        sample = simple_system.sample_state(size=1)
        assert sample.shape == (1, 2)

    def test_continuous_time_system(self) -> None:
        """Test continuous time system flag."""
        nx, ny = 1, 1
        f = lambda x: -x
        g = lambda x: x
        dist = stats.norm()

        system = DynamicalSystem(
            nx, ny, f, g, dist, dist, dist, discrete_time=False
        )
        assert system.discrete_time is False


class TestCovarianceFunctions:
    """Tests for covariance computation functions."""

    @pytest.fixture
    def test_setup(self):
        """Set up a test system for covariance tests."""
        nx, ny, n_features = 2, 1, 5
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn_sys = DynamicalSystem(
            nx, ny, f, g, dist_X, dist_dyn, dist_obs, discrete_time=True
        )

        kernel = Matern(length_scale=1.0, nu=0.5)
        koopman = KoopmanOperator(kernel, dyn_sys)

        # Compute EDMD to set up the Koopman operator
        x_init = np.array([0.5, 0.5])
        koopman.compute_edmd(n_features, optimize=False)

        return {
            "dyn_sys": dyn_sys,
            "koopman": koopman,
            "n_features": n_features,
            "x": x_init,
        }

    def test_initial_covariance_shape(self, test_setup: dict) -> None:
        """Test initial covariance has correct shape."""
        cov = compute_initial_covariance(
            test_setup["x"],
            test_setup["n_features"],
            test_setup["dyn_sys"].dist_X,
            test_setup["koopman"],
            n_samples=100,
        )
        assert cov.shape == (test_setup["n_features"], test_setup["n_features"])

    def test_initial_covariance_symmetry(self, test_setup: dict) -> None:
        """Test initial covariance is symmetric."""
        cov = compute_initial_covariance(
            test_setup["x"],
            test_setup["n_features"],
            test_setup["dyn_sys"].dist_X,
            test_setup["koopman"],
            n_samples=100,
        )
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_dynamics_covariance_shape(self, test_setup: dict) -> None:
        """Test dynamics covariance has correct shape."""
        cov = compute_dynamics_covariance(
            test_setup["x"],
            test_setup["n_features"],
            test_setup["dyn_sys"],
            test_setup["koopman"],
            n_samples=100,
        )
        assert cov.shape == (test_setup["n_features"], test_setup["n_features"])

    def test_observation_covariance_shape(self, test_setup: dict) -> None:
        """Test observation covariance has correct shape."""
        ny = test_setup["dyn_sys"].ny
        cov = compute_observation_covariance(
            test_setup["x"], ny, test_setup["dyn_sys"], n_samples=100
        )
        assert cov.shape == (ny, ny)


class TestKoopmanOperator:
    """Tests for KoopmanOperator class."""

    @pytest.fixture
    def koopman_setup(self):
        """Set up a Koopman operator for testing."""
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn_sys = DynamicalSystem(
            nx, ny, f, g, dist_X, dist_dyn, dist_obs, discrete_time=True
        )

        kernel = Matern(length_scale=1.0, nu=0.5)
        return KoopmanOperator(kernel, dyn_sys)

    def test_koopman_initialization(self, koopman_setup: KoopmanOperator) -> None:
        """Test KoopmanOperator initialization."""
        assert koopman_setup.X is None
        assert koopman_setup.phi is None
        assert koopman_setup.U is None

    def test_compute_edmd(self, koopman_setup: KoopmanOperator) -> None:
        """Test EDMD computation."""
        n_features = 10
        koopman_setup.compute_edmd(n_features, optimize=False)

        assert koopman_setup.X is not None
        assert koopman_setup.phi is not None
        assert koopman_setup.U is not None
        assert koopman_setup.X.shape[0] == n_features

    def test_feature_dimension_before_compute(
        self, koopman_setup: KoopmanOperator
    ) -> None:
        """Test get_feature_dimension returns None before EDMD."""
        assert koopman_setup.get_feature_dimension() is None

    def test_feature_dimension_after_compute(
        self, koopman_setup: KoopmanOperator
    ) -> None:
        """Test get_feature_dimension after EDMD."""
        n_features = 10
        koopman_setup.compute_edmd(n_features, optimize=False)
        assert koopman_setup.get_feature_dimension() == n_features

    def test_feature_map_output_shape(self, koopman_setup: KoopmanOperator) -> None:
        """Test feature map produces correct output shape."""
        n_features = 10
        koopman_setup.compute_edmd(n_features, optimize=False)

        x = np.array([0.5, 0.5])
        phi_x = koopman_setup.phi(x)

        assert len(phi_x) == n_features


class TestKoopmanKalmanFilterSolution:
    """Tests for KoopmanKalmanFilterSolution. """

    @pytest.fixture
    def solution(self):
        """Create a test solution."""
        n_timesteps, nx, ny, n_features = 10, 2, 1, 5

        x_plus = np.random.randn(n_timesteps, nx)
        x_minus = x_plus + np.random.randn(n_timesteps, nx) * 0.1
        Pz_plus = np.tile(np.eye(n_features), (n_timesteps, 1, 1))
        Pz_minus = Pz_plus + np.random.randn(n_timesteps, n_features, n_features) * 0.01
        Px_plus = np.tile(np.eye(nx), (n_timesteps, 1, 1))
        Px_minus = Px_plus + np.random.randn(n_timesteps, nx, nx) * 0.01
        S = np.tile(np.eye(ny), (n_timesteps, 1, 1))
        K = np.random.randn(n_timesteps, n_features, ny)

        return KoopmanKalmanFilterSolution(
            x_plus, x_minus, Pz_plus, Pz_minus, Px_plus, Px_minus, S, K
        )

    def test_solution_initialization(
        self, solution: KoopmanKalmanFilterSolution
    ) -> None:
        """Test solution initialization."""
        assert solution.x_plus.shape[0] == 10
        assert solution.x_plus.shape[1] == 2

    def test_get_state_dimension(self, solution: KoopmanKalmanFilterSolution) -> None:
        """Test get_state_dimension method."""
        assert solution.get_state_dimension() == 2

    def test_get_feature_dimension(self, solution: KoopmanKalmanFilterSolution) -> None:
        """Test get_feature_dimension method."""
        assert solution.get_feature_dimension() == 5

    def test_solution_post_init_ensures_arrays(self) -> None:
        """Test __post_init__ converts lists to arrays."""
        solution = KoopmanKalmanFilterSolution(
            x_plus=[1.0, 2.0],
            x_minus=[0.9, 1.9],
            Pz_plus=[[1.0, 0.0], [0.0, 1.0]],
            Pz_minus=[[1.0, 0.0], [0.0, 1.0]],
            Px_plus=[[1.0, 0.0], [0.0, 1.0]],
            Px_minus=[[1.0, 0.0], [0.0, 1.0]],
            S=[[1.0]],
            K=[[0.1]],
        )

        assert isinstance(solution.x_plus, np.ndarray)
        assert isinstance(solution.Pz_plus, np.ndarray)


@pytest.mark.slow
class TestIntegration:
    """Integration tests for the complete KKF workflow."""

    def test_simple_sir_model(self) -> None:
        """Test KKF with simple SIR model."""
        # System setup
        beta, gamma = 0.12, 0.04

        def f(x):
            return x + np.array([-beta * x[0] * x[1], beta * x[0] * x[1] - gamma * x[1], gamma * x[1]])

        def g(x):
            return np.array([x[1]])

        nx, ny = 3, 1
        N = 50

        X_dist = stats.dirichlet(alpha=np.ones(nx))
        dyn_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-5 * np.eye(3))
        obs_dist = stats.multivariate_normal(mean=np.zeros(ny), cov=1e-3 * np.eye(1))

        dyn = DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist, discrete_time=True)

        # Generate synthetic data
        iters = 20
        x0 = np.array([0.9, 0.1, 0.0])
        x = np.zeros((iters, nx))
        y = np.zeros((iters, ny))

        x[0] = x0
        y[0] = g(x[0]) + obs_dist.rvs()

        for i in range(1, iters):
            x[i] = f(x[i - 1]) + dyn_dist.rvs()
            y[i] = g(x[i]) + obs_dist.rvs()

        # Setup Koopman KF
        k = Matern(length_scale=N ** (-1 / nx), nu=0.5)
        Koop = KoopmanOperator(k, dyn)

        x0_prior = np.array([0.8, 0.15, 0.05])
        d0 = stats.multivariate_normal(mean=x0_prior, cov=0.1 * np.eye(3))

        # Apply filter
        sol = apply_koopman_kalman_filter(
            Koop, y, d0, N, optimize=False, noise_samples=50
        )

        # Basic assertions
        assert sol.x_plus.shape == (iters, nx)
        assert sol.x_minus.shape == (iters, nx)
        assert np.all(np.isfinite(sol.x_plus))
        assert np.all(np.isfinite(sol.x_minus))

    def test_filter_consistency(self) -> None:
        """Test filter produces consistent results."""
        # Setup
        f = lambda x: 0.9 * x
        g = lambda x: x

        nx, ny = 1, 1
        dist = stats.norm(scale=0.1)
        dyn = DynamicalSystem(nx, ny, f, g, dist, dist, dist, discrete_time=True)

        # Generate observations
        iters = 10
        observations = np.random.randn(iters, ny) * 0.1

        # Setup and apply filter
        kernel = RBF(length_scale=1.0)
        koop = KoopmanOperator(kernel, dyn)
        d0 = stats.norm(scale=0.1)

        sol = apply_koopman_kalman_filter(
            koop, observations, d0, n_features=5, optimize=False, noise_samples=20
        )

        # Verify output structure
        assert isinstance(sol, KoopmanKalmanFilterSolution)
        assert sol.x_plus.shape == (iters, nx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
