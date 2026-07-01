"""Utility and edge case tests for KKF library."""

import numpy as np
import pytest
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

from kkf import (
    DynamicalSystem,
    KoopmanOperator,
    compute_dynamics_covariance,
    compute_initial_covariance,
    create_additive_system,
)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_state_dimension_fails(self) -> None:
        """Test that zero state dimension raises a ValueError."""
        f = lambda x: x
        g = lambda x: x
        dist = stats.norm()
        with pytest.raises(ValueError):
            DynamicalSystem(0, 1, f, g, dist, dist, dist, discrete_time=True)

    def test_zero_measurement_dimension_fails(self) -> None:
        """Test that zero measurement dimension raises a ValueError."""
        f = lambda x: x
        g = lambda x: x
        dist = stats.norm()
        with pytest.raises(ValueError):
            DynamicalSystem(1, 0, f, g, dist, dist, dist, discrete_time=True)

    def test_single_state_dimension(self) -> None:
        """Test system with single state dimension."""
        f = lambda x: 0.9 * x
        g = lambda x: x
        dist = stats.norm()

        dyn = DynamicalSystem(1, 1, f, g, dist, dist, dist, discrete_time=True)
        assert dyn.nx == 1
        assert dyn.ny == 1

    def test_high_dimensional_system(self) -> None:
        """Test system with high-dimensional state."""
        nx, ny = 10, 5
        f = lambda x: 0.9 * x
        g = lambda x: x[:ny]
        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist_noise = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist_noise, dist_obs, discrete_time=True)
        assert dyn.nx == 10
        assert dyn.ny == 5

        samples = dyn.sample_state(size=50)
        assert samples.shape == (50, nx)

    def test_measurement_dimension_mismatch(self) -> None:
        """Test that measurement function can have different output dimension."""
        nx, ny = 3, 2
        f = lambda x: 0.9 * x
        g = lambda x: x[:ny]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)

        x = np.random.randn(nx)
        y = dyn.g(x)
        assert y.shape == (ny,)

    def test_noise_addition_correctness(self) -> None:
        """Test that noise is correctly added in dynamics and measurements."""
        nx, ny = 2, 1
        f = lambda x: x
        g = lambda x: x[0:1]
        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=0.1 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.1 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist_dyn, dist_obs, discrete_time=True)

        x = np.array([1.0, 2.0])

        # Test dynamics with zero noise
        w_zero = np.zeros(nx)
        result = dyn.dynamics(x, w_zero)
        np.testing.assert_array_almost_equal(result, f(x))

        # Test measurements with zero noise
        v_zero = np.zeros(ny)
        result = dyn.measurements(x, v_zero)
        np.testing.assert_array_almost_equal(result, g(x))

    def test_additive_system_creation_is_deprecated(self) -> None:
        """create_additive_system emits a DeprecationWarning and still builds a system."""
        nx, ny = 2, 1

        def f(x, w):
            return x + w

        def g(x, v):
            return x[0:1] + v

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        with pytest.warns(DeprecationWarning):
            add_sys = create_additive_system(
                nx, ny, f, g, dist_X, dist_dyn, dist_obs, N_samples=10, discrete_time=True
            )
        assert add_sys.nx == nx
        assert add_sys.ny == ny


class TestNumericalStability:
    """Test numerical stability of operations."""

    def test_small_covariance_values(self) -> None:
        """Test handling of small covariance values."""
        nx, ny = 2, 1
        f = lambda x: x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-10 * np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-10 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=1e-10 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist_dyn, dist_obs, discrete_time=True)
        assert dyn is not None

    def test_large_covariance_values(self) -> None:
        """Test handling of large covariance values."""
        nx, ny = 2, 1
        f = lambda x: x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=1e10 * np.eye(nx))
        dist_dyn = stats.multivariate_normal(mean=np.zeros(nx), cov=1e10 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=1e10 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist_dyn, dist_obs, discrete_time=True)
        assert dyn is not None

    def test_matrix_operations_with_small_values(self) -> None:
        """Test matrix operations with small values."""
        nx, ny = 2, 1
        f = lambda x: 1e-5 * x
        g = lambda x: 1e-5 * x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-5 * np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-6 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=1e-6 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)

        x = np.array([1.0, 2.0])
        w = np.array([1e-6, 1e-6])
        result = dyn.dynamics(x, w)

        assert np.all(np.isfinite(result))


class TestSampling:
    """Test sampling functionality."""

    def test_sample_state_distribution(self) -> None:
        """Test that sample_state follows the correct distribution."""
        nx = 2
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.1], [0.1, 1.0]])

        dist = stats.multivariate_normal(mean=mean, cov=cov)
        f = lambda x: x
        g = lambda x: x

        dyn = DynamicalSystem(nx, 2, f, g, dist, dist, dist, discrete_time=True)

        samples = dyn.sample_state(size=1000)

        # Check empirical mean is close to theoretical mean
        empirical_mean = np.mean(samples, axis=0)
        np.testing.assert_array_almost_equal(empirical_mean, mean, decimal=1)

    def test_single_sample_shape(self) -> None:
        """Test single sample returns correct shape."""
        nx = 3
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        f = lambda x: x
        g = lambda x: x

        dyn = DynamicalSystem(nx, 1, f, g, dist, dist, dist, discrete_time=True)

        sample = dyn.sample_state(size=1)
        assert sample.shape == (1, nx)
        assert sample.ndim == 2


class TestKoopmanOperatorAPI:
    """Tests for the deprecation aliases and descriptive property aliases."""

    @pytest.fixture
    def fitted_koopman(self) -> KoopmanOperator:
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]
        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))
        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)
        koop = KoopmanOperator(Matern(length_scale=1.0, nu=0.5), dyn)
        koop.compute_edmd(n_features=8, optimize=False)
        return koop

    def test_descriptive_property_aliases(self, fitted_koopman: KoopmanOperator) -> None:
        """Descriptive properties mirror the single-letter attributes."""
        assert fitted_koopman.koopman_matrix is fitted_koopman.U
        assert fitted_koopman.gram_matrix is fitted_koopman.G
        assert fitted_koopman.output_matrix is fitted_koopman.C
        assert fitted_koopman.state_matrix is fitted_koopman.B
        assert fitted_koopman.dictionary is fitted_koopman.X
        assert fitted_koopman.feature_map is fitted_koopman.phi

    def test_opt_kernel_is_deprecated_alias(self, fitted_koopman: KoopmanOperator) -> None:
        """opt_kernel warns and delegates to optimize_kernel."""
        with pytest.warns(DeprecationWarning):
            fitted_koopman.opt_kernel(fitted_koopman.X, n_restarts_optimizer=0)


class TestKoopmanOperatorEdgeCases:
    """Edge case tests for KoopmanOperator."""

    def test_single_feature(self) -> None:
        """Test Koopman operator with single feature."""
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)

        kernel = Matern(length_scale=1.0, nu=0.5)
        koop = KoopmanOperator(kernel, dyn)

        koop.compute_edmd(n_features=1, optimize=False)
        assert koop.get_feature_dimension() == 1

    def test_single_feature_covariances_are_2d(self) -> None:
        """Covariance helpers must return 2D matrices even with a single feature."""
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)
        koop = KoopmanOperator(Matern(length_scale=1.0, nu=0.5), dyn)
        koop.compute_edmd(n_features=1, optimize=False)

        x = np.array([0.5, 0.5])
        cov_init = compute_initial_covariance(x, 1, dist_X, koop, n_samples=50)
        cov_dyn = compute_dynamics_covariance(x, 1, dyn, koop, n_samples=50)

        assert cov_init.shape == (1, 1)
        assert cov_dyn.shape == (1, 1)

    def test_many_features(self) -> None:
        """Test Koopman operator with many features."""
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)

        kernel = Matern(length_scale=1.0, nu=0.5)
        koop = KoopmanOperator(kernel, dyn)

        koop.compute_edmd(n_features=100, optimize=False)
        assert koop.get_feature_dimension() == 100

    def test_phi_function_consistency(self) -> None:
        """Test that phi function is consistent."""
        nx, ny = 2, 1
        f = lambda x: 0.9 * x
        g = lambda x: x[0:1]

        dist_X = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
        dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01 * np.eye(nx))
        dist_obs = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01 * np.eye(ny))

        dyn = DynamicalSystem(nx, ny, f, g, dist_X, dist, dist_obs, discrete_time=True)

        kernel = Matern(length_scale=1.0, nu=0.5)
        koop = KoopmanOperator(kernel, dyn)

        koop.compute_edmd(n_features=10, optimize=False)

        x = np.array([0.5, 0.5])
        phi1 = koop.phi(x)
        phi2 = koop.phi(x)

        np.testing.assert_array_almost_equal(phi1, phi2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
