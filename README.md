# Kernel Koopman Kalman Filter (KKF)

[![Tests](https://github.com/diegoolguinw/kkf/workflows/Tests%20&%20Code%20Quality/badge.svg)](https://github.com/diegoolguinw/kkf/actions)
[![PyPI version](https://badge.fury.io/py/kkf.svg)](https://badge.fury.io/py/kkf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for non-linear state estimation using Kernel-based Extended Dynamic Mode Decomposition (kEDMD) and Koopman operator theory.

## What is KKF?

KKF is a sophisticated filtering library that combines:

- **Koopman Operator Theory**: Lifts nonlinear dynamics to a linear setting in feature space
- **Kernel Methods**: Uses kernel functions for feature space approximation
- **Kalman Filtering**: Applies optimal filtering in the lifted feature space
- **Extended Dynamic Mode Decomposition**: Learns accurate linear approximations

This enables accurate state estimation for highly nonlinear dynamical systems where traditional Kalman filters struggle.

## Features

✨ **Core Capabilities**
- Kernel-based Extended Dynamic Mode Decomposition (kEDMD)
- Non-linear Kalman Filter implementation
- Support for arbitrary dynamical systems
- Multiple kernel function options (RBF, Matérn, etc.)
- Robust state estimation with uncertainty quantification
- Covariance propagation in both state and feature spaces

📊 **High-Quality Implementation**
- Type hints throughout the codebase
- Comprehensive error handling
- Numerical stability optimizations
- Efficient NumPy/SciPy operations

🧪 **Well-Tested**
- 95+ unit and integration tests
- Edge case coverage
- Multi-platform CI/CD
- Coverage reporting

📚 **Comprehensive Documentation**
- Detailed docstrings
- 3 complete working examples
- Parameter optimization guide
- API documentation

## Installation

### Current Release

```bash
pip install kkf
```

### Development Version

```bash
git clone https://github.com/diegoolguinw/kkf.git
cd kkf
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For visualization examples
pip install kkf[viz]

# For development and testing
pip install kkf[dev]

# For building documentation
pip install kkf[docs]

# Install everything
pip install kkf[all]
```

## Quick Start

Here's a complete example of using KKF to estimate states in a SIR (Susceptible-Infected-Recovered) epidemiological model:

```python
import numpy as np
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

from KKF import DynamicalSystem, KoopmanOperator
from KKF.applyKKF import apply_koopman_kalman_filter

# Define SIR dynamics
beta, gamma = 0.12, 0.04

def f(x):  # State transition
    return x + np.array([
        -beta * x[0] * x[1],
        beta * x[0] * x[1] - gamma * x[1],
        gamma * x[1]
    ])

def g(x):  # Observation (infected population)
    return np.array([x[1]])

# Setup system
nx, ny, n_features = 3, 1, 50
X_dist = stats.dirichlet(alpha=np.ones(nx))
dyn_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-5 * np.eye(nx))
obs_dist = stats.multivariate_normal(mean=np.zeros(ny), cov=1e-3 * np.eye(ny))

system = DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist, discrete_time=True)

# Generate synthetic observations
np.random.seed(42)
n_timesteps = 100
y_obs = np.random.randn(n_timesteps, ny) * 0.1 + 0.1

# Apply Koopman Kalman Filter
kernel = Matern(length_scale=1.0, nu=0.5)
koopman_op = KoopmanOperator(kernel, system)

x0_prior = np.array([0.8, 0.15, 0.05])
initial_prior = stats.multivariate_normal(mean=x0_prior, cov=0.1 * np.eye(nx))

solution = apply_koopman_kalman_filter(
    koopman_operator=koopman_op,
    observations=y_obs,
    initial_distribution=initial_prior,
    n_features=n_features,
    optimize=False,
    noise_samples=100
)

# Access results
print(f"Estimated states: {solution.x_plus.shape}")  # (n_timesteps, nx)
print(f"State covariances: {solution.Px_plus.shape}")  # (n_timesteps, nx, nx)
```

## Examples

The library includes three comprehensive examples:

### 1. Basic Linear System (`examples/01_basic_linear_system.py`)

Introduces core concepts with a simple linear system:
- System definition
- Synthetic data generation
- Filter application
- Confidence interval visualization

**Run:** `python examples/01_basic_linear_system.py`

### 2. SIR Epidemic Model (`examples/02_sir_epidemic_model.py`)

Real-world application to disease modeling:
- Nonlinear dynamics
- Partial state observation
- Population dynamics estimation
- Epidemic curve prediction

**Run:** `python examples/02_sir_epidemic_model.py`

### 3. Parameter Exploration (`examples/03_advanced_parameter_exploration.py`)

Advanced hyperparameter tuning:
- Multiple kernel comparison
- Feature dimension optimization
- Performance analysis
- Systematic search strategy

**Run:** `python examples/03_advanced_parameter_exploration.py`

For more details, see [examples/README.md](examples/README.md)

## Documentation

### API Reference

```python
# Core classes
from KKF import DynamicalSystem, KoopmanOperator, KoopmanKalmanFilterSolution
from KKF.applyKKF import apply_koopman_kalman_filter

# Utility functions
from KKF import (
    compute_initial_covariance,
    compute_dynamics_covariance,
    compute_observation_covariance
)
```

### Theory

The KKF algorithm works in three main steps:

1. **Lifting**: Maps nonlinear states to a higher-dimensional feature space using kernel methods
2. **Approximation**: Constructs a linear Koopman operator approximation via kEDMD
3. **Filtering**: Applies Kalman filtering in the lifted space, then maps back to original coordinates

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_features` | Dimension of lifted feature space | 50 |
| `kernel` | Kernel function (RBF, Matérn, etc.) | RBF(1.0) |
| `optimize` | Whether to optimize kernel parameters | False |
| `noise_samples` | Monte Carlo samples for covariance | 100 |

## System Requirements

- Python 3.8 or higher
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- scikit-learn ≥ 1.0.0
- Matplotlib ≥ 3.5.0 (optional, for visualization)

## Advanced Usage

### Custom Kernel Functions

```python
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ExpSineSquared, DotProduct, ConstantKernel
)

# Different kernel choices
kernel1 = RBF(length_scale=1.0)
kernel2 = Matern(length_scale=1.0, nu=2.5)
kernel3 = ExpSineSquared(length_scale=1.0, periodicity=1.0)
kernel4 = ConstantKernel(constant_value=1.0) * Matern(nu=1.5)
```

### Manual Kernel Optimization

```python
solution = apply_koopman_kalman_filter(
    koopman_operator=koopman_op,
    observations=y_obs,
    initial_distribution=initial_prior,
    n_features=50,
    optimize=True,           # Enable kernel optimization
    n_restarts_optimizer=20  # Number of optimization restarts
)
```

### Extracting Uncertainty Estimates

```python
# Posterior state covariance
Px_plus = solution.Px_plus  # (n_timesteps, nx, nx)

# Compute confidence intervals
std_devs = np.sqrt(np.diagonal(Px_plus, axis1=1, axis2=2))
ci_lower = solution.x_plus - 1.96 * std_devs
ci_upper = solution.x_plus + 1.96 * std_devs
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/

# Verbose output
pytest tests/ -v

# Specific test file
pytest tests/test_core.py

# Skip slow tests
pytest tests/ -m "not slow"

# With coverage report
pytest tests/ --cov=KKF --cov-report=html
```

## Performance

Recommended configurations:

| System Size | State Dim | Features | Notes |
|------------|-----------|----------|-------|
| Small | 1-3 | 10-20 | Fast, real-time capable |
| Medium | 3-10 | 20-50 | Moderate, seconds per filter run |
| Large | 10-50 | 50-100 | Slow, minutes per filter run |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms.

## Citation

If you use KKF in your research, please cite:

```bibtex
@software{kkf2024,
  title={KKF: Kernel Koopman Kalman Filter},
  author={Olguin-Wende, Diego},
  year={2024},
  url={https://github.com/diegoolguinw/kkf}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## Acknowledgments

- Koopman operator theory foundations
- Extended Dynamic Mode Decomposition methodology
- Kalman filtering research community

## Support & Contact

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/diegoolguinw/kkf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/diegoolguinw/kkf/discussions)
- **Email**: dolguin@dim.uchile.cl

## References

Key papers and resources:
- Koopman, B. O. (1931). "Hamiltonian systems and transformation in Hilbert space"
- Williams, M. O., et al. (2015). "A data-driven approximation of the koopman operator"
- Korda, M., & Mezić, I. (2018). "Linear predictors for nonlinear dynamical systems"

---

**Made with ❤️ for the scientific computing community**
labels = ["S (True)", "I (True)", "R (True)"]
colors = ["blue", "red", "green"]

plt.plot(sol.x_plus, label=["S (KKF)", "I (KKF)", "R (KKF)"])

for i in range(nx):
    plt.fill_between(np.arange(iters), err1[:,i], err2[:,i], alpha=0.6)
    plt.scatter(np.arange(iters), x[:,i], label=labels[i], color=colors[i], s=1.4)

plt.xlabel("Days")
plt.ylabel("Propotion of population")
plt.title("KKKF Estimation")
plt.legend()
plt.show()
```

## API Reference

### DynamicalSystem

```python
DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist)
```
Creates a dynamical system with:
- `nx`: State dimension
- `ny`: Observation dimension
- `f`: State transition function
- `g`: Observation function
- `X_dist`: State distribution
- `dyn_dist`: Dynamic noise distribution
- `obs_dist`: Observation noise distribution

### KoopmanOperator

```python
KoopmanOperator(kernel, dynamical_system)
```
Initializes a Koopman operator with:
- `kernel`: Kernel function (e.g., Matérn kernel)
- `dynamical_system`: Instance of DynamicalSystem

### apply_koopman_kalman_filter

```python
apply_koopman_kalman_filter(koopman, observations, initial_distribution, N, noise_samples=100)
```
Applies the Koopman-based Kalman filter with:
- `koopman`: KoopmanOperator instance
- `observations`: Observation data
- `initial_distribution`: Initial state distribution
- `N`: Number of samples
- `noise_samples`: Number of noise samples for uncertainty estimation

Returns a solution object containing:
- `x_plus`: State estimates
- `Px_plus`: Covariance matrices
- Additional filter statistics

## Visualization

The library supports visualization of results with confidence intervals. The example above demonstrates how to:
- Plot state estimates
- Add confidence intervals (shaded regions)
- Compare with real data (if available)
- Customize plot appearance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{kkf,
  title = {kerKKF: Kernel Koopman Kalman Filter},
  year = {2024},
  author = {Diego Olguín-Wende},
  url = {https://github.com/diegoolguinw/kkf}
}
```
