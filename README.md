# Kernel Koopman Kalman Filter (KKF)

[![Tests](https://github.com/diegoolguinw/kkf/workflows/Tests%20&%20Code%20Quality/badge.svg)](https://github.com/diegoolguinw/kkf/actions)
[![PyPI version](https://badge.fury.io/py/kkf.svg)](https://badge.fury.io/py/kkf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for nonlinear state estimation using kernel-based Extended
Dynamic Mode Decomposition (kEDMD) and Koopman operator theory.

## What is KKF?

The Kalman filter is optimal for linear systems but degrades when the dynamics
are strongly nonlinear. KKF lifts the state into a feature space where the
dynamics are approximately linear, runs a Kalman filter there, and maps the
estimate back. The pieces are:

- **Koopman operator theory** — represent nonlinear dynamics as a linear
  operator acting on observables.
- **Kernel methods** — build the feature space from a kernel (RBF, Matérn, ...).
- **kEDMD** — estimate the linear operator from samples of the dynamics.
- **Kalman filtering** — propagate the mean and covariance in feature space.

## Features

- Kernel Extended Dynamic Mode Decomposition (kEDMD)
- A Kalman filter that operates in the lifted feature space
- Works with arbitrary user-supplied dynamics `f` and observation `g`
- Choice of kernel through scikit-learn's `gaussian_process.kernels`
- Posterior covariance in both state and feature space, with optional
  kernel-parameter optimization
- Type hints throughout, and unit/integration tests run in CI

## Installation

```bash
pip install kkf
```

From source, for development:

```bash
git clone https://github.com/diegoolguinw/kkf.git
cd kkf
pip install -e ".[dev]"
```

Optional dependency groups:

```bash
pip install kkf[viz]    # plotting for the examples
pip install kkf[dev]    # tests and linters
pip install kkf[docs]   # documentation build
pip install kkf[all]    # everything
```

## Quick Start

The example below estimates the states of an SIR (Susceptible-Infected-Recovered)
epidemic model from noisy observations of the infected fraction. A runnable
version is in [`examples/02_sir_epidemic_model.py`](examples/02_sir_epidemic_model.py).

```python
import numpy as np
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

from kkf import DynamicalSystem, KoopmanOperator, apply_koopman_kalman_filter

# SIR dynamics (discrete time)
beta, gamma = 0.12, 0.04

def f(x):  # state transition
    return x + np.array([
        -beta * x[0] * x[1],
        beta * x[0] * x[1] - gamma * x[1],
        gamma * x[1],
    ])

def g(x):  # observe the infected fraction
    return np.array([x[1]])

# Dimensions and noise model
nx, ny, n_features = 3, 1, 50
X_dist = stats.dirichlet(alpha=np.ones(nx))
dyn_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-5 * np.eye(nx))
obs_dist = stats.multivariate_normal(mean=np.zeros(ny), cov=1e-3 * np.eye(ny))

system = DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist, discrete_time=True)

# Synthetic observations of the infected fraction
np.random.seed(42)
n_timesteps = 100
y_obs = np.random.randn(n_timesteps, ny) * 0.1 + 0.1

# Build the Koopman operator and run the filter
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
    noise_samples=100,
)

# Posterior estimates and covariances at each step
print(solution.x_plus.shape)    # (n_timesteps, nx)
print(solution.Px_plus.shape)   # (n_timesteps, nx, nx)
```

`solution.x_plus` / `solution.x_minus` are the posterior / prior state estimates,
and `solution.Px_plus` / `solution.Px_minus` the corresponding covariances.

## Examples

The [`examples/`](examples/) directory contains three runnable scripts:

- [`01_basic_linear_system.py`](examples/01_basic_linear_system.py) — a linear
  system end to end: definition, synthetic data, filtering, confidence intervals.
- [`02_sir_epidemic_model.py`](examples/02_sir_epidemic_model.py) — the SIR model
  above, with partial observation and population estimates.
- [`03_advanced_parameter_exploration.py`](examples/03_advanced_parameter_exploration.py)
  — comparing kernels and feature dimensions.

Run any of them with, e.g., `python examples/02_sir_epidemic_model.py`. See
[examples/README.md](examples/README.md) for more.

## API at a glance

```python
from kkf import (
    DynamicalSystem,
    KoopmanOperator,
    KoopmanKalmanFilterSolution,
    apply_koopman_kalman_filter,
    compute_initial_covariance,
    compute_dynamics_covariance,
    compute_observation_covariance,
)
```

The algorithm proceeds in three steps:

1. **Lift** the state into feature space using the kernel.
2. **Approximate** the linear dynamics in that space with kEDMD.
3. **Filter** with a Kalman filter in feature space, then project the estimate
   and covariance back to the state coordinates.

Main arguments to `apply_koopman_kalman_filter`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `koopman_operator` | A `KoopmanOperator` built from your kernel and system | required |
| `observations` | Measurements, shape `(n_timesteps, n_outputs)` | required |
| `initial_distribution` | Prior over the initial state (a `scipy.stats` distribution) | required |
| `n_features` | Dimension of the lifted feature space | required |
| `optimize` | Optimize kernel parameters during fitting | `False` |
| `n_restarts_optimizer` | Optimizer restarts (used only when `optimize=True`) | 10 |
| `noise_samples` | Monte Carlo samples for covariance propagation | 100 |

The kernel is not passed here — choose it when constructing the
`KoopmanOperator` (see [Common variations](#common-variations) below).

## Requirements

- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- scikit-learn ≥ 1.0
- Matplotlib ≥ 3.5 (optional, only for the example plots)

## Common variations

Different kernels:

```python
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ExpSineSquared, ConstantKernel,
)

kernel = RBF(length_scale=1.0)
kernel = Matern(length_scale=1.0, nu=2.5)
kernel = ExpSineSquared(length_scale=1.0, periodicity=1.0)
kernel = ConstantKernel(1.0) * Matern(nu=1.5)
```

Optimizing the kernel parameters:

```python
solution = apply_koopman_kalman_filter(
    koopman_operator=koopman_op,
    observations=y_obs,
    initial_distribution=initial_prior,
    n_features=50,
    optimize=True,
    n_restarts_optimizer=20,
)
```

Confidence intervals from the posterior covariance:

```python
std = np.sqrt(np.diagonal(solution.Px_plus, axis1=1, axis2=2))
ci_lower = solution.x_plus - 1.96 * std
ci_upper = solution.x_plus + 1.96 * std
```

## Testing

```bash
pytest tests/                 # all tests
pytest tests/ -v              # verbose
pytest tests/ -m "not slow"   # skip slow tests
pytest tests/ --cov=kkf --cov-report=html
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). This project
follows the [Contributor Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

```bibtex
@software{kkf2024,
  title  = {KKF: Kernel Koopman Kalman Filter},
  author = {Olguin-Wende, Diego},
  year   = {2024},
  url    = {https://github.com/diegoolguinw/kkf}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Issues and bug reports: [GitHub Issues](https://github.com/diegoolguinw/kkf/issues)
- Email: dolguin@dim.uchile.cl
