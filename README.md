# KKKF - Kernel Koopman Kalman Filter

KKKF is a Python library that implements kernel Extended Dynamic Mode Decomposition (EDMD) of Koopman operators and provides a non-linear variant of the Kalman Filter. This library is particularly useful for state estimation in dynamical systems with non-linear behavior.

## Installation

You can install KKKF using pip:

```bash
pip install KKKF
```

## Features

- Kernel-based Extended Dynamic Mode Decomposition (EDMD)
- Non-linear Kalman Filter implementation
- Support for general dynamical systems
- Integration with various kernel functions (e.g., Matérn kernel)
- Robust state estimation with noise handling

## Dependencies

- NumPy
- SciPy
- scikit-learn (for kernel functions)
- Matplotlib (for visualization)

## Quick Start

Here's a complete example of using KKKF to estimate and visualize states in a SIR (Susceptible-Infected-Recovered) model:

```python
# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.gaussian_process.kernels import Matern
from KKKF.DynamicalSystems import DynamicalSystem
from KKKF.kEDMD import KoopmanOperator
from KKKF.applyKKKF import apply_koopman_kalman_filter

# Define system parameters
beta, gamma = 0.12, 0.04

# Define system dynamics
def f(x):
    return x + np.array([-beta*x[0]*x[1], beta*x[0]*x[1] - gamma*x[1], gamma*x[1]])

# Define system observations
def g(x):
    return np.array([x[1]])

# Setup system dimensions and kernel
N = 300
nx, ny = 3, 1
k = Matern(length_scale=N**(-1/nx), nu=0.5)

# Setup distributions
X_dist = stats.dirichlet(alpha=np.ones(nx))
dyn_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=1e-5*np.eye(3))
obs_dist = stats.multivariate_normal(mean=np.zeros(ny), cov=1e-3*np.eye(1))

# Create dynamical system
dyn = DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist)

# Generate synthetic data
iters = 100
x0 = np.array([0.9, 0.1, 0.0])
x = np.zeros((iters, nx))
y = np.zeros((iters, ny))

x[0] = x0
y[0] = g(x[0]) + obs_dist.rvs()

for i in range(1, iters):
    x[i] = f(x[i-1]) + dyn.dist_dyn.rvs()
    y[i] = g(x[i]) + obs_dist.rvs()

# Initialize and apply Koopman Kalman Filter

# Prior for the initial condition
x0_prior = np.array([0.8, 0.15, 0.05])
d0 = stats.multivariate_normal(mean=x0_prior, cov=0.1*np.eye(3))

# Koopman operator
Koop = KoopmanOperator(k, dyn)

# Solution
sol = apply_koopman_kalman_filter(Koop, y, d0, N, noise_samples=100)

# Visualization with confidence intervals
conf = np.zeros((iters, nx))
for i in range(iters):
    conf[i, :] = np.sqrt(np.diag(sol.Px_plus[i,:,:]))

# 95% confidence interval
err1 = sol.x_plus - 1.96*conf
err2 = sol.x_plus + 1.96*conf

# Plot elements
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
@software{kkkf,
  title = {KKKF: Kernel Koopman Kalman Filter},
  year = {2024},
  author = {Diego Olguín-Wende},
  url = {https://github.com/diegoolguinw/KKKF}
}
```
