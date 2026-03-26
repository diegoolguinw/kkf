# KKF Quick Reference

Quick commands and snippets for working with the KKF library.

## Installation

```bash
# Basic installation
pip install kkf

# With visualization
pip install kkf[viz]

# With development tools
pip install -e ".[dev]"

# All extras
pip install kkf[all]
```

## Basic Usage

### Import Core Components
```python
from KKF import (
    DynamicalSystem,
    KoopmanOperator,
    KoopmanKalmanFilterSolution,
    compute_initial_covariance,
    compute_dynamics_covariance,
    compute_observation_covariance,
)
from KKF.applyKKF import apply_koopman_kalman_filter
import numpy as np
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, Matern
```

### Create a System
```python
# Define functions
def f(x):
    return 0.9 * x  # Dynamics

def g(x):
    return x[:1]    # Measurement

# Define distributions
nx, ny = 2, 1
X_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=np.eye(nx))
dyn_dist = stats.multivariate_normal(mean=np.zeros(nx), cov=0.01*np.eye(nx))
obs_dist = stats.multivariate_normal(mean=np.zeros(ny), cov=0.01*np.eye(ny))

# Create system
system = DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist,
                        discrete_time=True)
```

### Create Koopman Operator
```python
# Choose kernel
kernel = RBF(length_scale=1.0)
# Alternative kernels:
# kernel = Matern(length_scale=1.0, nu=0.5)
# kernel = ExpSineSquared(length_scale=1.0, periodicity=1.0)

# Create operator
koopman_op = KoopmanOperator(kernel, system)

# Compute EDMD
koopman_op.compute_edmd(n_features=20, optimize=False)
```

### Apply Filter
```python
# Generate observations
n_timesteps = 100
observations = np.random.randn(n_timesteps, ny)

# Initial state prior
x0_prior = np.array([0.5, 0.5])
initial_dist = stats.multivariate_normal(mean=x0_prior, cov=0.1*np.eye(nx))

# Apply filter
solution = apply_koopman_kalman_filter(
    koopman_operator=koopman_op,
    observations=observations,
    initial_distribution=initial_dist,
    n_features=20,
    optimize=False,
    noise_samples=100
)
```

### Extract Results
```python
# Estimated states
states = solution.x_plus  # Shape: (n_timesteps, nx)

# State covariances
covariances = solution.Px_plus  # Shape: (n_timesteps, nx, nx)

# Confidence intervals (95%)
std_devs = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
ci_lower = solution.x_plus - 1.96 * std_devs
ci_upper = solution.x_plus + 1.96 * std_devs
```

## Common Patterns

### Pattern 1: Simple Linear System
```python
# For linear systems: x[k+1] = A @ x[k], y[k] = C @ x[k]
A = np.array([[0.9, 0.05], [0.0, 0.95]])
C = np.array([[1.0, 0.0]])

def f(x): return A @ x
def g(x): return C @ x

# Rest of setup...
```

### Pattern 2: Nonlinear System
```python
# For nonlinear systems
def f(x):
    return np.array([
        0.5 * x[0] + np.sin(x[1]),
        0.7 * x[1] - 0.1 * x[0]**2
    ])

def g(x):
    return np.array([x[0] + 0.1 * x[1]**2])
```

### Pattern 3: Parameter Sweep
```python
kernels = {
    'RBF_0.5': RBF(0.5),
    'RBF_1.0': RBF(1.0),
    'Matern': Matern(1.0, nu=0.5),
}

results = {}
for name, kernel in kernels.items():
    koop = KoopmanOperator(kernel, system)
    sol = apply_koopman_kalman_filter(...)
    results[name] = sol
```

### Pattern 4: Grid Search
```python
n_features_list = [10, 20, 30, 50]
errors = {}

for n_feat in n_features_list:
    koop = KoopmanOperator(kernel, system)
    sol = apply_koopman_kalman_filter(
        koopman_operator=koop,
        observations=observations,
        initial_distribution=initial_dist,
        n_features=n_feat
    )
    error = np.mean(np.linalg.norm(sol.x_plus - x_true, axis=1))
    errors[n_feat] = error
```

## Visualization

### Plot States with Confidence Intervals
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Extract time series
time = np.arange(solution.x_plus.shape[0])
state_0 = solution.x_plus[:, 0]
std_0 = np.sqrt(solution.Px_plus[:, 0, 0])

# Plot state
ax.plot(time, state_0, 'b-', label='Estimated')

# Plot CI
ax.fill_between(time, 
                state_0 - 1.96*std_0, 
                state_0 + 1.96*std_0,
                alpha=0.3, label='95% CI')

ax.set_xlabel('Time')
ax.set_ylabel('State')
ax.legend()
plt.show()
```

### Heatmap of Estimation Errors
```python
import matplotlib.pyplot as plt

errors = np.abs(solution.x_plus - x_true)
plt.imshow(errors.T, aspect='auto', cmap='hot')
plt.colorbar(label='Absolute Error')
plt.xlabel('Time Step')
plt.ylabel('State Dimension')
plt.title('Estimation Errors')
plt.show()
```

## Testing

```bash
# Run all tests
pytest tests/

# Verbose mode
pytest tests/ -v

# Specific test
pytest tests/test_core.py::TestDynamicalSystem -v

# With coverage
pytest tests/ --cov=KKF

# Skip slow tests
pytest tests/ -m "not slow"

# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s
```

## Code Formatting

```bash
# Format code
black KKF

# Sort imports
isort KKF

# Lint
flake8 KKF

# Type check
mypy KKF --ignore-missing-imports
```

## Common Issues

### Import Error: "No module named 'KKF'"
```bash
# Reinstall in development mode
pip install -e .
```

### AttributeError: Module has no attribute
```bash
# Check __init__.py exports
cat KKF/__init__.py

# Import directly if needed
from KKF.module_name import Class
```

### Test failures
```bash
# Clean cache
rm -rf .pytest_cache __pycache__

# Reinstall
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Numerical instability
```python
# Use larger state/noise covariances
# Reduce number of features
# Try different kernel
# Check if system is properly scaled
```

## Development Commands

```bash
# Create development environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -e ".[dev,viz]"

# Format and lint all
black KKF && isort KKF && flake8 KKF

# Run tests and check coverage
pytest tests/ --cov=KKF --cov-report=html

# Run example
python examples/01_basic_linear_system.py

# Build package
python -m build

# Upload to PyPI (for maintainers)
twine upload dist/*
```

## Kernel Selection

```python
from sklearn.gaussian_process.kernels import (
    RBF,              # Smooth functions
    Matern,           # Less smooth (nu parameter)
    RationalQuadratic,# Polynomial-like decay
    ExpSineSquared,   # Periodic functions
    DotProduct,       # Linear kernels
    ConstantKernel,   # Scale factor
)

# Simple comparison
kernels = [
    RBF(length_scale=1.0),
    Matern(length_scale=1.0, nu=0.5),
    Matern(length_scale=1.0, nu=1.5),
    Matern(length_scale=1.0, nu=2.5),
]

# Composite kernel
kernel = ConstantKernel(1.0) * Matern(1.0, nu=1.5)
```

## Documentation Links

- [Main README](README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Examples](examples/)
- [Development Setup](DEVELOPMENT.md)
- [Change Log](CHANGELOG.md)
- [Improvements Summary](IMPROVEMENTS_SUMMARY.md)

## Performance Tips

1. **Reduce features for speed**: Use smaller `n_features` value
2. **Reduce noise samples**: Lower `noise_samples` parameter
3. **Use simpler kernel**: RBF is often faster than Matern
4. **Vectorize when possible**: Use NumPy operations

## Getting Help

- **Documentation**: See README.md and examples/
- **API Reference**: Check docstrings with `help(KKF.ClassName)`
- **Examples**: Look at `examples/` for practical usage
- **Issues**: File bugs on GitHub
- **Discussions**: Start a GitHub discussion for questions

---

For more details, see the full documentation files!
