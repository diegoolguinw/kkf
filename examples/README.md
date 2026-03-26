# KKF Examples

This directory contains practical examples of using the Kernel Koopman Kalman Filter library.

## Examples

### 1. Basic Linear System (`01_basic_linear_system.py`)

**Difficulty:** Beginner

Demonstrates the basic workflow:
- Creating a simple linear dynamical system
- Generating synthetic observations
- Applying the Koopman Kalman Filter
- Visualizing results with confidence intervals

**Key concepts:**
- System definition (f, g functions)
- Noise distributions
- Kernel selection
- Filter application

**Run:**
```bash
python examples/01_basic_linear_system.py
```

### 2. SIR Epidemic Model (`02_sir_epidemic_model.py`)

**Difficulty:** Intermediate

Demonstrates application to epidemiological modeling:
- Nonlinear SIR system dynamics
- Modeling disease progression
- Estimating unobserved states (susceptible, recovered) from partial observations
- Confidence interval computation

**Key concepts:**
- Nonlinear dynamics
- Partial observations
- Epidemic modeling
- Matérn kernels

**Run:**
```bash
python examples/02_sir_epidemic_model.py
```

### 3. Advanced Parameter Exploration (`03_advanced_parameter_exploration.py`)

**Difficulty:** Advanced

Demonstrates parameter optimization and comparison:
- Testing different kernel functions
- Exploring number of features impact
- Performance comparison and visualization
- Systematic parameter search

**Key concepts:**
- Different kernel types (RBF, Matérn, ExpSineSquared)
- Hyperparameter sensitivity
- Performance metrics
- Visualization and analysis

**Run:**
```bash
python examples/03_advanced_parameter_exploration.py
```

## Getting Started

### Prerequisites

Ensure you have installed the KKF library and its dependencies:

```bash
pip install -e .
# or with visualization:
pip install -e ".[viz]"
```

### Running Examples

All examples are standalone and can be run directly:

```bash
python examples/01_basic_linear_system.py
```

Each example generates:
- Console output with analysis
- PNG figure with visualizations
- Prints intermediate results and statistics

## Example Templates

These examples can serve as templates for your own applications:

1. **Copy an example as a starting point**
2. **Replace the system dynamics** (f and g functions)
3. **Adjust the parameters** for your specific problem
4. **Run and evaluate** the filter performance

## Common Customizations

### Changing System Dynamics

```python
def f(x):
    """Your custom state transition function."""
    return your_function(x)

def g(x):
    """Your custom measurement function."""
    return your_observation(x)
```

### Adjusting Noise Levels

```python
# Increase process noise
dyn_noise = stats.multivariate_normal(
    mean=np.zeros(nx), 
    cov=larger_value * np.eye(nx)
)

# Increase measurement noise
obs_noise = stats.multivariate_normal(
    mean=np.zeros(ny), 
    cov=larger_value * np.eye(ny)
)
```

### Selecting Different Kernels

```python
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ExpSineSquared,
    ConstantKernel,
    DotProduct
)

# Different kernel choices
kernel = RBF(length_scale=1.0)
kernel = Matern(length_scale=1.0, nu=0.5)
kernel = DotProduct(sigma_0=1.0)
```

## Tips for Your Own Applications

1. **Start Simple**: Begin with a basic example and gradually add complexity
2. **Verify Results**: Always check if estimated states make physical sense
3. **Tune Hyperparameters**: Number of features, kernel parameters, noise estimates
4. **Visualize**: Plot states, observations, and errors to understand filter behavior
5. **Test Robustness**: Try different initial conditions and noise levels

## Troubleshooting

### Filter becomes unstable
- Reduce the number of features
- Adjust kernel parameters
- Check noise distribution assumptions

### Poor estimation accuracy
- Increase number of kernel features
- Try different kernel functions
- Verify measurement model is correct

### Slow computation
- Reduce number of features
- Use fewer monte carlo samples (noise_samples)
- Optimize kernel computation

## Further Reading

For more information, see:
- Main [README.md](../README.md)
- [API Documentation](../docs/)
- Scientific papers on Koopman operator theory

## Contributing

If you create interesting examples, consider contributing them back to the project!
See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
