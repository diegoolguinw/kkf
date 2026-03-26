# Contributing to KKF

Thank you for your interest in contributing to the Kernel Koopman Kalman Filter library! We welcome contributions from everyone in the community. This document provides guidelines and instructions for contributing.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Ways to Contribute

We appreciate contributions in many forms:

### Bug Reports
- Check if the issue already exists in [GitHub Issues](https://github.com/diegoolguinw/kkf/issues)
- Provide a minimal reproducible example
- Include:
  - Python version
  - KKF version
  - Error message and traceback
  - Steps to reproduce

### Feature Requests
- Check existing issues and discussions
- Describe the feature and its benefits
- Provide use case examples
- Discuss potential implementation approach

### Code Contributions
- Bug fixes
- New features
- Improved documentation
- Performance improvements
- Example applications

### Documentation
- Fix typos or unclear documentation
- Add examples and tutorials
- Improve API documentation
- Create guides for common tasks

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/kkf.git
cd kkf
```

### 2. Create Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,viz]"
```

### 3. Verify Setup

```bash
# Run tests
pytest tests/

# Check formatting
black --check KKF
isort --check-only KKF

# Run linting
flake8 KKF
```

## Making Changes

### 1. Create a Branch

```bash
# Create a feature branch from develop
git checkout -b feature/your-feature-name
```

### 2. Code Style Guidelines

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these tools:

- **Black**: Code formatting (max line length: 100 characters)
- **isort**: Import sorting
- **Flake8**: Linting
- **mypy**: Type checking (when applicable)

```bash
# Auto-format your code
black KKF
isort KKF

# Check for issues
flake8 KKF
mypy KKF --ignore-missing-imports
```

### 3. Type Hints

Please include type hints for better code quality:

```python
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

def my_function(
    x: NDArray[np.float64], 
    n: int
) -> Tuple[NDArray[np.float64], float]:
    """Function with type hints."""
    return x * 2, float(np.mean(x))
```

### 4. Documentation

Add docstrings following NumPy style:

```python
def apply_filter(
    system: DynamicalSystem,
    observations: NDArray[np.float64],
    n_features: int,
) -> KoopmanKalmanFilterSolution:
    """Brief description of function.
    
    Longer description with more details about what the function does,
    how it works, and when to use it.
    
    Parameters
    ----------
    system : DynamicalSystem
        The dynamical system to filter.
    observations : np.ndarray
        Array of shape (n_timesteps, n_outputs) containing observations.
    n_features : int
        Number of kernel features to use.
    
    Returns
    -------
    KoopmanKalmanFilterSolution
        Filter solution containing state estimates.
    
    Raises
    ------
    ValueError
        If n_features <= 0.
    
    Notes
    -----
    Additional implementation details and mathematical background.
    
    Examples
    --------
    >>> system = create_system()
    >>> obs = np.random.randn(100, 2)
    >>> sol = apply_filter(system, obs, n_features=20)
    """
```

### 5. Testing

Add tests for new features:

```python
def test_my_new_feature():
    """Test description."""
    # Setup
    x = np.array([1.0, 2.0])
    
    # Execute
    result = my_function(x, n=5)
    
    # Assert
    np.testing.assert_array_almost_equal(result, expected)
```

Run tests before committing:

```bash
pytest tests/
pytest tests/ -v  # Verbose output
pytest tests/test_core.py::TestClass::test_method  # Specific test
pytest tests/ -m "not slow"  # Skip slow tests
```

### 6. Commit Messages

Follow good commit message practices:

```
Short summary (50 characters or less)

Longer explanation if necessary. Wrap at 72 characters.
Explain WHY the change was made, not WHAT was changed.

- Use bullet points for multiple concepts
- Reference issues when relevant (#123)
- Be clear and descriptive
```

Examples:
- `Fix numerical stability in matrix inversion`
- `Add support for custom kernel functions (#42)`
- `Improve documentation for Koopman operator`

## Pull Requests

### 1. Before Submitting

- [ ] Code passes all tests: `pytest tests/`
- [ ] Code is formatted: `black KKF && isort KKF`
- [ ] No linting errors: `flake8 KKF`
- [ ] Type hints are present where applicable
- [ ] Documentation is updated
- [ ] A changelog entry is added

### 2. Create Pull Request

1. Push your branch to your fork
2. Go to the main repository
3. Click "New Pull Request"
4. Describe your changes:
   - What problem does it solve?
   - How does it solve it?
   - Are there breaking changes?
   - Are there any limitations?

### 3. PR Template

```markdown
## Description
Brief description of the changes.

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Specific change 1
- Specific change 2

## Testing
Describe the tests added or modified.

## Checklist
- [ ] Tests pass
- [ ] Code is formatted
- [ ] Documentation updated
- [ ] Changelog entry added
```

## Review Process

1. **Automated Checks**: GitHub Actions runs:
   - Tests on multiple Python versions
   - Formatting and linting checks
   - Type checking

2. **Code Review**: Maintainers will:
   - Review code quality and style
   - Check for correctness
   - Ensure adequate documentation
   - Request changes if needed

3. **Approval**: Once approved:
   - Your PR will be merged into `develop`
   - Your contribution will be acknowledged

## Repository Structure

```
kkf/
├── KKF/                    # Main package
│   ├── __init__.py
│   ├── DynamicalSystems.py
│   ├── kEDMD.py
│   ├── KKFsol.py
│   ├── applyKKF.py
│   ├── covariances.py
│   └── utils.py
├── tests/                  # Test suite
│   ├── conftest.py
│   ├── test_core.py
│   └── test_utils.py
├── examples/               # Example usage
│   ├── 01_basic_linear_system.py
│   ├── 02_sir_epidemic_model.py
│   └── 03_advanced_parameter_exploration.py
├── docs/                   # Documentation
├── README.md
├── pyproject.toml          # Project configuration
├── setup.py                # Setup configuration
└── .github/workflows/      # CI/CD workflows
```

## Reporting Issues

### Good Bug Reports Include

- **Title**: Short, descriptive summary
- **Description**: Clear explanation of the issue
- **Reproducible Example**: Minimal code that reproduces the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, KKF version, OS
- **Error Message**: Full traceback if available

### Example

```python
# Minimal reproducible example
import numpy as np
from scipy import stats
from KKF import DynamicalSystem, KoopmanOperator

# System setup...
# ... code that triggers the bug ...
# Resulting error message...
```

## Development Best Practices

1. **Keep PRs Focused**: One feature/fix per PR
2. **Stay Updated**: Sync with main regularly
3. **Test Thoroughly**: Add tests for edge cases
4. **Document Changes**: Update docstrings and guides
5. **Be Responsive**: Respond to review comments
6. **Ask Questions**: If anything is unclear

## Performance Considerations

When optimizing or adding new features:

- Profile code to identify bottlenecks
- Benchmark changes: `pytest tests/ --benchmark`
- Avoid unnecessary copies of large arrays
- Consider memory usage for high-dimensional systems

## Release Process

Maintainers follow semantic versioning (Major.Minor.Patch):

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Tag release: `git tag vX.Y.Z`
5. Build and publish to PyPI

## Questions?

- Create an issue for questions
- Join discussions for feature ideas
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- Release notes
- Contributors list
- GitHub contributors page

Thank you for contributing to KKF! 🚀
