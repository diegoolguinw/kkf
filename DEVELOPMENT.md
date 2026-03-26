# Getting Started for Development

This guide will help you set up a development environment and contribute to KKF.

## Quick Setup (5 minutes)

### 1. Prerequisites
- Python 3.8 or higher
- Git
- pip (Python package manager)

### 2. Clone and Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/kkf.git
cd kkf

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,viz]"
```

### 3. Verify Installation
```bash
# Run tests
pytest tests/ -v

# Check formatting
black --check KKF

# Run examples (optional, requires matplotlib)
python examples/01_basic_linear_system.py
```

If all pass, you're ready to develop! ✨

## Development Workflow

### Working on Features

1. **Create a branch**
```bash
git checkout -b feature/my-feature-name
```

2. **Make changes** to the code

3. **Format and lint**
```bash
# Format code
black KKF

# Sort imports
isort KKF

# Check for lint issues
flake8 KKF

# Type check
mypy KKF --ignore-missing-imports
```

4. **Write tests** in `tests/`
```python
# Example test
def test_my_feature():
    from KKF import MyClass
    obj = MyClass()
    assert obj.my_method() == expected_result
```

5. **Run tests**
```bash
# All tests
pytest tests/

# Run with coverage
pytest tests/ --cov=KKF

# Run specific test
pytest tests/test_core.py::TestClass::test_method -v
```

6. **Commit changes**
```bash
git add .
git commit -m "Add my feature

Detailed explanation of what was added and why."
```

7. **Push and create PR**
```bash
git push origin feature/my-feature-name
# Then create PR on GitHub
```

## Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project configuration, dependencies, tool settings |
| `setup.py` | Legacy setup (delegates to pyproject.toml) |
| `KKF/__init__.py` | Package exports |
| `tests/test_core.py` | Main test suite |
| `tests/conftest.py` | Pytest configuration |
| `.github/workflows/` | CI/CD pipelines |
| `CONTRIBUTING.md` | Detailed contribution guidelines |

## Running and Debugging Options

### Run Tests with Different Options
```bash
# Verbose output
pytest tests/ -v

# Show local variables on error
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Recently modified files
pytest tests/ --lf

# Interactive pdb on failure
pytest tests/ --pdb

# Benchmark pytest
pytest tests/ --benchmark

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto
```

### Debug an Issue
```bash
# Run specific test with verbose output
pytest tests/test_core.py::TestDynamicalSystem::test_initialization -vv

# Run with Python debugger
python -m pdb examples/01_basic_linear_system.py

# Run in IPython for interactive debugging
ipython -c "%run examples/02_sir_epidemic_model.py"
```

## Code Style Guide

### Naming Conventions
```python
# Constants
MAX_ITERATIONS = 100

# Functions and methods
def compute_filter_solution():
    pass

# Classes
class KoopmanOperator:
    pass

# Private/internal
def _internal_helper():
    pass
```

### Docstring Format
```python
def apply_filter(system, observations, n_features):
    """Short description in one line.
    
    Longer description with more details. Can span
    multiple lines explaining the functionality.
    
    Parameters
    ----------
    system : DynamicalSystem
        Description of parameter.
    observations : np.ndarray
        Description, shape (n_timesteps, n_outputs).
    n_features : int
        Number of kernel features to use.
    
    Returns
    -------
    np.ndarray
        Description of return value.
    
    Raises
    ------
    ValueError
        If n_features <= 0.
    
    Notes
    -----
    Additional implementation details.
    
    Examples
    --------
    >>> system = create_system()
    >>> result = apply_filter(system, obs, 20)
    """
```

### Type Hints
```python
from typing import Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray

def my_function(
    x: NDArray[np.float64],
    n: int,
    callback: Optional[Callable] = None
) -> Tuple[NDArray[np.float64], float]:
    """Function with type hints."""
    result = process(x)
    return result, float(np.mean(x))
```

## Common Tasks

### Adding a New Feature
1. Create feature branch: `git checkout -b feature/description`
2. Add code to appropriate module in `KKF/`
3. Write tests in `tests/test_*.py`
4. Run tests and format: `pytest` and `black KKF`
5. Update docstrings
6. Create pull request

### Fixing a Bug
1. Create branch: `git checkout -b bugfix/description`
2. Write a test that reproduces the bug
3. Fix the code to make test pass
4. Run full test suite: `pytest tests/`
5. Create pull request with "Fixes #ISSUE_NUMBER"

### Improving Documentation
1. Edit `.md` files for documentation
2. Update docstrings in code
3. Add examples or clarify existing ones
4. Run spell checker if available
5. Create pull request

### Preparing for Release
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit and push
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to v0.3.0"

# Create and push tag
git tag v0.3.0
git push origin v0.3.0
# GitHub Actions handles the rest!
```

## Troubleshooting

### Tests Fail After I Install
```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Clear cache
rm -rf .pytest_cache __pycache__

# Run tests again
pytest tests/
```

### Formatting/Linting Issues
```bash
# Auto-fix with black
black KKF

# Auto-fix imports
isort KKF

# See what flake8 complains about
flake8 KKF

# See type errors
mypy KKF --ignore-missing-imports
```

### Import Errors
```bash
# Verify installation
python -c "import KKF; print(KKF.__file__)"

# Check installed packages
pip list | grep -E "KKF|numpy|scipy"

# Reinstall
pip uninstall KKF
pip install -e "."
```

### Tests Hang or Take Too Long
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run in parallel
pip install pytest-xdist
pytest tests/ -n auto

# Run single test file
pytest tests/test_core.py
```

## Tips & Tricks

1. **Use git aliases** for common commands
```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
```

2. **Enable pre-commit hooks** (if available)
```bash
# Would auto-run formatting/linting before commit
# pip install pre-commit
# pre-commit install
```

3. **Use VS Code extensions**
- Python
- Pylance
- Black Formatter
- Pytest

4. **Keep your fork updated**
```bash
# Add upstream remote
git remote add upstream https://github.com/diegoolguinw/kkf.git

# Fetch latest
git fetch upstream

# Update main branch
git checkout main
git rebase upstream/main
```

## Getting Help

- **Questions**: Create a GitHub Discussion
- **Bugs**: Create a GitHub Issue with reproducible example
- **Ideas**: Start a discussion before opening an issue
- **Contact**: dolguin@dim.uchile.cl

## Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Git Documentation](https://git-scm.com/doc)
- [Semantic Versioning](https://semver.org/)

## What's Next?

After setting up, read:
1. [CONTRIBUTING.md](CONTRIBUTING.md) - Full contribution guidelines
2. [examples/](examples/) - See how the library is used
3. [tests/](tests/) - Understand the test structure
4. [KKF/](KKF/) - Explore the actual implementation

Happy coding! 🚀
