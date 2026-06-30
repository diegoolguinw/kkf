# Changelog

All notable changes to the Kernel Koopman Kalman Filter (KKF) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-03-26

### Added
- Test suite covering system initialization, EDMD computation, and filtering:
  - `tests/test_core.py` for the main functionality
  - `tests/test_utils.py` for edge cases and numerical stability
  - `tests/conftest.py` with shared pytest fixtures
- `pyproject.toml` with the full project configuration, supporting Python 3.8–3.12,
  and `dev`, `docs`, and `viz` extras.
- GitHub Actions for testing on Ubuntu, Windows, and macOS, with code-quality
  checks (black, flake8, mypy, isort) and automatic publishing to PyPI on release.
- Tooling configuration: black (100-char lines), flake8 (`.flake8`), mypy, isort,
  and `.editorconfig`.
- Three runnable examples with their own README:
  - `examples/01_basic_linear_system.py`
  - `examples/02_sir_epidemic_model.py`
  - `examples/03_advanced_parameter_exploration.py`
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- `matplotlib` added to the dependencies (it was used by the examples but missing),
  with explicit version constraints across all dependencies.

### Changed
- `setup.py` now delegates to `pyproject.toml`.
- Version is specified in `pyproject.toml` (0.2.0).
- Minimum Python raised to 3.8 (was 3.6).
- Added project URLs and classifiers to the package metadata.

### Fixed
- Missing `matplotlib` dependency that the README referenced.

## [0.1.1] - 2024-01-15

### Fixed
- Bug in covariance computation.

## [0.1.0] - 2024-01-09

### Added
- Initial release of KKF.
- Core modules: system definition and sampling, Koopman operator approximation
  via kernel EDMD, the Koopman Kalman Filter, the solution data structure, and
  covariance computation.
- README with an SIR model example.
- MIT License.

---

## Roadmap

Ideas under consideration, not commitments:

- Extended and Unscented Kalman Filter variants
- Adaptive noise estimation and multi-rate filtering
- GPU acceleration and benchmarking
- A stable 1.0 API with expanded documentation and examples

## Release process

For maintainers:

1. Run the tests: `pytest tests/`
2. Bump the version in `pyproject.toml`
3. Update this changelog
4. Tag the release (`git tag vX.Y`) and push tags — GitHub Actions builds and
   publishes to PyPI

For bug reports and feature requests, see the GitHub Issues page.
