# Changelog

All notable changes to the Kernel Koopman Kalman Filter (KKF) project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0] - 2026-06-30

### Added
- `reg` parameter on `KoopmanOperator.compute_edmd` and `apply_koopman_kalman_filter`:
  Tikhonov (jitter) regularization for the Gram-matrix inversion, improving numerical
  stability on ill-conditioned kernels. Defaults to `1e-10` (previous behavior ≈ no jitter).
- `KoopmanOperator.optimize_kernel`: clearer name for the former `opt_kernel`.
- Descriptive read-only properties on `KoopmanOperator` (`koopman_matrix`, `gram_matrix`,
  `output_matrix`, `state_matrix`, `dictionary`, `feature_map`) aliasing the single-letter
  attributes `U`, `G`, `C`, `B`, `X`, `phi` for discoverability.

### Changed
- **BEHAVIOR:** The default of `optimize` is now `False` in both
  `apply_koopman_kalman_filter` and `KoopmanOperator.compute_edmd` (was `True`). The fast,
  no-optimization path is now the default; pass `optimize=True` to fit kernel
  hyperparameters as before.

### Deprecated
- `create_additive_system` now emits a `DeprecationWarning` and was removed from the
  top-level `__all__` (still importable). It is incompatible with the filter's covariance
  sampling; build a `DynamicalSystem` directly instead.
- `KoopmanOperator.opt_kernel` is a deprecated alias for `optimize_kernel`.

### Fixed
- Covariance helpers (`compute_initial_covariance`, `compute_dynamics_covariance`) now
  always return 2D matrices, even with a single feature (`n_features=1`), matching
  `compute_observation_covariance`.
- `DynamicalSystem` now validates that the measurement dimension `ny` is positive,
  mirroring the existing `nx` check.

### Removed
- Deleted the empty, unused `kkf/utils.py` module.

## [0.25] - 2026-06-30

### Changed
- **BREAKING:** Renamed the import package from `KKF` to `kkf` (lowercase) to
  match the PyPI distribution name and PEP 8. User code must update imports from
  `from KKF import ...` to `from kkf import ...`.

### Fixed
- Corrected stale module names in the `CONTRIBUTING.md` project tree and fixed an
  outdated `kkf.applyKKF` import in the bug-report issue template (the module is
  `kkf.filter`).

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
- Expanded documentation and examples on top of the stable 1.0 API

## Release process

For maintainers:

1. Run the tests: `pytest tests/`
2. Bump the version in `pyproject.toml`
3. Update this changelog
4. Tag the release (`git tag vX.Y`) and push tags — GitHub Actions builds and
   publishes to PyPI

For bug reports and feature requests, see the GitHub Issues page.
