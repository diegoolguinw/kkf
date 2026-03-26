"""Pytest configuration and shared fixtures."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    import numpy as np
    
    np.random.seed(42)
    return 42
