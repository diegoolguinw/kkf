"""Backward compatibility wrapper for applyKKF.

This module is deprecated. Please import from KKF.filter instead:
    from KKF.filter import apply_koopman_kalman_filter

Or use the main package imports:
    from KKF import apply_koopman_kalman_filter
"""

# Re-export from the new module location for backward compatibility
from .filter import apply_koopman_kalman_filter

__all__ = ["apply_koopman_kalman_filter"]