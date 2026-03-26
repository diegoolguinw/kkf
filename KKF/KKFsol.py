"""Backward compatibility wrapper for KKFsol.

This module is deprecated. Please import from KKF.solution instead:
    from KKF.solution import KoopmanKalmanFilterSolution

Or use the main package imports:
    from KKF import KoopmanKalmanFilterSolution
"""

# Re-export from the new module location for backward compatibility
from .solution import KoopmanKalmanFilterSolution

__all__ = ["KoopmanKalmanFilterSolution"]