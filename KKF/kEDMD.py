"""Backward compatibility wrapper for kEDMD.

This module is deprecated. Please import from KKF.koopman instead:
    from KKF.koopman import KoopmanOperator

Or use the main package imports:
    from KKF import KoopmanOperator
"""

# Re-export from the new module location for backward compatibility
from .koopman import KoopmanOperator

__all__ = ["KoopmanOperator"]
        