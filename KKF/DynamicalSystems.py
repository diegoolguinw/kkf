"""Backward compatibility wrapper.

This module is deprecated. Please import from KKF.systems instead:
    from KKF.systems import DynamicalSystem, create_additive_system

Or use the main package imports:
    from KKF import DynamicalSystem, create_additive_system
"""

# Re-export from the new module location for backward compatibility
from .systems import DynamicalSystem, create_additive_system

__all__ = ["DynamicalSystem", "create_additive_system"]