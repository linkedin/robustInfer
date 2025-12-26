"""
JAX-based implementations for RobustInfer.

This subpackage contains JAX implementations that require JAX as a dependency.
Install with: pip install robustinfer[jax]
"""

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
except ImportError as e:
    raise ImportError(
        'JAX is required for this subpackage. '
        'Install with: pip install robustinfer[jax] or pip install jax'
    ) from e

from .drgu import DRGUJax

__all__ = ['DRGUJax']
