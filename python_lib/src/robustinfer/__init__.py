"""
robustinfer: A Python library for robust inference.
"""

__version__ = '0.1.0'

from .drgu import DRGU  # PyTorch implementation (default)

# JAX implementation (optional dependency)
try:
    from .jax import DRGUJax  # JAX implementation

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    DRGUJax = None
from .ee import EstimatingEquation
from .io.pair_dataset import PairBatchIterableDataset
from .io.samplers import KPartnersSampler

# Streamlined mini-batch implementation (simplified for debugging)
from .minibatch.drgu_minibatch import DRGUMiniBatch
from .minibatch.estimating_equations import drgu_compute_B_U
from .minibatch.minibatch_fisher import MiniBatchFisherScoring, Penalty
from .mwu import zero_trimmed_u

# TODO: Add back when advanced features are restored
# from .minibatch.estimating_equations import example_compute_B_U
# from .io.samplers import StratifiedKPartnersSampler, ReservoirKPartnersSampler

__all__ = [
    'DRGU',  # Default PyTorch implementation
    'DRGUMiniBatch',
    'EstimatingEquation',
    'zero_trimmed_u',
    # Essential mini-batch functionality
    'MiniBatchFisherScoring',
    'Penalty',
    'drgu_compute_B_U',
    'KPartnersSampler',
    'PairBatchIterableDataset',
    # TODO: Add back advanced features when restored:
    # 'example_compute_B_U',
    # 'StratifiedKPartnersSampler',
    # 'ReservoirKPartnersSampler',
]

# Add JAX version if available
if _HAS_JAX:
    __all__.append('DRGUJax')
