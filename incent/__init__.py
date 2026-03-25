"""
__init__.py — INCENT package (with INCENT-Align extension)
"""
from .core import (
    pairwise_align,
    pairwise_align_unbalanced,
    pairwise_align_chiral,
    neighborhood_distribution,
    cosine_distance,
)
from .utils import (
    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,
)

__all__ = [
    # Original INCENT
    'pairwise_align',
    'pairwise_align_unbalanced',
    'pairwise_align_chiral',
    'neighborhood_distribution',
    'cosine_distance',
    'fused_gromov_wasserstein_incent',
    'jensenshannon_divergence_backend',
    'pairwise_msd',
    'to_dense_array',
    'extract_data_matrix',
]