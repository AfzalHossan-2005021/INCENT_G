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
from .align import (
    compute_lntd,
    lntd_cost_matrix,
    compute_coupling,
)
from .transform import (
    recover_transform,
    apply_transform,
    weighted_procrustes,
    enumerate_candidates,
    RigidTransform,
)
from .temporal import (
    rank_harmonize,
    celltype_weights,
    temporal_cost_blend,
)
from .pipeline import (
    align_slices,
    AlignmentResult,
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
    # INCENT-Align: topology-aware coupling
    'compute_lntd',
    'lntd_cost_matrix',
    'compute_coupling',
    # INCENT-Align: transform recovery
    'recover_transform',
    'apply_transform',
    'weighted_procrustes',
    'enumerate_candidates',
    'RigidTransform',
    # INCENT-Align: cross-timepoint
    'rank_harmonize',
    'celltype_weights',
    'temporal_cost_blend',
    # INCENT-Align: pipeline
    'align_slices',
    'AlignmentResult',
]