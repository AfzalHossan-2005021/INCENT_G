# Implementation Plan: Chirality-Aware Spatial Alignment

This plan outlines the steps to implement the proposed CHIRAL-ST algorithm, to effectively handle partial overlaps, symmetric tissue regions, and cross-timepoint variations.

## Proposed Changes

### INCENT Module
We will add a new suite of algorithms into the `incent` package, alongside the existing FGW methods.

#### [MODIFY] [incent/core.py](file:///c:/Users/afzal/OneDrive/Desktop/New%20folder/INCENT_G/incent/core.py)
We will append the new CHIRAL-ST functions to the core module.
* `compute_spatial_barcodes(adata, radii_list)`: Computes a concatenated robust feature vector representing the local topology (cell type distributions at multiple radii).
* `weighted_procrustes(X, Y, pi)`: Implements the Kabsch algorithm with a strict $SO(d)$ (Rotation only, no reflection) constraint. It calculates optimal rotation matrix $R$ and translation vector $t$.
* `chiral_co_ot_align(...)`: The primary EM-loop function:
  1. Computes Spatial Barcodes.
  2. Runs a fast coarse rotation search (testing multiple angles for both the *unflipped* and *flipped* states) to find the best initialization, solving the arbitrary slide-flipping problem.
  3. Alternates between computing Unbalanced Optimal Transport (using `ot.unbalanced.sinkhorn_unbalanced` which operates on physical spatial metrics) and updating the Rigid spatial transformation (maintaining the solved determinant parity to prevent tearing/symmetric cross-mapping).
  4. Optionally applies a Coherent Point Drift (CPD) or similar non-rigid displacement for cross-timepoint deformation.

#### [MODIFY] [incent/__init__.py](file:///c:/Users/afzal/OneDrive/Desktop/New%20folder/INCENT_G/incent/__init__.py)
* Expose `chiral_co_ot_align` and necessary utilities in the top-level package namespace.

## User Review Required
> [!IMPORTANT]
> The approach shifts away from Gromov-Wasserstein (which is intrinsically reflection-invariant) and uses a **Joint Rigid/Non-Rigid Unbalanced Wasserstein** optimization with a strict Procrustes chirality constraint. This directly solves the issue of symmetric regions (e.g. mapping a left-cut exactly back to the left hemisphere despite rotated inputs). Please review the mathematical research in the [algorithm_research_chiral_st.md](file:///c:/Users/afzal/.gemini/antigravity/brain/8c0ee1c5-5048-4a2d-bb47-9b62298c962a/algorithm_research_chiral_st.md) artifact. Let me know if you approve this SOTA approach!

## Verification Plan

### Automated / Synthetic Testing
1. I will write a script `tests/test_chiral_symmetry.py` that generates a synthetic 2D organ (e.g., two symmetric circles representing left and right hemispheres) with distinct but spatially symmetric cell types.
2. I will cut a small "slice" out of the left hemisphere and randomly rotate and translate it.
3. I will test the existing [pairwise_align_unbalanced](file:///c:/Users/afzal/OneDrive/Desktop/New%20folder/INCENT_G/incent/core.py#492-738) and demonstrate that it sometimes maps it to the right hemisphere and fails to find the true rigid rotation.
4. I will test the new `chiral_co_ot_align` and assert that it perfectly recovers the true rotation, true translation, and maps exclusively to the left hemisphere.

### Manual Verification
The user can apply this algorithm directly to their MERFISH brain datasets. The rotation matrix $R$ and translation $t$ outputted by the algorithm can be directly applied to the spatial coordinates of the source slice, physically moving it back to its original location perfectly.
