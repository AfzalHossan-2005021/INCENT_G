# CHIRAL-ST: Chirality-Aware Highly Integrated Robust Alignment of Spatial Transcriptomics

## 1. The Core Problem with Existing State-of-the-Art (SOTA)
Current SOTA spatial alignment algorithms (PASTE, PASTE2, INCENT) rely heavily on **Gromov-Wasserstein (GW)** or **Fused Gromov-Wasserstein (FGW)** optimal transport. 
FGW aligns slices by comparing their intrinsic pairwise distance matrices ($D_A$ and $D_B$). It looks for an alignment where distances between points in slice A match the distances between matched points in slice B.

**The Fatal Flaw (The Symmetry Problem):** Pairwise distance matrices are invariant to all Euclidean isometries—including **reflections**. In a symmetric organ like the brain, the left hemisphere and the right hemisphere are mirror images. Their intrinsic pairwise distance matrices are statistically identical. When a small slice is cut from the left hemisphere, FGW cannot mathematically distinguish between mapping it back to the true left hemisphere (a rotation + translation) versus mapping it to the right hemisphere (which requires a reflection). Because biological tissues have chiral (handedness) properties, right and left are not interchangeable, but FGW is blind to chirality.

Furthermore, cross-timepoint alignment involves massive gene expression shifts and spatial deformations (growth, morphological changes), which breaks both strict gene-cosine similarities and strict pairwise distance conservation.

## 2. Novel Algorithmic Framework: Unbalanced Co-Optimal Transport with Chirality Preservation

To solve this, we propose abandoning Gromov-Wasserstein in favor of a **Joint Spatial-Transformation and Unbalanced Wasserstein Optimization**, alternating between finding an alignment plan and explicitly computing the spatial transformation.

### 2.1 Spatially Coherent Transformation (Preventing "Tearing" and Symmetry Misalignment)
Instead of aligning pairwise matrices $D_A$ and $D_B$ independently for each point, we map coordinates $X \in \mathbb{R}^{N \times d}$ directly to $Y \in \mathbb{R}^{M \times d}$ using an explicit global transformation function $T(X) = X R + t$.
Because FGW lacks a global spatial framework, it often "tears" a slice apart—for instance, mapping the top half of a slice to the left hemisphere and the bottom half to the right hemisphere. By enforcing a unified spatial transformation $T(X)$, we guarantee that the slice remains physically intact.

Crucially, to handle arbitrary reflections (e.g., if a tissue slice was physically flipped upside-down on the glass slide in the lab), we evaluate the optimal transformation over the full Orthogonal Group $O(d)$. We do this by explicitly computing the best pure rotation ($SO(d)$, determinant +1) and the best rotation with a reflection (determinant -1). The physical chirality of the tissue boundary features will naturally yield a significantly lower cost for the correct orientation, mathematically forbidding the tissue from being erroneously mapped to the *wrong* symmetric hemisphere, while correctly "un-flipping" lab-induced reflections.

### 2.2 Unbalanced Sinkhorn for Unknown Partial Overlap
Because either the source or the target may have missing or extra parts (or both), a strict 1-to-1 mapping fails. We use **Unbalanced Entropic Optimal Transport (FUGW equivalent but on Euclidean space)**.
The cost function becomes:
$$ C_{ij} = \alpha \| X_i R + t - Y_j \|^2 + (1 - \alpha) \text{Dist}_{bio}(F^A_i, F^B_j) $$
Mass relaxation allows unmatched cells (e.g., border regions cut off) to be "destroyed" (assigned zero transport mass) rather than forced into erroneous matches.

### 2.3 Cross-Timepoint Biological Robustness
Gene expression changes over time, rendering naive cosine distance ineffective. However, **local tissue topology is highly conserved**.
We replace raw gene expression with **Multi-Scale Spatial Barcodes**:
For each cell, we compute the distribution of cell types (or coarse spatial domain annotations) at multiple radii (e.g., $r=\{50, 100, 200\} \mu m$). This creates a structural topological vector that remains stable even if individual gene expressions drift over developmental timepoints.

### 2.4 Non-Rigid Deformation for Temporal Morphology
After a robust rigid initialization, cross-timepoint morphology differences are resolved by upgrading $T(X)$ from a rigid transformation to a **Coherent Point Drift (CPD) / Thin Plate Spline (TPS)** displacement field. The displacement field is regularized to maintain positive Jacobian determinants, preserving topology and preventing symmetric collapsing.

## 3. Algorithm Summary (The EM Loop)

**Input:** Source Slice $A$ (time $t_1$), Target Slice $B$ (time $t_2$). The sizes and overlap fractions are entirely unconstrained and symmetric regions may exist in either.
1. **Bio-Feature Extraction:** Compute Multi-Scale Spatial Barcodes $F^A, F^B$ based on cell-type neighborhoods to ensure time-invariant biological matching.
2. **Global Anchor Search (Chirality-Aware):** To avoid local minima and detect physical slide-flips, test $k$ coarse rotations $\theta_k$ for both the unflipped state and the flipped state (reflected along one axis). Calculate a fast approximation of the Unbalanced Wasserstein distance for each. Pick the rotation and flip-state with the minimum cost as the initialization $R^{(0)}, t^{(0)}$.
3. **Alternating Optimization:**
   * **E-Step (Coupling):** Given $R^{(k)}, t^{(k)}$, compute cost matrix $C$ combining transformed spatial distance and barcode distance. Solve Unbalanced Entropic OT to get coupling matrix $\pi$.
   * **M-Step (Spatial):** Given $\pi$, perform **Weighted Kabsch Algorithm (Procrustes)**. 
     * Compute cross-covariance $H = \tilde{X}^T \pi \tilde{Y}$.
     * $U, S, V^T = \text{SVD}(H)$.
     * Update $R^{(k+1)}$ while preserving the determinant (+1 or -1) established in the anchor search to maintain the correct physical chirality constraint.
     * Solve for $t^{(k+1)}$.
4. **Non-Rigid Fine-Tuning (Optional):** Once aligned rigidly to the correct symmetric hemisphere, compute a non-rigid B-spline or Gaussian displacement to accommodate temporal organ growth/deformation.

## 4. Why this is Publishable in Top Journals
* **Biological Grounding:** Recognizes that biological tissues are chiral. Explicitly models this, unlike all existing GW-based methods.
* **Solves the Overlap-Symmetry-Tearing Paradox:** Previously, partial overlap + symmetry was an unsolved problem because FGW tears slices apart to map onto symmetric regions. Global spatial transformation keeps slices physically coherent, and strict determinant tracking prevents mapping left-cuts to right-hemispheres while gracefully allowing valid flipped-slides.
* **Low Hyperparameters:** Eliminates complex GW tuning. Requires only $\alpha$ (space vs biology ratio) and a marginal relaxation penalty.
* **Cross-time robustness:** Shifts reliance from volatile gene expression to conserved multi-scale topological barcodes.
