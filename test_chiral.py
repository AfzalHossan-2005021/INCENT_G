import numpy as np
import anndata as ad
import pandas as pd
from incent import pairwise_align_unbalanced, pairwise_align_chiral

def make_symmetric_dataset():
    # Target: 2 symmetric circles (Left and Right)
    np.random.seed(42)
    # Left hemisphere
    theta = np.random.uniform(0, 2*np.pi, 200)
    r = np.random.uniform(0, 50, 200)
    left_x = r * np.cos(theta) - 100
    left_y = r * np.sin(theta)
    
    # Right hemisphere
    right_x = r * np.cos(theta) + 100
    right_y = r * np.sin(theta)
    
    X_target = np.vstack([np.column_stack([left_x, left_y]), np.column_stack([right_x, right_y])])
    # Cell types: somewhat symmetric, say distance from center determines type
    ct_target = np.array(['CT_A' if r_val < 25 else 'CT_B' for r_val in np.concatenate([r, r])])
    
    adata_target = ad.AnnData(np.random.rand(400, 10))
    adata_target.obsm['spatial'] = X_target
    adata_target.obs['cell_type_annot'] = pd.Categorical(ct_target)
    
    # Source: A smaller cut from the left hemisphere
    # Let's take x < -75 and y > 0
    mask = (left_x < -75) & (left_y > 0)
    source_x = left_x[mask]
    source_y = left_y[mask]
    source_ct = ct_target[:200][mask]
    
    X_source = np.column_stack([source_x, source_y])
    
    # Introduce arbitrary transformation (e.g. 90 degree rotation and a FLIP!)
    R_flip = np.array([[-1, 0], [0, 1]]) # Flip X
    R_rot = np.array([[0, -1], [1, 0]]) # 90 deg rot
    
    X_source_transformed = X_source @ R_flip @ R_rot + np.array([50, -200])
    
    adata_source = ad.AnnData(np.random.rand(len(X_source), 10))
    adata_source.obsm['spatial'] = X_source_transformed
    adata_source.obs['cell_type_annot'] = pd.Categorical(source_ct)
    
    return adata_source, adata_target, X_source

def main():
    print("Generating synthetic symmetric dataset...")
    adata_source, adata_target, original_X = make_symmetric_dataset()
    print(f"Source cells: {len(adata_source)}, Target cells: {len(adata_target)}")
    
    print("\nRunning standard FGW (Incent Unbalanced)...")
    try:
        pi_fgw = pairwise_align_unbalanced(
            adata_source, adata_target, alpha=0.5, beta=1.0, gamma=0.0, radius=20,
            filePath='./tmp_out', sliceA_name='source', sliceB_name='target'
        )
        print("FGW coupling shape:", pi_fgw.shape)
        # Check alignment mapping (does it map to Left or Right?)
        # Indices 0..199 are Left, 200..399 are Right
        mapped_to_right = np.sum(pi_fgw[:, 200:])
        mapped_to_left = np.sum(pi_fgw[:, :200])
        print(f"FGW Mass to Left: {mapped_to_left:.2f}, Mass to Right: {mapped_to_right:.2f}")
    except Exception as e:
        print("FGW error:", e)
        
    print("\nRunning novel CHIRAL-ST...")
    try:
        pi_chiral, R, t = pairwise_align_chiral(
            adata_source, adata_target, alpha=0.5, gamma=1.0, radii=[10, 20],
            filePath='./tmp_out', sliceA_name='source', sliceB_name='target',
            reg_marginals=1.0, epsilon=0.01, max_iter_em=10
        )
        mapped_to_right_c = np.sum(pi_chiral[:, 200:])
        mapped_to_left_c = np.sum(pi_chiral[:, :200])
        print(f"CHIRAL Mass to Left: {mapped_to_left_c:.2f}, Mass to Right: {mapped_to_right_c:.2f}")
        print("Recovered R:", R)
        print("Recovered T:", t)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
