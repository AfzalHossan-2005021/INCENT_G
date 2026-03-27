"""
core.py — INCENT  (balanced FGW  +  unbalanced FUGW variant)
=============================================================
Two public functions share one private preprocessing helper:

  pairwise_align()            — original INCENT FGW (balanced)
  pairwise_align_unbalanced() — FUGW via ot.gromov.fused_unbalanced_gromov_wasserstein

Shared fix in both: D_A and D_B are normalised by the SAME scale
(max of D_B) so that GW correctly embeds A as a spatial subregion of B.
"""

import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import ot

from typing import Optional, Tuple, Union
from numpy.typing import NDArray
from anndata import AnnData

from .utils import (
    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood distribution
# ─────────────────────────────────────────────────────────────────────────────

def neighborhood_distribution(curr_slice: AnnData, radius: float) -> np.ndarray:
    """
    Normalised cell-type neighbourhood distribution for every cell.

    Parameters
    ----------
    curr_slice : AnnData — .obsm['spatial'], .obs['cell_type_annot'] required
    radius     : float  — Euclidean radius of the local neighbourhood

    Returns
    -------
    dist : (n_cells, n_cell_types) float64, rows sum to 1
    """
    from tqdm import tqdm
    from sklearn.neighbors import BallTree

    cell_types     = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_ct      = np.unique(cell_types)
    ct2idx         = {c: i for i, c in enumerate(unique_ct)}
    coords         = curr_slice.obsm['spatial']
    n, K           = curr_slice.shape[0], len(unique_ct)

    tree           = BallTree(coords)
    neighbor_lists = tree.query_radius(coords, r=radius)

    dist = np.zeros((n, K), dtype=np.float64)
    for i in tqdm(range(n), desc="Neighbourhood distribution"):
        for idx in neighbor_lists[i]:
            dist[i, ct2idx[cell_types[idx]]] += 1.0

    row_sums = dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dist / row_sums


# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance on gene expression
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name,
                    filePath, use_rep=None, use_gpu=False,
                    nx=ot.backend.NumpyBackend(), overwrite=False):
    """Pairwise cosine distance on gene expression. Results cached to filePath."""
    A_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep)))
    B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, use_rep)))

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"

    if os.path.exists(fileName) and not overwrite:
        print("Loading cached cosine distance matrix")
        mat = np.load(fileName)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            return torch.from_numpy(mat).cuda()
        return mat

    print("Computing cosine distance matrix")
    if isinstance(s_A, torch.Tensor) and isinstance(s_B, torch.Tensor):
        norm_A = s_A / s_A.norm(dim=1, keepdim=True)
        norm_B = s_B / s_B.norm(dim=1, keepdim=True)
        mat    = 1.0 - torch.mm(norm_A, norm_B.T)
        np.save(fileName, mat.cpu().detach().numpy())
        return mat
    else:
        from sklearn.metrics.pairwise import cosine_distances
        mat = cosine_distances(
            to_dense_array(s_A) if not isinstance(s_A, np.ndarray) else s_A,
            to_dense_array(s_B) if not isinstance(s_B, np.ndarray) else s_B,
        )
        np.save(fileName, mat)
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# Helper: bring any matrix to numpy float64
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _hard_assignment_from_coupling(pi: np.ndarray) -> np.ndarray:
    """Project a soft coupling to a hard one-to-one assignment matrix."""
    from scipy.optimize import linear_sum_assignment

    pi_np = np.asarray(pi, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(-pi_np)

    hard_pi = np.zeros_like(pi_np, dtype=np.float64)
    hard_pi[row_ind, col_ind] = 1.0
    return hard_pi


def _ensure_rigid_transform(R, t, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce a rotation/translation pair into valid rigid-transform shapes."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)

    if R.ndim != 2 or R.shape != (dim, dim):
        R = np.eye(dim, dtype=np.float64)

    if t.size != dim:
        t = np.zeros(dim, dtype=np.float64)

    return R, t


def _sanitize_coupling(pi):
    """Convert a coupling to a finite float64 numpy array."""
    pi = np.asarray(pi, dtype=np.float64)
    if not np.isfinite(pi).all():
        pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    return pi


# ─────────────────────────────────────────────────────────────────────────────
# Private preprocessing helper — shared by both align functions
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(
        sliceA, sliceB,
        alpha, beta, gamma, radius, filePath,
        use_rep, G_init, a_distribution, b_distribution,
        numItermax, backend, use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite,
        neighborhood_dissimilarity,
        logFile,
):
    """
    All preprocessing shared by pairwise_align and pairwise_align_unbalanced.
    Returns a dict with every artefact both callers need.
    """
    # ── GPU / backend ─────────────────────────────────────────────────────────
    if use_gpu:
        if torch.cuda.is_available():
            backend = ot.backend.TorchBackend()
            if gpu_verbose:
                print("GPU available — using CUDA.")
        else:
            use_gpu = False
            backend = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("GPU requested but not available — using CPU.")
    else:
        backend = ot.backend.NumpyBackend()
        if gpu_verbose:
            print("Using CPU backend.")
    nx = backend

    # ── Input validation ───────────────────────────────────────────────────────
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Empty AnnData: {s}")

    # ── Shared genes ───────────────────────────────────────────────────────────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between slices.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]

    # ── Shared cell types ──────────────────────────────────────────────────────
    shared_ct = (pd.Index(sliceA.obs['cell_type_annot'])
                 .unique()
                 .intersection(pd.Index(sliceB.obs['cell_type_annot']).unique()))
    if len(shared_ct) == 0:
        raise ValueError("No shared cell types between slices.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_ct)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_ct)]

    logFile.write(f"n_A={sliceA.shape[0]}  n_B={sliceB.shape[0]}\n")
    logFile.write(f"shared_genes={len(shared_genes)}  shared_ct={len(shared_ct)}\n\n")

    # ── Spatial distance matrices ──────────────────────────────────────────────
    coordsA = nx.from_numpy(sliceA.obsm['spatial'].copy())
    coordsB = nx.from_numpy(sliceB.obsm['spatial'].copy())
    if isinstance(nx, ot.backend.TorchBackend):
        coordsA = coordsA.float()
        coordsB = coordsB.float()

    D_A = ot.dist(coordsA, coordsA, metric='euclidean')
    D_B = ot.dist(coordsB, coordsB, metric='euclidean')

    # ★ CRITICAL FIX: Shared-scale normalization ★
    # To map a smaller partial slice into a larger slice realistically, 
    # their spatial distances MUST be measured on the exact same scale. 
    # Independent normalization blows the small slice up to the size of the 
    # large slice, causing drastic spatial misalignments.
    scale = max(float(nx.max(D_A)), float(nx.max(D_B)))
    if scale < 1e-12:
        raise ValueError("Spatial coordinates are collapsed.")
        
    D_A = D_A / scale
    D_B = D_B / scale

    # Get max for logging
    logFile.write(f"Shared-scale normalisation factor: {scale:.4f}\n")
    logFile.write(f"Normalized by max: D_A max={float(nx.max(D_A)):.6f}, D_B max={float(nx.max(D_B)):.6f}\n")

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # ── Gene-expression cost ───────────────────────────────────────────────────
    cosine_dist_gene_expr = cosine_distance(
        sliceA, sliceB, sliceA_name, sliceB_name, filePath,
        use_rep=use_rep, use_gpu=use_gpu, nx=nx, overwrite=overwrite)

    # ── Cell-type mismatch penalty ─────────────────────────────────────────────
    lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (lab_A[:, None] != lab_B[None, :]).astype(np.float64)

    if isinstance(cosine_dist_gene_expr, torch.Tensor):
        M_ct = torch.from_numpy(M_celltype).to(cosine_dist_gene_expr.device)
        M1   = (1.0 - beta) * cosine_dist_gene_expr + beta * M_ct
    else:
        M1 = nx.from_numpy(
            (1.0 - beta) * cosine_dist_gene_expr + beta * M_celltype)

    logFile.write(f"M_celltype shape={M_celltype.shape}  beta={beta}\n")

    # ── Neighbourhood distributions ────────────────────────────────────────────
    nd_cache_A = f"{filePath}/nd_{sliceA_name}.npy"
    nd_cache_B = f"{filePath}/nd_{sliceB_name}.npy"

    if os.path.exists(nd_cache_A) and not overwrite:
        print("Loading cached neighbourhood distribution A")
        nd_A = np.load(nd_cache_A)
    else:
        print("Computing neighbourhood distribution A")
        nd_A = neighborhood_distribution(sliceA, radius=radius)
        np.save(nd_cache_A, nd_A)

    if os.path.exists(nd_cache_B) and not overwrite:
        print("Loading cached neighbourhood distribution B")
        nd_B = np.load(nd_cache_B)
    else:
        print("Computing neighbourhood distribution B")
        nd_B = neighborhood_distribution(sliceB, radius=radius)
        np.save(nd_cache_B, nd_B)

    nd_A += 0.01
    nd_B += 0.01

    if use_gpu:
        if isinstance(nd_A, np.ndarray):
            nd_A = torch.from_numpy(nd_A).cuda()
        if isinstance(nd_B, np.ndarray):
            nd_B = torch.from_numpy(nd_B).cuda()

    # ── Neighbourhood dissimilarity M2 ────────────────────────────────────────
    if neighborhood_dissimilarity == 'jsd':
        jsd_cache = f"{filePath}/jsd_{sliceA_name}_{sliceB_name}.npy"
        if os.path.exists(jsd_cache) and not overwrite:
            print("Loading cached JSD matrix")
            js_dist = np.load(jsd_cache)
            M2 = (torch.from_numpy(js_dist).cuda()
                  if use_gpu and isinstance(nx, ot.backend.TorchBackend)
                  else nx.from_numpy(js_dist))
        else:
            print("Computing JSD matrix")
            js_dist = jensenshannon_divergence_backend(nd_A, nd_B)
            if isinstance(js_dist, torch.Tensor):
                np.save(jsd_cache, js_dist.cpu().numpy())
                M2 = js_dist
            else:
                np.save(jsd_cache, js_dist)
                M2 = nx.from_numpy(js_dist)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(nd_A, torch.Tensor):
            na  = nd_A.cuda() if use_gpu else nd_A
            nb  = nd_B.cuda() if use_gpu else nd_B
            num = na @ nb.T
            den = na.norm(dim=1)[:, None] * nb.norm(dim=1)[None, :]
            M2  = 1.0 - num / (den + 1e-12)
        else:
            na  = np.asarray(nd_A)
            nb  = np.asarray(nd_B)
            num = na @ nb.T
            den = (np.linalg.norm(na, axis=1)[:, None]
                   * np.linalg.norm(nb, axis=1)[None, :])
            M2  = nx.from_numpy(1.0 - num / (den + 1e-12))

    elif neighborhood_dissimilarity == 'msd':
        na = nd_A.cpu().numpy() if isinstance(nd_A, torch.Tensor) else np.asarray(nd_A)
        nb = nd_B.cpu().numpy() if isinstance(nd_B, torch.Tensor) else np.asarray(nd_B)
        M2 = nx.from_numpy(pairwise_msd(na, nb))

    else:
        raise ValueError(
            f"neighborhood_dissimilarity must be 'jsd', 'cosine', or 'msd'. "
            f"Got: {neighborhood_dissimilarity!r}")

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        if not isinstance(M1, torch.Tensor):
            M1 = torch.from_numpy(np.asarray(M1)).cuda()
        if not isinstance(M2, torch.Tensor):
            M2 = torch.from_numpy(np.asarray(M2)).cuda()
        M1, M2 = M1.cuda(), M2.cuda()

    # ── Marginals ──────────────────────────────────────────────────────────────
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)

    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        a = a.cuda()
        b = b.cuda()

    # ── Initial transport plan ─────────────────────────────────────────────────
    if G_init is not None:
        G_init_t = nx.from_numpy(G_init)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init_t = G_init_t.float()
            if use_gpu:
                G_init_t = G_init_t.cuda()
    else:
        G_init_t = None

    return dict(
        nx=nx, use_gpu=use_gpu,
        sliceA=sliceA, sliceB=sliceB,
        D_A=D_A, D_B=D_B,
        M1=M1, M2=M2,
        cosine_dist_gene_expr=cosine_dist_gene_expr,
        a=a, b=b,
        G_init_t=G_init_t,
        nd_dissim=neighborhood_dissimilarity,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 1 — balanced FGW  (original INCENT, unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align(
    sliceA:    AnnData,
    sliceB:    AnnData,
    alpha:     float,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    use_rep:   Optional[str]   = None,
    G_init                     = None,
    a_distribution             = None,
    b_distribution             = None,
    norm:      bool            = False,
    numItermax: int            = 6000,
    backend                    = ot.backend.NumpyBackend(),
    use_gpu:   bool            = False,
    return_obj: bool           = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    hard_assignment: bool      = False,
    **kwargs,
) -> Union[NDArray[np.floating],
           Tuple[NDArray[np.floating], float, float, float, float]]:
    """
    Balanced Fused Gromov-Wasserstein alignment (original INCENT).

    Parameters
    ----------
    alpha  : weight of the GW spatial term  (0 = biology only, 1 = space only)
    beta   : weight of cell-type mismatch inside M1
    gamma  : weight of neighbourhood dissimilarity M2
    radius : neighbourhood radius (same units as spatial coordinates)

    Key fix vs original INCENT
    --------------------------
    D_A and D_B are both normalised by max(D_B), preserving the true size
    relationship so GW embeds A as a spatial subregion of B.
    """
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align — INCENT balanced FGW\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n\n")

    p = _preprocess(
        sliceA, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, G_init, a_distribution, b_distribution,
        numItermax, backend, use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        logFile,
    )
    nx     = p['nx']
    M1     = p['M1']
    M2     = p['M2']
    D_A    = p['D_A']
    D_B    = p['D_B']
    a      = p['a']
    b      = p['b']
    sliceA = p['sliceA']
    sliceB = p['sliceB']

    # Initial objective logging
    G0_np = np.ones((sliceA.shape[0], sliceB.shape[0])) / (
        sliceA.shape[0] * sliceB.shape[0])

    init_nb = 0.0
    if p['nd_dissim'] == 'jsd':
        init_nb = float(np.sum(_to_np(M2) * G0_np))
        logFile.write(f"Initial obj neighbour (jsd): {init_nb:.6f}\n")
    init_gene = float(np.sum(_to_np(p['cosine_dist_gene_expr']) * G0_np))
    logFile.write(f"Initial obj gene (cosine):    {init_gene:.6f}\n\n")

    # ── Solve balanced FGW ────────────────────────────────────────────────────
    pi, logw = fused_gromov_wasserstein_incent(
        M1, M2, D_A, D_B, a, b,
        G_init=p['G_init_t'],
        loss_fun='square_loss',
        alpha=alpha,
        gamma=gamma,
        log=True,
        numItermax=numItermax,
        verbose=verbose,
        use_gpu=p['use_gpu'],
    )
    pi = nx.to_numpy(pi)

    if hard_assignment:
        pi = _hard_assignment_from_coupling(pi)

    # Final objective logging
    final_nb = 0.0
    if p['nd_dissim'] == 'jsd':
        max_idx  = np.argmax(pi, axis=1)
        jsd_np   = _to_np(M2)
        final_nb = float(sum(pi[i, max_idx[i]] * jsd_np[i, max_idx[i]]
                             for i in range(len(max_idx))))
        logFile.write(f"Final obj neighbour (jsd): {final_nb:.6f}\n")

    final_gene = float(np.sum(_to_np(p['cosine_dist_gene_expr']) * pi))
    logFile.write(f"Final obj gene (cosine):   {final_gene:.6f}\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    if p['use_gpu'] and isinstance(nx, ot.backend.TorchBackend):
        torch.cuda.empty_cache()

    if return_obj:
        return pi, init_nb, init_gene, final_nb, final_gene
    return pi


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 2 — unbalanced FUGW  (new)
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align_unbalanced(
    sliceA:    AnnData,
    sliceB:    AnnData,
    alpha:     float,
    beta:      float,
    gamma:     float,
    radius:    float,
    filePath:  str,
    # ── new FUGW parameters ───────────────────────────────────────────────────
    reg_marginals:     float = 0.1,
    epsilon:           float = 0.01,
    divergence:        str   = 'kl',
    unbalanced_solver: str   = 'sinkhorn_log',
    max_iter:          int   = 1000,
    tol:               float = 1e-7,
    max_iter_ot:       int   = 500,
    tol_ot:            float = 1e-7,
    # ── identical to pairwise_align ───────────────────────────────────────────
    use_rep:   Optional[str]   = None,
    G_init                     = None,
    a_distribution             = None,
    b_distribution             = None,
    norm:      bool            = False,
    numItermax: int            = 6000,   # kept for API compat (unused by FUGW)
    backend                    = ot.backend.NumpyBackend(),
    use_gpu:   bool            = False,
    return_obj: bool           = False,
    verbose:   bool            = False,
    gpu_verbose: bool          = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    hard_assignment: bool      = False,
    **kwargs,
) -> Union[NDArray[np.floating],
           Tuple[NDArray[np.floating], float, float, float, float]]:
    """
    Unbalanced Fused Gromov-Wasserstein alignment.

    Uses ``ot.gromov.fused_unbalanced_gromov_wasserstein`` as the solver.
    Everything before the solver call is identical to ``pairwise_align``:
    same shared-scale normalisation, same M1 / M2 construction.

    The unbalanced marginal relaxation allows cells with no good counterpart
    in the other slice to remain (partially) unmatched, which naturally
    handles unknown partial overlap without specifying the overlap fraction.

    Parameters shared with pairwise_align
    ----------------------------------------
    alpha  : GW spatial weight [0, 1].
             Converted internally to FUGW's linear-term weight:
               alpha_fugw = (1 - alpha) / alpha
             so the GW/biology ratio is the same as in the balanced version.
             alpha=0.5 → alpha_fugw=1.0 (equal weighting).
    beta   : cell-type mismatch weight inside M1
    gamma  : neighbourhood dissimilarity weight
    radius : neighbourhood radius (same units as spatial coords)

    New FUGW-specific parameters
    --------------------------------
    reg_marginals : float, default 1.0
        KL (or L2) penalty on marginal violations.
        Smaller  → more cells can be "destroyed" → stronger partial-overlap.
        Larger   → approaches the balanced solution.
        Typical range:  0.1 (strongly unbalanced) … 10.0 (nearly balanced).
        Start with 1.0 and lower if the plan mass is close to 1.0
        (meaning the solver is behaving like balanced OT).

    epsilon : float, default 0.0
        Entropic regularisation (Sinkhorn smoothing).
        0.0 uses the MM solver (exact, recommended for small problems).
        > 0 uses Sinkhorn (faster for large problems). Try 0.01–0.1.

    divergence : 'kl' | 'l2', default 'kl'
        Divergence for marginal relaxation and entropic term.

    unbalanced_solver : 'mm' | 'lbfgsb' | 'sinkhorn' | 'sinkhorn_log'
        Inner OT solver.
        'mm'  works for any divergence and epsilon=0 (default, recommended).
        'sinkhorn' requires epsilon > 0 and divergence='kl'.

    max_iter : int, default 100
        BCD outer iterations.

    tol : float, default 1e-7
        BCD convergence tolerance.

    max_iter_ot, tol_ot : inner solver budget per BCD step.

    Returns
    -------
    pi  : (n_A, n_B) float64  — FUGW sample coupling (alignment plan).
          Rows no longer sum to 1/n_A for unmatched cells (mass is "destroyed").
          pi.sum() < 1 indicates partial overlap was detected.

    If return_obj=True: (pi, init_nb, init_gene, final_nb, final_gene)
    """
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_ub_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name
                else f"{filePath}/log_ub.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_unbalanced — INCENT FUGW\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    logFile.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n")
    logFile.write(f"reg_marginals={reg_marginals}  epsilon={epsilon}  "
                  f"divergence={divergence}  solver={unbalanced_solver}\n\n")

    # ── All preprocessing identical to pairwise_align ─────────────────────────
    p = _preprocess(
        sliceA, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, G_init, a_distribution, b_distribution,
        numItermax, backend, use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        logFile,
    )

    sliceA = p['sliceA']
    sliceB = p['sliceB']
    M1     = p['M1']
    M2     = p['M2']
    D_A    = p['D_A']
    D_B    = p['D_B']
    a      = p['a']
    b      = p['b']

    # Initial objective logging
    G0_np = np.ones((sliceA.shape[0], sliceB.shape[0])) / (
        sliceA.shape[0] * sliceB.shape[0])

    init_nb = 0.0
    if p['nd_dissim'] == 'jsd':
        init_nb = float(np.sum(_to_np(M2) * G0_np))
        logFile.write(f"Initial obj neighbour (jsd): {init_nb:.6f}\n")
    init_gene = float(np.sum(_to_np(p['cosine_dist_gene_expr']) * G0_np))
    logFile.write(f"Initial obj gene (cosine):    {init_gene:.6f}\n\n")

    # ── Convert to numpy float64 for FUGW ─────────────────────────────────────
    # ot.gromov.fused_unbalanced_gromov_wasserstein accepts any POT-backend
    # array; numpy float64 is always safe and avoids dtype surprises.
    D_A_np = _to_np(D_A)
    D_B_np = _to_np(D_B)
    a_np   = _to_np(a)
    b_np   = _to_np(b)

    # ── Build FUGW linear cost  M_bio = M1 + gamma * M2 ──────────────────────
    #
    # Balanced INCENT objective:
    #   (1-alpha) * [M1 + gamma*M2]  +  alpha * GW(D_A, D_B, pi)
    #
    # FUGW objective (POT convention):
    #   GW(D_A, D_B, pi)  +  alpha_fugw * <M_bio, pi>  +  unbalanced terms
    #
    # Matching the GW/biology ratio gives:
    #   alpha_fugw = (1 - alpha) / alpha
    #
    # Examples:
    #   alpha=0.5  →  alpha_fugw=1.0  (equal weight)
    #   alpha=0.3  →  alpha_fugw=2.33 (biology dominates)
    #   alpha=0.7  →  alpha_fugw=0.43 (space dominates)
    #
    M_bio_np = _to_np(M1) + gamma * _to_np(M2)   # (n_A, n_B) float64

    if alpha < 1e-6:
        alpha_fugw = 1e6        # effectively biology only
    elif alpha > 1.0 - 1e-6:
        alpha_fugw = 0.0        # effectively space only
    else:
        alpha_fugw = (1.0 - alpha) / alpha

    logFile.write(f"alpha → alpha_fugw: {alpha} → {alpha_fugw:.6f}\n")
    logFile.write(f"M_bio range: [{M_bio_np.min():.4f}, {M_bio_np.max():.4f}]\n\n")

    # ── Initial plan ──────────────────────────────────────────────────────────
    init_pi_np = _to_np(p['G_init_t']) if p['G_init_t'] is not None else None

    # ── Solve FUGW ────────────────────────────────────────────────────────────
    #
    # Returns (pi_samp, pi_feat, log_dict) when log=True.
    #   pi_samp  — the sample coupling  ← this is our alignment plan
    #   pi_feat  — second coupling (identical to pi_samp for pure GW; ignore)
    #   log_dict — cost breakdown
    #
    pi_samp, _pi_feat, log_dict = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=D_A_np,
        Cy=D_B_np,
        wx=a_np,
        wy=b_np,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha_fugw,
        M=M_bio_np,
        init_pi=init_pi_np,
        init_duals=None,
        max_iter=max_iter,
        tol=tol,
        max_iter_ot=max_iter_ot,
        tol_ot=tol_ot,
        log=True,
        verbose=verbose,
    )

    pi = np.asarray(pi_samp, dtype=np.float64)

    if hard_assignment:
        pi = _hard_assignment_from_coupling(pi)

    # ── Log ───────────────────────────────────────────────────────────────────
    linear_cost = float(log_dict.get('linear_cost', 0.0))
    fugw_cost   = float(log_dict.get('fugw_cost',   0.0))
    pi_mass     = float(pi.sum())

    # Final objective logging
    final_nb = 0.0
    if p['nd_dissim'] == 'jsd':
        max_idx  = np.argmax(pi, axis=1)
        jsd_np   = _to_np(M2)
        final_nb = float(sum(pi[i, max_idx[i]] * jsd_np[i, max_idx[i]]
                             for i in range(len(max_idx))))
        logFile.write(f"Final obj neighbour (jsd): {final_nb:.6f}\n")

    final_gene = float(np.sum(_to_np(p['cosine_dist_gene_expr']) * pi))
    logFile.write(f"Final obj gene (cosine):   {final_gene:.6f}\n")

    logFile.write(f"FUGW linear cost: {linear_cost:.6f}\n")
    logFile.write(f"FUGW total cost:  {fugw_cost:.6f}\n")
    logFile.write(f"pi mass:          {pi_mass:.6f}  "
                  f"(< 1.0 = partial overlap detected)\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    if p['use_gpu'] and isinstance(p['nx'], ot.backend.TorchBackend):
        torch.cuda.empty_cache()

    print(f"[FUGW] pi_mass={pi_mass:.4f}  "
          f"linear_cost={linear_cost:.4f}  fugw_cost={fugw_cost:.4f}")

    if return_obj:
        return pi, init_nb, init_gene, final_nb, final_gene
    return pi


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION 3 — CHIRAL-ST: Chirality-Aware Co-Optimal Transport
# ═════════════════════════════════════════════════════════════════════════════

def compute_spatial_barcodes(curr_slice: AnnData, radii: list) -> np.ndarray:
    """Computes a multi-scale topological signature for each cell."""
    barcodes = []
    for r in radii:
        barcodes.append(neighborhood_distribution(curr_slice, r))
    return np.concatenate(barcodes, axis=1) / max(1, len(radii))

def weighted_procrustes(X: np.ndarray, Y: np.ndarray, pi: np.ndarray, enforce_det: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves for rigid transformation (X @ R + t = Y) given optimal transport plan pi.
    If enforce_det is provided (+1 for SO(d), -1 for reflection), it strictly constrains R.
    """
    dim = X.shape[1]
    if pi is None:
        return np.eye(dim), np.zeros((dim,))

    pi = np.asarray(pi, dtype=np.float64)
    if pi.ndim != 2 or pi.shape[0] != X.shape[0] or pi.shape[1] != Y.shape[0]:
        return np.eye(dim), np.zeros((dim,))
    if not np.isfinite(pi).all():
        pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)

    W = np.sum(pi)
    if W < 1e-12:
        return np.eye(dim), np.zeros((dim,))
        
    w_x = np.sum(pi, axis=1)
    w_y = np.sum(pi, axis=0)
    
    mu_X = np.sum(w_x[:, None] * X, axis=0) / W
    mu_Y = np.sum(w_y[:, None] * Y, axis=0) / W
    
    X_c = X - mu_X
    Y_c = Y - mu_Y
    
    H = X_c.T @ pi @ Y_c

    if not np.isfinite(H).all():
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        U, S, Vh = np.linalg.svd(H) # Vh is V^T
    except np.linalg.LinAlgError:
        return np.eye(dim), np.zeros((dim,))
    
    d = np.sign(np.linalg.det(U @ Vh))
    
    if enforce_det is not None and d != np.sign(enforce_det):
        D = np.eye(dim)
        D[-1, -1] = -1
        R = U @ D @ Vh
    else:
        R = U @ Vh
        
    t = mu_Y - mu_X @ R
    return R, t

def coarse_anchor_search(X, Y, M_bio, a, b, alpha, reg, reg_m, angles_deg):
    """Searches over global coarse alignments in both O(d) chiral states."""
    k = 800
    idx_X = np.random.choice(X.shape[0], k, replace=False) if X.shape[0] > k else np.arange(X.shape[0])
    idx_Y = np.random.choice(Y.shape[0], k, replace=False) if Y.shape[0] > k else np.arange(Y.shape[0])
    
    X_sub = X[idx_X]
    Y_sub = Y[idx_Y]
    M_bio_sub = M_bio[np.ix_(idx_X, idx_Y)]
    a_sub = np.ones(len(idx_X)) / len(idx_X)
    b_sub = np.ones(len(idx_Y)) / len(idx_Y)
    
    mu_X_sub = np.mean(X_sub, axis=0)
    mu_Y_sub = np.mean(Y_sub, axis=0)
    
    best_cost = np.inf
    best_R = np.eye(X.shape[1], dtype=np.float64)
    best_t = np.zeros(X.shape[1], dtype=np.float64)
    best_det = 1
    
    num_anchors = 6
    if Y_sub.shape[0] > num_anchors:
        anchor_idx = np.random.choice(Y_sub.shape[0], num_anchors, replace=False)
        target_anchors = Y_sub[anchor_idx]
    else:
        target_anchors = Y_sub

    target_anchors = np.vstack([target_anchors, mu_Y_sub])
    
    for angle in angles_deg:
        rad = np.deg2rad(angle)
        c, s = np.cos(rad), np.sin(rad)
        
        R_unf = np.array([[c, -s], [s, c]])
        R_flip = np.array([[-c, s], [s, c]])
        
        for R_core, det in [(R_unf, 1), (R_flip, -1)]:
            R = np.eye(X.shape[1])
            R[:2, :2] = R_core
            
            for Y_target_center in target_anchors:
                t = Y_target_center - mu_X_sub @ R
                X_rot = X_sub @ R + t
                
                M_space = ot.dist(X_rot, Y_sub, metric='sqeuclidean')
                M_space /= np.max(M_space) + 1e-12
                C = alpha * M_space + (1.0 - alpha) * M_bio_sub
                
                try:
                    cost = ot.unbalanced.sinkhorn_unbalanced2(a_sub, b_sub, C, reg=reg, reg_m=reg_m, method='sinkhorn_log')
                    val = float(cost.item() if hasattr(cost, 'item') else cost[0])
                    if not np.isfinite(val):
                        continue
                except Exception:
                    continue
                    
                if val < best_cost:
                    best_cost = val
                    best_R = R
                    best_t = t
                    best_det = det
                    
    return best_R, best_t, best_det

def pairwise_align_chiral(
    sliceA:    AnnData,
    sliceB:    AnnData,
    alpha:     float = 0.5,
    gamma:     float = 0.5, 
    radii:     list  = [50.0, 100.0, 200.0],
    filePath:  str   = "./",
    reg_marginals: float = 1.0,
    epsilon:   float = 0.01,
    max_iter_em: int = 20,
    tol_em:    float = 1e-4,
    use_rep:   Optional[str] = None,
    G_init                     = None,
    a_distribution             = None,
    b_distribution             = None,
    backend                    = ot.backend.NumpyBackend(),
    use_gpu:   bool            = False,
    return_obj: bool           = True,
    return_transform: bool     = False,  
    verbose:   bool            = False,
    gpu_verbose: bool          = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool            = False,
    neighborhood_dissimilarity: str = 'jsd',
    hard_assignment: bool      = False,
    angles_deg: list           = [0, 45, 90, 135, 180, 225, 270, 315],
    **kwargs,
):
    """
    CHIRAL-ST Alignment:
    Performs joint optimization of an explicit un-tearable Euclidean spatial transformation (rotation/translation)
    and an unbalanced Wasserstein optimal transport plan.
    """
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_chiral_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log_chiral.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_chiral — INCENT CHIRAL-ST\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
    
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]
    
    shared_ct = (pd.Index(sliceA.obs['cell_type_annot']).unique().intersection(pd.Index(sliceB.obs['cell_type_annot']).unique()))
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_ct)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_ct)]
    
    X = sliceA.obsm['spatial'].copy()
    Y = sliceB.obsm['spatial'].copy()
    
    D_A_max = np.max(ot.dist(X, X))
    D_B_max = np.max(ot.dist(Y, Y))
    scale = np.sqrt(max(D_A_max, D_B_max))
    if scale < 1e-12:
        scale = 1.0
    X_scaled = X / scale
    Y_scaled = Y / scale
    
    nd_cache_A = f"{filePath}/barcodes_{sliceA_name}.npy"
    if sliceA_name is not None and os.path.exists(nd_cache_A) and not overwrite:
        F_A = np.load(nd_cache_A)
        if F_A.shape[0] != sliceA.shape[0]:
            F_A = compute_spatial_barcodes(sliceA, radii)
            np.save(nd_cache_A, F_A)
    else:
        F_A = compute_spatial_barcodes(sliceA, radii)
        if sliceA_name is not None:
            np.save(nd_cache_A, F_A)
        
    nd_cache_B = f"{filePath}/barcodes_{sliceB_name}.npy"
    if sliceB_name is not None and os.path.exists(nd_cache_B) and not overwrite:
        F_B = np.load(nd_cache_B)
        if F_B.shape[0] != sliceB.shape[0]:
            F_B = compute_spatial_barcodes(sliceB, radii)
            np.save(nd_cache_B, F_B)
    else:
        F_B = compute_spatial_barcodes(sliceB, radii)
        if sliceB_name is not None:
            np.save(nd_cache_B, F_B)
        
    F_A_tensor = torch.from_numpy(F_A).float().cuda() if use_gpu and torch.cuda.is_available() else torch.from_numpy(F_A).float()
    F_B_tensor = torch.from_numpy(F_B).float().cuda() if use_gpu and torch.cuda.is_available() else torch.from_numpy(F_B).float()
    M2 = jensenshannon_divergence_backend(F_A_tensor, F_B_tensor)
    if isinstance(M2, torch.Tensor):
        M2 = M2.cpu().numpy()
        
    M_bio = M2
    
    a = a_distribution if a_distribution is not None else np.ones(X.shape[0]) / X.shape[0]
    b = b_distribution if b_distribution is not None else np.ones(Y.shape[0]) / Y.shape[0]
    
    logFile.write("Starting Chirality-Aware Anchor Search...\n")
    R, t, det = coarse_anchor_search(X_scaled, Y_scaled, M_bio, a, b, alpha, epsilon, reg_marginals, angles_deg)
    R, t = _ensure_rigid_transform(R, t, X_scaled.shape[1])
    
    pi = None
    for iteration in range(max_iter_em):
        X_mapped = X_scaled @ R + t
        M_space = ot.dist(X_mapped, Y_scaled, metric='sqeuclidean')
        M_space /= np.max(M_space) + 1e-12
        C = alpha * M_space + (1.0 - alpha) * M_bio
        
        pi = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=epsilon, reg_m=reg_marginals, method='sinkhorn')
        pi = _sanitize_coupling(pi)

        # Fallback to sparse identity connection to retain mass instead of uniform 0 matrix
        if np.sum(pi) < 1e-12:
            pi = np.eye(pi.shape[0], pi.shape[1], dtype=np.float64) / max(pi.shape[0], pi.shape[1])

        R_new, t_new = weighted_procrustes(X_scaled, Y_scaled, pi, enforce_det=det)
        R_new, t_new = _ensure_rigid_transform(R_new, t_new, X_scaled.shape[1])
        
        delta_R = np.linalg.norm(R_new - R)
        R = R_new
        t = t_new
        
        if delta_R < tol_em:
            break
            
    t_unscaled = t * scale
    
    if hard_assignment:
        pi = _hard_assignment_from_coupling(pi)
        
    logFile.close()
    
    # Clean up memory
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if return_transform:
        if return_obj:
            return pi, R, t_unscaled, 0, 0, 0, 0
        return pi, R, t_unscaled
        
    if return_obj:
        return pi, 0, 0, 0, 0
    return pi
