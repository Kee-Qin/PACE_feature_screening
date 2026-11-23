from src import model_path as mp
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import gc
import json
import math
import time
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from scipy.special import digamma, ndtri
from scipy.interpolate import BSpline as _ScipyBSpline

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.screening_methods_torch import (
    DEVICE, CHUNK_SIZE, set_device,
    _to_tensor, _prepare_y_torch, _double_center, _double_center_kappa,
    _auto_chunk_size, _get_available_gpu_mem, EMB_FEAT_CHUNK,
    TorchAcceleratedScreening, TorchAcceleratedEmbedder, TABICL_PATH,
)


# ===========================================================================

# ---------------------------------------------------------------------------


from src.screening_utils import (
    EMB_LAYERS,
    MAX_TRAIN_FEATURES,
    LGB_PARAMS_CLS,
    LGB_PARAMS_REG,
    CB_PARAMS_CLS,
    CB_PARAMS_REG,
    RF_PARAMS_CLS,
    RF_PARAMS_REG,
    PFI_REPEATS,
    VAL_FRACTION,
    _fill_missing,
    _train_val_split,
    _model_njobs,
    _metric_loss,
    _metric_loss_C,
    _metric_loss_D
)
from src.screening_methods_torch import (
    _midranks_gpu,
    _median_bandwidth_gpu,
    _hsic_gpu,
    _gpu_hist_nmi,
    TRACK_A_RAW_METHODS,
    TRACK_A_EMB_BASE_METHODS,
    TRACK_A_METHODS,
    _EMB_LEGACY_TO_L3,
    _EMB_ALL_LEGACY,
    _parse_emb_methods,
    _ucenter,
    _median_bandwidth_1d,
    _midranks,
    TrackAScreening,
    _RawTrackA,
    _PaceA,
    _HAS_PACEA_A,
    TRACK_A_DISPATCH,
    TRACK_A_EMB_DISPATCH,
    compute_trackA_scores,
    compute_trackA_emb_scores,
    _tiny_meanshift,
    validate_gpu_scorers,
    _HAS_PACEA_REG_A,
    TRACK_A_RAW_METHODS_REG,
    TRACK_A_EMB_BASE_METHODS_REG,
    TRACK_A_METHODS_REG,
    _parse_emb_methods_reg,
    _ucenter_reg,
    _median_bandwidth_1d_reg,
    _midranks_reg,
    _median_bandwidth_general_reg,
    TrackAScreeningReg,
    TRACK_A_DISPATCH_REG,
    TRACK_A_EMB_DISPATCH_REG,
    compute_trackA_scores_reg,
    _tiny_reg_nonlinear_A,
    validate_gpu_scorers_reg,
    _A_median_bw_gpu,
    _HAS_TORCH_BASELINE,
    PAceRawA,
    PAceEmbA,
    get_merged_embeddings,
    get_embeddings_incremental,
    PACE_TRACK_A_RAW,
    PACE_TRACK_A_EMB_BASE,
    PACE_TRACK_A_EMB,
    compute_paceA_raw_scores,
    compute_paceA_emb_scores,
    compute_paceA_emb_scores_multilayer,
    compute_paceA_emb_scores_incremental
)

TRACK_B_METHODS = ['relieff', 'rrelieff', 'legendre_wmc', 'spline_wmc', 'spline_dc',
                    'rbf_dc', 'rff_dc', 'qbin_hsic', 'unitree_cv',

                    'legendre_sis', 'legendre_dc', 'legendre_hsic',
                    'rbf_sis', 'rbf_hsic', 'rff_sis', 'rff_hsic',
                    'qbin_sis', 'qbin_dc', 'spline_sis', 'spline_hsic']

REPR_DIM = 128

RELIEFF_M = 1000

RELIEFF_K = 10

RRELIEFF_SIGMA_FACTOR = 0.5

POLY_DEGREE = 8

SPLINE_DEGREE = 3

RFF_GAMMA_DEFAULT = 2.0

QBIN_B = 16

UNITREE_MAX_DEPTH = 3

UNITREE_MIN_LEAF_FACTOR = 0.02

CV_FOLDS = 5

RELIEFF_P_MAX = 3_000

EPS_Q = None

def _midranks_B(Xt: torch.Tensor) -> torch.Tensor:
    n = Xt.shape[0]
    single = (Xt.dim() == 1)
    if single:
        Xt = Xt.unsqueeze(1)
    dt = Xt.dtype
    order = torch.argsort(Xt, dim=0, stable=True)
    sv = torch.gather(Xt, 0, order)
    pos = torch.arange(1, n + 1, device=Xt.device, dtype=dt).unsqueeze(1)
    ranks = pos.expand(-1, Xt.shape[1]).clone()
    eq = (sv[1:] == sv[:-1])
    for c in range(Xt.shape[1]):
        col_eq = eq[:, c]
        i = 0
        while i < n:
            if i < n - 1 and col_eq[i]:
                j = i
                while j < n - 1 and col_eq[j]:
                    j += 1
                ranks[i:j + 1, c] = pos[i:j + 1, 0].mean()
                i = j + 1
            else:
                i += 1
    out = torch.empty_like(ranks)
    out.scatter_(0, order, ranks)
    return out.squeeze(1) if single else out

def _rank_gaussian(x_np: np.ndarray) -> np.ndarray:
    x = np.asarray(x_np, dtype=np.float64)
    n = len(x)
    if n < 2 or np.std(x) < 1e-12:
        return x.copy()
    eps = 1.0 / (2 * n)
    # mid-rank
    order = np.argsort(x, kind='stable')
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    u = (ranks - 0.5) / n
    u = np.clip(u, eps, 1 - eps)
    return ndtri(u)

def _legendre_basis(x_col: np.ndarray, degree: int) -> np.ndarray:
    z = np.asarray(x_col, dtype=np.float64)
    n = len(z)
    zmin, zmax = z.min(), z.max()
    t = 2 * (z - zmin) / (zmax - zmin + 1e-30) - 1
    t = np.clip(t, -1, 1)
    P = np.zeros((n, max(degree, 1)), dtype=np.float64)
    P_prev2 = np.ones(n)         # P0
    if degree >= 1:
        P_prev1 = t.copy()       # P1
        P[:, 0] = P_prev1
    for m in range(1, degree):
        Pm = ((2 * m + 1) * t * P_prev1 - m * P_prev2) / (m + 1)
        P[:, m] = Pm
        P_prev2, P_prev1 = P_prev1, Pm
    H = P - P.mean(axis=0, keepdims=True)
    std = H.std(axis=0) + 1e-12
    return H / std

def _bspline_basis(x_col: np.ndarray, d: int, degree: int = 3) -> np.ndarray:
    z = np.asarray(x_col, dtype=np.float64)
    n = len(z)
    if n < 8 or np.std(z) < 1e-12:
        return np.zeros((n, 1))
    lo, hi = z.min(), z.max()

    n_inner = max(d - degree, 1)
    qs = np.linspace(0, 1, n_inner + 2)[1:-1]
    inner = np.quantile(z, qs)
    inner = np.unique(inner)
    knots = np.concatenate([[lo] * (degree + 1), inner, [hi] * (degree + 1)])
    kk = knots
    t_arr = np.clip(z, lo, hi)
    nb = len(kk) - degree - 1
    if nb <= 0:
        return np.zeros((n, 1))


    B = np.zeros((nb, n), dtype=np.float64)

    for m in range(nb):
        if kk[m] == kk[m + 1]:
            B[m] = 0.0
        else:
            B[m] = ((t_arr >= kk[m]) & (t_arr < kk[m + 1])).astype(np.float64)

    if nb >= 1 and kk[nb - 1] < kk[nb]:
        B[nb - 1] = np.where(t_arr >= hi, 1.0, B[nb - 1])
    for q in range(1, degree + 1):
        B_new = np.zeros_like(B)
        for m in range(nb - q):
            d1 = (kk[m + q] - kk[m])
            d2 = (kk[m + q + 1] - kk[m + 1])
            term1 = 0.0 if d1 == 0 else (t_arr - kk[m]) / d1 * B[m]
            term2 = 0.0 if d2 == 0 else (kk[m + q + 1] - t_arr) / d2 * B[m + 1]
            B_new[m] = term1 + term2
        B = B_new
    H = B.T                                       # (n, nb)

    H = H - H.mean(axis=0, keepdims=True)
    keep = H.std(axis=0) > 1e-10
    H = H[:, keep]
    if H.shape[1] == 0:
        return np.zeros((n, 1))
    return H

def _rbf_basis(x_col: np.ndarray, d: int, gamma: float = RFF_GAMMA_DEFAULT) -> Tuple[np.ndarray, float]:
    z = np.asarray(x_col, dtype=np.float64)
    n = len(z)
    if n < 2 or np.std(z) < 1e-12:
        return np.zeros((n, 1)), 1.0
    qs = (np.arange(1, d + 1) - 0.5) / d
    centers = np.quantile(z, qs)
    centers = np.unique(centers)
    if len(centers) < 2:
        return np.zeros((n, 1)), 1.0
    spacings = np.diff(centers)
    sigma = np.median(spacings) * gamma
    if sigma <= 0:
        sigma = np.mean(spacings) * gamma if spacings.size else 1.0
    H = np.exp(-((z[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2 + 1e-30))
    H = H - H.mean(axis=0, keepdims=True)
    keep = H.std(axis=0) > 1e-10
    return H[:, keep], float(sigma)

def _rff_basis(x_col: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    z = np.asarray(x_col, dtype=np.float64).reshape(-1)
    n = len(z)
    if n < 2 or np.std(z) < 1e-12:
        return np.zeros((n, 1))
    rng = np.random.RandomState(seed)
    nz = z[z != z[0]]
    sigma = np.median(np.abs(z[:, None] - z[None, :])[np.triu_indices(n, 1)]) if n > 1 else 1.0
    if (sigma == 0 or np.isnan(sigma)) and nz.size:
        sigma = np.std(z)
    if sigma == 0:
        sigma = 1.0
    omega = rng.normal(0, 1.0 / (sigma + 1e-30), size=d)
    b = rng.uniform(0, 2 * np.pi, size=d)
    H = np.sqrt(2.0 / d) * np.cos(z[:, None] * omega[None, :] + b[None, :])
    return H - H.mean(axis=0, keepdims=True)

def _quantile_bin_basis(x_col: np.ndarray, B: int, cat_col: bool = False) -> np.ndarray:
    x = np.asarray(x_col)
    n = len(x)
    if cat_col:
        cats, inv = np.unique(x, return_inverse=True)
        min_freq = max(2, int(np.ceil(0.005 * n)))
        counts = np.bincount(inv)
        rare = cats[counts < min_freq]

        x_mapped = x.copy()
        if len(rare):
            x_mapped[np.isin(x, rare)] = -1
        cats2 = np.unique(x_mapped)
        idx = np.searchsorted(cats2, x_mapped)
        H = np.zeros((n, len(cats2)), dtype=np.float64)
        H[np.arange(n), idx] = 1.0
    else:
        qs = np.quantile(x, np.linspace(0, 1, B + 1))
        qs = np.unique(qs)
        if len(qs) < 2:
            return np.zeros((n, 1))
        idx = np.clip(np.digitize(x, qs[1:-1], right=False), 0, len(qs) - 2)
        H = np.zeros((n, len(qs) - 1), dtype=np.float64)
        H[np.arange(n), idx] = 1.0
    return H - H.mean(axis=0, keepdims=True) if H.shape[1] > 0 else H

def _vector_dc_scores_gpu(H_list: List[np.ndarray], y, task, n) -> np.ndarray:
    p = len(H_list)
    Y = _to_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float64) if task == 'regression' \
        else _y_onehot(y, torch.float64)
    b = torch.cdist(Y, Y)                       # (n, n) 2D
    Bc = _double_center(b)
    b_var = (Bc ** 2).sum() + 1e-30
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        H = H_list[j]
        d_j = H.shape[1]
        if d_j == 0 or np.std(H) < 1e-12:
            continue
        Ht = _to_tensor(H, dtype=torch.float64)
        a = torch.cdist(Ht, Ht)                 # (n, n) 2D
        A = _double_center(a)
        num = (A * Bc).sum()
        a_var = (A ** 2).sum() + 1e-30
        dc_sq = num / torch.sqrt(a_var * b_var)
        scores[j] = float(torch.sqrt(torch.clamp(dc_sq, min=0, max=1)))
    return scores

def _vector_nhsic_scores_gpu(H_list, y, task, n) -> np.ndarray:
    p = len(H_list)
    if task == 'classification':
        Ly = _y_delta_kernel(y, torch.float64)
    else:
        Ly = _y_rbf_kernel_regression(y, torch.float64)
    Hm = torch.eye(n, device=Ly.device, dtype=Ly.dtype) - 1.0 / n
    HLH = Hm @ Ly @ Hm
    Lc_norm = torch.sqrt((HLH ** 2).sum() + 1e-30)
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        H = H_list[j]
        if H.std() < 1e-12 or H.shape[1] == 0:
            continue
        Ht = _to_tensor(H, dtype=torch.float64)
        dist = torch.cdist(Ht, Ht)
        nz = dist[dist > 0]
        s = nz.median() if nz.numel() > 0 else 1.0
        s = s if s > 0 else (nz.mean() if nz.numel() > 0 else 1.0)
        if s == 0:
            continue
        K = torch.exp(-dist ** 2 / (2 * s ** 2 + 1e-30))
        Kc = Hm @ K @ Hm
        num = (Kc * HLH).sum()
        knorm = torch.sqrt((Kc ** 2).sum() + 1e-30)
        scores[j] = float(num / (knorm * Lc_norm + 1e-30))
    return scores

_VBATCH_DTYPE = torch.float64

def _prepare_y_vec(y, task, n, device, dtype=_VBATCH_DTYPE):
    y = np.asarray(y)
    if task == 'classification':
        classes = np.unique(y)
        Y = np.zeros((n, len(classes)), dtype=np.float64)
        for i, c in enumerate(classes):
            Y[:, i] = (y == c)
        return _to_tensor(Y, dtype=dtype, device=device), True
    Y = y.reshape(-1, 1).astype(np.float64)
    return _to_tensor(Y, dtype=dtype, device=device), False

def _y_kernel_vec(y, task, n, device, dtype=_VBATCH_DTYPE):
    if task == 'classification':
        return _y_delta_kernel(y, dtype)
    Yt = _to_tensor(np.asarray(y).reshape(-1, 1), dtype=dtype)
    d = torch.cdist(Yt, Yt)
    nz = d[d > 0]
    s = nz.median() if nz.numel() > 0 else torch.tensor(1.0, device=Yt.device, dtype=dtype)
    if not s.item() > 0:
        s = nz.mean() if nz.numel() > 0 else torch.tensor(1.0, device=Yt.device, dtype=dtype)
    return torch.exp(-d ** 2 / (2 * s ** 2 + 1e-30))

def _vector_dc_scores_batch(H_pad, y, task, n, p):
    dev = H_pad.device; dt = H_pad.dtype
    Yt, is_cat = _prepare_y_vec(y, task, n, dev, dt)
    b = torch.cdist(Yt, Yt)                       # (n, n)
    Bc = b - b.mean(dim=1, keepdim=True) - b.mean(dim=0, keepdim=True) + b.mean()
    b_var = (Bc ** 2).sum() + 1e-30
    scores = torch.zeros(p, device=dev, dtype=dt)
    chunk = CHUNK_SIZE
    for start in range(0, p, chunk):
        end = min(start + chunk, p); C = end - start
        Hc = H_pad[start:end]
        a = torch.cdist(Hc, Hc)                    # (C, n, n)


        A = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) \
            + a.mean(dim=(1, 2), keepdim=True)
        num = (A * Bc.unsqueeze(0)).sum(dim=(1, 2))
        a_var = (A ** 2).sum(dim=(1, 2))
        dc_sq = num / torch.sqrt(a_var * b_var + 1e-30)
        scores[start:end] = torch.sqrt(torch.clamp(dc_sq, min=0, max=1))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores.cpu().numpy()

def _vector_hsic_scores_batch(H_pad, y, task, n, p, normalized=False):
    dev = H_pad.device; dt = H_pad.dtype
    Ly = _y_kernel_vec(y, task, n, dev, dt)        # (n, n)
    Hm = torch.eye(n, device=dev, dtype=dt) - 1.0 / n
    HLH = Hm @ Ly @ Hm
    Lc_norm = torch.sqrt((HLH ** 2).sum() + 1e-30)
    scores = torch.zeros(p, device=dev, dtype=dt)
    chunk = CHUNK_SIZE
    mbw = _A_median_bw_gpu if _A_median_bw_gpu is not None else None
    for start in range(0, p, chunk):
        end = min(start + chunk, p); C = end - start
        Hc = H_pad[start:end]
        dist = torch.cdist(Hc, Hc)                 # (C, n, n)
        if mbw is not None:
            sg = mbw(dist)
        else:

            idx = torch.triu_indices(n, n, offset=1, device=dev)
            d_triu = dist[:, idx[0], idx[1]]        # (C, m)
            sg = torch.zeros(C, device=dev, dtype=dt)
            for j in range(C):
                dn = d_triu[j][d_triu[j] > 0]
                if dn.numel() == 0:
                    continue
                m = dn.median()
                sg[j] = m if m.item() > 0 else dn.mean()
        sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
        K = torch.exp(-dist ** 2 / (2 * sg_safe.view(C, 1, 1) ** 2 + 1e-30))   # (C,n,n)
        const_mask = sg == 0
        if const_mask.any():
            K[const_mask] = 0.0
        HKxH = Hm @ K @ Hm
        if normalized:   # NHSIC
            num = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2))
            k_norm = torch.sqrt((HKxH ** 2).sum(dim=(1, 2)) + 1e-30)
            sc = num / (k_norm * Lc_norm + 1e-30)
            sc = torch.clamp(sc, min=0)
            sc = sc.masked_fill(const_mask, 0.0)
        else:
            sc = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2)) / (n * n)
            sc = torch.clamp(sc, min=0)
            sc = sc.masked_fill(const_mask, 0.0)
        scores[start:end] = sc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores.cpu().numpy()

def _vector_sis_scores_batch(H_pad, y, task, n, p, D=128):
    dev = H_pad.device; dt = H_pad.dtype
    Hm = lambda v: v - v.mean(dim=0, keepdim=True)
    if task == 'classification':
        y = np.asarray(y); classes = np.unique(y)
        Yo = np.zeros((n, len(classes)), dtype=np.float64)
        for i, c in enumerate(classes):
            Yo[:, i] = (y == c)
        Yt = _to_tensor(Yo, dtype=dt, device=dev)              # (n, C)
        Yc = Yt - Yt.mean(dim=0, keepdim=True)
        y_std = Yt.std(dim=0, keepdim=True) + 1e-12            # (1, C)
        scores = torch.zeros(p, device=dev, dtype=dt)
        for dim in range(D):
            z = H_pad[:, :, dim].t()
            zc = z - z.mean(dim=0, keepdim=True)
            z_std = z.std(dim=0) + 1e-12              # (p,)
            # corr (p, C) = |zc^T Yc| / (n * z_std * y_std)
            num = zc.t() @ Yc                         # (p, C)
            corr = torch.abs(num) / (n * z_std.unsqueeze(1) * y_std + 1e-30)
            scores += corr.max(dim=1)[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (scores / D).cpu().numpy()

    yc = _to_tensor(np.asarray(y).reshape(-1, 1), dtype=dt, device=dev)
    yc = yc - yc.mean(dim=0, keepdim=True)
    y_std = yc.std() + 1e-12
    scores = torch.zeros(p, device=dev, dtype=dt)
    for dim in range(D):
        z = H_pad[:, :, dim].t()                     # (n, p)
        zc = z - z.mean(dim=0, keepdim=True)
        z_std = z.std(dim=0) + 1e-12
        corr = torch.abs((zc.t() @ yc).squeeze(1)) / (n * z_std * y_std + 1e-30)
        scores += corr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (scores / D).cpu().numpy()

_BASIS_NJOBS = int(os.environ.get("PACE_BASIS_NJOBS", "16"))

def _build_padded_H_gpu(Xtr, p, basis_fn, cat_indices, cat_fn, D=128):
    dev = DEVICE
    n = Xtr.shape[0]
    cat_set = set(cat_indices or [])

    def _one(j):
        xj = Xtr[:, j]
        Hj = cat_fn(xj) if (j in cat_set and cat_fn is not None) else basis_fn(xj)
        dj = 0 if Hj is None else Hj.shape[1]
        if dj == 0:
            return None, 0
        dj = min(dj, D)
        return np.ascontiguousarray(Hj[:, :dj], dtype=np.float64), dj


    out = Parallel(n_jobs=_BASIS_NJOBS, prefer='threads')(
        delayed(_one)(j) for j in range(p))

    H_cpu = np.zeros((p, n, D), dtype=np.float64)
    for j, (Hj, dj) in enumerate(out):
        if dj == 0:
            continue
        H_cpu[j, :, :dj] = Hj
    H_pad = _to_tensor(H_cpu, dtype=_VBATCH_DTYPE, device=dev)
    return H_pad

def _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, basis_kind, measure):
    if p > MAX_TRAIN_FEATURES:
        return None

    deg_lg = min(POLY_DEGREE, REPR_DIM)
    if basis_kind == 'legendre':
        bf = lambda x: _legendre_basis(x, deg_lg)
    elif basis_kind == 'spline':
        bf = lambda x: _bspline_basis(x, REPR_DIM, SPLINE_DEGREE)
    elif basis_kind == 'rbf':
        bf = lambda x: _rbf_basis(x, REPR_DIM)[0]
    elif basis_kind == 'rff':
        bf = lambda x: _rff_basis(x, REPR_DIM, seed=0)
    elif basis_kind == 'qbin':
        bf = lambda x: _quantile_bin_basis(x, QBIN_B, cat_col=False)
    else:
        raise ValueError(f"unknown basis_kind {basis_kind}")
    cat_fn = lambda x: _quantile_bin_basis(x, QBIN_B, cat_col=True)
    H_pad = _build_padded_H_gpu(Xtr, p, bf, cat_indices, cat_fn)
    n = Xtr.shape[0]
    if measure == 'dc':
        return _vector_dc_scores_batch(H_pad, ytr, task, n, p)
    elif measure == 'hsic':
        return _vector_hsic_scores_batch(H_pad, ytr, task, n, p, normalized=True)
    elif measure == 'sis':
        return _vector_sis_scores_batch(H_pad, ytr, task, n, p)
    else:
        raise ValueError(f"unknown measure {measure}")

def _y_onehot(y, dtype):
    y = np.asarray(y)
    classes = np.unique(y)
    Y = np.zeros((len(y), len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Y[:, i] = (y == c)
    return _to_tensor(Y, dtype=dtype)

def _y_delta_kernel(y, dtype):
    y = np.asarray(y)
    classes = np.unique(y)
    Ym = np.zeros((len(y), len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Ym[:, i] = (y == c)
    return _to_tensor(Ym @ Ym.T, dtype=dtype)

def _y_rbf_kernel_regression(y, dtype):
    Y = _to_tensor(np.asarray(y).reshape(-1, 1), dtype=dtype)
    d = torch.cdist(Y, Y)
    nz = d[d > 0]
    s = nz.median().item() if nz.numel() > 0 else 1.0
    s = s if s > 0 else (nz.mean().item() if nz.numel() > 0 else 1.0)
    return torch.exp(-d ** 2 / (2 * s * s + 1e-30))

def _ridge_whitened_quad(Sigma: np.ndarray, c: np.ndarray, lam: float, d: int) -> float:
    try:
        A = (Sigma + (lam * max(np.trace(Sigma) / max(d, 1), 1e-12)) * np.eye(d))
        e, V = np.linalg.eigh((A + A.T) / 2)
        e = np.clip(e, 1e-10, None)
        # (Σ+λI)^-1/2 = V diag(1/sqrt(e)) V^T
        inv_sqrt = (V * (1.0 / np.sqrt(e))) @ V.T
        val = float(c @ (inv_sqrt @ c))
    except Exception:
        inv_sqrt = np.linalg.pinv(Sigma + lam * np.eye(d))
        val = float(c @ (inv_sqrt @ c))

    fallback = float(np.sqrt((c ** 2).sum()))
    return max(val, 0.0, 1e-12 * fallback) if False else max(float(val), 0.0)

def _wmc_score(H: np.ndarray, y: np.ndarray, task: str, lam: float = 1e-2) -> float:
    n, d = H.shape
    if d == 0 or H.std() < 1e-12:
        return 0.0
    Hc = H - H.mean(axis=0, keepdims=True)
    Sigma = (Hc.T @ Hc) / n
    if task == 'classification':
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) <= 2:
            yn = (y == classes[1]).astype(np.float64)
            c = (Hc.T @ (yn - yn.mean())) / n
            val = _ridge_whitened_quad(Sigma, c, lam, d)
            vy = yn.var() + 1e-12
            return float(np.sqrt(val) / np.sqrt(vy))
        else:
            Ym = np.zeros((n, len(classes)))
            for i, cl in enumerate(classes):
                Ym[:, i] = (y == cl)
            Yc = Ym - Ym.mean(axis=0)
            CHY = (Hc.T @ Yc) / n          # (d, C)
            SigY = (Yc.T @ Yc) / n
            lam_h = lam * max(np.trace(Sigma) / d, 1e-12)
            lam_y = lam * max(np.trace(SigY) / len(classes), 1e-12)
            try:
                Sh_inv_sqrt = _matrix_inv_sqrt(Sigma + lam_h * np.eye(d))
                Sy_inv_sqrt = _matrix_inv_sqrt(SigY + lam_y * np.eye(len(classes)))
                M = Sh_inv_sqrt @ CHY @ Sy_inv_sqrt
                return float(np.linalg.norm(M, 'fro'))
            except Exception:
                return 0.0
    else:
        yn = np.asarray(y, dtype=np.float64)
        c = (Hc.T @ (yn - yn.mean())) / n
        val = _ridge_whitened_quad(Sigma, c, lam, d)
        vy = yn.var() + 1e-12
        return float(np.sqrt(val) / np.sqrt(vy))

def _matrix_inv_sqrt(M):
    e, V = np.linalg.eigh((M + M.T) / 2)
    e = np.clip(e, 1e-12, None)
    return (V * (1.0 / np.sqrt(e))) @ V.T

class TrackBScreening:


    @staticmethod
    def relieff_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if task != 'classification' or p > RELIEFF_P_MAX:
            if p > RELIEFF_P_MAX:
                print(f"  [TrackB] relieff: p={p}>{RELIEFF_P_MAX}, skipping", flush=True)
            return None
        X = np.asarray(Xtr, dtype=np.float64); y = np.asarray(ytr)
        n = len(y)
        classes = np.unique(y)
        if len(classes) < 2:
            return None
        rng = np.random.RandomState(0)
        ranges = X.max(axis=0) - X.min(axis=0) + 1e-12
        cat_set = set(cat_indices or [])
        m = min(n, RELIEFF_M)
        refs = rng.choice(n, size=m, replace=(m > n))
        k = min(RELIEFF_K, max(1, n // (2 * len(classes))))
        pc = np.array([(y == c).mean() for c in classes], dtype=np.float32)

        Xt = _to_tensor(X, dtype=torch.float32)                  # (n, p)
        rngs_t = _to_tensor(ranges, dtype=torch.float32)         # (p,)
        cat_cols = torch.as_tensor(list(cat_set), device=Xt.device, dtype=torch.long) if cat_set else None
        y_t = _to_tensor(y, dtype=torch.float32)
        ref_t = _to_tensor(refs, dtype=torch.long)
        W = torch.zeros(p, device=Xt.device, dtype=torch.float32)

        pc_t = _to_tensor(pc, dtype=torch.float32)               # (n_cls,)
        cls_t = torch.as_tensor(classes, device=Xt.device, dtype=torch.float32)

        ref_class = y_t[ref_t]                                    # (mb,)
        yr_prior = (ref_class.reshape(-1, 1) == cls_t.reshape(1, -1)).float().mean(dim=1)  # approx

        REF_CHUNK = 64
        for s in range(0, m, REF_CHUNK):
            e = min(s + REF_CHUNK, m)
            ridx = ref_t[s:e]                                       # (mb,)
            Xr = Xt[ridx]                                           # (mb, p)
            d = (Xt.unsqueeze(0) - Xr.unsqueeze(1)).abs() / rngs_t  # (mb, n, p)
            if cat_cols is not None:
                Xc = Xt[:, cat_cols]
                Xrc = Xr[:, cat_cols]
                cat_d = (Xc.unsqueeze(0) != Xrc.unsqueeze(1)).float()
                d = d.clone()
                d[:, :, cat_cols] = cat_d
            dsum = d.sum(dim=2)                                     # (mb, n)
            mb = e - s
            row_idx = torch.arange(mb, device=Xt.device).unsqueeze(1)
            dsum[row_idx, ridx.unsqueeze(1)] = 1e18
            knn_d, knn_idx = torch.topk(-dsum, k, dim=1)           # (mb, k)
            gather_idx = knn_idx.unsqueeze(2).expand(-1, -1, p)
            nb_d = torch.gather(d, 1, gather_idx)                 # (mb, k, p)
            nb_y = y_t[knn_idx]                                     # (mb, k)
            yr_chunk = ref_class[s:e]                              # (mb,)
            for ci, c in enumerate(classes):
                c_t = float(c)
                is_c_nb = (nb_y == c_t).float().unsqueeze(2)      # (mb,k,1)
                yr_is_c = (yr_chunk == c_t).float().reshape(-1, 1, 1)  # (mb,1,1)
                is_hit = is_c_nb * yr_is_c                          # (mb,k,1)
                is_miss = is_c_nb * (1 - yr_is_c)                  # (mb,k,1)
                W -= (is_hit * nb_d).sum(dim=(0, 1)) / (m * k)

                yr_prior_chunk = yr_prior[s:e].reshape(-1, 1, 1)
                w_miss = (pc_t[ci] / (1 - yr_prior_chunk + 1e-12))  # (mb,1,1)
                W += (is_miss * nb_d * w_miss).sum(dim=(0, 1)) / (m * k)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return W.cpu().numpy()


    @staticmethod
    def rrelieff_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if task != 'regression' or p > RELIEFF_P_MAX:
            if p > RELIEFF_P_MAX:
                print(f"  [TrackB] rrelieff: p={p}>{RELIEFF_P_MAX}, skipping", flush=True)
            return None
        X = np.asarray(Xtr, dtype=np.float64); y = np.asarray(ytr, dtype=np.float64)
        n = len(y)
        if n < 4:
            return None
        rng = np.random.RandomState(0)
        ranges = X.max(axis=0) - X.min(axis=0) + 1e-12
        cat_set = set(cat_indices or [])
        yrange = y.max() - y.min() + 1e-12
        m = min(n, RELIEFF_M)
        k = min(RELIEFF_K, n - 1)
        sigma = k / 2
        refs = rng.choice(n, size=m, replace=(m > n))

        Xt = _to_tensor(X, dtype=torch.float32)
        rngs_t = _to_tensor(ranges, dtype=torch.float32)
        cat_cols = torch.as_tensor(list(cat_set), device=Xt.device, dtype=torch.long) if cat_set else None
        y_t = _to_tensor(y, dtype=torch.float32)
        ref_t = _to_tensor(refs, dtype=torch.long)
        yrange_t = float(yrange)
        NdC = torch.zeros(1, device=Xt.device, dtype=torch.float32)
        NdA = torch.zeros(p, device=Xt.device, dtype=torch.float32)
        NdC_A = torch.zeros(p, device=Xt.device, dtype=torch.float32)

        REF_CHUNK = 64
        for s in range(0, m, REF_CHUNK):
            e = min(s + REF_CHUNK, m)
            ridx = ref_t[s:e]                                       # (mb,)
            Xr = Xt[ridx]
            d = (Xt.unsqueeze(0) - Xr.unsqueeze(1)).abs() / rngs_t  # (mb, n, p)
            if cat_cols is not None:
                Xc = Xt[:, cat_cols]; Xrc = Xr[:, cat_cols]
                cat_d = (Xc.unsqueeze(0) != Xrc.unsqueeze(1)).float()
                d = d.clone(); d[:, :, cat_cols] = cat_d
            dsum = d.sum(dim=2)                                     # (mb, n)
            mb = e - s
            row_idx = torch.arange(mb, device=Xt.device).unsqueeze(1)
            dsum[row_idx, ridx.unsqueeze(1)] = 1e18
            knn_d, knn_idx = torch.topk(-dsum, k, dim=1)           # (mb, k)
            gather_idx = knn_idx.unsqueeze(2).expand(-1, -1, p)
            nb_d = torch.gather(d, 1, gather_idx)                 # (mb, k, p)
            ref_y = y_t[ridx].unsqueeze(1)                        # (mb, 1)
            nb_y = y_t[knn_idx]                                     # (mb, k)
            diff_y = (nb_y - ref_y).abs() / yrange_t              # (mb, k)
            ranks = torch.arange(1, k + 1, device=Xt.device, dtype=torch.float32)  # (k,)
            d1 = torch.exp(-(ranks / sigma) ** 2)                 # (k,)
            d = d1 / (d1.sum() + 1e-30)                            # (k,) normalized per ref
            NdC += (d * diff_y).sum()
            dw = (d * diff_y)                          # (mb, k)

            NdA  += (d.reshape(1, -1, 1) * nb_d).sum(dim=(0, 1))
            NdC_A += (dw.unsqueeze(2) * nb_d).sum(dim=(0, 1))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        T = float(m)
        W = NdC_A / (NdC + 1e-12) - (NdA - NdC_A) / (T - NdC + 1e-12)
        return W.cpu().numpy()

    # ---- 3. Legendre-WMC ----
    @staticmethod
    def legendre_wmc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if p > MAX_TRAIN_FEATURES:
            return None
        deg = min(POLY_DEGREE, REPR_DIM)
        cat_set = set(cat_indices or [])

        def _one(j):
            Hj = _quantile_bin_basis(Xtr[:, j], QBIN_B, cat_col=True) if j in cat_set \
                else _legendre_basis(Xtr[:, j], deg)
            return _wmc_score(Hj, ytr, task)
        return np.array(Parallel(n_jobs=-1, prefer='threads')(delayed(_one)(j) for j in range(p)),
                        dtype=np.float64)

    # ---- 4. Spline-WMC ----
    @staticmethod
    def spline_wmc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if p > MAX_TRAIN_FEATURES:
            return None
        cat_set = set(cat_indices or [])
        def _one(j):
            Hj = _quantile_bin_basis(Xtr[:, j], QBIN_B, cat_col=True) if j in cat_set \
                else _bspline_basis(Xtr[:, j], REPR_DIM, SPLINE_DEGREE)
            return _wmc_score(Hj, ytr, task)
        return np.array(Parallel(n_jobs=-1, prefer='threads')(delayed(_one)(j) for j in range(p)),
                        dtype=np.float64)


    @staticmethod
    def spline_dc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'spline', 'dc')

    @staticmethod
    def rbf_dc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'rbf', 'dc')

    @staticmethod
    def rff_dc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'rff', 'dc')

    @staticmethod
    def qbin_hsic_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'qbin', 'hsic')


    @staticmethod
    def legendre_sis_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'legendre', 'sis')

    @staticmethod
    def legendre_dc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'legendre', 'dc')

    @staticmethod
    def legendre_hsic_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'legendre', 'hsic')

    @staticmethod
    def rbf_sis_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'rbf', 'sis')

    @staticmethod
    def rbf_hsic_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'rbf', 'hsic')

    @staticmethod
    def rff_sis_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'rff', 'sis')

    @staticmethod
    def rff_hsic_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'rff', 'hsic')

    @staticmethod
    def qbin_sis_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'qbin', 'sis')

    @staticmethod
    def qbin_dc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'qbin', 'dc')

    @staticmethod
    def spline_sis_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'spline', 'sis')

    @staticmethod
    def spline_hsic_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return _basis_x_measure_score(Xtr, ytr, p, task, cat_indices, 'spline', 'hsic')

    # ---- 9. UniTree-CV(§12) ----
    @staticmethod
    def unitree_cv_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if p > MAX_TRAIN_FEATURES:
            return None
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.model_selection import KFold
        n = Xtr.shape[0]
        min_leaf = max(5, int(np.ceil(UNITREE_MIN_LEAF_FACTOR * n)))
        kf = KFold(n_splits=min(CV_FOLDS, n), shuffle=True, random_state=0)
        scores = np.zeros(p, dtype=np.float64)
        if task == 'classification':
            classes = np.unique(ytr)
            L0 = _cv_baseline_logloss(ytr, kf)
            for j in range(p):
                xj = Xtr[:, j].reshape(-1, 1)
                Lj = _cv_logloss_clf(xj, ytr, kf, min_leaf, classes)
                scores[j] = max(L0 - Lj, 0.0)
        else:
            L0 = _cv_baseline_mse(ytr, kf)
            for j in range(p):
                xj = Xtr[:, j].reshape(-1, 1)
                Lj = _cv_mse_reg(xj, ytr, kf, min_leaf)
                scores[j] = max(L0 - Lj, 0.0)
        return scores

def _cv_baseline_logloss(y, kf):
    ys = np.asarray(y)
    L = 0.0; n = len(ys)
    for tr, te in kf.split(ys):
        pi = np.array([(ys[tr] == c).mean() for c in np.unique(ys)])
        pi = np.clip(pi, 1e-12, 1)
        Lk = 0.0
        for i in te:
            pri = np.clip((ys[tr] == ys[i]).mean(), 1e-12, 1)
            Lk -= np.log(pri)
        L += Lk
    return L / n

def _cv_baseline_mse(y, kf):
    ys = np.asarray(y, dtype=np.float64); L = 0.0; n = len(ys)
    for tr, te in kf.split(ys):
        ytr_mean = ys[tr].mean()
        L += np.sum((ys[te] - ytr_mean) ** 2)
    return L / n

def _cv_logloss_clf(xj, y, kf, min_leaf, classes):
    from sklearn.tree import DecisionTreeClassifier
    n = len(y); L = 0.0
    for tr, te in kf.split(xj):
        m = DecisionTreeClassifier(max_depth=UNITREE_MAX_DEPTH, min_samples_leaf=min_leaf,
                                    random_state=0)
        m.fit(xj[tr], y[tr])
        proba = m.predict_proba(xj[te])

        prob = np.clip(_gather_proba(proba, m.classes_, np.asarray(y)[te]), 1e-12, 1.0)
        L -= np.sum(np.log(prob))
    return L / n

def _gather_proba(proba, model_classes, y_true):
    out = np.zeros(len(y_true))
    cmap = {c: i for i, c in enumerate(model_classes)}
    for i, c in enumerate(y_true):
        out[i] = proba[i, cmap[c]] if c in cmap else 1e-12
    return out

def _cv_mse_reg(xj, y, kf, min_leaf):
    from sklearn.tree import DecisionTreeRegressor
    n = len(y); L = 0.0
    for tr, te in kf.split(xj):
        m = DecisionTreeRegressor(max_depth=UNITREE_MAX_DEPTH, min_samples_leaf=min_leaf,
                                   random_state=0)
        m.fit(xj[tr], y[tr])
        pred = m.predict(xj[te])
        L += np.sum((y[te] - pred) ** 2)
    return L / n

TRACK_B_DISPATCH = {
    'relieff':     TrackBScreening.relieff_score,
    'rrelieff':    TrackBScreening.rrelieff_score,
    'legendre_wmc': TrackBScreening.legendre_wmc_score,
    'spline_wmc':  TrackBScreening.spline_wmc_score,
    'spline_dc':   TrackBScreening.spline_dc_score,
    'rbf_dc':      TrackBScreening.rbf_dc_score,
    'rff_dc':      TrackBScreening.rff_dc_score,
    'qbin_hsic':   TrackBScreening.qbin_hsic_score,
    'unitree_cv':  TrackBScreening.unitree_cv_score,

    'legendre_sis':  TrackBScreening.legendre_sis_score,
    'legendre_dc':   TrackBScreening.legendre_dc_score,
    'legendre_hsic': TrackBScreening.legendre_hsic_score,
    'rbf_sis':  TrackBScreening.rbf_sis_score,
    'rbf_hsic': TrackBScreening.rbf_hsic_score,
    'rff_sis':  TrackBScreening.rff_sis_score,
    'rff_hsic': TrackBScreening.rff_hsic_score,
    'qbin_sis': TrackBScreening.qbin_sis_score,
    'qbin_dc':  TrackBScreening.qbin_dc_score,
    'spline_sis':  TrackBScreening.spline_sis_score,
    'spline_hsic': TrackBScreening.spline_hsic_score,
}

_METHOD_REPR = {
    'relieff': 'neighborhood', 'rrelieff': 'neighborhood',
    'legendre_wmc': 'poly-WMC', 'spline_wmc': 'spline-WMC',
    'spline_dc': 'spline-VectorDC', 'rbf_dc': 'rbf-VectorDC',
    'rff_dc': 'rff-VectorDC', 'qbin_hsic': 'quantile-NHSIC',
    'unitree_cv': 'univariate-tree-CV',

    'legendre_sis': 'legendre-SIS', 'legendre_dc': 'legendre-VectorDC',
    'legendre_hsic': 'legendre-NHSIC',
    'rbf_sis': 'rbf-SIS', 'rbf_hsic': 'rbf-NHSIC',
    'rff_sis': 'rff-SIS', 'rff_hsic': 'rff-NHSIC',
    'qbin_sis': 'qbin-SIS', 'qbin_dc': 'qbin-VectorDC',
    'spline_sis': 'spline-SIS', 'spline_hsic': 'spline-NHSIC',
}

def compute_trackB_scores(Xtr, ytr, Xval, yval, p, task, cat_indices, methods):
    results = {}
    for m in methods:
        t0 = time.time()
        try:
            sc = TRACK_B_DISPATCH[m](Xtr, ytr, Xval, yval, p, task, cat_indices)
            if sc is not None and not np.isnan(np.asarray(sc, dtype=np.float64)).all():
                results[m] = sc
        except Exception as e:
            print(f"  [TrackB] {m} failed (logged as missing): {e}", flush=True)
        dt = time.time() - t0
        if dt > 120:
            print(f"  [TrackB] {m} took {dt:.1f}s (p={p})", flush=True)
    return results

def _tiny_cls_mean_shift_B(n=200, p=60, s=6, mu=0.6, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n); X = np.zeros((n, p))
    for j in range(p):
        Z = rng.randn(n)
        if j < s:
            X[y == 0, j] = Z[y == 0] - mu; X[y == 1, j] = Z[y == 1] + mu
        else:
            X[:, j] = Z
    return X, y, list(range(s))

def _tiny_reg_nonlinear_B(n=200, p=60, s=4, c=4.0, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    Y = (2.0 * X[:, 0] + 2.0 * X[:, 1] ** 2 + 2.0 * X[:, 2] ** 2
         + 2.0 * X[:, 3] ** 2 + c * rng.randn(n))
    return X, Y, [0, 1, 2, 3]

def validate_trackB_scorers(methods, task='classification'):
    print("\n" + "=" * 70)
    print(f"Track-B validation (task={task})")
    print("=" * 70, flush=True)
    np.random.seed(0)
    if task == 'classification':
        X, y, tf = _tiny_cls_mean_shift_B(n=200, p=60, s=6, mu=0.6, seed=0)
    else:
        X, y, tf = _tiny_reg_nonlinear_B(n=200, p=60, s=4, c=4.0, seed=0)
    n, p = X.shape; s = len(tf)
    Xf, cat = _fill_missing(X, None)
    from sklearn.model_selection import train_test_split
    is_cls = (len(np.unique(y)) < n / 2)
    strat = y if (is_cls and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xval, ytr, yval = train_test_split(Xf, y, test_size=0.35, random_state=0, stratify=strat)

    failures = []
    relax = {'rff_dc', 'rff_sis', 'rff_hsic'}

    task_only = {'rrelieff': 'regression', 'relieff': 'classification'}
    for m in methods:
        if m in task_only and task_only[m] != task:
            print(f"  [VALIDATE] {m}: N/A for task={task} (expects {task_only[m]}), skipped (not a failure)",
                  flush=True)
            continue
        t0 = time.time()
        try:
            sc = TRACK_B_DISPATCH[m](Xtr, ytr, Xval, yval, p, task, cat)
        except Exception as e:
            print(f"  [VALIDATE] {m}: SCORER EXCEPTION -> {e}", flush=True)
            failures.append(m); continue
        dt = time.time() - t0
        if sc is None:
            print(f"  [VALIDATE] {m}: returned None", flush=True)
            failures.append(m); continue
        sc = np.nan_to_num(np.asarray(sc, dtype=np.float64), nan=-np.inf)
        ranks = np.argsort(np.argsort(-sc)) + 1
        sig_r = ranks[tf]; null_r = np.delete(ranks, tf)
        sig_mean = float(sig_r.mean()); null_mean = float(null_r.mean())
        thr = min(3 * s if m in relax else 2 * s, p)
        n_in = len(set(tf) & set(np.argsort(-sc)[:thr].tolist()))


        reg_critical = {'spline_wmc', 'rbf_sis', 'rbf_hsic', 'spline_sis', 'spline_hsic'}
        if task == 'regression' and m in reg_critical:
            ok = (sig_mean < null_mean) or (n_in >= 1)
        else:
            ok = (sig_mean < null_mean) and (n_in >= int(np.ceil(s / 2)))
        if not ok:
            failures.append(m)
        print(f"  [VALIDATE] {m:<12} {dt:5.1f}s: {'OK' if ok else 'FAIL'} "
              f"(sig mean={sig_mean:.0f} < null mean={null_mean:.0f},  in top{thr}={n_in}/{s})", flush=True)
    print("-" * 70)
    if failures:
        print(f"[VALIDATE] FAILED: {failures}", flush=True)
        print("[VALIDATE] Aborting. Use --skip-validation to override.", flush=True)
        return False
    print("[VALIDATE] All checks passed.", flush=True)
    return True

_HAS_TORCH_BASELINE = True

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    _HAS_SK_RF = True
except Exception:
    _HAS_SK_RF = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import catboost as cb
    _HAS_CB = True
except Exception:
    _HAS_CB = False

TRACK_C_METHODS = ['rf_mdi', 'rf_pfi_oob', 'rf_pfi_val', 'cpi',
                    'lgb_split', 'lgb_gain', 'lgb_pfi', 'cb_pvc', 'cb_lfc']

CPI_TOP_Q = 5

CPI_N_GROUPS = 8

_C_METHOD_NJOBS = int(os.environ.get('PACE_C_METHOD_NJOBS', '1'))

def _rf_params(cls):
    base = dict(RF_PARAMS_CLS if cls else RF_PARAMS_REG)
    nj = _model_njobs()
    if nj != -1:
        base['n_jobs'] = nj
    return base

def _lgb_params(cls):
    base = dict(LGB_PARAMS_CLS if cls else LGB_PARAMS_REG)
    nj = _model_njobs()
    if nj != -1:
        base['n_jobs'] = nj
    return base

def _cb_params(cls):
    base = dict(CB_PARAMS_CLS if cls else CB_PARAMS_REG)
    nj = _model_njobs()
    if nj != -1 and nj > 0:
        base['thread_count'] = nj
    return base

class TrackCScreening:

    # ---- 1. RF MDI (§1) ----
    @staticmethod
    def rf_mdi_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_SK_RF:
            raise RuntimeError("sklearn RF unavailable")
        if p > 2 * MAX_TRAIN_FEATURES:
            print(f"  [TrackC] rf_mdi: p={p}>{2*MAX_TRAIN_FEATURES}, skipping (tree too slow)",
                  flush=True)
            return None
        if task == 'classification':
            m = RandomForestClassifier(**_rf_params(True))
        else:
            m = RandomForestRegressor(**_rf_params(False))
        m.fit(Xtr, ytr)
        raw = m.feature_importances_.astype(np.float64)
        if len(raw) != p:
            return None
        return raw

    # ---- 2. RF OOB permutation (§3) ----
    @staticmethod
    def rf_pfi_oob_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_SK_RF or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackC] rf_pfi_oob: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None


        if task == 'classification':
            m = RandomForestClassifier(**_rf_params(True))
        else:
            m = RandomForestRegressor(**_rf_params(False))
        m.fit(Xtr, ytr)
        ev_X = Xval if Xval is not None else Xtr
        ev_y = yval if yval is not None else ytr
        try:
            s = _pfi_batched(m, ev_X, ev_y, task, n_repeats=PFI_REPEATS)
        except Exception as e:
            print(f"  [TrackC] rf_pfi_oob batched failed -> fallback generic: {e}", flush=True)
            s = _pfi_generic(m, ev_X, ev_y, task, n_repeats=PFI_REPEATS)
        if s is None or len(s) != p:
            return None
        return s

    # ---- 3. RF PFI on val (§2) ----
    @staticmethod
    def rf_pfi_val_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_SK_RF or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackC] rf_pfi_val: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None
        if Xval is None:
            return None
        if task == 'classification':
            m = RandomForestClassifier(**_rf_params(True))
        else:
            m = RandomForestRegressor(**_rf_params(False))
        m.fit(Xtr, ytr)
        try:
            s = _pfi_batched(m, Xval, yval, task, n_repeats=PFI_REPEATS)
        except Exception as e:
            print(f"  [TrackC] rf_pfi_val batched failed -> fallback generic: {e}", flush=True)
            s = _pfi_generic(m, Xval, yval, task, n_repeats=PFI_REPEATS)
        return s if (s is not None and len(s) == p) else None

    # ---- 4. Conditional PFI subgroup (§4.1) ----
    @staticmethod
    def cpi_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_SK_RF or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackC] cpi: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None
        if task == 'classification':
            m = RandomForestClassifier(**_rf_params(True))
        else:
            m = RandomForestRegressor(**_rf_params(False))
        m.fit(Xtr, ytr)
        ev_X = Xval if Xval is not None else Xtr
        ev_y = yval if yval is not None else ytr
        n_ev = ev_X.shape[0]
        is_cls = (task == 'classification')

        if is_cls:
            classes = np.asarray(getattr(m, 'classes_', np.arange(2))).astype(int)
            cmap = {int(c): i for i, c in enumerate(classes)}
            y_idx = np.array([cmap.get(int(v), 0) for v in ev_y])
        L0 = _metric_loss_C(task, m, ev_X, ev_y)
        scores = np.zeros(p, dtype=np.float64)

        Xtr_std = Xtr.std(axis=0) + 1e-12
        Xtr_c = (Xtr - Xtr.mean(axis=0)) / Xtr_std
        n_tr = Xtr.shape[0]
        cmat = (Xtr_c.T @ Xtr_c) / n_tr
        np.fill_diagonal(cmat, 0.0)
        q = min(CPI_TOP_Q, max(1, p - 1))
        top_q_idx = np.argsort(-np.abs(cmat), axis=1)[:, :q]
        rng = np.random.RandomState(0)
        n_groups = min(CPI_N_GROUPS, n_ev)

        grp_all = np.zeros((p, n_ev), dtype=np.int16)
        for j in range(p):
            cm = top_q_idx[j]
            Cmat_ev = ev_X[:, cm]
            if Cmat_ev.shape[1] == 0 or n_ev < 4:
                continue
            lead = Cmat_ev[:, 0]
            bins = np.quantile(lead, np.linspace(0, 1, n_groups + 1)[1:-1])
            grp_all[j] = (np.digitize(lead, bins) if len(bins) > 0 else np.zeros(n_ev, dtype=int))
        CHUNK = 32
        for r in range(PFI_REPEATS):
            for j0 in range(0, p, CHUNK):
                js = np.arange(j0, min(j0 + CHUNK, p))
                blocks = []
                for j in js:
                    Xp = ev_X.copy()
                    grp = grp_all[j]
                    for g in np.unique(grp):
                        idg = np.where(grp == g)[0]
                        if len(idg) < 2:
                            continue
                        perm = rng.permutation(idg)
                        Xp[idg, j] = ev_X[perm, j]
                    blocks.append(Xp)
                Xbig = np.vstack(blocks)
                pbig = _model_predict_proba_or_prob(m, Xbig).reshape(len(js), n_ev, -1)
                for ki, j in enumerate(js):
                    if is_cls:
                        l = -np.mean(np.log(np.clip(pbig[ki][np.arange(n_ev), y_idx], 1e-12, 1.0)))
                    else:
                        pred = pbig[ki, :, 0]
                        l = float(np.mean((ev_y - pred) ** 2))
                    scores[j] += l - L0
        scores /= PFI_REPEATS
        return scores

    # ---- 5/6/7. LightGBM ----
    @staticmethod
    def _lgb_fit(Xtr, ytr, Xval, yval, task, cat_indices):
        params = _lgb_params(task == 'classification')

        data_kw = dict()
        if cat_indices:

            data_kw['categorical_feature'] = list(cat_indices)
        if task == 'classification':
            cls = len(np.unique(ytr))
            dtr = lgb.Dataset(Xtr, label=ytr, **data_kw)
            model = lgb.LGBMClassifier(objective='multiclass' if cls > 2 else 'binary',
                                       num_class=cls if cls > 2 else None, **params)
        else:
            dtr = lgb.Dataset(Xtr, label=ytr, **data_kw)
            model = lgb.LGBMRegressor(objective='regression', **params)
        es = None
        if Xval is not None:
            es = [lgb.early_stopping(20, verbose=False)] if hasattr(lgb, 'early_stopping') else None
        try:
            if Xval is not None:
                model.fit(Xtr, ytr, eval_set=[(Xval, yval)], categorical_feature=cat_indices,
                           callbacks=es)
            else:
                model.fit(Xtr, ytr, categorical_feature=cat_indices)
            gpu_used = False
        except Exception:

            m_cpu = type(model)(**{k: v for k, v in params.items() if k != 'device'})
            if Xval is not None:
                m_cpu.fit(Xtr, ytr, eval_set=[(Xval, yval)], categorical_feature=cat_indices, callbacks=es)
            else:
                m_cpu.fit(Xtr, ytr, categorical_feature=cat_indices)
            model = m_cpu
            gpu_used = False
        else:
            gpu_used = False
        return model, gpu_used

    @staticmethod
    def lgb_split_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_LGB or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackC] lgb_split: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None
        model, _ = TrackCScreening._lgb_fit(Xtr, ytr, Xval, yval, task, cat_indices)

        bo = model.booster_ if hasattr(model, 'booster_') else model
        imp = bo.feature_importance(importance_type='split')
        s = np.asarray(imp, dtype=np.float64)
        return s if len(s) == p else None

    @staticmethod
    def lgb_gain_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_LGB or p > MAX_TRAIN_FEATURES:
            return None
        model, _ = TrackCScreening._lgb_fit(Xtr, ytr, Xval, yval, task, cat_indices)
        bo = model.booster_ if hasattr(model, 'booster_') else model
        imp = bo.feature_importance(importance_type='gain')
        s = np.asarray(imp, dtype=np.float64)
        return s if len(s) == p else None

    @staticmethod
    def lgb_pfi_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_LGB or p > MAX_TRAIN_FEATURES:
            return None
        model, _ = TrackCScreening._lgb_fit(Xtr, ytr, Xval, yval, task, cat_indices)
        ev_X = Xval if Xval is not None else Xtr
        ev_y = yval if yval is not None else ytr

        try:
            s = _pfi_batched(model, ev_X, ev_y, task, n_repeats=PFI_REPEATS)
        except Exception as e:
            print(f"  [TrackC] lgb_pfi batched failed -> fall back generic: {e}", flush=True)
            s = _pfi_generic(model, ev_X, ev_y, task, n_repeats=PFI_REPEATS)
        return s if (s is not None and len(s) == p) else None

    # ---- 8/9. CatBoost ----
    @staticmethod
    def _cb_fit(Xtr, ytr, Xval, yval, task, cat_indices):
        params = _cb_params(task == 'classification')
        pool_args = {}
        if cat_indices:

            pool_args['cat_features'] = list(cat_indices)

        try:
            pool_tmp = None
            if task == 'classification':
                model = cb.CatBoostClassifier(**params, task_type='GPU', devices='0')
            else:
                model = cb.CatBoostRegressor(**params, task_type='GPU', devices='0')
            pool_tr = cb.Pool(Xtr, ytr, **pool_args)
            eval_set = (cb.Pool(Xval, yval, **pool_args)) if Xval is not None else None
            if eval_set is not None:
                model.fit(pool_tr, eval_set=eval_set, use_best_model=True)
            else:
                model.fit(pool_tr)
            return model, True     # GPU
        except Exception as e:

            params_cpu = dict(params)
            if task == 'classification':
                model = cb.CatBoostClassifier(**params_cpu)
            else:
                model = cb.CatBoostRegressor(**params_cpu)
            pool_tr = cb.Pool(Xtr, ytr, **pool_args)
            eval_set = (cb.Pool(Xval, yval, **pool_args)) if Xval is not None else None
            if eval_set is not None:
                model.fit(pool_tr, eval_set=eval_set, use_best_model=True)
            else:
                model.fit(pool_tr)
            return model, False

    @staticmethod
    def cb_pvc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_CB or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackC] cb_pvc: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None
        model, gpu = TrackCScreening._cb_fit(Xtr, ytr, Xval, yval, task, cat_indices)

        pool = cb.Pool(Xtr, ytr,
                        cat_features=list(cat_indices) if cat_indices else None)
        try:
            imp = model.get_feature_importance(data=pool, type='PredictionValuesChange')
        except Exception:
            imp = model.get_feature_importance(type='PredictionValuesChange')
        s = np.asarray(imp, dtype=np.float64)
        return s if len(s) == p else None

    @staticmethod
    def cb_lfc_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_CB or p > MAX_TRAIN_FEATURES:
            return None
        model, gpu = TrackCScreening._cb_fit(Xtr, ytr, Xval, yval, task, cat_indices)
        pool_args = dict(cat_features=list(cat_indices)) if cat_indices else {}

        if Xval is not None:
            pool = cb.Pool(Xval, yval, **pool_args)
        else:
            pool = cb.Pool(Xtr, ytr, **pool_args)
        try:
            imp = model.get_feature_importance(data=pool, type='LossFunctionChange')
        except Exception as e:
            print(f"  [TrackC] cb_lfc LFC failed: {e}", flush=True)
            return None
        s = np.asarray(imp, dtype=np.float64)
        return s if len(s) == p else None

def _pfi_generic(model, X_eval, y_eval, task, n_repeats=PFI_REPEATS):
    y_eval = np.asarray(y_eval)
    rng = np.random.RandomState(0)
    p = X_eval.shape[1]
    L0 = _metric_loss_C(task, model, X_eval, y_eval)
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        losses = []
        for _ in range(n_repeats):
            Xp = X_eval.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            losses.append(_metric_loss_C(task, model, Xp, y_eval))
        scores[j] = float(np.mean(losses)) - L0
    return scores

def _model_predict_proba_or_prob(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        return np.asarray(model.predict(X)).reshape(-1, 1)

def _pfi_batched(model, X_eval, y_eval, task, n_repeats=PFI_REPEATS, chunk=128):
    y_eval = np.asarray(y_eval)
    n_ev, p = X_eval.shape
    L0 = _metric_loss_C(task, model, X_eval, y_eval)
    is_cls = (task == 'classification')
    if is_cls:
        classes = np.asarray(getattr(model, 'classes_',
            np.arange(_model_predict_proba_or_prob(model, X_eval).shape[1]))).astype(int)
        cmap = {int(c): i for i, c in enumerate(classes)}
        y_idx = np.array([cmap.get(int(v), 0) for v in y_eval])

    rng = np.random.RandomState(0)
    perm_repo = np.empty((p, n_repeats, n_ev), dtype=np.int64)
    for j in range(p):
        for r in range(n_repeats):
            perm_repo[j, r] = rng.permutation(n_ev)
    scores = np.zeros(p, dtype=np.float64)
    for r in range(n_repeats):
        for j0 in range(0, p, chunk):
            js = np.arange(j0, min(j0 + chunk, p))
            blocks = [X_eval.copy() for _ in range(len(js))]
            for ki, j in enumerate(js):
                blocks[ki][:, j] = X_eval[perm_repo[j, r], j]
            Xbig = np.vstack(blocks)                              # (len(js)*n_ev, p)
            pbig = _model_predict_proba_or_prob(model, Xbig)      # (len(js)*n_ev, C)
            C = pbig.shape[1]
            pbig = pbig.reshape(len(js), n_ev, C)
            for ki, j in enumerate(js):
                if is_cls:
                    l = -np.mean(np.log(np.clip(pbig[ki][np.arange(n_ev), y_idx], 1e-12, 1.0)))
                else:
                    pred = pbig[ki, :, 0]
                    l = float(np.mean((y_eval - pred) ** 2))
                scores[j] += l - L0
    return scores / n_repeats

TRACK_C_DISPATCH = {
    'rf_mdi':     TrackCScreening.rf_mdi_score,
    'rf_pfi_oob': TrackCScreening.rf_pfi_oob_score,
    'rf_pfi_val': TrackCScreening.rf_pfi_val_score,
    'cpi':        TrackCScreening.cpi_score,
    'lgb_split':  TrackCScreening.lgb_split_score,
    'lgb_gain':   TrackCScreening.lgb_gain_score,
    'lgb_pfi':    TrackCScreening.lgb_pfi_score,
    'cb_pvc':     TrackCScreening.cb_pvc_score,
    'cb_lfc':     TrackCScreening.cb_lfc_score,
}

_METHOD_MODEL_C = {
    'rf_mdi': 'RF', 'rf_pfi_oob': 'RF', 'rf_pfi_val': 'RF', 'cpi': 'RF',
    'lgb_split': 'LGB', 'lgb_gain': 'LGB', 'lgb_pfi': 'LGB',
    'cb_pvc': 'CB', 'cb_lfc': 'CB',
}

def compute_trackC_scores(Xtr, ytr, Xval, yval, p, task, cat_indices, methods):
    def _one(m):
        t0 = time.time()
        try:
            sc = TRACK_C_DISPATCH[m](Xtr, ytr, Xval, yval, p, task, cat_indices)
        except Exception as e:
            print(f"  [TrackC] {m} failed (logged as missing): {e}", flush=True)
            sc = None
        return m, sc, time.time() - t0

    if _C_METHOD_NJOBS > 1 and len(methods) > 1:
        pairs = Parallel(n_jobs=min(_C_METHOD_NJOBS, len(methods)), prefer='threads')(
            delayed(_one)(m) for m in methods)
    else:
        pairs = [_one(m) for m in methods]

    results = {}
    for m, sc, dt in pairs:
        if dt > 120:
            print(f"  [TrackC] {m} took {dt:.1f}s (p={p})", flush=True)
        if sc is not None and not np.isnan(np.asarray(sc, dtype=np.float64)).all():
            results[m] = sc
    return results

def _tiny_cls_mean_shift_C(n=200, p=80, s=6, mu=0.6, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    X = np.zeros((n, p))
    for j in range(p):
        Z = rng.randn(n)
        if j < s:
            X[y == 0, j] = Z[y == 0] - mu
            X[y == 1, j] = Z[y == 1] + mu
        else:
            X[:, j] = Z
    return X, y, list(range(s))

def _tiny_reg_nonlinear_C(n=200, p=80, s=4, c=4.0, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    Y = (2.0 * X[:, 0] + 2.0 * X[:, 1] ** 2 + 2.0 * np.exp(X[:, 2])
         + 2.0 * np.log(np.abs(X[:, 3]) + 1e-8) + c * rng.randn(n))
    return X, Y, [0, 1, 2, 3]

def validate_trackC_scorers(methods, task='classification'):
    print("\n" + "=" * 70)
    print(f"Track-C tree-importance validation (task={task})")
    print("=" * 70, flush=True)
    np.random.seed(0)
    if task == 'classification':
        X, y, tf = _tiny_cls_mean_shift_C(n=200, p=80, s=6, mu=0.6, seed=0)
    else:
        X, y, tf = _tiny_reg_nonlinear_C(n=200, p=80, s=4, c=4.0, seed=0)
    n, p = X.shape
    s = len(tf)

    Xf, cat = _fill_missing(X, None)
    Xtr, ytr, Xval, yval, has_val = _train_val_split(Xf, y, n, random_state=0)

    failures = []
    for m in methods:
        try:
            sc = TRACK_C_DISPATCH[m](Xtr, ytr, Xval if has_val else None,
                                       yval if has_val else None, p, task, cat)
        except Exception as e:
            print(f"  [VALIDATE] {m}: SCORER EXCEPTION -> {e}", flush=True)
            failures.append(m); continue
        if sc is None:
            print(f"  [VALIDATE] {m}: returned None", flush=True)
            failures.append(m); continue
        sc = np.nan_to_num(np.asarray(sc, dtype=np.float64), nan=-np.inf)
        ranks = np.argsort(np.argsort(-sc)) + 1
        sig_r = ranks[tf]; null_r = np.delete(ranks, tf)
        sig_mean = float(sig_r.mean()); null_mean = float(null_r.mean())
        top2s = set(np.argsort(-sc)[:min(2 * s, p)].tolist())
        n_in = len(set(tf) & top2s)
        ok = (sig_mean < null_mean) and (n_in >= int(np.ceil(s / 2)))
        if not ok:
            failures.append(m)
        print(f"  [VALIDATE] {m:<11} model={_METHOD_MODEL_C[m]:<3}: {'OK' if ok else 'FAIL'} "
              f"(sig_ranks={sig_r.tolist()}, sig mean={sig_mean:.0f} < null mean={null_mean:.0f}, "
              f" in top{2*s}={n_in}/{s})", flush=True)


    if 'cb_lfc' in methods and 'cb_lfc' not in failures:
        try:
            sc = TrackCScreening.cb_lfc_score(Xtr, ytr, Xval, yval, p, task, cat)
            if sc is not None:
                has_neg = (np.nan_to_num(sc) < 0).any()
                print(f"  [VALIDATE] cb_lfc allows negatives (uncropped): {'OK' if has_neg else 'OK(no-neg-but-allowed)'}",
                      flush=True)
        except Exception:
            pass

    print("-" * 70)
    if failures:
        print(f"[VALIDATE] FAILED: {failures}", flush=True)
        print("[VALIDATE] Aborting. Use --skip-validation to override.", flush=True)
        return False
    print("[VALIDATE] All checks passed.", flush=True)
    return True

_HAS_TORCH_BASELINE = True

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    _HAS_SK_RF = True
except Exception:
    _HAS_SK_RF = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import catboost as cb
    _HAS_CB = True
except Exception:
    _HAS_CB = False

try:
    import shap as _shap_lib
    _HAS_SHAP_LIB = True
except Exception:
    _HAS_SHAP_LIB = False

try:
    import sage as _sage_lib
    _HAS_SAGE_LIB = True
except Exception:
    _HAS_SAGE_LIB = False

TRACK_D_METHODS = ['loco_simple', 'loco_cv', 'lgb_shap', 'cb_shap', 'sage_lgb', 'sage_cb', 'tabicl_shap']

LOCO_P_MAX = 500

LOCO_P_MAX_K = 300

LOCO_CV_FOLDS = 5

SAGE_P_MAX = 3000

SHAP_BACKGROUND = 200

SHAP_EVAL = 1000

SAGE_N_PERM = 32

SAGE_N_IMPUTE = 32

SAGE_TAU = 0.01


TABICL_V2_CLF = str(mp.checkpoint('tabicl_v2_clf'))
TABICL_V2_REG = str(mp.checkpoint('tabicl_v2_reg'))
TABICL_SHAP_N_EST = 2
TABICL_SHAP_EVAL = 256
TABICL_SHAP_P_MAX = 5000

def _background_sample(Xtr, n_bg, random_state=0):
    n = Xtr.shape[0]
    rng = np.random.RandomState(random_state)
    if n <= n_bg:
        return Xtr.copy()
    idx = rng.choice(n, size=n_bg, replace=False)
    return Xtr[idx].copy()

def _fit_lgb(Xtr, ytr, task, cat_indices):
    params = dict(LGB_PARAMS_CLS if task == 'classification' else LGB_PARAMS_REG)
    if task == 'classification':
        cls = len(np.unique(ytr))
        model = lgb.LGBMClassifier(objective='multiclass' if cls > 2 else 'binary',
                                    num_class=cls if cls > 2 else None, **params)
    else:
        model = lgb.LGBMRegressor(objective='regression', **params)
    try:
        model.fit(Xtr, ytr, categorical_feature=cat_indices or 'auto')
    except Exception:
        model.fit(Xtr, ytr)
    return model

def _fit_cb(Xtr, ytr, task, cat_indices, use_gpu=False):
    params = dict(CB_PARAMS_CLS if task == 'classification' else CB_PARAMS_REG)
    pool_args = dict(cat_features=list(cat_indices)) if cat_indices else {}
    pool_tr = cb.Pool(Xtr, ytr, **pool_args)
    if use_gpu:
        try:
            if task == 'classification':
                model = cb.CatBoostClassifier(**params, task_type='GPU', devices='0')
            else:
                model = cb.CatBoostRegressor(**params, task_type='GPU', devices='0')
            model.fit(pool_tr)
            return model, True
        except Exception:
            pass
    if task == 'classification':
        model = cb.CatBoostClassifier(**params)
    else:
        model = cb.CatBoostRegressor(**params)
    model.fit(pool_tr)
    return model, False

def _fit_rf(Xtr, ytr, task):
    if not _HAS_SK_RF:
        raise RuntimeError("sklearn RF unavailable")
    if task == 'classification':
        m = RandomForestClassifier(n_estimators=300, max_depth=6, max_features=0.8,
                                    random_state=0, n_jobs=-1, oob_score=True, bootstrap=True)
    else:
        m = RandomForestRegressor(n_estimators=300, max_depth=6, max_features=0.8,
                                   random_state=0, n_jobs=-1, oob_score=True, bootstrap=True)
    m.fit(Xtr, ytr)
    return m

class TrackDScreening:

    # ---- 1. LOCO simple (§2, val-split) ----
    @staticmethod
    def loco_simple_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return compute_scores(Xtr, ytr, "loco_simple", task=normalize_task(task, ytr),
                              validation_data=(Xval, yval), cat_indices=cat_indices)

    # ---- 2. Cross-fitted LOCO (§2.1) ----
    @staticmethod
    def loco_cv_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return compute_scores(Xtr, ytr, "loco_cv", task=normalize_task(task, ytr),
                              validation_data=(Xval, yval), cat_indices=cat_indices)


    # =================================================================

    # =================================================================

    @staticmethod
    def _lgb_shap_values(model, X_eval, task):
        booster = model.booster_ if hasattr(model, 'booster_') else model
        X_eval = np.asarray(X_eval, dtype=np.float64)
        contrib = booster.predict(X_eval, pred_contrib=True)

        if isinstance(contrib, list):
            C = len(contrib)
            phis = [np.array(c)[:, :-1] for c in contrib]
            stack = np.stack(phis, axis=0)                   # (C, n, p)
            return np.mean(np.abs(stack), axis=0).astype(np.float64)  # (n, p)

        contrib = np.array(contrib)
        if contrib.ndim == 2:


            from lightgbm.sklearn import LGBMModel as _LGBMModel
            n_cls = getattr(model, 'n_classes_', None)
            if n_cls is not None and n_cls > 2:

                p1 = contrib.shape[1] // n_cls
                if p1 * n_cls == contrib.shape[1] and p1 >= 2:
                    contrib = contrib.reshape(n_cls, contrib.shape[0], p1)  # (C, n, p+1)
                    contrib = contrib[:, :, :-1]
                    return np.mean(np.abs(contrib), axis=0).astype(np.float64)

            if contrib.shape[1] >= 1:
                return contrib[:, :-1].astype(np.float64)
        return contrib.astype(np.float64)

    @staticmethod
    def lgb_shap_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_LGB or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackD] lgb_shap: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None
        model = _fit_lgb(Xtr, ytr, task, cat_indices)


        ev_X = Xval if Xval is not None else Xtr
        n_ev = ev_X.shape[0]
        if n_ev > SHAP_EVAL:
            rng = np.random.RandomState(0)
            idx = rng.choice(n_ev, size=SHAP_EVAL, replace=False)
            ev_Xs = ev_X[idx]
        else:
            ev_Xs = ev_X
        try:
            phi = TrackDScreening._lgb_shap_values(model, ev_Xs, task)  # (n, p)
        except Exception as e:
            print(f"  [TrackD] lgb_shap pred_contrib failed: {e}", flush=True)
            return None
        if phi is None or phi.ndim != 2 or phi.shape[1] != p:
            return None

        return np.mean(np.abs(phi), axis=0)

    @staticmethod
    def cb_shap_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        if not _HAS_CB or p > MAX_TRAIN_FEATURES:
            if p > MAX_TRAIN_FEATURES:
                print(f"  [TrackD] cb_shap: p={p}>{MAX_TRAIN_FEATURES}, skipping", flush=True)
            return None
        model, _ = _fit_cb(Xtr, ytr, task, cat_indices, use_gpu=False)
        ev_X = Xval if Xval is not None else Xtr
        n_ev = ev_X.shape[0]
        if n_ev > SHAP_EVAL:
            rng = np.random.RandomState(0)
            idx = rng.choice(n_ev, size=SHAP_EVAL, replace=False)
            ev_Xs = ev_X[idx]
        else:
            ev_Xs = ev_X
        pool_args = dict(cat_features=list(cat_indices)) if cat_indices else {}
        pool_ev = cb.Pool(ev_Xs, **pool_args)
        try:
            sv = model.get_feature_importance(pool_ev, type='ShapValues')
        except Exception as e:
            print(f"  [TrackD] cb_shap ShapValues failed: {e}", flush=True)
            return None
        sv = np.array(sv)


        if sv.ndim == 3 and sv.shape[2] >= p + 1:
            phi = sv[:, :, :-1]                       # (n, C, p)
            phi = np.mean(np.abs(phi), axis=1)        # (n, p)
            return np.mean(np.abs(phi), axis=0)        # (p,)
        if sv.ndim != 2 or sv.shape[1] < p + 1:
            return None
        phi = sv[:, :-1]
        return np.mean(np.abs(phi), axis=0)

    # ---- TabICL-v2 SHAP importance (GPU) ----
    @staticmethod
    def tabicl_shap_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return compute_scores(Xtr, ytr, "tabicl_shap", task=normalize_task(task, ytr),
                              validation_data=(Xval, yval), cat_indices=cat_indices)


    # =================================================================

    # =================================================================


    @staticmethod
    def _marginal_imputer(model, X_bg, X_S, S, B):
        n_eval = X_S.shape[0]
        X_full = np.array(X_S, dtype=np.float64, copy=True)
        S_bar = ~S
        if not S_bar.any():

            return np.tile(X_full, (B, 1)), n_eval


        rng = np.random.RandomState(0)
        n_bg = X_bg.shape[0]
        rep_X = []
        b_idx = rng.choice(n_bg, size=B, replace=True) if n_bg < B else \
                rng.choice(n_bg, size=B, replace=False)

        blocks = []
        for b in b_idx:
            Xb = np.array(X_S, dtype=np.float64, copy=True)
            Xb[:, S_bar] = X_bg[b, S_bar]
            blocks.append(Xb)
        return np.vstack(blocks), n_eval     # (n_eval*B, p)

    @staticmethod
    def _sage_mc(model, Xtr, ytr, Xval, yval, p, task, cat_indices,
                 n_perm=SAGE_N_PERM, B=SAGE_N_IMPUTE):
        ev_X = Xval if Xval is not None else Xtr
        ev_y = yval if yval is not None else ytr
        y_arr = np.asarray(ev_y)
        n_eval = ev_X.shape[0]

        X_bg = _background_sample(Xtr, SHAP_BACKGROUND, random_state=0)
        rng = np.random.RandomState(0)


        def _subset_loss(S_mask):
            if not S_mask.any():


                blocks = []
                for b in rng.choice(X_bg.shape[0], size=min(B, X_bg.shape[0]), replace=False):
                    Xb = np.tile(X_bg[b], (n_eval, 1))
                    blocks.append(Xb)
                Xp = np.vstack(blocks)
                L = _sage_predict_loss(model, Xp, y_arr, task,
                                        n_eval=n_eval, n_blocks=len(blocks))
                return L

            Xp, _ne = TrackDScreening._marginal_imputer(model, X_bg, ev_X, S_mask, B)
            L = _sage_predict_loss(model, Xp, y_arr, task, n_eval=_ne, n_blocks=B)
            return L

        phi = np.zeros(p, dtype=np.float64)
        for r in range(n_perm):
            perm = rng.permutation(p)
            S_mask = np.zeros(p, dtype=bool)
            L_prev = _subset_loss(S_mask)            # L(S_0=empty)
            for t in range(p):
                j = perm[t]
                S_mask = S_mask.copy(); S_mask[j] = True
                L_cur = _subset_loss(S_mask)         # L(S_t)
                phi[j] += (L_prev - L_cur)           # delta = L(S_{t-1}) - L(S_t)
                L_prev = L_cur
        phi /= n_perm
        return phi

    @staticmethod
    def sage_lgb_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return compute_scores(Xtr, ytr, "sage_lgb", task=normalize_task(task, ytr),
                              validation_data=(Xval, yval), cat_indices=cat_indices)

    @staticmethod
    def sage_cb_score(Xtr, ytr, Xval, yval, p, task, cat_indices):
        return compute_scores(Xtr, ytr, "sage_cb", task=normalize_task(task, ytr),
                              validation_data=(Xval, yval), cat_indices=cat_indices)

def _sage_predict_loss(model, X, y, task, n_eval, n_blocks):
    N = X.shape[0]
    y = np.asarray(y)
    try:
        if task == 'classification':

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)         # (N, C)
            else:
                pred = np.asarray(model.predict(X)).ravel()

                return float(np.mean(pred != np.repeat(y, n_blocks))) if N > 0 else 0.0
            C = proba.shape[1]
            proba = proba.reshape(n_blocks, n_eval, C).mean(axis=0)   # (n_eval, C)
            classes = np.asarray(getattr(model, 'classes_', np.arange(C))).astype(int)
            cmap = {c: i for i, c in enumerate(classes)}
            idx = np.array([cmap.get(int(v), 0) for v in y])
            p_true = np.clip(proba[np.arange(n_eval), idx], 1e-12, 1.0)
            return float(np.mean(-np.log(p_true)))
        else:
            pred = np.asarray(model.predict(X)).ravel()    # (N,)
            pred = pred.reshape(n_blocks, n_eval).mean(axis=0)   # (n_eval,)
            return float(np.mean((y - pred) ** 2))
    except Exception as e:

        try:
            pred = np.asarray(model.predict(X)).ravel()
            if task == 'classification':
                pred = pred.reshape(n_blocks, n_eval).mean(axis=0).round().astype(int)
                return float(np.mean(pred != y))
            pred = pred.reshape(n_blocks, n_eval).mean(axis=0)
            return float(np.mean((y - pred) ** 2))
        except Exception:
            return float('inf')

TRACK_D_DISPATCH = {
    'loco_simple': TrackDScreening.loco_simple_score,
    'loco_cv':     TrackDScreening.loco_cv_score,
    'lgb_shap':    TrackDScreening.lgb_shap_score,
    'cb_shap':     TrackDScreening.cb_shap_score,
    'sage_lgb':    TrackDScreening.sage_lgb_score,
    'sage_cb':     TrackDScreening.sage_cb_score,
    'tabicl_shap': TrackDScreening.tabicl_shap_score,
}

_METHOD_MODEL_D = {
    'loco_simple': 'RF', 'loco_cv': 'RF',
    'lgb_shap': 'LGB', 'cb_shap': 'CB',
    'sage_lgb': 'LGB', 'sage_cb': 'CB',
    'tabicl_shap': 'TabICL',
}

def compute_trackD_scores(Xtr, ytr, Xval, yval, p, task, cat_indices, methods):
    results = {}
    for m in methods:
        t0 = time.time()
        try:
            sc = TRACK_D_DISPATCH[m](Xtr, ytr, Xval, yval, p, task, cat_indices)
            if sc is not None and len(sc) == p and not np.isnan(np.asarray(sc, dtype=np.float64)).all():
                results[m] = np.asarray(sc, dtype=np.float64)
        except Exception as e:
            print(f"  [TrackD] {m} failed (logged as missing): {e}", flush=True)
        dt = time.time() - t0
        if dt > 120:
            print(f"  [TrackD] {m} took {dt:.1f}s (p={p})", flush=True)
    return results

def _tiny_cls_mean_shift_D(n=200, p=80, s=6, mu=0.6, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    X = np.zeros((n, p))
    for j in range(p):
        Z = rng.randn(n)
        if j < s:
            X[y == 0, j] = Z[y == 0] - mu
            X[y == 1, j] = Z[y == 1] + mu
        else:
            X[:, j] = Z
    return X, y, list(range(s))

def _tiny_reg_nonlinear_D(n=200, p=80, s=4, c=4.0, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    Y = (2.0 * X[:, 0] + 2.0 * X[:, 1] ** 2
         + 2.0 * np.exp(X[:, 2] - X[:, 2].mean())
         + 2.0 * np.log(np.abs(X[:, 3]) + 1e-8) + c * rng.randn(n))
    return X, Y, [0, 1, 2, 3]

def validate_trackD_scorers(methods, task='classification'):
    print("\n" + "=" * 70)
    print(f"Track-D predictive-contribution validation (task={task})")
    print("=" * 70, flush=True)
    if task == 'classification':
        X, y, tf = _tiny_cls_mean_shift_D(n=200, p=80, s=6, mu=0.6, seed=0)
    else:
        X, y, tf = _tiny_reg_nonlinear_D(n=200, p=80, s=4, c=4.0, seed=0)
    n, p = X.shape
    s = len(tf)

    Xf, cat = _fill_missing(X, None)
    Xtr, ytr, Xval, yval, has_val = _train_val_split(Xf, y, n, random_state=0)

    failures = []
    for m in methods:
        try:
            sc = TRACK_D_DISPATCH[m](Xtr, ytr, Xval if has_val else None,
                                     yval if has_val else None, p, task, cat)
        except Exception as e:
            print(f"  [VALIDATE] {m}: SCORER EXCEPTION -> {e}", flush=True)
            failures.append(m); continue
        if sc is None:
            print(f"  [VALIDATE] {m}: returned None", flush=True)
            failures.append(m); continue
        sc = np.nan_to_num(np.asarray(sc, dtype=np.float64), nan=-np.inf)

        if m in ('lgb_shap', 'cb_shap', 'tabicl_shap') and (sc < 0).any():
            print(f"  [VALIDATE] {m}: mean-|SHAP| must be nonnegative; negative value detected: FAIL", flush=True)
            failures.append(m); continue
        ranks = np.argsort(np.argsort(-sc)) + 1
        sig_r = ranks[tf]; null_r = np.delete(ranks, tf)
        sig_mean = float(sig_r.mean()); null_mean = float(null_r.mean())
        top2s = set(np.argsort(-sc)[:min(2 * s, p)].tolist())
        n_in = len(set(tf) & top2s)
        ok = (sig_mean < null_mean) and (n_in >= int(np.ceil(s / 2)))
        if not ok:
            failures.append(m)
        extra = ''
        if m.startswith('sage'):
            has_neg = (sc < 0).any()
            extra = f" {'(neg allowed/uncropped OK)' if has_neg else '(no-neg-but-allowed OK)'}"
        print(f"  [VALIDATE] {m:<11} model={_METHOD_MODEL_D[m]:<3}: {'OK' if ok else 'FAIL'} "
              f"(sig_ranks={sig_r.tolist()}, sig mean={sig_mean:.0f}<null mean={null_mean:.0f}, "
              f" in top{2*s}={n_in}/{s}){extra}", flush=True)

    print("-" * 70)
    if failures:
        print(f"[VALIDATE] FAILED: {failures}", flush=True)
        print("[VALIDATE] Aborting. Use --skip-validation to override.", flush=True)
        return False
    print("[VALIDATE] All checks passed.", flush=True)
    return True

_APPEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "append")

for _sub in ("tabular-feature-selection-main", "GradEnFS-main", "SAND-main", "xRFM-main"):
    _p = os.path.join(_APPEND, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

MAX_TRAIN_FEATURES_E = 100_000

TRACK_E_METHODS = ['xrfm_agop', 'lassonet_path', 'gradenfs', 'sand_score', 'deep_lasso']

XRFM_N_TREES = 1

XRFM_MAX_LEAF = 60_000

LASSONET_HIDDEN = (64,)

LASSONET_M = 10

LASSONET_N_ITERS = (300, 30)

LASSONET_PATIENCE = (30, 10)

GRADENFS_HIDDEN = 512

GRADENFS_EPOCH = 50

GRADENFS_LR = 1e-3

GRADENFS_EPSILON = 1

GRADENFS_ALPHA = 0.3

GRADENFS_BETA = 0.9

GRADENFS_BATCH = 100

SAND_K0 = 100

SAND_SIGMA = 0.5

SAND_EPOCHS = 100

SAND_LR = 1e-3

SAND_BATCH = 64

DEEPLASSO_ALPHA = 0.1

DEEPLASSO_LAYERS = (128, 128)

DEEPLASSO_DROPOUT = 0.1

DEEPLASSO_EPOCHS = 100

DEEPLASSO_LR = 1e-3

DEEPLASSO_WD = 2e-4

DEEPLASSO_BATCH = 256

EPOCHS_SCALER = 1.0

RANK_TYPE = {
    'xrfm_agop': 'native-full-score',
    'lassonet_path': 'official-path',
    'gradenfs': 'native-full-score',
    'sand_score': 'budget-conditional-gate',
    'deep_lasso': 'native-full-score',
}

def _dev_str(device):
    if isinstance(device, str):
        return 'cuda' if 'cuda' in device else 'cpu'
    return 'cuda' if (device is not None and device.type == 'cuda') else 'cpu'

def _standardize(Xtr, Xval):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xval = np.asarray(Xval, dtype=np.float64)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xval - mu) / sd

def _torch_seed_all(seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def compute_mms_stable(scores, true_features):
    scores = np.asarray(scores, dtype=np.float64)
    ranked = np.argsort(-scores, kind='stable')
    rank_of = np.empty(scores.shape[0], dtype=np.int64)
    rank_of[ranked] = np.arange(scores.shape[0])
    tf = np.asarray(list(true_features), dtype=np.int64)
    tf = tf[(tf >= 0) & (tf < scores.shape[0])]
    if tf.size == 0:
        return int(scores.shape[0])
    return int(rank_of[tf].max()) + 1

def compute_mms_worst_tie(scores, true_features):
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    tf = np.asarray(list(true_features), dtype=np.int64)
    tf = tf[(tf >= 0) & (tf < n)]
    if tf.size == 0:
        return int(n)
    order = np.argsort(-scores, kind='stable')
    sv = scores[order]
    r_worst_sorted = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        r_worst_sorted[i:j + 1] = j + 1
        i = j + 1
    rank_of_worst = np.empty(n, dtype=np.int64)
    rank_of_worst[order] = r_worst_sorted
    return int(rank_of_worst[tf].max())

def _topk_stable(scores, k):
    ranked = np.argsort(-np.asarray(scores, dtype=np.float64), kind='stable')
    return set(ranked[:min(k, len(ranked))].tolist())

def evaluate_screening_E(scores, true_features, s, p, n, rank_type=None, status='ok'):
    if scores is None:
        return {'mms': p, 'mms_worst_tie': p, 'tpr_n1': 0.0,
                'fpr_n1': 1.0 if (p - s) > 0 else 0.0, 'tpr_nlogn': 0.0,
                'fpr_nlogn': 1.0 if (p - s) > 0 else 0.0,
                'rank_type': rank_type, 'status': status}
    scores = np.asarray(scores, dtype=np.float64)
    assert len(scores) == p, f"score len {len(scores)} != p {p}"
    mms = compute_mms_stable(scores, true_features)
    mw = compute_mms_worst_tie(scores, true_features)
    k1 = min(n - 1, p)
    knlog = min(max(1, int(n / np.log(n))) if n > 1 else 1, p)
    ts = set(true_features)
    t1 = len(_topk_stable(scores, k1) & ts) / max(1, len(ts))
    tnl = len(_topk_stable(scores, knlog) & ts) / max(1, len(ts))
    n_neg = p - len(ts)
    f1 = len(_topk_stable(scores, k1) - ts) / n_neg if n_neg > 0 else 0.0
    fnl = len(_topk_stable(scores, knlog) - ts) / n_neg if n_neg > 0 else 0.0
    return {'mms': mms, 'mms_worst_tie': mw, 'tpr_n1': t1, 'fpr_n1': f1,
            'tpr_nlogn': tnl, 'fpr_nlogn': fnl,
            'rank_type': rank_type, 'status': status}

def xrfm_agop_score(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):


    if p >= 100_000:
        return None, 'na_oom'
    if p > MAX_TRAIN_FEATURES_E:
        return None, 'na_resource'
    from xrfm import xRFM
    Xtr_s, Xval_s = _standardize(Xtr, Xval)
    metric = 'brier' if task == 'classification' else 'mse'
    if task == 'classification':
        ytr_ = np.asarray(ytr).astype(np.int64)
        yval_ = np.asarray(yval).astype(np.int64)
    else:
        ytr_ = np.asarray(ytr, dtype=np.float32)
        yval_ = np.asarray(yval, dtype=np.float32)
    model = xRFM(rfm_params=None, device=device, tuning_metric=metric,
                 n_trees=XRFM_N_TREES, max_leaf_size=XRFM_MAX_LEAF,
                 use_temperature_tuning=False, verbose=False, random_state=int(seed))
    model.fit(Xtr_s, ytr_, Xval_s, yval_)

    model.split_temperature = None
    grads = model.get_grads(Xtr_s)                       # (n, C, p)
    imp = (grads.float() ** 2).sum(dim=(0, 1)).cpu().numpy() / float(grads.shape[0])
    return imp.astype(np.float64), 'ok'

def lassonet_path_score(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):
    if p > MAX_TRAIN_FEATURES_E:
        return None, 'na_resource'
    from lassonet import LassoNetClassifier, LassoNetRegressor
    Xtr_s, Xval_s = _standardize(Xtr, Xval)
    dev = torch.device(device) if isinstance(device, str) else device
    common = dict(hidden_dims=LASSONET_HIDDEN, lambda_start='auto', path_multiplier=1.02,
                  M=LASSONET_M, gamma=0.0, dropout=0, batch_size=None,
                  n_iters=LASSONET_N_ITERS, patience=LASSONET_PATIENCE, tol=0.99,
                  backtrack=False, val_size=0.1, device=dev, verbose=0,
                  random_state=int(seed), torch_seed=int(seed))
    if task == 'classification':
        ytr_ = np.asarray(ytr).astype(np.int64)
        yval_ = np.asarray(yval).astype(np.int64)
        model = LassoNetClassifier(**common)
    else:
        ytr_ = np.asarray(ytr, dtype=np.float32)
        yval_ = np.asarray(yval, dtype=np.float32)
        model = LassoNetRegressor(**common)           # MSELoss
    model.path(Xtr_s, ytr_, X_val=Xval_s, y_val=yval_, return_state_dicts=False)

    fi = model.feature_importances_.detach().cpu().numpy().astype(np.float64)
    assert fi.shape[0] == p, f"feature_importances_ len {fi.shape[0]} != p {p}"
    return fi, 'ok'

def gradenfs_score(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):
    if task != 'classification':
        return None, 'na_task'                         # cls-only（methodology P1-C）
    if p > MAX_TRAIN_FEATURES_E:
        return None, 'na_resource'
    from model import MLP                              # append/GradEnFS-main/model.py
    from gradenfs import GradEnFS                      # append/GradEnFS-main/gradenfs.py
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import argparse, logging
    dev = _dev_str(device)
    _torch_seed_all(seed)
    Xtr_s, _ = _standardize(Xtr, Xval)
    ytr_int = np.asarray(ytr).astype(np.int64).ravel()
    n_classes = int(ytr_int.max()) + 1
    ytr_oh = np.eye(n_classes, dtype='float32')[ytr_int]
    bs = min(Xtr_s.shape[0], GRADENFS_BATCH)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_s.astype('float32')),
                                       torch.from_numpy(ytr_oh)),
                         batch_size=bs, shuffle=True, drop_last=False)
    args = argparse.Namespace(
        input_dim=p, hidden_dim=GRADENFS_HIDDEN, output_dim=n_classes,
        epoch=int(round(GRADENFS_EPOCH * EPOCHS_SCALER)), lr=GRADENFS_LR, device=dev,
        network='sparse', epsilon=GRADENFS_EPSILON, alpha=GRADENFS_ALPHA, beta=GRADENFS_BETA,
        k_list=[min(GRADENFS_BATCH, p)], batch_update=False,
        training_batch_size=GRADENFS_BATCH)
    logger = logging.getLogger('gradenfs_wrap'); logger.setLevel(logging.CRITICAL)
    mlp = MLP(p, GRADENFS_HIDDEN, n_classes).to(dev)
    optimizer = optim.Adam(mlp.parameters(), lr=GRADENFS_LR)
    loss_fn = nn.CrossEntropyLoss()
    args.batch = len(loader)
    ge = GradEnFS(mlp, args, logger); ge.apply_mask()
    mlp.train()
    n_ep = int(round(GRADENFS_EPOCH * EPOCHS_SCALER))
    for ep in range(1, n_ep + 1):
        for bi, (x, y) in enumerate(loader, start=1):
            x, y = x.to(dev), y.to(dev); x.requires_grad = True
            optimizer.zero_grad(); yh = mlp(x); loss = loss_fn(yh, y); loss.backward()
            grad = torch.mean(torch.abs(x.grad), dim=0)
            optimizer.step(); ge.update_importance_scores(grad)

            if bi == len(loader):
                ge.update_mask(bi, ep); ge.apply_mask()
    sc = ge.neuron_importance_scores.detach().cpu().numpy().astype(np.float64)
    return sc, 'ok'

def sand_score(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):
    if p > MAX_TRAIN_FEATURES_E:
        return None, 'na_resource'
    import math
    from torch import nn
    _torch_seed_all(seed)
    Xtr_s, _ = _standardize(Xtr, Xval)
    dev_obj = torch.device(device) if isinstance(device, str) else device
    is_clf = (task == 'classification')
    if is_clf:
        cat = np.unique(np.asarray(ytr)); remap = {c: i for i, c in enumerate(cat)}; n_out = len(cat)
        yt = np.vectorize(remap.get)(np.asarray(ytr)).astype(np.int64)
    else:
        n_out = 1
        ym = float(np.mean(ytr)); ys = float(np.std(ytr)); ys = ys if ys > 0 else 1.0
        yt = ((np.asarray(ytr, dtype=np.float64) - ym) / ys).astype(np.float32)

    class _SAND(nn.Module):


        def __init__(self, p, h, n_out, is_clf, k0, sigma):
            super().__init__()
            self.a = nn.Parameter(torch.full((p,), float(math.sqrt(k0 / max(p, 1)))))
            self.fc1 = nn.Linear(p, h); self.act = nn.LeakyReLU(0.0)
            self.head = nn.Linear(h, n_out)
            self.is_clf = is_clf; self.k0 = float(k0); self.sigma = float(sigma)

        def _gate(self):
            w = torch.clamp(torch.abs(self.a), max=1.0)               # wabs - relu(wabs-1) = min(wabs,1)
            return w * math.sqrt(self.k0) / (torch.sqrt((w ** 2).sum()) + 1e-12)

        def forward(self, x, training=False):
            wn = self._gate()
            out = x * wn.unsqueeze(0)
            if training:
                out = out + torch.randn_like(out) * (self.sigma * (1.0 - wn.unsqueeze(0)))
            return self.head(self.act(self.fc1(out))), wn

    hidden = min(256, max(32, p // 4))
    net = _SAND(p, hidden, n_out, is_clf, SAND_K0, SAND_SIGMA).to(dev_obj)
    opt = torch.optim.Adam(net.parameters(), lr=SAND_LR)
    crit = torch.nn.CrossEntropyLoss() if is_clf else torch.nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr_s.astype('float32')).to(dev_obj)
    ytr_t = torch.from_numpy(yt).to(dev_obj)
    n_tr = Xtr_t.shape[0]; bs = min(n_tr, SAND_BATCH)
    n_ep = int(round(SAND_EPOCHS * EPOCHS_SCALER))
    net.train()
    for _ in range(n_ep):
        perm = torch.randperm(n_tr, device=dev_obj)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            out, _ = net(xb, training=True)
            loss = crit(out, yb) if is_clf else crit(out.view(-1), yb.view(-1))
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        _, wn = net(Xtr_t, training=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gates = np.asarray(wn.detach().cpu().numpy(), dtype=np.float64)   # ≥0, Σa²≈k0
    return gates, 'ok'

def deep_lasso_score(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):
    if p > MAX_TRAIN_FEATURES_E:
        return None, 'na_resource'


    import importlib.util as _ilu
    _dl_base = os.path.join(_APPEND, "tabular-feature-selection-main")
    def _dl_load(modname, relpath):
        full = os.path.join(_dl_base, relpath) + ".py"
        spec = _ilu.spec_from_file_location(modname, full)
        m = _ilu.module_from_spec(spec); sys.modules[modname] = m
        spec.loader.exec_module(m); return m
    _dl_mlp = _dl_load("final_experiments_E._dl_mlp", "deep_tabular/models/mlp")
    _dl_tr  = _dl_load("final_experiments_E._dl_training", "deep_tabular/utils/training")
    _dl_te  = _dl_load("final_experiments_E._dl_testing", "deep_tabular/utils/testing")
    _dl_wu  = _dl_load("final_experiments_E._dl_warmup", "deep_tabular/utils/warmup")
    MLP = _dl_mlp.MLP
    default_training_loop, TrainingSetup = _dl_tr.default_training_loop, _dl_tr.TrainingSetup
    get_feat_importance_deeplasso = _dl_te.get_feat_importance_deeplasso
    LinearWarmup = _dl_wu.LinearWarmup
    from torch.utils.data import DataLoader, TensorDataset
    _torch_seed_all(seed)
    Xtr_s, Xval_s = _standardize(Xtr, Xval)
    dev_obj = torch.device(device) if isinstance(device, str) else device


    if task == 'classification':
        uniq = np.unique(np.asarray(ytr))
        if len(uniq) <= 2:
            sub, n_out = 'binclass', 1
            yt = (np.asarray(ytr) == uniq[-1]).astype(np.float32)
            yv = (np.asarray(yval) == uniq[-1]).astype(np.float32)
        else:
            remap = {c: i for i, c in enumerate(uniq)}; sub, n_out = 'multiclass', len(uniq)
            yt = np.vectorize(remap.get)(np.asarray(ytr)).astype(np.int64)
            yv = np.vectorize(remap.get)(np.asarray(yval)).astype(np.int64)
    else:
        sub, n_out = 'regression', 1
        ym = np.mean(ytr); ys = np.std(ytr); ys = ys if ys > 0 else 1.0
        yt = ((np.asarray(ytr, dtype=np.float64) - ym) / ys).astype(np.float32)
        yv = ((np.asarray(yval, dtype=np.float64) - ym) / ys).astype(np.float32)

    def _loader(X, yarr, shuffle):
        empty = torch.empty((X.shape[0], 0), dtype=torch.float32)
        ydt = 'int64' if sub == 'multiclass' else 'float32'
        return DataLoader(TensorDataset(torch.from_numpy(X.astype('float32')), empty,
                                         torch.from_numpy(np.asarray(yarr, dtype=ydt))),
                         batch_size=min(X.shape[0], DEEPLASSO_BATCH), shuffle=shuffle, drop_last=False)

    criterion = (torch.nn.CrossEntropyLoss() if sub == 'multiclass'
                 else torch.nn.BCEWithLogitsLoss() if sub == 'binclass'
                 else torch.nn.MSELoss())
    net = MLP(d_in=p, d_out=n_out, categories=None, d_embedding=1,
              d_layers=list(DEEPLASSO_LAYERS), dropout=DEEPLASSO_DROPOUT).to(dev_obj)
    optimizer = torch.optim.AdamW(net.parameters(), lr=DEEPLASSO_LR, weight_decay=DEEPLASSO_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                 T_max=int(round(DEEPLASSO_EPOCHS * EPOCHS_SCALER)), eta_min=0.0)
    warmup = LinearWarmup(optimizer, warmup_period=5)
    train_setup = TrainingSetup(criterions=criterion, optimizer=optimizer,
                                 scheduler=scheduler, warmup=warmup)

    class _Hyp:
        regularization = 'deep_lasso'
        reg_weight = float(DEEPLASSO_ALPHA)
    n_ep = int(round(DEEPLASSO_EPOCHS * EPOCHS_SCALER))
    tr_loader = _loader(Xtr_s, yt, True); val_loader = _loader(Xval_s, yv, False)
    net.train()
    for _ in range(n_ep):
        default_training_loop(net, tr_loader, train_setup, dev_obj, _Hyp())
    imp = get_feat_importance_deeplasso(net, val_loader, criterion, dev_obj)
    imp = np.asarray(imp.detach().cpu().numpy(), dtype=np.float64)
    return imp, 'ok'

TRACK_E_DISPATCH = {
    'xrfm_agop':     xrfm_agop_score,
    'lassonet_path': lassonet_path_score,
    'gradenfs':      gradenfs_score,
    'sand_score':    sand_score,
    'deep_lasso':    deep_lasso_score,
}

def _scorer_call(method, Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):
    fn = TRACK_E_DISPATCH[method]
    try:
        return fn(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device)
    except (ImportError, ModuleNotFoundError) as e:


        print(f"  [TrackE] {method}: NA (dep missing, cached): {str(e)[:120]}", flush=True)
        return None, 'na_dep'
    except Exception as e:
        print(f"  [TrackE] {method}: EXCEPTION -> NA: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return None, 'error'

def compute_trackE_scores(Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device, methods):
    results = {}
    for m in methods:
        if m not in results:
            t0 = time.time()
            sc, st = _scorer_call(m, Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device)
            if sc is not None:
                assert sc.shape[0] == p, f"{m} score shape {sc.shape} != p {p}"
            ev = evaluate_screening_E(sc, [0], 1, p, 1, rank_type=RANK_TYPE[m], status=st)
            results[m] = {'scores': sc, 'status': st, 'runtime_sec': time.time() - t0,
                           'eval': ev}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results

def validate_trackE_scorers(methods, task='classification'):
    print("\n" + "=" * 70)
    print(f"Track-E validation (task={task}) — imports + signal-vs-null sanity")
    print("=" * 70, flush=True)
    if task == 'classification':
        X, y, tf = _tiny_cls_mean_shift_B(n=200, p=60, s=6, mu=0.6, seed=0)
    else:
        X, y, tf = _tiny_reg_nonlinear_B(n=200, p=60, s=4, c=4.0, seed=0)
    n, p = X.shape; s = len(tf)
    Xf, cat = _fill_missing(X, None)
    from sklearn.model_selection import train_test_split
    is_cls = (len(np.unique(y)) < n / 2)
    strat = y if (is_cls and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xval, ytr, yval = train_test_split(Xf, y, test_size=0.35, random_state=0, stratify=strat)
    task_only = {'gradenfs': 'classification'}
    failures = []
    for m in methods:
        if m in task_only and task_only[m] != task:
            print(f"  [VALIDATE] {m}: N/A for task={task} (expects {task_only[m]}), skipped", flush=True)
            continue
        t0 = time.time()
        sc, st = _scorer_call(m, Xtr, ytr, Xval, yval, p, task, cat, 12345, DEVICE)
        dt = time.time() - t0
        if sc is None:
            print(f"  [VALIDATE] {m:<12} {dt:5.1f}s: {st} (returned None)", flush=True)
            if st == 'error':
                failures.append(m)
            continue
        sc = np.nan_to_num(np.asarray(sc, dtype=np.float64), nan=-np.inf)
        ranks = np.argsort(np.argsort(-sc, kind='stable'), kind='stable') + 1
        sig_r = ranks[tf]; null_r = np.delete(ranks, tf)
        thr = min(3 * s, p)
        n_in = len(set(tf) & set(np.argsort(-sc, kind='stable')[:thr].tolist()))
        ok = (sig_r.mean() < null_r.mean()) or (n_in >= 1)
        if not ok:
            failures.append(m)
        print(f"  [VALIDATE] {m:<12} {dt:5.1f}s: {'OK' if ok else 'FAIL'} "
              f"(sig_mean={sig_r.mean():.0f} < null_mean={null_r.mean():.0f}, top{thr}={n_in}/{s})", flush=True)
    print("-" * 70)
    if failures:
        print(f"[VALIDATE] FAILED: {failures}", flush=True)
        print("[VALIDATE] Aborting. Use --skip-validation to override.", flush=True)
        return False
    print("[VALIDATE] All checks passed.", flush=True)
    return True

def _method_hyperparams(m):
    if m == 'xrfm_agop':
        return {'n_trees': XRFM_N_TREES, 'max_leaf_size': XRFM_MAX_LEAF,
                'rfm_params': 'repo_default', 'use_temperature_tuning': False}
    if m == 'lassonet_path':
        return {'hidden_dims': list(LASSONET_HIDDEN), 'M': LASSONET_M,
                'n_iters': list(LASSONET_N_ITERS), 'patience': list(LASSONET_PATIENCE),
                'lambda_start': 'auto', 'path_multiplier': 1.02}
    if m == 'gradenfs':
        return {'hidden_dim': GRADENFS_HIDDEN, 'epoch': int(round(GRADENFS_EPOCH * EPOCHS_SCALER)),
                'lr': GRADENFS_LR, 'epsilon': GRADENFS_EPSILON, 'alpha': GRADENFS_ALPHA,
                'beta': GRADENFS_BETA, 'batch_update': False}
    if m == 'sand_score':
        return {'k0': SAND_K0, 'sigma': SAND_SIGMA, 'epochs': int(round(SAND_EPOCHS * EPOCHS_SCALER)),
                'lr': SAND_LR, 'loss_reg': 'MSE(repo MAE->MSE)'}
    if m == 'deep_lasso':
        return {'alpha(reg_weight)': DEEPLASSO_ALPHA, 'd_layers': list(DEEPLASSO_LAYERS),
                'dropout': DEEPLASSO_DROPOUT, 'epochs': int(round(DEEPLASSO_EPOCHS * EPOCHS_SCALER)),
                'lr': DEEPLASSO_LR, 'weight_decay': DEEPLASSO_WD}
    return {}

# ==========================================================================

# ==========================================================================


# ===========================================================================

# ===========================================================================




# Complete-candidate attribution methods.
import argparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from src.screening_utils import screen_split as _screen_split, atomic_save_npz, score_workspace, normalize_task, validate_scores
TABICL_SHAP_MAX_SAMPLES = 500
TABICL_SHAP_VALIDATION_FRACTION = 0.2
TABICL_SHAP_SAMPLING_ID = "rows500_split80_20_v1"
SAGE_METHODS = ("sage_lgb", "sage_cb")
SAGE_MAX_SAMPLES = 500
SAGE_VALIDATION_FRACTION = 0.2
SAGE_SAMPLING_ID = "rows500_split80_20_v1"
TASK_FULL = {"bin": "classification", "multi": "classification", "reg": "regression"}
_elapsed_base = 0.0
_elapsed_start = 0.0
def _method_sample_cap(method, args):
    if method in SAGE_METHODS:
        return min(args.max_screen_samples, SAGE_MAX_SAMPLES)
    return (min(args.max_screen_samples, TABICL_SHAP_MAX_SAMPLES)
            if method == "tabicl_shap" else args.max_screen_samples)


def _method_screen_split(method, prepared, args):
    options = ({"test_size": TABICL_SHAP_VALIDATION_FRACTION,
                "allow_tiny_overlap": False} if method == "tabicl_shap" else {})
    if method in SAGE_METHODS:
        options = {"test_size": SAGE_VALIDATION_FRACTION, "allow_tiny_overlap": False}
    return _screen_split(prepared["Xc_train"], prepared["y_train"],
                         args.current_task, _method_sample_cap(method, args), **options)


def _partial_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.stem + ".partial.npz")


def _config_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _bind_config(config, args):
    config.update(args.run_identity)
    return config


def _load_partial(path: Path, config: dict, **shapes):
    global _elapsed_base
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["config_json"].item()) != _config_json(config):
                return None
            out = {}
            for name, shape in shapes.items():
                a = np.asarray(z[name])
                if a.shape != shape:
                    return None
                out[name] = a
            _elapsed_base = float(z["elapsed_seconds"][0]) if "elapsed_seconds" in z else 0.0
            return out
    except Exception:
        return None


def _save_partial(path: Path, config: dict, **arrays):
    atomic_save_npz(path, config_json=np.array(_config_json(config)),
                    elapsed_seconds=np.array([_elapsed_base + time.perf_counter() - _elapsed_start]),
                    **arrays)


def _rf_fit(X, y, task, trees, max_depth, max_features, n_jobs):
    cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
    model = cls(
        n_estimators=trees,
        max_depth=max_depth,
        max_features=max_features,
        random_state=0,
        n_jobs=n_jobs,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(X, y)
    return model


def _loss(model, X, y, task):
    if task == "regression":
        pred = np.asarray(model.predict(X)).reshape(-1)
        return float(np.mean((np.asarray(y) - pred) ** 2))
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = np.asarray(model.classes_).astype(int)
        cmap = {c: i for i, c in enumerate(classes)}
        idx = np.array([cmap.get(int(v), -1) for v in y])
        p_true = np.zeros(len(y))
        known = idx >= 0
        p_true[known] = proba[np.flatnonzero(known), idx[known]]
        p_true = np.clip(p_true, 1e-12, 1.0)
        return float(-np.mean(np.log(p_true)))
    return float(np.mean(np.asarray(model.predict(X)).reshape(-1) != np.asarray(y)))


def _score_loco_simple(Xtr, ytr, Xv, yv, task, args, final_path):
    P = Xtr.shape[1]
    cfg = {
        "method": "loco_simple", "P": P, "rf_trees": args.rf_trees,
        "rf_max_depth": args.rf_max_depth, "rf_max_features": args.rf_max_features,
    }
    _bind_config(cfg, args)
    part = _load_partial(_partial_path(final_path), cfg, scores=(P,), done=(P,))
    scores = np.full(P, np.nan) if part is None else part["scores"].astype(float)
    done = np.zeros(P, dtype=bool) if part is None else part["done"].astype(bool)
    full = _rf_fit(Xtr, ytr, task, args.rf_trees, args.rf_max_depth,
                   args.rf_max_features, args.rf_n_jobs)
    loss_full = _loss(full, Xv, yv, task)
    def one(j):
        keep = np.ones(P, dtype=bool)
        keep[j] = False
        reduced = _rf_fit(Xtr[:, keep], ytr, task, args.rf_trees, args.rf_max_depth,
                          args.rf_max_features, args.rf_n_jobs)
        return int(j), _loss(reduced, Xv[:, keep], yv, task) - loss_full

    pending = list(map(int, np.flatnonzero(~done)))
    batch_size = max(1, args.checkpoint_every, args.loco_feature_workers)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        if args.loco_feature_workers > 1:
            from joblib import Parallel, delayed
            values = Parallel(n_jobs=args.loco_feature_workers, prefer="threads")(
                delayed(one)(j) for j in batch)
        else:
            values = [one(j) for j in batch]
        for j, value in values:
            scores[j] = value
            done[j] = True
        _save_partial(_partial_path(final_path), cfg, scores=scores, done=done)
        print(f"      loco_simple {done.sum()}/{P}", flush=True)
    return scores, cfg


def _score_loco_cv(Xtr, ytr, task, args, final_path):
    P = Xtr.shape[1]
    K = min(args.loco_cv_folds, max(2, len(Xtr) // 10))
    cfg = {
        "method": "loco_cv", "P": P, "folds": K, "rf_trees": args.rf_trees,
        "rf_max_depth": args.rf_max_depth, "rf_max_features": args.rf_max_features,
    }
    shape = (K, P)
    _bind_config(cfg, args)
    part = _load_partial(_partial_path(final_path), cfg, delta=(K, P), done=shape)
    delta = np.full(shape, np.nan) if part is None else part["delta"].astype(float)
    done = np.zeros(shape, dtype=bool) if part is None else part["done"].astype(bool)
    folds = list(KFold(n_splits=K, shuffle=True, random_state=0).split(Xtr))
    for fold, (fit_idx, eval_idx) in enumerate(folds):
        if done[fold].all():
            continue
        Xf, yf = Xtr[fit_idx], ytr[fit_idx]
        Xe, ye = Xtr[eval_idx], ytr[eval_idx]
        full = _rf_fit(Xf, yf, task, args.rf_trees, args.rf_max_depth,
                       args.rf_max_features, args.rf_n_jobs)
        loss_full = _loss(full, Xe, ye, task)
        def one(j):
            keep = np.ones(P, dtype=bool)
            keep[j] = False
            reduced = _rf_fit(Xf[:, keep], yf, task, args.rf_trees, args.rf_max_depth,
                              args.rf_max_features, args.rf_n_jobs)
            return int(j), _loss(reduced, Xe[:, keep], ye, task) - loss_full

        pending = list(map(int, np.flatnonzero(~done[fold])))
        batch_size = max(1, args.checkpoint_every, args.loco_feature_workers)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if args.loco_feature_workers > 1:
                from joblib import Parallel, delayed
                values = Parallel(n_jobs=args.loco_feature_workers, prefer="threads")(
                    delayed(one)(j) for j in batch)
            else:
                values = [one(j) for j in batch]
            for j, value in values:
                delta[fold, j] = value
                done[fold, j] = True
            total_done = int(done.sum())
            _save_partial(_partial_path(final_path), cfg, delta=delta, done=done)
            print(f"      loco_cv {total_done}/{K * P} (fold {fold + 1}/{K})", flush=True)
    return np.mean(delta, axis=0), cfg


def _fit_sage_model(method, Xtr, ytr, task, threads=8):
    for params in (LGB_PARAMS_CLS, LGB_PARAMS_REG):
        params["n_jobs"] = threads
    for params in (CB_PARAMS_CLS, CB_PARAMS_REG):
        params["thread_count"] = threads
    if method == "sage_lgb":
        model = _fit_lgb(Xtr, ytr, task, None)
    else:
        model, _ = _fit_cb(Xtr, ytr, task, None, use_gpu=False)
    # Explicitly bound inference threads as well as fit threads.
    original_predict = model.predict
    if method == "sage_cb":
        model.predict = lambda X: original_predict(X, thread_count=threads)
        if hasattr(model, "predict_proba"):
            original_proba = model.predict_proba
            model.predict_proba = lambda X: original_proba(X, thread_count=threads)
    return model


def _sage_loss(model, work, y, task):
    B, n_eval, P = work.shape
    X = work.reshape(B * n_eval, P)
    if task == "regression":
        pred = np.asarray(model.predict(X)).reshape(B, n_eval).mean(axis=0)
        result = float(np.mean((np.asarray(y) - pred) ** 2))
    else:
        proba = np.asarray(model.predict_proba(X))
        proba = proba.reshape(B, n_eval, -1).mean(axis=0)
        cmap = {int(c): i for i, c in enumerate(model.classes_)}
        idx = np.array([cmap.get(int(v), -1) for v in y])
        p_true = np.zeros(n_eval)
        known = idx >= 0
        p_true[known] = proba[np.flatnonzero(known), idx[known]]
        result = float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))
    if not np.isfinite(result):
        raise FloatingPointError("SAGE subset loss is non-finite")
    return result


def _score_sage(method, Xtr, ytr, Xv, yv, task, args, final_path):
    P = Xtr.shape[1]
    if args.sage_eval_max and len(Xv) > args.sage_eval_max:
        idx = np.random.RandomState(17).choice(len(Xv), args.sage_eval_max, replace=False)
        Xv, yv = Xv[idx], yv[idx]
    n_bg = min(args.sage_background, len(Xtr))
    B = args.sage_imputations
    cfg = {
        "method": method, "P": P, "permutations": args.sage_permutations,
        "imputations": B, "background": n_bg, "eval_rows": len(Xv),
        "fit_rows": len(Xtr), "sampling_protocol": SAGE_SAMPLING_ID,
        "fixed_marginal_draws_per_permutation": True,
    }
    _bind_config(cfg, args)
    part = _load_partial(
        _partial_path(final_path), cfg,
        phi_sum=(P,), done=(args.sage_permutations,))
    phi_sum = np.zeros(P) if part is None else part["phi_sum"].astype(float)
    done = np.zeros(args.sage_permutations, dtype=bool) if part is None else part["done"].astype(bool)
    model = _fit_sage_model(method, Xtr, ytr, task, args.threads)
    bg_idx = np.random.RandomState(0).choice(len(Xtr), n_bg, replace=False)
    Xbg = np.asarray(Xtr[bg_idx])
    for r in np.flatnonzero(~done):
        rng = np.random.RandomState(1009 + int(r))
        perm = rng.permutation(P)
        draw = rng.choice(n_bg, size=B, replace=(n_bg < B))
        # Shape (B,n_eval,P). Every prediction still contains all P columns.
        work = np.repeat(Xbg[draw, None, :], len(Xv), axis=1)
        previous = _sage_loss(model, work, yv, task)
        contribution = np.zeros(P, dtype=float)
        for step, j in enumerate(perm, start=1):
            work[:, :, j] = Xv[None, :, j]
            current = _sage_loss(model, work, yv, task)
            contribution[j] = previous - current
            previous = current
            if step % max(1, args.progress_every) == 0 or step == P:
                print(f"      {method} permutation {r + 1}/{args.sage_permutations}: "
                      f"{step}/{P} subsets", flush=True)
        phi_sum += contribution
        done[r] = True
        _save_partial(_partial_path(final_path), cfg, phi_sum=phi_sum, done=done)
        print(f"      {method} completed permutations {done.sum()}/{len(done)}", flush=True)
    return phi_sum / max(1, done.sum()), cfg


def _tabicl_model(task_type, args):
    from tabicl import TabICLClassifier, TabICLRegressor
    if task_type == "reg":
        path = mp.checkpoint('tabicl_v2_reg', args.model_root)
        if not path.exists():
            raise FileNotFoundError(path)
        return TabICLRegressor(
            model_path=str(path), checkpoint_version=path.name,
            n_estimators=args.tabicl_estimators, random_state=42,
            batch_size=2, device=args.device)
    path = mp.checkpoint('tabicl_v2_clf', args.model_root)
    if not path.exists():
        raise FileNotFoundError(path)
    return TabICLClassifier(
        model_path=str(path), checkpoint_version=path.name,
        n_estimators=args.tabicl_estimators, random_state=42,
        batch_size=2, device=args.device)


def _score_tabicl_shap(Xtr, ytr, Xv, task_type, args, final_path):
    import shap
    P = Xtr.shape[1]
    if args.tabicl_eval_max and len(Xv) > args.tabicl_eval_max:
        idx = np.random.RandomState(0).choice(len(Xv), args.tabicl_eval_max, replace=False)
        Xv = Xv[idx]
    cfg = {
        "method": "tabicl_shap", "P": P, "eval_rows": len(Xv),
        "fit_rows": len(Xtr),
        "sampling_protocol": TABICL_SHAP_SAMPLING_ID,
        "eval_batch": args.tabicl_eval_batch,
        "predict_batch": args.tabicl_predict_batch,
        "n_estimators": args.tabicl_estimators,
        "background": "one_all_nan_row",
        "max_evals": 2 * P + 1,
    }
    _bind_config(cfg, args)
    n_batches = (len(Xv) + args.tabicl_eval_batch - 1) // args.tabicl_eval_batch
    part = _load_partial(
        _partial_path(final_path), cfg,
        abs_sum=(P,), denominator=(1,), done=(n_batches,))
    abs_sum = np.zeros(P) if part is None else part["abs_sum"].astype(float)
    denominator = np.zeros(1) if part is None else part["denominator"].astype(float)
    done = np.zeros(n_batches, dtype=bool) if part is None else part["done"].astype(bool)

    model = _tabicl_model(task_type, args)
    model.fit(np.ascontiguousarray(Xtr, dtype=np.float64), np.asarray(ytr))
    raw_predict = model.predict if task_type == "reg" else model.predict_proba

    def predict_batched(z):
        z = np.ascontiguousarray(z, dtype=np.float64)
        out = []
        for start in range(0, len(z), args.tabicl_predict_batch):
            out.append(np.asarray(raw_predict(z[start:start + args.tabicl_predict_batch])))
        return np.concatenate(out, axis=0)

    background = np.full((1, P), np.nan, dtype=np.float64)
    for b in np.flatnonzero(~done):
        start = int(b) * args.tabicl_eval_batch
        stop = min(len(Xv), start + args.tabicl_eval_batch)
        ev = np.ascontiguousarray(Xv[start:stop], dtype=np.float64)
        try:
            explainer = shap.PermutationExplainer(predict_batched, background, seed=42000 + int(b))
        except TypeError:
            np.random.seed(42000 + int(b))
            explainer = shap.PermutationExplainer(predict_batched, background)
        values = np.asarray(
            explainer(ev, max_evals=2 * P + 1, silent=True).values,
            dtype=np.float64)
        if values.ndim == 3:
            abs_sum += np.abs(values).sum(axis=(0, 2))
            denominator[0] += values.shape[0] * values.shape[2]
        elif values.ndim == 2:
            abs_sum += np.abs(values).sum(axis=0)
            denominator[0] += values.shape[0]
        else:
            raise ValueError(f"unexpected TabICL-SHAP shape {values.shape}")
        done[b] = True
        _save_partial(_partial_path(final_path), cfg,
                      abs_sum=abs_sum, denominator=denominator, done=done)
        print(f"      tabicl_shap eval batches {done.sum()}/{len(done)}", flush=True)
    if denominator[0] <= 0:
        raise RuntimeError("TabICL-SHAP produced no explained rows")
    return abs_sum / denominator[0], cfg


def _validation_probe(method, Xtr, ytr, Xv, yv, task_type, args, final_path):
    """Fit full-width models and probe selected feature perturbations."""
    P = Xtr.shape[1]
    assert P > 0 and Xv.shape[1] == P
    task = TASK_FULL[task_type]
    if method.startswith("loco"):
        if method == "loco_cv":
            fit_idx, eval_idx = next(KFold(n_splits=2, shuffle=True, random_state=0).split(Xtr))
            fit_X, fit_y, eval_X, eval_y = Xtr[fit_idx], ytr[fit_idx], Xtr[eval_idx], ytr[eval_idx]
        else:
            fit_X, fit_y, eval_X, eval_y = Xtr, ytr, Xv, yv
        full = _rf_fit(fit_X, fit_y, task, min(5, args.rf_trees), args.rf_max_depth,
                       args.rf_max_features, args.rf_n_jobs)
        base = _loss(full, eval_X, eval_y, task)
        for j in (0, P - 1):
            keep = np.ones(P, dtype=bool); keep[j] = False
            reduced = _rf_fit(fit_X[:, keep], fit_y, task, min(5, args.rf_trees),
                              args.rf_max_depth, args.rf_max_features, args.rf_n_jobs)
            value = _loss(reduced, eval_X[:, keep], eval_y, task) - base
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite LOCO probe at feature {j}")
        return {"probe": "full-P fit and first/last feature deletion", "P": P}
    if method.startswith("sage"):
        ev_n = min(8, len(Xv))
        bg_n = min(8, len(Xtr))
        model = _fit_sage_model(method, Xtr, ytr, task, args.threads)
        Xbg = Xtr[:bg_n]
        B = min(2, bg_n)
        work = np.repeat(Xbg[:B, None, :], ev_n, axis=1)
        previous = _sage_loss(model, work, yv[:ev_n], task)
        for j in (0, P - 1):
            work[:, :, j] = Xv[None, :ev_n, j]
            current = _sage_loss(model, work, yv[:ev_n], task)
            if not np.isfinite(previous - current):
                raise RuntimeError(f"non-finite SAGE probe at feature {j}")
            previous = current
        return {"probe": "full-P SAGE model and first/last subset transitions", "P": P}
    model = _tabicl_model(task_type, args)
    model.fit(np.asarray(Xtr, dtype=np.float64), ytr)
    predict = model.predict if task_type == "reg" else model.predict_proba
    ev = np.asarray(Xv[:2], dtype=np.float64).copy()
    full = np.asarray(predict(ev))
    ev[:, [0, P - 1]] = np.nan
    masked = np.asarray(predict(ev))
    empty = np.asarray(predict(np.full_like(ev[:1], np.nan)))
    if not all(np.isfinite(a).all() for a in (full, masked, empty)):
        raise RuntimeError("TabICL full-P masked prediction returned non-finite values")
    return {"probe": "full-P fit, unmasked and first/last-feature masked predictions", "P": P}




ATTRIBUTION_METHODS = ('tabicl_shap', 'sage_lgb', 'sage_cb', 'loco_simple', 'loco_cv')


def attribution_options(**values):
    """Build explicit method defaults independently of any dataset runner."""
    defaults = dict(max_screen_samples=2000, rf_trees=300, rf_max_depth=6,
        rf_max_features=0.8, rf_n_jobs=-1, loco_feature_workers=1, loco_cv_folds=5,
        sage_permutations=32, sage_imputations=32, sage_background=200, sage_eval_max=0,
        tabicl_estimators=2, tabicl_eval_max=100, tabicl_eval_batch=1,
        tabicl_predict_batch=16, checkpoint_every=8, progress_every=100,
        device='cuda:0' if torch.cuda.is_available() else 'cpu', threads=8,
        model_root=str(mp.MODEL_ROOT), run_identity={})
    defaults.update({key: value for key, value in values.items() if key in defaults})
    return argparse.Namespace(**defaults)


def compute_scores(X, y, method, *, task='bin', device=None, aggregation='native',
                   validation_data=None, cat_indices=None, seed=42, work_path=None,
                   **options):
    """Compute a non-PACE screening method on all input features."""
    from src.screening_utils import validate_scores
    global DEVICE, _elapsed_base, _elapsed_start
    if device is None:
        from src import screening_methods_torch as base
        DEVICE = base.DEVICE
    if device is not None:
        DEVICE = torch.device(device)
        from src import screening_methods_torch as base
        base.DEVICE = DEVICE
    task_full = 'regression' if task == 'reg' else 'classification'
    if method in ATTRIBUTION_METHODS:
        args = attribution_options(device=str(DEVICE), **options)
        args.current_task = task
        args.run_identity = {'task': task, 'n_rows': len(X), 'n_features': X.shape[1]}
        if args.loco_feature_workers > 1:
            args.rf_n_jobs = 1
        if validation_data is None:
            Xtr, Xv, ytr, yv = _method_screen_split(method, {'Xc_train': X, 'y_train': y}, args)
        else:
            Xtr, ytr = X, y
            Xv, yv = validation_data
            if Xv is None:
                Xtr, Xv, ytr, yv = _screen_split(X, y, task, args.max_screen_samples)
        with score_workspace(work_path) as path:
            _elapsed_base, _elapsed_start = 0.0, time.perf_counter()
            if method == 'loco_simple':
                result = _score_loco_simple(Xtr, ytr, Xv, yv, task_full, args, path)
            elif method == 'loco_cv':
                result = _score_loco_cv(Xtr, ytr, task_full, args, path)
            elif method.startswith('sage'):
                result = _score_sage(method, Xtr, ytr, Xv, yv, task_full, args, path)
            else:
                result = _score_tabicl_shap(Xtr, ytr, Xv, task, args, path)
        return validate_scores(result[0], X.shape[1], method)
    if validation_data is None:
        Xtr, Xv, ytr, yv = _screen_split(X, y, task, len(X), test_size=VAL_FRACTION)
    else:
        Xtr, ytr = X, y
        Xv, yv = validation_data
    if method in TRACK_E_DISPATCH:
        scores, status = TRACK_E_DISPATCH[method](Xtr, ytr, Xv, yv,
            X.shape[1], task_full, cat_indices, seed, str(DEVICE))
        if scores is None or status not in (None, 'ok'):
            raise RuntimeError(f'{method}: {status}')
    else:
        dispatch = {**TRACK_B_DISPATCH, **TRACK_C_DISPATCH, **TRACK_D_DISPATCH}
        if method not in dispatch:
            raise ValueError(f'Unknown screening method: {method}')
        scores = dispatch[method](Xtr, ytr, Xv, yv, X.shape[1], task_full, cat_indices)
    return validate_scores(scores, X.shape[1], method)
