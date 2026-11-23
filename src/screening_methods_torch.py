from src import model_path as mp
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import math
import os
import numpy as np
import torch
from scipy import stats as sp_stats
from typing import Dict, List, Optional, Tuple
from pathlib import Path


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TABICL_PATH = mp.TABICL_PATH
CHUNK_SIZE = 256
EMB_FEAT_CHUNK = 5000


def _get_available_gpu_mem(device=None):
    if device is None:
        device = DEVICE
    if torch.cuda.is_available() and device.type == 'cuda':
        return torch.cuda.mem_get_info(device)[0]
    return 8 * 1024 ** 3


def _auto_chunk_size(n, bytes_per_feat, min_chunk=32, max_chunk=None,
                     gpu_mem_frac=0.25, device=None):
    if max_chunk is None:
        max_chunk = CHUNK_SIZE
    available = _get_available_gpu_mem(device) * gpu_mem_frac
    chunk = max(min_chunk, int(available / max(bytes_per_feat, 1)))
    return min(chunk, max_chunk)


def set_device(device_id: int = 0):
    global DEVICE
    DEVICE = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    return DEVICE


# =============================================================================

# =============================================================================

def _to_tensor(x, device=None, dtype=None):
    """NumPy / torch → torch.Tensor on device"""
    if device is None:
        device = DEVICE
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if dtype is not None:
        return torch.as_tensor(x, dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _prepare_y_torch(y: np.ndarray, n: int, device=None, dtype=torch.float32):
    if device is None:
        device = DEVICE
    unique_vals = np.unique(y)
    is_cat = len(unique_vals) < n / 2 or not np.issubdtype(y.dtype, np.number)
    if is_cat:
        classes = unique_vals
        Y = np.zeros((n, len(classes)), dtype=np.float64 if dtype == torch.float64 else np.float32)
        for i, c in enumerate(classes):
            Y[:, i] = (y == c).astype(float)
        return _to_tensor(Y, device, dtype=dtype), is_cat, classes
    else:
        return _to_tensor(y.reshape(-1, 1), device, dtype=dtype), is_cat, None


def _double_center(D: torch.Tensor) -> torch.Tensor:
    mu = D.mean()
    mu_r = D.mean(dim=1, keepdim=True)
    mu_c = D.mean(dim=0, keepdim=True)
    return D - mu_r - mu_c + mu


def _double_center_kappa(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    K_off = K.clone()
    K_off.fill_diagonal_(0)
    row_sums = K_off.sum(dim=1)
    col_sums = K_off.sum(dim=0)
    total = K_off.sum()
    a_row = row_sums / (n - 2)
    a_col = col_sums / (n - 2)
    a_total = total / ((n - 1) * (n - 2))
    K_c = K - a_row.unsqueeze(1) - a_col.unsqueeze(0) + a_total
    K_c.fill_diagonal_(0)
    return K_c


def _kappa_matrix(Z: torch.Tensor) -> torch.Tensor:
    dot_ZZ = Z @ Z.T
    norm_sq = (Z ** 2).sum(dim=1)
    num = 1 + dot_ZZ
    denom = torch.sqrt((1 + norm_sq.unsqueeze(1)) * (1 + norm_sq.unsqueeze(0)) + 1e-15)
    cos_val = torch.clamp(num / denom, -1, 1)
    return torch.asin(cos_val)


def _ids_transform(X: torch.Tensor, num_terms: int = 6,
                   bandwidth_term: float = 0.5) -> torch.Tensor:
    B = bandwidth_term
    n, d = X.shape
    device = X.device


    X64 = X.double()
    neg_B_x2 = -B * X64 ** 2                        # (n, d)


    log_fact = torch.zeros(num_terms, dtype=torch.float64, device=device)
    for i in range(1, num_terms):
        log_fact[i] = log_fact[i - 1] + math.log(i)
    inv_sqrt_fact = 1.0 / torch.exp(0.5 * log_fact)  # (num_terms,), float64


    log_abs_x = torch.log(torch.abs(X64) + 1e-300)
    sign_x = torch.sign(X64)


    terms = []
    cum_log_abs_x = torch.zeros(n, d, dtype=torch.float64, device=device)  # i=0: log(1)=0
    for i in range(num_terms):
        # log(|phi_i(x)|) = -B*x^2 + i*log|x| - 0.5*log(i!)
        log_abs_term = neg_B_x2 + cum_log_abs_x + math.log(inv_sqrt_fact[i])

        if i % 2 == 0:
            abs_term = torch.exp(log_abs_term)
        else:
            abs_term = torch.exp(log_abs_term) * sign_x
        terms.append(abs_term.float())
        cum_log_abs_x = cum_log_abs_x + log_abs_x

    return torch.cat(terms, dim=-1)                    # (n, d * num_terms)


def _mann_whitney_batched(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    p, n = x.shape
    R = len(masks)
    gammas = torch.zeros(p, R, dtype=x.dtype, device=x.device)

    for r_idx, mask_r in enumerate(masks):
        n_r = mask_r.sum().item()
        n_not_r = n - n_r
        if n_r == 0 or n_not_r == 0:
            continue

        mask_r_bool = mask_r.bool()
        x_r = x[:, mask_r_bool]         # (p, n_r)
        x_not_r = x[:, ~mask_r_bool]    # (p, n_not_r)

        # broadcasting: (p, n_r, 1) < (p, 1, n_not_r) → (p, n_r, n_not_r)
        count = (x_r.unsqueeze(2) < x_not_r.unsqueeze(1)).sum(dim=(1, 2))
        gammas[:, r_idx] = count.float() / (n_r * n_not_r)

    return gammas


# =============================================================================
# TorchAcceleratedScreening
# =============================================================================

class TorchAcceleratedScreening:


    @staticmethod
    def sis_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)                    # (n, p)
        n, p = Xt.shape
        y_prep, is_cat, classes = _prepare_y_torch(y, n)

        if is_cat and classes is not None:

            X_std = Xt.std(dim=0, unbiased=False)         # (p,)
            X_centered = Xt - Xt.mean(dim=0, keepdim=True)
            Y_centered = y_prep - y_prep.mean(dim=0, keepdim=True)
            Y_std = y_prep.std(dim=0, unbiased=False)     # (R,)

            # corr_matrix: (p, R)
            corr = (X_centered.T @ Y_centered) / (n * X_std.unsqueeze(1) * Y_std.unsqueeze(0) + 1e-10)
            corr = corr.abs()

            mask = (X_std.unsqueeze(1) < 1e-10) | (Y_std.unsqueeze(0) < 1e-10)
            corr[mask] = 0
            scores = corr.max(dim=1).values
        else:
            X_std = Xt.std(dim=0, unbiased=False)
            y_std = y_prep.std()
            X_centered = Xt - Xt.mean(dim=0, keepdim=True)
            y_centered = y_prep - y_prep.mean()
            scores = (X_centered.T @ y_centered).squeeze() / (n * X_std * y_std + 1e-10)
            scores = scores.abs()
            scores[X_std < 1e-10] = 0

        return scores.cpu().numpy()

    @staticmethod
    def dc_sis_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        b = torch.cdist(y_prep, y_prep)              # (n, n)
        b_c = _double_center(b)
        b_var = (b_c ** 2).sum()

        scores = torch.zeros(p, device=Xt.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            # (chunk, n, 1) → cdist → (chunk, n, n)
            X_chunk = Xt[:, start:end].T.unsqueeze(2)  # (C, n, 1)
            a = torch.cdist(X_chunk, X_chunk)           # (C, n, n)
            a_c = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(1, 2), keepdim=True)
            num = (a_c * b_c.unsqueeze(0)).sum(dim=(1, 2))
            a_var = (a_c ** 2).sum(dim=(1, 2))
            denom = torch.sqrt(a_var * b_var + 1e-30)
            dc_sq = num / denom
            scores[start:end] = torch.sqrt(torch.clamp(dc_sq.abs(), min=0, max=1))

        return scores.cpu().numpy()

    @staticmethod
    def sc_sis_accelerated(X: np.ndarray, y: np.ndarray, a: float = 0.5, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        dist_y = torch.cdist(y_prep, y_prep)
        Ky = torch.exp(-dist_y.pow(a))


        E2_y = (Ky.sum() - Ky.diagonal().sum()) / (n * (n - 1))
        Kyy = torch.exp(-dist_y.pow(a) * 2)
        E1_yy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
        E2_yy = E2_y ** 2


        row_sums_Ky = Ky.sum(dim=1)
        E3_yy = ((row_sums_Ky - Ky.diagonal()) ** 2).sum() / (n * (n - 1) * (n - 2))
        Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

        scores = torch.zeros(p, device=Xt.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            X_chunk = Xt[:, start:end].T.unsqueeze(2)   # (C, n, 1)
            dist_x = torch.cdist(X_chunk, X_chunk)      # (C, n, n)

            Kx = torch.exp(-dist_x.pow(a))
            Kxy = Kx * Ky.unsqueeze(0)
            E1 = (Kxy.sum(dim=(1, 2)) - Kxy.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))

            E2_x = (Kx.sum(dim=(1, 2)) - Kx.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
            E2 = E2_x * E2_y


            Kx_row_sums = Kx.sum(dim=2)       # (C, n)
            Kx_diag = Kx.diagonal(dim1=1, dim2=2)  # (C, n)
            Ky_row_sums = Ky.sum(dim=1)       # (n,)
            Ky_diag = Ky.diagonal()           # (n,)
            E3 = ((Kx_row_sums - Kx_diag) * (Ky_row_sums - Ky_diag).unsqueeze(0)).sum(dim=1) / (n * (n - 1) * (n - 2))

            Scov2 = E1 + E2 - 2 * E3

            Kxx = torch.exp(-dist_x.pow(a) * 2)
            E1_xx = (Kxx.sum(dim=(1, 2)) - Kxx.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
            E2_xx = E2_x ** 2

            E3_xx = ((Kx_row_sums - Kx_diag) ** 2).sum(dim=1) / (n * (n - 1) * (n - 2))
            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx

            valid = (Svar_x2 > 0) & (Svar_y2 > 0)
            SC2 = torch.zeros(C, device=Xt.device)
            SC2[valid] = Scov2[valid] / torch.sqrt(Svar_x2[valid] * Svar_y2 + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(SC2.abs(), min=0, max=1))

        return scores.cpu().numpy()

    @staticmethod
    def ipc_sis_accelerated(X: np.ndarray, y: np.ndarray, sigma: float = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        y_prep, _, _ = _prepare_y_torch(y, n, dtype=torch.float64)

        y_scaled = (y_prep - y_prep.mean(dim=0)) / (y_prep.std(dim=0) + 1e-10)
        if sigma is None:
            sigma_y = torch.sqrt(torch.median((y_scaled ** 2).sum(dim=1)))
        else:
            sigma_y = _to_tensor(sigma, dtype=torch.float64)

        y_sq = (y_scaled ** 2).sum(dim=1)
        y_norm = torch.sqrt(sigma_y ** 2 + y_sq)
        Gy = y_scaled / (y_norm.unsqueeze(1) * torch.sqrt(sigma_y ** 2))
        N = Gy @ Gy.T
        H = torch.eye(n, dtype=torch.float64, device=Xt.device) - 1.0 / n
        NH = N @ H
        tr_NH_NH = torch.trace(NH @ NH)

        scores = torch.zeros(p, dtype=torch.float64, device=Xt.device)

        for j in range(p):
            z = Xt[:, j].unsqueeze(1)              # (n, 1)
            z_c = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-10)

            if sigma is None:
                sigma_j = torch.sqrt(torch.median(z_c ** 2))  # scalar
            else:
                sigma_j = sigma_y

            z_sq = z_c ** 2
            z_norm = torch.sqrt(sigma_j ** 2 + z_sq)  # (n, 1)
            Gz = z_c / (z_norm * torch.sqrt(sigma_j ** 2))  # (n, 1)

            M = Gz @ Gz.T                          # (n, n)
            MH = M @ H
            tr_MH_NH = torch.trace(MH @ NH)
            tr_MH_MH = torch.trace(MH @ MH)
            if tr_MH_MH * tr_NH_NH > 1e-15:
                ipc = tr_MH_NH / torch.sqrt(tr_MH_MH * tr_NH_NH)
                scores[j] = torch.clamp(ipc, min=0, max=1)

        return scores.cpu().numpy()

    @staticmethod
    def pc_sis_accelerated(X: np.ndarray, y: np.ndarray, sigma: float = 1.0, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        y_scaled = (y_prep - y_prep.mean(dim=0)) / (y_prep.std(dim=0) + 1e-10)
        sigma_t = _to_tensor(sigma)

        y_sq = (y_scaled ** 2).sum(dim=1)
        dot_y = y_scaled @ y_scaled.T
        denom_y = torch.sqrt((sigma_t ** 2 + y_sq.unsqueeze(1)) * (sigma_t ** 2 + y_sq.unsqueeze(0)) + 1e-10)
        cos_y = torch.clamp((sigma_t ** 2 + dot_y) / denom_y, -1, 1)
        Ay = torch.acos(cos_y)

        Ay_mean = Ay.mean()
        Ay_row_mean = Ay.mean(dim=1)
        n_pairs = n * (n - 1)
        pcov_yy = (Ay ** 2).sum() / n_pairs - 2 * (Ay_row_mean ** 2).mean() + Ay_mean ** 2

        scores = torch.zeros(p, device=Xt.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            Z = Xt[:, start:end]
            Z_c = (Z - Z.mean(dim=0)) / (Z.std(dim=0) + 1e-10)

            z_sq = (Z_c ** 2).sum(dim=0)        # (C,)
            # dot_x per feature: (C, n, n)
            dot_x = Z_c.T.unsqueeze(2) * Z_c.T.unsqueeze(1)


            for j_off in range(C):
                z = Z_c[:, j_off].unsqueeze(1)   # (n, 1)
                dot_xj = z @ z.T                  # (n, n)
                zsq_j = z_sq[j_off]
                denom_x = torch.sqrt((sigma_t ** 2 + zsq_j + 0 * dot_xj) * (sigma_t ** 2 + zsq_j) + 1e-10)
                # Actually: denom_x[i,k] = sqrt((σ²+z[i]²)(σ²+z[k]²))
                denom_x2 = torch.sqrt((sigma_t ** 2 + (z ** 2).sum(dim=1)).unsqueeze(1) *
                                      (sigma_t ** 2 + (z ** 2).sum(dim=1)).unsqueeze(0) + 1e-10)
                cos_x = torch.clamp((sigma_t ** 2 + dot_xj) / denom_x2, -1, 1)
                Ax = torch.acos(cos_x)

                Ax_mean = Ax.mean()
                Ax_row_mean = Ax.mean(dim=1)

                term1 = (Ax * Ay).sum() / n_pairs
                term2 = 2 * (Ax_row_mean * Ay_row_mean).mean()
                term3 = Ax_mean * Ay_mean
                pcov_xy = term1 - term2 + term3

                pcov_xx = (Ax ** 2).sum() / n_pairs - 2 * (Ax_row_mean ** 2).mean() + Ax_mean ** 2

                if pcov_xx > 1e-10 and pcov_yy > 1e-10:
                    pc_sq = pcov_xy / torch.sqrt(pcov_xx * pcov_yy)
                    scores[start + j_off] = torch.clamp(pc_sq, min=0, max=1)

        return scores.cpu().numpy()

    @staticmethod
    def fkf_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=Xt.device) for r in classes]
        R = len(classes)

        scores = torch.zeros(p, device=Xt.device)

        for r_idx, mask_r in enumerate(masks):
            n_r = mask_r.sum().item()
            n_not_r = n - n_r
            if n_r == 0 or n_not_r == 0:
                continue
            x_r = Xt[mask_r, :]       # (n_r, p)
            x_not_r = Xt[~mask_r, :]  # (n_not_r, p)

            # KS statistic via sorted CDF difference
            x_r_sorted = torch.sort(x_r, dim=0).values      # (n_r, p)
            x_not_r_sorted = torch.sort(x_not_r, dim=0).values  # (n_not_r, p)


            combined = torch.cat([x_r_sorted, x_not_r_sorted], dim=0)  # (n, p)
            combined_sorted, combined_idx = torch.sort(combined, dim=0)


            is_r = mask_r.unsqueeze(1).expand(-1, p)


            ecdf_r = torch.cumsum(combined_idx < n_r, dim=0).float() / n_r
            ecdf_not_r = torch.cumsum(combined_idx >= n_r, dim=0).float() / n_not_r
            ks = (ecdf_r - ecdf_not_r).abs().max(dim=0).values
            scores = torch.max(scores, ks)

        return scores.cpu().numpy()

    @staticmethod
    def cru_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape

        Xt = Xt + torch.empty_like(Xt).uniform_(-0.001, 0.001)
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=Xt.device) for r in classes]
        pi_r = torch.tensor([mask.float().mean() for mask in masks], device=Xt.device)

        gammas = _mann_whitney_batched(Xt.T, masks)  # (p, R)
        weights = (pi_r * (1 - pi_r) ** 2).unsqueeze(0)  # (1, R)
        scores = (weights * (gammas - 0.5) ** 2).sum(dim=1)
        return scores.cpu().numpy()

    @staticmethod
    def cavs_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=Xt.device) for r in classes]
        pi_r = torch.tensor([mask.float().mean() for mask in masks], device=Xt.device)

        gammas = _mann_whitney_batched(Xt.T, masks)  # (p, R)
        weights = (1 - pi_r).unsqueeze(0)  # (1, R)
        scores = (weights * (gammas - 0.5).abs()).max(dim=1).values
        return scores.cpu().numpy()

    @staticmethod
    def mv_sis_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=Xt.device) for r in classes]
        pi_r = torch.tensor([mask.float().mean() for mask in masks], device=Xt.device)

        gammas = _mann_whitney_batched(Xt.T, masks)  # (p, R)
        weights = (pi_r * (1 - pi_r) ** 2).unsqueeze(0)
        scores = (weights * (gammas - 0.5) ** 2).sum(dim=1)
        return scores.cpu().numpy()

    @staticmethod
    def lr_ffs_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=Xt.device) for r in classes]

        gammas = _mann_whitney_batched(Xt.T, masks)  # (p, R)
        scores = (gammas - 0.5).abs().max(dim=1).values
        return scores.cpu().numpy()

    @staticmethod
    def psis_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        classes = np.unique(y)
        R = len(classes)
        masks_np = [(y == r) for r in classes]
        masks_t = [torch.as_tensor(m, dtype=torch.bool, device=Xt.device) for m in masks_np]


        medians = torch.zeros(R, p, device=Xt.device)
        for r_idx, mask in enumerate(masks_t):
            x_r = Xt[mask, :]  # (n_r, p)
            medians[r_idx] = x_r.median(dim=0).values


        max_diff = torch.zeros(p, device=Xt.device)
        for i in range(R):
            for k in range(i + 1, R):
                diff = (medians[i] - medians[k]).abs()
                max_diff = torch.max(max_diff, diff)

        return max_diff.cpu().numpy()

    @staticmethod
    def swc_accelerated(X: np.ndarray, y: np.ndarray, L: int = 20,
                        p_order: int = 1, n_shuf: int = 11,
                        qmc: bool = True, seed: int = 42, **kw) -> np.ndarray:
        from src.screening_methods import CorrelationMeasures, AcceleratedScreening as OrigAcceleratedScreening
        n, n_features = X.shape


        gaussian_quantiles = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        def rank_gauss(col):
            ranks = sp_stats.rankdata(col, method='average')
            return gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]

        y_prep_np = OrigAcceleratedScreening._prepare_y_for_screening(y, n) if hasattr(OrigAcceleratedScreening, '_prepare_y_for_screening') else y.reshape(-1, 1)
        if y_prep_np.ndim == 1:
            y_prep_np = y_prep_np.reshape(-1, 1)
        dy = y_prep_np.shape[1]

        gry_np = np.column_stack([rank_gauss(y_prep_np[:, k]) for k in range(dy)])
        X_t_np = np.column_stack([rank_gauss(X[:, j]) for j in range(n_features)])


        X_t = _to_tensor(X_t_np)   # (n, p)
        gry = _to_tensor(gry_np)   # (n, dy)


        total_dir = (dy + 1) * L
        dir_xy_np = CorrelationMeasures._uniform2sphere_mc(total_dir, dy + 1, seed) if not qmc else \
                    np.array([CorrelationMeasures._uniform2sphere_qmc(
                        CorrelationMeasures._uniform2sphere_mc(1, dy + 1, seed + i)[0])
                        for i in range(total_dir)])
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(d=dy, scramble=True, seed=seed)
            vec_meta = sobol.random(total_dir)
            dir_xy_np = np.array([CorrelationMeasures._uniform2sphere_qmc(v) for v in vec_meta])
        except Exception:
            dir_xy_np = CorrelationMeasures._uniform2sphere_mc(total_dir, dy + 1, seed)

        dir_xy = _to_tensor(dir_xy_np)  # (total_dir, dy+1)

        scores = torch.zeros(n_features, device=X_t.device)
        chunk = CHUNK_SIZE

        for start in range(0, n_features, chunk):
            end = min(start + chunk, n_features)
            C = end - start
            x_chunk = X_t[:, start:end]   # (n, C)

            scores_shuf = torch.zeros(n_shuf, C, device=X_t.device)

            for s in range(n_shuf):
                rng = np.random.RandomState(seed + s + 1)
                shuf_idx = rng.permutation(n)
                x_shuf = x_chunk[shuf_idx]   # (n, C)

                L_mat = torch.zeros(C, total_dir, device=X_t.device)

                for i in range(total_dir):
                    # joint/indep projections: (n, C)
                    proj_joint = x_chunk * dir_xy[i, 0] + (gry * dir_xy[i, 1:dy + 1]).sum(dim=1, keepdim=True)
                    proj_indep = x_shuf * dir_xy[i, 0] + (gry * dir_xy[i, 1:dy + 1]).sum(dim=1, keepdim=True)

                    p_joint = torch.sort(proj_joint, dim=0).values  # (n, C)
                    p_indep = torch.sort(proj_indep, dim=0).values
                    diff = (p_joint - p_indep).abs()
                    if p_order == 1:
                        L_mat[:, i] = diff.mean(dim=0)
                    else:
                        L_mat[:, i] = diff.pow(p_order).mean(dim=0)

                scores_shuf[s] = L_mat.mean(dim=1).pow(1.0 / p_order)

            scores[start:end] = scores_shuf.median(dim=0).values

        return scores.cpu().numpy()

    @staticmethod
    def hd_sis_accelerated(X: np.ndarray, y: np.ndarray, K: int = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        classes = np.unique(y)
        is_cat = len(classes) < n / 2 or not np.issubdtype(y.dtype, np.number)

        if is_cat:
            masks = [torch.as_tensor((y == r), dtype=torch.bool, device=Xt.device) for r in classes]
            R = len(masks)
            pi_r = torch.tensor([m.float().mean() for m in masks], device=Xt.device)


            scores = torch.zeros(p, device=Xt.device)
            chunk = CHUNK_SIZE

            for start in range(0, p, chunk):
                end = min(start + chunk, p)
                C = end - start
                X_chunk = Xt[:, start:end]  # (n, C)
                _, sort_idx = torch.sort(X_chunk, dim=0)


                rank_vals = torch.arange(1, n + 1, device=Xt.device).unsqueeze(1).float() / n  # (n, 1)
                sqrt_F = torch.sqrt(rank_vals).expand(-1, C)
                sqrt_1_F = torch.sqrt(1.0 - rank_vals).expand(-1, C)

                hd_chunk = torch.zeros(C, device=Xt.device)
                for r_idx, mask in enumerate(masks):
                    n_r = mask.sum().item()
                    if n_r == 0:
                        continue


                    mask_float = mask.float().unsqueeze(1).expand(-1, C)  # (n, C)
                    y_at_sorted = torch.gather(mask_float, 0, sort_idx)  # (n, C)
                    cum_r = torch.cumsum(y_at_sorted, dim=0) / n_r
                    sqrt_cdf_r = torch.sqrt(cum_r.clamp(min=0, max=1))
                    sqrt_1_cdf_r = torch.sqrt((1 - cum_r).clamp(min=0, max=1))
                    integrand = (sqrt_cdf_r - sqrt_F) ** 2 + (sqrt_1_cdf_r - sqrt_1_F) ** 2
                    hd_chunk += pi_r[r_idx] * integrand.mean(dim=0)

                scores[start:end] = hd_chunk

            return scores.cpu().numpy()
        else:

            if K is None:
                K = max(1, int(np.ceil(np.log(n))) - 2)

            yt = _to_tensor(y)
            scores = torch.zeros(p, device=Xt.device)

            for k in range(1, K + 1):
                R_k = k + 3
                quantiles = torch.quantile(yt, torch.linspace(0, 1, R_k + 1, device=Xt.device))
                groups = torch.bucketize(yt, quantiles[1:-1])  # values 0..R_k-1

                masks_k = [(groups == r) for r in range(R_k)]
                pi_k = torch.tensor([m.float().mean() for m in masks_k], device=Xt.device)

                chunk = CHUNK_SIZE
                hd_k = torch.zeros(p, device=Xt.device)

                for start in range(0, p, chunk):
                    end = min(start + chunk, p)
                    C = end - start
                    X_chunk = Xt[:, start:end]
                    _, sort_idx = torch.sort(X_chunk, dim=0)

                    rank_vals = torch.arange(1, n + 1, device=Xt.device).unsqueeze(1).float() / n
                    sqrt_F = torch.sqrt(rank_vals).expand(-1, C)
                    sqrt_1_F = torch.sqrt(1.0 - rank_vals).expand(-1, C)

                    hd_chunk = torch.zeros(C, device=Xt.device)
                    for r_idx in range(R_k):
                        n_r = masks_k[r_idx].sum().item()
                        if n_r == 0:
                            continue
                        mask_exp = masks_k[r_idx].float().unsqueeze(1).expand(-1, C)
                        y_at_sorted = torch.gather(mask_exp, 0, sort_idx)
                        cum_r = torch.cumsum(y_at_sorted, dim=0) / n_r
                        sqrt_cdf_r = torch.sqrt(cum_r.clamp(min=0, max=1))
                        sqrt_1_cdf_r = torch.sqrt((1 - cum_r).clamp(min=0, max=1))
                        integrand = (sqrt_cdf_r - sqrt_F) ** 2 + (sqrt_1_cdf_r - sqrt_1_F) ** 2
                        hd_chunk += pi_k[r_idx] * integrand.mean(dim=0)

                    hd_k[start:end] = hd_chunk

                scores += hd_k

            scores /= K
            return scores.cpu().numpy()

    @staticmethod
    def gcor_sis_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        n, p = Xt.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        Ky = _kappa_matrix(y_prep)
        Ky_c = _double_center_kappa(Ky)
        gcov2_yy = (Ky_c * Ky_c).sum() / (n * (n - 3))

        scores = torch.zeros(p, device=Xt.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            for j in range(start, end):
                x_j = Xt[:, j].unsqueeze(1)   # (n, 1)
                Kx = _kappa_matrix(x_j)
                Kx_c = _double_center_kappa(Kx)
                gcov2_xy = (Kx_c * Ky_c).sum() / (n * (n - 3))
                gcov2_xx = (Kx_c * Kx_c).sum() / (n * (n - 3))
                if gcov2_xx * gcov2_yy > 1e-15:
                    gcor2 = gcov2_xy / torch.sqrt(gcov2_xx * gcov2_yy)
                    scores[j] = torch.clamp(torch.sqrt(gcor2.abs()), min=0, max=1)

        return scores.cpu().numpy()

    @staticmethod
    def ids_accelerated(X: np.ndarray, y: np.ndarray, num_terms: int = 6,
                        p_norm: str = 'max', bandwidth_term: float = 0.5, **kw) -> np.ndarray:
        Xt = _to_tensor(X)                    # (n, p)
        n, p = Xt.shape
        y_prep, is_cat, classes = _prepare_y_torch(y, n)

        dy = y_prep.shape[1]


        Y_t = _ids_transform(y_prep, num_terms, bandwidth_term)   # (n, dy * num_terms)
        Y_t = Y_t - Y_t.mean(dim=0, keepdim=True)                # center
        Y_std = torch.sqrt((Y_t ** 2).sum(dim=0) + 1e-10)         # (dy * num_terms,)

        scores = torch.zeros(p, device=Xt.device, dtype=torch.float32)


        chunk = max(4, CHUNK_SIZE * 4 // num_terms)

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            X_chunk = Xt[:, start:end]                            # (n, C)


            X_t = _ids_transform(X_chunk, num_terms, bandwidth_term)  # (n, C * num_terms)
            X_t = X_t - X_t.mean(dim=0, keepdim=True)                # center
            X_std = torch.sqrt((X_t ** 2).sum(dim=0) + 1e-10)         # (C * num_terms,)


            cov = X_t.T @ Y_t                                        # (C*num_terms, dy*num_terms)
            correlations = cov / (X_std.unsqueeze(1) + 1e-10) / (Y_std.unsqueeze(0) + 1e-10)


            C_mat = correlations.reshape(num_terms, C, num_terms, dy)
            C_mat = torch.nan_to_num(C_mat, nan=0, posinf=0, neginf=0)
            C_mat = torch.abs(C_mat)

            if p_norm == 'max':
                C_red = torch.amax(C_mat, dim=(0, 2))               # (C, dy)
            elif p_norm == 2:
                C_red = (C_mat ** 2).mean(dim=0).mean(dim=1)        # (C, dy)
                C_red = torch.sqrt(C_red)
            elif p_norm == 1:
                C_red = C_mat.mean(dim=0).mean(dim=1)               # (C, dy)
            else:
                C_red = torch.amax(C_mat, dim=(0, 2))               # default max


            if is_cat and classes is not None:
                scores[start:end] = C_red.max(dim=1).values        # (C,) max over classes
            else:
                scores[start:end] = C_red.squeeze(1)               # (C,) dy=1

        return scores.cpu().numpy()


    @staticmethod
    def emb_dc_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        b = torch.cdist(y_prep, y_prep)
        b_c = _double_center(b)
        b_var = (b_c ** 2).sum()

        scores = torch.zeros(p, device=embeddings.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            emb_chunk = embeddings[:, start:end, :].permute(1, 0, 2)  # (C, n, d)
            a = torch.cdist(emb_chunk, emb_chunk)  # (C, n, n)
            a_c = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(1, 2), keepdim=True)
            num = (a_c * b_c.unsqueeze(0)).sum(dim=(1, 2))
            a_var = (a_c ** 2).sum(dim=(1, 2))
            dc_sq = num / torch.sqrt(a_var * b_var + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(dc_sq.abs(), min=0, max=1))

        return scores.cpu().numpy()

    @staticmethod
    def emb_sc_accelerated(embeddings, y: np.ndarray, a: float = 0.5, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        dist_y = torch.cdist(y_prep, y_prep)
        Ky = torch.exp(-dist_y.pow(a))
        E2_y = (Ky.sum() - Ky.diagonal().sum()) / (n * (n - 1))
        Kyy = torch.exp(-dist_y.pow(a) * 2)
        E1_yy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
        E2_yy = E2_y ** 2
        Ky_row_sums = Ky.sum(dim=1)
        E3_yy = ((Ky_row_sums - Ky.diagonal()) ** 2).sum() / (n * (n - 1) * (n - 2))
        Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

        Ky_diag = Ky.diagonal()

        scores = torch.zeros(p, device=embeddings.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            emb_chunk = embeddings[:, start:end, :].permute(1, 0, 2)
            dist_x = torch.cdist(emb_chunk, emb_chunk)
            Kx = torch.exp(-dist_x.pow(a))

            Kx_row_sums = Kx.sum(dim=2)                    # (C, n)
            Kx_diag = Kx.diagonal(dim1=1, dim2=2)          # (C, n)

            Kxy = Kx * Ky.unsqueeze(0)
            E1 = (Kxy.sum(dim=(1, 2)) - Kxy.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
            E2_x = (Kx_row_sums.sum(dim=1) - Kx_diag.sum(dim=1)) / (n * (n - 1))
            E2 = E2_x * E2_y
            E3 = ((Kx_row_sums - Kx_diag) * (Ky_row_sums - Ky_diag).unsqueeze(0)).sum(dim=1) / (n * (n - 1) * (n - 2))
            Scov2 = E1 + E2 - 2 * E3

            Kxx = torch.exp(-dist_x.pow(a) * 2)
            E1_xx = (Kxx.sum(dim=(1, 2)) - Kxx.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
            E2_xx = E2_x ** 2
            E3_xx = ((Kx_row_sums - Kx_diag) ** 2).sum(dim=1) / (n * (n - 1) * (n - 2))
            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx

            valid = (Svar_x2 > 0) & (Svar_y2 > 0)
            SC2 = torch.zeros(C, device=embeddings.device)
            SC2[valid] = Scov2[valid] / torch.sqrt(Svar_x2[valid] * Svar_y2 + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(SC2.abs(), min=0, max=1))

        return scores.cpu().numpy()

    @staticmethod
    def emb_ipc_accelerated(embeddings, y: np.ndarray, sigma: float = None, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        y_scaled = (y_prep - y_prep.mean(dim=0)) / (y_prep.std(dim=0) + 1e-10)
        if sigma is None:
            sigma_y = torch.sqrt(torch.median((y_scaled ** 2).sum(dim=1)))
        else:
            sigma_y = _to_tensor(sigma)

        y_sq = (y_scaled ** 2).sum(dim=1)
        y_norm = torch.sqrt(sigma_y ** 2 + y_sq)
        Gy = y_scaled / (y_norm.unsqueeze(1) * torch.sqrt(sigma_y ** 2))
        N = Gy @ Gy.T
        H = torch.eye(n, device=embeddings.device) - 1.0 / n
        NH = N @ H
        tr_NH_NH = torch.trace(NH @ NH)

        scores = torch.zeros(p, device=embeddings.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            for j in range(start, end):
                Z = embeddings[:, j, :]   # (n, d)
                Z_c = (Z - Z.mean(dim=0)) / (Z.std(dim=0) + 1e-10)
                if sigma is None:
                    sigma_j = torch.sqrt(torch.median((Z_c ** 2).sum(dim=1)))
                else:
                    sigma_j = sigma_y
                z_sq = (Z_c ** 2).sum(dim=1)
                z_norm = torch.sqrt(sigma_j ** 2 + z_sq)
                Gz = Z_c / (z_norm.unsqueeze(1) * torch.sqrt(sigma_j ** 2))
                M = Gz @ Gz.T
                MH = M @ H
                tr_MH_NH = torch.trace(MH @ NH)
                tr_MH_MH = torch.trace(MH @ MH)
                if tr_MH_MH * tr_NH_NH > 1e-15:
                    ipc = tr_MH_NH / torch.sqrt(tr_MH_MH * tr_NH_NH)
                    scores[j] = torch.clamp(ipc, min=0, max=1)

        return scores.cpu().numpy()

    @staticmethod
    def emb_pc_accelerated(embeddings, y: np.ndarray, sigma: float = 1.0, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        y_scaled = (y_prep - y_prep.mean(dim=0)) / (y_prep.std(dim=0) + 1e-10)
        sigma_t = _to_tensor(sigma)
        y_sq = (y_scaled ** 2).sum(dim=1)
        dot_y = y_scaled @ y_scaled.T
        denom_y = torch.sqrt((sigma_t ** 2 + y_sq.unsqueeze(1)) * (sigma_t ** 2 + y_sq.unsqueeze(0)) + 1e-10)
        cos_y = torch.clamp((sigma_t ** 2 + dot_y) / denom_y, -1, 1)
        Ay = torch.acos(cos_y)
        Ay_mean = Ay.mean()
        Ay_row_mean = Ay.mean(dim=1)
        n_pairs = n * (n - 1)
        pcov_yy = (Ay ** 2).sum() / n_pairs - 2 * (Ay_row_mean ** 2).mean() + Ay_mean ** 2

        scores = torch.zeros(p, device=embeddings.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            for j in range(start, end):
                Z = embeddings[:, j, :]
                Z_c = (Z - Z.mean(dim=0)) / (Z.std(dim=0) + 1e-10)
                z_sq = (Z_c ** 2).sum(dim=1)
                dot_z = Z_c @ Z_c.T
                denom_z = torch.sqrt((sigma_t ** 2 + z_sq.unsqueeze(1)) * (sigma_t ** 2 + z_sq.unsqueeze(0)) + 1e-10)
                cos_z = torch.clamp((sigma_t ** 2 + dot_z) / denom_z, -1, 1)
                Az = torch.acos(cos_z)
                Az_mean = Az.mean()
                Az_row_mean = Az.mean(dim=1)
                term1 = (Az * Ay).sum() / n_pairs
                term2 = 2 * (Az_row_mean * Ay_row_mean).mean()
                term3 = Az_mean * Ay_mean
                pcov_xy = term1 - term2 + term3
                pcov_xx = (Az ** 2).sum() / n_pairs - 2 * (Az_row_mean ** 2).mean() + Az_mean ** 2
                if pcov_xx > 1e-10 and pcov_yy > 1e-10:
                    pc_sq = pcov_xy / torch.sqrt(pcov_xx * pcov_yy)
                    scores[j] = torch.clamp(pc_sq, min=0, max=1)

        return scores.cpu().numpy()

    @staticmethod
    def emb_hsic_accelerated(embeddings, y: np.ndarray, sigma=None, *, task=None, **kw) -> np.ndarray:
        from src.screening_utils import pace_hsic_target

        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, _ = embeddings.shape
        target, categorical = pace_hsic_target(y, task)
        target = torch.as_tensor(target, device=embeddings.device, dtype=embeddings.dtype)

        def gaussian_kernel(values, width):
            distances = torch.cdist(values, values, compute_mode='donot_use_mm_for_euclid_dist')
            if width is None:
                widths = _median_bandwidth_gpu(distances)
                width = torch.where(widths > 0, widths, 1.0)[:, None, None]
            return torch.exp(-0.5 * (distances / width).square())

        response = gaussian_kernel(target.unsqueeze(0), 1.0 if categorical and sigma is None else sigma)[0]
        response = _double_center(response)
        scores = embeddings.new_zeros(p)
        for start in range(0, p, CHUNK_SIZE):
            block = embeddings[:, start:start + CHUNK_SIZE].transpose(0, 1)
            kernel = gaussian_kernel(block, sigma)
            centered = kernel - kernel.mean(1, keepdim=True) - kernel.mean(2, keepdim=True) + kernel.mean((1, 2), keepdim=True)
            scores[start:start + len(block)] = (centered * response).mean((1, 2)).clamp_min(0)
        return scores.detach().cpu().numpy()

    @staticmethod
    def emb_sis_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        y_prep, _, _ = _prepare_y_torch(y, n)


        y_c = y_prep - y_prep.mean(dim=0, keepdim=True)
        y_std = y_prep.std(dim=0, unbiased=False)

        scores = torch.zeros(p, device=embeddings.device)


        if y_prep.shape[1] > 1:
            for dim in range(d):
                z = embeddings[:, :, dim]  # (n, p)
                z_c = z - z.mean(dim=0, keepdim=True)
                z_std = z.std(dim=0, unbiased=False)
                # corr: (p, R)
                corr = (z_c.T @ y_c) / (n * z_std.unsqueeze(1) * y_std.unsqueeze(0) + 1e-10)
                corr = corr.abs()
                mask = (z_std.unsqueeze(1) < 1e-10) | (y_std.unsqueeze(0) < 1e-10)
                corr[mask] = 0
                scores += corr.max(dim=1).values
            scores /= d
        else:
            y_c1 = y_c.squeeze(1)
            y_s = y_std.item()
            for dim in range(d):
                z = embeddings[:, :, dim]
                z_c = z - z.mean(dim=0, keepdim=True)
                z_std = z.std(dim=0, unbiased=False)
                corr = (z_c.T @ y_c1).abs() / (n * z_std * y_s + 1e-10)
                corr[z_std < 1e-10] = 0
                scores += corr
            scores /= d

        return scores.cpu().numpy()

    @staticmethod
    def emb_fkf_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=embeddings.device) for r in classes]

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            Xd = embeddings[:, :, dim]  # (n, p)
            dim_max_ks = torch.zeros(p, device=embeddings.device)
            for mask in masks:
                n_r = mask.sum().item()
                n_not_r = n - n_r
                if n_r == 0 or n_not_r == 0:
                    continue
                x_r = Xd[mask, :]
                x_not_r = Xd[~mask, :]
                x_r_sorted = torch.sort(x_r, dim=0).values
                x_not_r_sorted = torch.sort(x_not_r, dim=0).values
                combined = torch.cat([x_r_sorted, x_not_r_sorted], dim=0)
                _, idx = torch.sort(combined, dim=0)
                ecdf_r = torch.cumsum(idx < n_r, dim=0).float() / n_r
                ecdf_not_r = torch.cumsum(idx >= n_r, dim=0).float() / n_not_r
                ks = (ecdf_r - ecdf_not_r).abs().max(dim=0).values
                dim_max_ks = torch.max(dim_max_ks, ks)
            scores += dim_max_ks  # sum over dims, then divide by d

        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_cru_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=embeddings.device) for r in classes]
        pi_r = torch.tensor([m.float().mean() for m in masks], device=embeddings.device)

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            Xd = embeddings[:, :, dim]  # (n, p)

            Xd = Xd + torch.empty_like(Xd).uniform_(-0.001, 0.001)
            gammas = _mann_whitney_batched(Xd.T, masks)  # (p, R)
            weights = (pi_r * (1 - pi_r) ** 2).unsqueeze(0)
            scores += (weights * (gammas - 0.5) ** 2).sum(dim=1)
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_cavs_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=embeddings.device) for r in classes]
        pi_r = torch.tensor([m.float().mean() for m in masks], device=embeddings.device)

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            Xd = embeddings[:, :, dim]
            gammas = _mann_whitney_batched(Xd.T, masks)
            weights = (1 - pi_r).unsqueeze(0)
            scores += (weights * (gammas - 0.5).abs()).max(dim=1).values
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_mv_sis_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=embeddings.device) for r in classes]
        pi_r = torch.tensor([m.float().mean() for m in masks], device=embeddings.device)

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            Xd = embeddings[:, :, dim]
            gammas = _mann_whitney_batched(Xd.T, masks)
            weights = (pi_r * (1 - pi_r) ** 2).unsqueeze(0)
            scores += (weights * (gammas - 0.5) ** 2).sum(dim=1)
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_lr_ffs_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        classes = np.unique(y)
        masks = [torch.as_tensor((y == r), dtype=torch.bool, device=embeddings.device) for r in classes]

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            Xd = embeddings[:, :, dim]
            gammas = _mann_whitney_batched(Xd.T, masks)
            scores += (gammas - 0.5).abs().max(dim=1).values
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_psis_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        classes = np.unique(y)
        R = len(classes)
        masks_np = [(y == r) for r in classes]

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            Xt = embeddings[:, :, dim]  # (n, p)
            medians = torch.zeros(R, p, device=embeddings.device)
            for r_idx, cls in enumerate(classes):
                mask = torch.as_tensor(masks_np[r_idx], dtype=torch.bool, device=embeddings.device)
                medians[r_idx] = Xt[mask, :].median(dim=0).values

            max_diff = torch.zeros(p, device=embeddings.device)
            for i in range(R):
                for k in range(i + 1, R):
                    diff = (medians[i] - medians[k]).abs()
                    max_diff = torch.max(max_diff, diff)
            scores += max_diff
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_swc_accelerated(embeddings, y: np.ndarray, L: int = 20, p_order: int = 1,
                            n_shuf: int = 11, qmc: bool = True, seed: int = 42, **kw) -> np.ndarray:

        if isinstance(embeddings, np.ndarray):
            embeddings_np = embeddings
        else:
            embeddings_np = embeddings.cpu().numpy()

        n, p, d = embeddings_np.shape
        from src.screening_methods import AcceleratedScreening as OrigAS


        scores = np.zeros(p)
        for j in range(p):

            x_2d = embeddings_np[:, j, :]  # (n, d)


            from src.screening_methods import CorrelationMeasures
            y_prep_np = OrigAS._prepare_y_for_screening(y, n)
            if y_prep_np.ndim == 1:
                y_prep_np = y_prep_np.reshape(-1, 1)
            dy = y_prep_np.shape[1]
            scores[j] = CorrelationMeasures.sliced_wasserstein_correlation(
                x_2d, y_prep_np, L=L, p=p_order, qmc=qmc, seed=seed)
        return scores

    @staticmethod
    def emb_hd_accelerated(embeddings, y: np.ndarray, K: int = None, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):

            X_dim = embeddings[:, :, dim].cpu().numpy()
            scores_dim = TorchAcceleratedScreening.hd_sis_accelerated(X_dim, y, K=K)
            scores += _to_tensor(scores_dim, embeddings.device)
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_ids_accelerated(embeddings, y: np.ndarray, num_terms: int = 6,
                            p_norm: str = 'max', bandwidth_term: float = 0.5, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape

        scores = torch.zeros(p, device=embeddings.device)
        for dim in range(d):
            X_dim = embeddings[:, :, dim].cpu().numpy()
            scores_dim = TorchAcceleratedScreening.ids_accelerated(
                X_dim, y, num_terms=num_terms, p_norm=p_norm, bandwidth_term=bandwidth_term)
            scores += _to_tensor(scores_dim, embeddings.device)
        scores /= d
        return scores.cpu().numpy()

    @staticmethod
    def emb_gcor_accelerated(embeddings, y: np.ndarray, **kw) -> np.ndarray:
        if isinstance(embeddings, np.ndarray):
            embeddings = _to_tensor(embeddings)
        n, p, d = embeddings.shape
        y_prep, _, _ = _prepare_y_torch(y, n)

        Ky = _kappa_matrix(y_prep)
        Ky_c = _double_center_kappa(Ky)
        gcov2_yy = (Ky_c * Ky_c).sum() / (n * (n - 3))

        scores = torch.zeros(p, device=embeddings.device)
        chunk = CHUNK_SIZE

        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            emb_chunk = embeddings[:, start:end, :].permute(1, 0, 2)  # (C, n, d)
            for j_off in range(C):
                Z = emb_chunk[j_off]   # (n, d)
                Kx = _kappa_matrix(Z)
                Kx_c = _double_center_kappa(Kx)
                gcov2_xy = (Kx_c * Ky_c).sum() / (n * (n - 3))
                gcov2_xx = (Kx_c * Kx_c).sum() / (n * (n - 3))
                if gcov2_xx * gcov2_yy > 1e-15:
                    gcor2 = gcov2_xy / torch.sqrt(gcov2_xx * gcov2_yy)
                    scores[start + j_off] = torch.clamp(torch.sqrt(gcor2.abs()), min=0, max=1)

        return scores.cpu().numpy()


    @staticmethod
    def compute_all_scores(X: np.ndarray, y: np.ndarray,
                           methods: List[str] = None,
                           n_jobs: int = -1) -> Dict[str, np.ndarray]:
        if methods is None:
            methods = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                       'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'ids']


        n, p = X.shape
        if n > 500:
            methods = [m for m in methods if m not in ('pc',)]

        results = {}
        X_t = _to_tensor(X)

        dispatch = {
            'sis': lambda: TorchAcceleratedScreening.sis_accelerated(X, y),
            'dc': lambda: TorchAcceleratedScreening.dc_sis_accelerated(X, y),
            'sc': lambda: TorchAcceleratedScreening.sc_sis_accelerated(X, y),
            'ipc': lambda: TorchAcceleratedScreening.ipc_sis_accelerated(X, y),
            'pc': lambda: TorchAcceleratedScreening.pc_sis_accelerated(X, y),
            'fkf': lambda: TorchAcceleratedScreening.fkf_accelerated(X, y),
            'cru': lambda: TorchAcceleratedScreening.cru_accelerated(X, y),
            'cavs': lambda: TorchAcceleratedScreening.cavs_accelerated(X, y),
            'mv_sis': lambda: TorchAcceleratedScreening.mv_sis_accelerated(X, y),
            'lr_ffs': lambda: TorchAcceleratedScreening.lr_ffs_accelerated(X, y),
            'psis': lambda: TorchAcceleratedScreening.psis_accelerated(X, y),
            'hd': lambda: TorchAcceleratedScreening.hd_sis_accelerated(X, y),
            'gcor': lambda: TorchAcceleratedScreening.gcor_sis_accelerated(X, y),
            'ids': lambda: TorchAcceleratedScreening.ids_accelerated(X, y),
        }

        for method in methods:
            if method in dispatch:
                try:
                    results[method] = dispatch[method]()
                except Exception as e:
                    print(f"  [Torch] {method} failed: {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    @staticmethod
    def emb_compute_all_scores(embeddings, y: np.ndarray,
                                methods: List[str] = None,
                                n_jobs: int = -1) -> Dict[str, np.ndarray]:
        if methods is None:
            methods = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hsic', 'fkf', 'cru',
                       'cavs', 'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'ids']

        n = y.shape[0] if isinstance(y, np.ndarray) else y.shape[0]
        if n > 500:
            methods = [m for m in methods if m not in ('pc',)]

        results = {}

        dispatch = {
            'sis': lambda: TorchAcceleratedScreening.emb_sis_accelerated(embeddings, y),
            'dc': lambda: TorchAcceleratedScreening.emb_dc_accelerated(embeddings, y),
            'sc': lambda: TorchAcceleratedScreening.emb_sc_accelerated(embeddings, y),
            'ipc': lambda: TorchAcceleratedScreening.emb_ipc_accelerated(embeddings, y),
            'pc': lambda: TorchAcceleratedScreening.emb_pc_accelerated(embeddings, y),
            'hsic': lambda: TorchAcceleratedScreening.emb_hsic_accelerated(embeddings, y),
            'fkf': lambda: TorchAcceleratedScreening.emb_fkf_accelerated(embeddings, y),
            'cru': lambda: TorchAcceleratedScreening.emb_cru_accelerated(embeddings, y),
            'cavs': lambda: TorchAcceleratedScreening.emb_cavs_accelerated(embeddings, y),
            'mv_sis': lambda: TorchAcceleratedScreening.emb_mv_sis_accelerated(embeddings, y),
            'lr_ffs': lambda: TorchAcceleratedScreening.emb_lr_ffs_accelerated(embeddings, y),
            'psis': lambda: TorchAcceleratedScreening.emb_psis_accelerated(embeddings, y),
            'hd': lambda: TorchAcceleratedScreening.emb_hd_accelerated(embeddings, y),
            'gcor': lambda: TorchAcceleratedScreening.emb_gcor_accelerated(embeddings, y),
            'ids': lambda: TorchAcceleratedScreening.emb_ids_accelerated(embeddings, y),
        }

        for method in methods:
            if method in dispatch:
                try:
                    results[f'emb_{method}'] = dispatch[method]()
                except Exception as e:
                    print(f"  [Torch] emb_{method} failed: {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    @staticmethod
    def compute_all_scores_with_embeddings(X: np.ndarray, y: np.ndarray,
                                            embedder: 'TorchAcceleratedEmbedder',
                                            emb_layers: List[int] = [1, 2, 3],
                                            methods: List[str] = None,
                                            n_jobs: int = -1) -> Dict[str, np.ndarray]:
        if not embedder.loaded:
            return {}

        if methods is None:
            methods = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hsic', 'fkf', 'cru',
                       'cavs', 'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'ids']

        n = y.shape[0] if isinstance(y, np.ndarray) else y.shape[0]
        p = X.shape[1]


        if n > 500:
            methods = [m for m in methods if m not in ('pc',)]


        embed_dim = embedder.embed_dim
        available = _get_available_gpu_mem() * 0.5
        feat_chunk_size = max(1, int(available / (n * embed_dim * 4)))
        feat_chunk_size = min(feat_chunk_size, p, EMB_FEAT_CHUNK)
        print(f"  [Torch-Incremental] p={p}, feat_chunk_size={feat_chunk_size}, "
              f"layers={emb_layers}, methods={methods}", flush=True)


        results = {}
        for layer in emb_layers:
            for m in methods:
                results[f'emb_{m}_L{layer}'] = np.zeros(p, dtype=np.float64)

        for layer in emb_layers:
            print(f"  [Torch-Incremental] Layer {layer}: processing {p} features in chunks of {feat_chunk_size}",
                  flush=True)
            layer_scores = {m: np.zeros(p, dtype=np.float64) for m in methods}

            for feat_start in range(0, p, feat_chunk_size):
                feat_end = min(feat_start + feat_chunk_size, p)
                chunk_p = feat_end - feat_start


                chunk_emb = embedder.get_embeddings_for_range(
                    X, layer_idx=layer,
                    feat_start=feat_start, feat_end=feat_end,
                    batch_size=32
                )


                chunk_scores = TorchAcceleratedScreening.emb_compute_all_scores(
                    chunk_emb, y, methods=methods, n_jobs=n_jobs
                )


                for m in methods:
                    key = f'emb_{m}'
                    if key in chunk_scores and chunk_scores[key] is not None:
                        layer_scores[m][feat_start:feat_end] = chunk_scores[key]


                del chunk_emb, chunk_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            for m in methods:
                results[f'emb_{m}_L{layer}'] = layer_scores[m]

            del layer_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results
# =============================================================================

class TorchAcceleratedGroupScreening:

    # =========================================================================

    # =========================================================================

    @staticmethod
    def group_dc_accelerated(X: np.ndarray, y: np.ndarray,
                              groups: List[Tuple], **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        yt, _, _ = _prepare_y_torch(y, Xt.shape[0])
        n_groups = len(groups)

        b = torch.cdist(yt, yt)
        b_c = _double_center(b)
        b_var = (b_c ** 2).sum()

        scores = np.zeros(n_groups)
        chunk = CHUNK_SIZE

        for start in range(0, n_groups, chunk):
            end = min(start + chunk, n_groups)
            batch_groups = groups[start:end]
            C = len(batch_groups)


            k_max = max(len(g) for g in batch_groups)
            X_batch = torch.zeros(C, Xt.shape[0], k_max, device=Xt.device)
            for i, g in enumerate(batch_groups):
                idx = list(g)
                X_batch[i, :, :len(idx)] = Xt[:, idx]

            a = torch.cdist(X_batch, X_batch)  # (C, n, n)
            a_c = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(1, 2), keepdim=True)
            num = (a_c * b_c.unsqueeze(0)).sum(dim=(1, 2))
            a_var = (a_c ** 2).sum(dim=(1, 2))
            dc_sq = num / torch.sqrt(a_var * b_var + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(dc_sq.abs(), min=0, max=1)).cpu().numpy()

        return scores

    @staticmethod
    def group_sc_accelerated(X: np.ndarray, y: np.ndarray,
                              groups: List[Tuple], a: float = 0.5, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        yt, _, _ = _prepare_y_torch(y, Xt.shape[0])
        n = Xt.shape[0]
        n_groups = len(groups)

        dist_y = torch.cdist(yt, yt)
        Ky = torch.exp(-dist_y.pow(a))
        E2_y = (Ky.sum() - Ky.diagonal().sum()) / (n * (n - 1))
        Kyy = torch.exp(-dist_y.pow(a) * 2)
        E1_yy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
        E2_yy = E2_y ** 2
        Ky_row_sums = Ky.sum(dim=1)
        Ky_diag = Ky.diagonal()
        E3_yy = ((Ky_row_sums - Ky_diag) ** 2).sum() / (n * (n - 1) * (n - 2))
        Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

        scores = np.zeros(n_groups)
        chunk = CHUNK_SIZE

        for start in range(0, n_groups, chunk):
            end = min(start + chunk, n_groups)
            batch_groups = groups[start:end]
            C = len(batch_groups)

            k_max = max(len(g) for g in batch_groups)
            X_batch = torch.zeros(C, n, k_max, device=Xt.device)
            for i, g in enumerate(batch_groups):
                idx = list(g)
                X_batch[i, :, :len(idx)] = Xt[:, idx]

            dist_x = torch.cdist(X_batch, X_batch)
            Kx = torch.exp(-dist_x.pow(a))

            Kx_row_sums = Kx.sum(dim=2)
            Kx_diag = Kx.diagonal(dim1=1, dim2=2)

            Kxy = Kx * Ky.unsqueeze(0)
            E1 = (Kxy.sum(dim=(1, 2)) - Kxy.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
            E2_x = (Kx_row_sums.sum(dim=1) - Kx_diag.sum(dim=1)) / (n * (n - 1))
            E2 = E2_x * E2_y
            E3 = ((Kx_row_sums - Kx_diag) * (Ky_row_sums - Ky_diag).unsqueeze(0)).sum(dim=1) / (n * (n - 1) * (n - 2))
            Scov2 = E1 + E2 - 2 * E3

            Kxx = torch.exp(-dist_x.pow(a) * 2)
            E1_xx = (Kxx.sum(dim=(1, 2)) - Kxx.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
            E2_xx = E2_x ** 2
            E3_xx = ((Kx_row_sums - Kx_diag) ** 2).sum(dim=1) / (n * (n - 1) * (n - 2))
            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx

            valid = (Svar_x2 > 0) & (Svar_y2 > 0)
            SC2 = torch.zeros(C, device=Xt.device)
            SC2[valid] = Scov2[valid] / torch.sqrt(Svar_x2[valid] * Svar_y2 + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(SC2.abs(), min=0, max=1)).cpu().numpy()

        return scores

    @staticmethod
    def group_ipc_accelerated(X: np.ndarray, y: np.ndarray,
                               groups: List[Tuple], sigma: float = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        yt, _, _ = _prepare_y_torch(y, Xt.shape[0])
        n = Xt.shape[0]
        n_groups = len(groups)

        y_scaled = (yt - yt.mean(dim=0)) / (yt.std(dim=0) + 1e-10)
        if sigma is None:
            sigma_y = torch.sqrt(torch.median((y_scaled ** 2).sum(dim=1)))
        else:
            sigma_y = _to_tensor(sigma)

        y_sq = (y_scaled ** 2).sum(dim=1)
        y_norm = torch.sqrt(sigma_y ** 2 + y_sq)
        Gy = y_scaled / (y_norm.unsqueeze(1) * torch.sqrt(sigma_y ** 2))
        N = Gy @ Gy.T
        H = torch.eye(n, device=Xt.device) - 1.0 / n
        NH = N @ H
        tr_NH_NH = torch.trace(NH @ NH)

        scores = np.zeros(n_groups)
        chunk = min(CHUNK_SIZE, 64)

        for start in range(0, n_groups, chunk):
            end = min(start + chunk, n_groups)
            batch_groups = groups[start:end]
            C = len(batch_groups)

            for i, g in enumerate(batch_groups):
                idx = list(g)
                X_group = Xt[:, idx]  # (n, k)


                Z = (X_group - X_group.mean(dim=0)) / (X_group.std(dim=0) + 1e-10)

                if sigma is None:
                    sigma_g = torch.sqrt(torch.median((Z ** 2).sum(dim=1)).clamp(min=1e-10))
                else:
                    sigma_g = _to_tensor(sigma)

                z_sq = (Z ** 2).sum(dim=1)
                z_norm = torch.sqrt(sigma_g ** 2 + z_sq)
                Gz = Z / (z_norm.unsqueeze(1) * torch.sqrt(sigma_g ** 2))

                M = Gz @ Gz.T
                MH = M @ H

                ipc = torch.trace(MH @ NH) / torch.sqrt(
                    torch.trace(MH @ MH) * tr_NH_NH + 1e-30)
                scores[start + i] = max(0, min(1, ipc.item()))

        return scores

    @staticmethod
    def group_pc_accelerated(X: np.ndarray, y: np.ndarray,
                              groups: List[Tuple], sigma: float = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X)
        yt, _, _ = _prepare_y_torch(y, Xt.shape[0])
        n = Xt.shape[0]
        n_groups = len(groups)


        y_scaled = (yt - yt.mean(dim=0)) / (yt.std(dim=0) + 1e-10)

        if sigma is None:
            sigma_y = torch.sqrt(torch.median((y_scaled ** 2).sum(dim=1)))
        else:
            sigma_y = _to_tensor(sigma)


        def _compute_A(z, sig):
            zz = z @ z.T
            z_sq = (z ** 2).sum(dim=1)
            num = sig ** 2 + zz
            denom = torch.sqrt((sig ** 2 + z_sq.unsqueeze(1)) *
                               (sig ** 2 + z_sq.unsqueeze(0)) + 1e-30)
            cos_val = torch.clamp(num / (denom + 1e-10), -1, 1)
            return torch.arccos(cos_val)

        Ay = _compute_A(y_scaled, sigma_y)
        n_pairs = n * (n - 1)
        E2_y = 2 * (Ay.mean(dim=1) ** 2).mean()
        E3_y = Ay.mean() ** 2
        pcov_yy = (Ay ** 2).sum() / n_pairs - E2_y + E3_y

        scores = np.zeros(n_groups)

        for gi in range(n_groups):
            group = groups[gi]
            idx = list(group)
            X_group = Xt[:, idx]

            Z = (X_group - X_group.mean(dim=0)) / (X_group.std(dim=0) + 1e-10)

            if sigma is None:
                sigma_g = torch.sqrt(torch.median((Z ** 2).sum(dim=1)).clamp(min=1e-10))
            else:
                sigma_g = _to_tensor(sigma)

            Az = _compute_A(Z, sigma_g)

            term1 = (Az * Ay).sum() / n_pairs
            term2 = 2 * (Az.mean(dim=1) * Ay.mean(dim=1)).mean()
            term3 = Az.mean() * Ay.mean()

            pcov_xy = term1 - term2 + term3
            pcov_xx = (Az ** 2).sum() / n_pairs - 2 * (Az.mean(dim=1) ** 2).mean() + Az.mean() ** 2

            if pcov_xx < 1e-10 or pcov_yy < 1e-10:
                scores[gi] = 0.0
            else:
                pc_sq = pcov_xy / torch.sqrt(pcov_xx * pcov_yy + 1e-30)
                scores[gi] = max(0, min(1, pc_sq.item()))

        return scores


    @staticmethod
    def group_sis_accelerated(X: np.ndarray, y: np.ndarray,
                              groups: List[Tuple], **kw) -> np.ndarray:

        all_scores = TorchAcceleratedScreening.sis_accelerated(X, y)  # (p,)


        n_groups = len(groups)
        scores = np.zeros(n_groups)
        for g in range(n_groups):
            idx = list(groups[g])
            scores[g] = all_scores[idx].sum()

        return scores

    @staticmethod
    def group_hd_accelerated(X: np.ndarray, y: np.ndarray,
                              groups: List[Tuple], **kw) -> np.ndarray:
        all_scores = TorchAcceleratedScreening.hd_sis_accelerated(X, y)  # (p,)

        n_groups = len(groups)
        scores = np.zeros(n_groups)
        for g in range(n_groups):
            idx = list(groups[g])
            scores[g] = all_scores[idx].sum()

        return scores

    @staticmethod
    def group_gcor_accelerated(X: np.ndarray, y: np.ndarray,
                                groups: List[Tuple], **kw) -> np.ndarray:
        all_scores = TorchAcceleratedScreening.gcor_sis_accelerated(X, y)  # (p,)

        n_groups = len(groups)
        scores = np.zeros(n_groups)
        for g in range(n_groups):
            idx = list(groups[g])
            scores[g] = all_scores[idx].sum()

        return scores

    @staticmethod
    def group_ids_accelerated(X: np.ndarray, y: np.ndarray,
                               groups: List[Tuple], **kw) -> np.ndarray:
        all_scores = TorchAcceleratedScreening.ids_accelerated(X, y)  # (p,)

        n_groups = len(groups)
        scores = np.zeros(n_groups)
        for g in range(n_groups):
            idx = list(groups[g])
            scores[g] = all_scores[idx].sum()

        return scores

    @staticmethod
    def group_compute_all_scores(X: np.ndarray, y: np.ndarray,
                                  groups: List[Tuple],
                                  metrics: List[str] = None,
                                  **kw) -> Dict[str, np.ndarray]:
        if metrics is None:
            metrics = ['dc', 'sc', 'ipc', 'pc']

        n = X.shape[0]
        if n > 500:
            metrics = [m for m in metrics if m != 'pc']

        results = {}
        dispatch = {
            'dc': lambda: TorchAcceleratedGroupScreening.group_dc_accelerated(X, y, groups),
            'sc': lambda: TorchAcceleratedGroupScreening.group_sc_accelerated(X, y, groups),
            'ipc': lambda: TorchAcceleratedGroupScreening.group_ipc_accelerated(X, y, groups),
            'pc': lambda: TorchAcceleratedGroupScreening.group_pc_accelerated(X, y, groups),
            'sis': lambda: TorchAcceleratedGroupScreening.group_sis_accelerated(X, y, groups),
            'hd': lambda: TorchAcceleratedGroupScreening.group_hd_accelerated(X, y, groups),
            'gcor': lambda: TorchAcceleratedGroupScreening.group_gcor_accelerated(X, y, groups),
            'ids': lambda: TorchAcceleratedGroupScreening.group_ids_accelerated(X, y, groups),
        }

        for method in metrics:
            if method in dispatch:
                try:
                    results[method] = dispatch[method]()
                except Exception as e:
                    print(f"  [Torch-Group] {method} failed: {e}", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    # =========================================================================

    # =========================================================================

    @staticmethod
    def group_cls_compute_all_scores(embedder: 'TorchAcceleratedEmbedder',
                                      X: np.ndarray, y: np.ndarray,
                                      groups: List[Tuple],
                                      metrics: List[str] = None,
                                      layer_idx: int = 2,
                                      batch_size: int = 32) -> Dict[str, np.ndarray]:
        if not embedder.loaded:
            print("  [Torch-Group-CLS] Embedder not loaded!", flush=True)
            return {}

        if metrics is None:
            metrics = ['dc', 'sc', 'ipc', 'pc']

        n = X.shape[0]
        if n > 500:
            metrics = [m for m in metrics if m != 'pc']


        col_embeddings = embedder.get_all_embeddings(X, layer_idx=layer_idx,
                                                      batch_size=batch_size)
        # col_embeddings: Tensor(n, p, embed_dim) on GPU
        n_samples, n_features, embed_dim = col_embeddings.shape


        if embedder._row_interactor is None:
            print("  [Torch-Group-CLS] row_interactor not found in embedder, "
                  "cannot compute CLS scores", flush=True)
            del col_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {}

        group_emb = embedder.get_group_cls_embeddings(
            col_embeddings, groups, group_batch_size=64)


        del col_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        results = {}

        if 'dc' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_dc(
                group_emb, y, yt=None)
            results['cls_dc'] = scores

        if 'sc' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_sc(
                group_emb, y, yt=None, a_sc=0.5)
            results['cls_sc'] = scores

        if 'ipc' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_ipc(
                group_emb, y, yt=None)
            results['cls_ipc'] = scores

        if 'pc' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_pc(
                group_emb, y, yt=None)
            results['cls_pc'] = scores

        if 'sis' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_sis(
                group_emb, y)
            results['cls_sis'] = scores

        if 'hd' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_hd(
                group_emb, y)
            results['cls_hd'] = scores

        if 'gcor' in metrics:
            scores = TorchAcceleratedGroupScreening._emb_group_gcor(
                group_emb, y)
            results['cls_gcor'] = scores


        del group_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


    @staticmethod
    def _emb_group_dc(group_emb, y, yt=None):
        return TorchAcceleratedScreening.emb_dc_accelerated(group_emb, y)

    @staticmethod
    def _emb_group_sc(group_emb, y, yt=None, a_sc=0.5):
        return TorchAcceleratedScreening.emb_sc_accelerated(group_emb, y, a=a_sc)

    @staticmethod
    def _emb_group_ipc(group_emb, y, yt=None):
        return TorchAcceleratedScreening.emb_ipc_accelerated(group_emb, y)

    @staticmethod
    def _emb_group_pc(group_emb, y, yt=None):
        return TorchAcceleratedScreening.emb_pc_accelerated(group_emb, y)

    @staticmethod
    def _emb_group_sis(group_emb, y, **kw):
        return TorchAcceleratedScreening.emb_sis_accelerated(group_emb, y)

    @staticmethod
    def _emb_group_hd(group_emb, y, **kw):
        return TorchAcceleratedScreening.emb_hd_accelerated(group_emb, y)

    @staticmethod
    def _emb_group_gcor(group_emb, y, **kw):
        return TorchAcceleratedScreening.emb_gcor_accelerated(group_emb, y)


# =============================================================================

# =============================================================================

class TorchAcceleratedEmbedder:

    def __init__(self, col_embedder=None, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.col_embedder = col_embedder
        self.embed_dim = 128
        self.n_cls_tokens = 4
        self.n_blocks = 3
        self.loaded = False
        self._row_interactor = None

        if col_embedder is not None:
            try:
                self.embed_dim = col_embedder.in_linear.out_features
                self.n_blocks = len(col_embedder.tf_col.blocks)
                self.loaded = True
            except Exception:
                pass

    def load_model(self, model_path: str, seed: int = 42, device: str = None) -> bool:
        if self.loaded:
            return True
        if device is not None:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        try:
            from tabicl import TabICLClassifier
            model = TabICLClassifier(
                model_path=model_path,
                n_estimators=1,
                random_state=seed,
                device=str(self.device)
            )

            model.fit([[0, 1], [1, 0]], [0, 1])
            self.col_embedder = model.model_.col_embedder
            self.embed_dim = self.col_embedder.embed_dim
            self.n_cls_tokens = getattr(self.col_embedder, 'reserve_cls_tokens', 4)
            self.n_blocks = len(self.col_embedder.tf_col.blocks)

            self._row_interactor = getattr(model.model_, 'row_interactor', None)
            self.loaded = True
            ri_status = "with row_interactor" if self._row_interactor is not None else "without row_interactor"
            print(f"TabICL loaded (Torch): embed_dim={self.embed_dim}, blocks={self.n_blocks}, {ri_status}", flush=True)
            return True
        except Exception as e:
            print(f"Failed to load TabICL (Torch): {e}", flush=True)
            return False

    def _preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        gaussian_quantiles = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        result = np.zeros((n, p), dtype=np.float32)
        for j in range(p):
            ranks = sp_stats.rankdata(X[:, j], method='average')
            result[:, j] = gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]
        return result

    def get_embeddings_batch_all_layers(self, X, layers: List[int] = [1, 2, 3],
                                         batch_size: int = 32) -> Dict[int, torch.Tensor]:
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)

        n, p = X_np.shape
        X_processed = self._preprocess_batch(X_np)
        layer_set = set(layers)
        max_layer = max(layers)
        results = {l: torch.zeros(n, p, self.embed_dim, device=self.device) for l in layers}

        n_cls = self.n_cls_tokens
        self.col_embedder.eval()
        with torch.no_grad():
            for start in range(0, p, batch_size):
                end = min(start + batch_size, p)
                batch_p = end - start


                x_input = np.zeros((batch_p, 1 + n_cls, n, 1), dtype=np.float32)
                x_input[:, n_cls, :, 0] = X_processed[:, start:end].T
                x_tensor = torch.as_tensor(x_input, device=self.device)

                out = self.col_embedder.in_linear(x_tensor)
                for block_idx, block in enumerate(self.col_embedder.tf_col.blocks):
                    out = block(out)
                    layer_num = block_idx + 1
                    if layer_num in layer_set:
                        # out[:, n_cls:, :, :] → (batch_p, 1, n, embed_dim)

                        emb = out[:, n_cls:, :, :].squeeze(1).permute(1, 0, 2)
                        results[layer_num][:, start:end, :] = emb.detach()
                    if layer_num >= max_layer:
                        break

        return results

    def get_all_embeddings(self, X, layer_idx: int = 2, batch_size: int = 32) -> torch.Tensor:
        result = self.get_embeddings_batch_all_layers(X, layers=[layer_idx], batch_size=batch_size)
        return result[layer_idx]

    def get_embeddings_for_range(self, X, layer_idx: int = 2,
                                  feat_start: int = 0, feat_end: int = None,
                                  batch_size: int = 32) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)

        n, p = X_np.shape
        if feat_end is None:
            feat_end = p
        chunk_p = feat_end - feat_start


        X_slice = X_np[:, feat_start:feat_end]  # (n, chunk_p)
        X_processed = self._preprocess_batch(X_slice)

        results = torch.zeros(n, chunk_p, self.embed_dim, device=self.device)

        n_cls = self.n_cls_tokens
        self.col_embedder.eval()
        with torch.no_grad():
            for start in range(0, chunk_p, batch_size):
                end = min(start + batch_size, chunk_p)
                bp = end - start

                x_input = np.zeros((bp, 1 + n_cls, n, 1), dtype=np.float32)
                x_input[:, n_cls, :, 0] = X_processed[:, start:end].T
                x_tensor = torch.as_tensor(x_input, device=self.device)

                out = self.col_embedder.in_linear(x_tensor)
                for block_idx, block in enumerate(self.col_embedder.tf_col.blocks):
                    out = block(out)
                    layer_num = block_idx + 1
                    if layer_num == layer_idx:
                        emb = out[:, n_cls:, :, :].squeeze(1).permute(1, 0, 2)
                        results[:, start:end, :] = emb.detach()
                    if layer_num >= layer_idx:
                        break

        return results

    def get_group_cls_embeddings(self, col_embeddings: torch.Tensor,
                                  groups: List[Tuple],
                                  group_batch_size: int = 64) -> torch.Tensor:
        if self._row_interactor is None:
            raise RuntimeError("row_interactor not loaded. Call load_model() first.")

        row_interactor = self._row_interactor
        n_samples, n_features, embed_dim = col_embeddings.shape
        n_groups = len(groups)
        n_cls = getattr(row_interactor, 'row_num_cls',
                        getattr(row_interactor, 'num_cls', 4))
        cls_tokens_raw = row_interactor.cls_tokens
        cls_dim = n_cls * embed_dim
        device = col_embeddings.device

        all_cls_embeddings = torch.zeros(n_groups, n_samples, cls_dim, device=device)

        with torch.no_grad():
            row_interactor.eval()

            for start in range(0, n_groups, group_batch_size):
                end = min(start + group_batch_size, n_groups)
                batch_groups = groups[start:end]
                B = end - start

                for i, group in enumerate(batch_groups):
                    gi = start + i
                    idx = list(group)
                    k = len(idx)


                    group_col_emb = col_embeddings[:, idx, :]
                    col_tensor = group_col_emb.unsqueeze(0)  # (1, n, k, embed_dim)
                    _, T, H, E = col_tensor.shape


                    cls_expanded = cls_tokens_raw.expand(1, T, n_cls, E).to(device)

                    full_emb = torch.cat([cls_expanded, col_tensor], dim=2)
                    row_outputs = row_interactor.tf_row(full_emb)

                    if getattr(row_interactor, 'norm_first', False):
                        if hasattr(row_interactor, 'out_ln'):
                            row_outputs = row_interactor.out_ln(row_outputs)

                    cls_emb = row_outputs[:, :, :n_cls, :].flatten(-2)
                    all_cls_embeddings[gi] = cls_emb.squeeze(0)

        # (n_groups, n, cls_dim) → (n, n_groups, cls_dim)
        return all_cls_embeddings.permute(1, 0, 2).contiguous()


# =============================================================================

# =============================================================================

def expand_categorical_groups(X: np.ndarray, groups: List[Tuple],
                               active_groups: List[Tuple],
                               cat_info: dict) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
    if cat_info is None or not cat_info.get('n_categories'):
        return X, groups, active_groups

    n, p = X.shape
    is_discrete = cat_info['is_discrete']
    n_cats_map = cat_info['n_categories']


    col_mapping = {}  # old_idx -> list of new indices
    new_col_idx = 0
    for j in range(p):
        if is_discrete[j] and j in n_cats_map:
            C = n_cats_map[j]
            col_mapping[j] = list(range(new_col_idx, new_col_idx + C))
            new_col_idx += C
        else:
            col_mapping[j] = [new_col_idx]
            new_col_idx += 1

    p_expanded = new_col_idx
    X_expanded = np.zeros((n, p_expanded), dtype=X.dtype)


    for j in range(p):
        if is_discrete[j] and j in n_cats_map:
            new_cols = col_mapping[j]
            cat_vals = X[:, j].astype(int)
            for i_row in range(n):
                c = cat_vals[i_row]
                if 0 <= c < len(new_cols):
                    X_expanded[i_row, new_cols[c]] = 1.0
        else:
            new_col = col_mapping[j][0]
            X_expanded[:, new_col] = X[:, j]


    def _remap_groups(gs):
        return [tuple(idx for old_idx in g for idx in col_mapping[old_idx]) for g in gs]

    groups_expanded = _remap_groups(groups)
    active_groups_expanded = _remap_groups(active_groups)

    return X_expanded, groups_expanded, active_groups_expanded


# =============================================================================

# =============================================================================

def cls_pca_group_compute_all_scores(embedder: 'TorchAcceleratedEmbedder',
                                      X: np.ndarray, y: np.ndarray,
                                      groups: List[Tuple],
                                      n_components: int = None,
                                      metrics: List[str] = None,
                                      layer_idx: int = 2,
                                      batch_size: int = 32) -> Dict[str, np.ndarray]:
    if not embedder.loaded:
        return {}

    if metrics is None:
        metrics = ['dc', 'sc', 'ipc', 'pc']

    n = X.shape[0]
    if n > 500:
        metrics = [m for m in metrics if m != 'pc']


    col_embeddings = embedder.get_all_embeddings(X, layer_idx=layer_idx, batch_size=batch_size)
    group_emb = embedder.get_group_cls_embeddings(col_embeddings, groups, group_batch_size=64)
    del col_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # group_emb: Tensor(n, n_groups, cls_dim) on GPU
    n_samples, n_groups, cls_dim = group_emb.shape


    if n_components is None:

        n_components = 128


    group_emb_np = group_emb.cpu().numpy()  # (n, n_groups, cls_dim)
    del group_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # group_emb_np: (n, n_groups, cls_dim) → PCA → (n, n_groups, n_components)
    group_means = group_emb_np.mean(axis=0, keepdims=True)
    group_emb_centered = group_emb_np - group_means
    del group_emb_np

    if n_components >= min(n_samples, cls_dim):
        group_emb_pca = group_emb_centered
    else:

        emb_t = _to_tensor(group_emb_centered)  # (n, n_groups, cls_dim)
        del group_emb_centered

        group_emb_pca_t = torch.zeros(n_samples, n_groups, n_components,
                                        dtype=emb_t.dtype, device=emb_t.device)


        pca_chunk = 64
        for start in range(0, n_groups, pca_chunk):
            end = min(start + pca_chunk, n_groups)
            # (n, chunk_size, cls_dim)
            chunk = emb_t[:, start:end, :]

            for i in range(end - start):
                eg = chunk[:, i, :]  # (n, cls_dim)
                try:
                    U, S, V = torch.pca_lowrank(eg, q=n_components, center=False)
                    group_emb_pca_t[:, start + i, :] = U * S.unsqueeze(0)
                except Exception:
                    group_emb_pca_t[:, start + i, :] = eg[:, :n_components]

        del emb_t
        group_emb_pca = group_emb_pca_t.cpu().numpy().astype(np.float32)
        del group_emb_pca_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    group_emb_pca_t = _to_tensor(group_emb_pca)  # (n, n_groups, n_components)

    results = {}

    if 'dc' in metrics:
        scores = TorchAcceleratedScreening.emb_dc_accelerated(group_emb_pca_t, y)
        results['cls_pca_dc'] = scores

    if 'sc' in metrics:
        scores = TorchAcceleratedScreening.emb_sc_accelerated(group_emb_pca_t, y, a=0.5)
        results['cls_pca_sc'] = scores

    if 'ipc' in metrics:
        scores = TorchAcceleratedScreening.emb_ipc_accelerated(group_emb_pca_t, y)
        results['cls_pca_ipc'] = scores

    if 'pc' in metrics:
        scores = TorchAcceleratedScreening.emb_pc_accelerated(group_emb_pca_t, y)
        results['cls_pca_pc'] = scores

    if 'sis' in metrics:
        scores = TorchAcceleratedScreening.emb_sis_accelerated(group_emb_pca_t, y)
        results['cls_pca_sis'] = scores

    if 'hd' in metrics:
        scores = TorchAcceleratedScreening.emb_hd_accelerated(group_emb_pca_t, y)
        results['cls_pca_hd'] = scores

    if 'gcor' in metrics:
        scores = TorchAcceleratedScreening.emb_gcor_accelerated(group_emb_pca_t, y)
        results['cls_pca_gcor'] = scores

    del group_emb_pca_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================================

# =============================================================================

from src.screening_methods import ScreeningEvaluator


# =============================================================================

# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    n, p = 200, 500
    X = np.random.randn(n, p)
    y = np.random.randint(0, 2, n)
    true_features = list(range(8))

    print(f"Test data: n={n}, p={p}")

    TS = TorchAcceleratedScreening

    method_map = {
        'sis': 'sis_accelerated', 'dc': 'dc_sis_accelerated', 'sc': 'sc_sis_accelerated',
        'fkf': 'fkf_accelerated', 'cru': 'cru_accelerated', 'cavs': 'cavs_accelerated',
        'mv_sis': 'mv_sis_accelerated', 'lr_ffs': 'lr_ffs_accelerated',
        'psis': 'psis_accelerated', 'gcor': 'gcor_sis_accelerated',
    }
    for method, fn_name in method_map.items():
        try:
            scores = getattr(TS, fn_name)(X, y)
            mms = ScreeningEvaluator.compute_mms(scores, true_features)
            print(f"  {method}: MMS={mms}, scores[:5]={scores[:5].round(4)}")
        except Exception as e:
            print(f"  {method}: FAILED - {e}")


# Statistical utilities and PACE representations.
import sys
import json
import time
import random
from datetime import datetime
from scipy.special import digamma, ndtri
from scipy.interpolate import BSpline as _ScipyBSpline
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
def _midranks_gpu(X: torch.Tensor) -> torch.Tensor:
    if X.dim() == 1:
        Xt = X.unsqueeze(1)
        squeeze = False
    else:
        Xt = X
        squeeze = False
    n, p = Xt.shape
    dev = Xt.device
    Xt = Xt.to(torch.float64)
    order = torch.argsort(Xt, dim=0, stable=True)             # (n, p)
    sortedX = torch.gather(Xt, 0, order)
    pos = torch.arange(1, n + 1, device=dev, dtype=torch.float64).unsqueeze(1).expand(n, p).contiguous()

    seg_starts = torch.ones_like(pos)
    seg_starts[1:, :] = (sortedX[1:, :] != sortedX[:-1, :]).to(torch.float64)
    seg_id = torch.cumsum(seg_starts, dim=0)

    Kmax = int(seg_id.max().item())


    sid = (seg_id - 1).long()
    pos_min = torch.full((Kmax, p), float('inf'), device=dev, dtype=torch.float64)
    pos_max = torch.full((Kmax, p), float('-inf'), device=dev, dtype=torch.float64)

    pos_min = pos_min.scatter_reduce(0, sid, pos, reduce='amin', include_self=False)

    pos_max = pos_max.scatter_reduce(0, sid, pos, reduce='amax', include_self=False)
    pos_mean = (pos_min + pos_max) / 2.0

    ranks_sorted = pos_mean.gather(0, sid)

    ranks = torch.empty_like(ranks_sorted)
    ranks.scatter_(0, order, ranks_sorted)
    return ranks.squeeze(1) if squeeze else ranks


# ===========================================================================

# ---------------------------------------------------------------------------


def _median_bandwidth_gpu(dist_block: torch.Tensor) -> torch.Tensor:
    Cn, n, _ = dist_block.shape
    dev = dist_block.device
    dt = dist_block.dtype

    idx = torch.triu_indices(n, n, offset=1, device=dev)          # (2, m), m=n*(n-1)/2
    d = dist_block[:, idx[0], idx[1]]                              # (C, m)
    nz = d > 0

    sigma = torch.zeros(Cn, device=dev, dtype=dt)


    for j in range(Cn):
        dn = d[j][nz[j]]
        if dn.numel() == 0:
            sigma[j] = 0.0
        else:
            m = dn.median()
            sigma[j] = m if m.item() > 0 else dn.mean()
    return sigma


# ===========================================================================

# ---------------------------------------------------------------------------
# cls exp `hsic_accelerated`:


def _hsic_gpu(X: np.ndarray, y: np.ndarray, normalized: bool = False) -> np.ndarray:
    Xt = _to_tensor(X, dtype=torch.float64)
    n, p = Xt.shape
    Y, is_cat, _ = _prepare_y_torch(y, n, dtype=torch.float64)
    L = _RawTrackA._y_kernel_response(Y, is_cat, n)             # (n, n)
    H = torch.eye(n, device=Xt.device, dtype=Xt.dtype) - 1.0 / n
    HLH = H @ L @ H                                              # (n, n)
    if normalized:
        Lc_norm = torch.sqrt((HLH ** 2).sum() + 1e-30)
    scores = torch.zeros(p, device=Xt.device, dtype=Xt.dtype)
    chunk = CHUNK_SIZE
    for start in range(0, p, chunk):
        end = min(start + chunk, p)
        C = end - start
        Xc = Xt[:, start:end].T.unsqueeze(2)                     # (C, n, 1)
        dist = torch.cdist(Xc, Xc)                               # (C, n, n)
        sg = _median_bandwidth_gpu(dist)
        sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
        K = torch.exp(-dist ** 2 / (2 * sg_safe.unsqueeze(1).unsqueeze(2) ** 2 + 1e-30))   # (C,n,n)
        const_mask = sg == 0
        if const_mask.any():
            K[const_mask] = 0.0

        HKxH = H @ K @ H                                         # (C, n, n)
        if normalized:

            num = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2))     # (C,)
            k_norm = torch.sqrt((HKxH ** 2).sum(dim=(1, 2)) + 1e-30)   # (C,)
            sc = num / (k_norm * Lc_norm + 1e-30)
            sc = torch.clamp(sc, min=0)
            sc = sc.masked_fill(const_mask, 0.0)
            scores[start:end] = sc
        else:
            hsic_b = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2)) / (n * n)
            hsic_b = torch.clamp(hsic_b, min=0)
            hsic_b = hsic_b.masked_fill(const_mask, 0.0)
            scores[start:end] = hsic_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores.cpu().numpy()


# ===========================================================================

# ---------------------------------------------------------------------------


# ===========================================================================

def _gpu_hist_nmi(X_t: torch.Tensor, y_t: torch.Tensor, is_cat_y: bool,
                   B: int = 64, eps: float = 1e-30) -> torch.Tensor:
    n = X_t.shape[0]
    X = X_t.reshape(n, -1) if X_t.dim() > 1 else X_t.reshape(n, 1)
    F = X.shape[1]
    dev = X.device
    dtype = torch.float64
    X = X.to(dtype)

    if is_cat_y:
        ycls = y_t.argmax(dim=1).long()
        By = int(y_t.shape[1])
        ycls_tot = ycls
    else:
        yv = y_t.reshape(n).to(dtype)
        Bc = min(B, max(2, int(n // 8)))
        qs = torch.linspace(1.0 / Bc, 1 - 1.0 / Bc, Bc - 1, device=dev, dtype=dtype)
        brk = torch.quantile(yv.detach(), qs).clamp_min(yv.min().detach())
        ycls_tot = torch.bucketize(yv, brk).clamp(0, Bc - 1)
        By = Bc


    Bx = min(B, max(2, int(n // 8)))
    BB = Bx * By


    try:
        avail = _get_available_gpu_mem(dev)
    except Exception:
        avail = 8 * 1024 ** 3
    bytes_per_f = (BB + 2 * n) * 8 + 8 * n
    _NMI_F_CHUNK = max(256, min(32768, int(avail * 0.4 / max(bytes_per_f, 1))))
    nmi_out = torch.zeros(F, device=dev, dtype=dtype)
    ycls_view = ycls_tot
    for f0 in range(0, F, _NMI_F_CHUNK):
        f1 = min(f0 + _NMI_F_CHUNK, F)
        Xc = X[:, f0:f1]

        order = torch.argsort(Xc, dim=0)                        # (n, fc)
        ranks = torch.empty_like(order, dtype=dtype)
        fc = f1 - f0
        col = torch.arange(fc, device=dev).view(1, fc).expand(n, fc)
        ranks[order, col] = torch.arange(n, device=dev, dtype=dtype).view(n, 1).expand(n, fc)
        Xb = (ranks * Bx / n).to(torch.long).clamp(0, Bx - 1)    # (n, fc) bin idx
        del order, ranks, col

        key = Xb * By + ycls_view.unsqueeze(1)                  # (n, fc), 0..BB-1
        ones = torch.ones(fc, n, device=dev, dtype=dtype)
        C2 = torch.zeros(fc, BB, device=dev, dtype=dtype)
        C2.scatter_add_(1, key.t().contiguous(), ones)           # (fc, BB)
        del Xb, key, ones

        Pf = (C2 / n).reshape(fc, Bx, By)
        Px2 = Pf.sum(dim=2); Py2 = Pf.sum(dim=1)
        mx = Px2 > 0; my = Py2 > 0
        lpx = torch.where(mx, torch.log(Px2 + eps), torch.zeros_like(Px2))
        lpy = torch.where(my, torch.log(Py2 + eps), torch.zeros_like(Py2))
        Hxv = -(Px2 * lpx).sum(dim=1); Hyv = -(Py2 * lpy).sum(dim=1)
        mp = Pf > 0
        lpf = torch.where(mp, torch.log(Pf + eps), torch.zeros_like(Pf))
        ljoint = lpf - lpx.unsqueeze(2) - lpy.unsqueeze(1)
        IV = (Pf * ljoint).sum(dim=(1, 2))
        denom = torch.sqrt(torch.clamp(Hxv * Hyv, min=eps))
        sc = IV / denom
        sc = torch.where(torch.isfinite(sc) & (denom > eps), sc, torch.zeros_like(sc))
        nmi_out[f0:f1] = torch.clamp(sc, min=0.0)
        del C2, Pf, Px2, Py2, lpx, lpy, lpf, ljoint, Hxv, Hyv, IV, denom, sc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return nmi_out

# ==========================================================================


# ==========================================================================


# ======================================================================

# ======================================================================

TRACK_A_RAW_METHODS = ['pearson', 'spearman', 'dc', 'hsic', 'nhsic', 'nmi']

TRACK_A_EMB_BASE_METHODS = ['pearson', 'spearman', 'dc', 'hsic', 'nhsic', 'nmi']

TRACK_A_METHODS = list(TRACK_A_RAW_METHODS) + [
    f'emb_{m}_L{L}' for m in TRACK_A_EMB_BASE_METHODS for L in EMB_LAYERS]

_EMB_LEGACY_TO_L3 = {f'emb_{m}': f'emb_{m}_L3' for m in TRACK_A_EMB_BASE_METHODS}

_EMB_ALL_LEGACY = set(_EMB_LEGACY_TO_L3.keys())

def _parse_emb_methods(track_methods):
    out = []
    for m in track_methods:
        if m in _EMB_ALL_LEGACY:
            out.append((m.replace('emb_', ''), 3))
        if m.startswith('emb_') and m.endswith(('_L1', '_L2', '_L3')):
            L = int(m.split('_L')[-1])
            base = m[4:].rsplit('_L', 1)[0]
            out.append((base, L))
    return out

def _ucenter(D: torch.Tensor) -> torch.Tensor:
    if D.dim() == 2:
        n = D.shape[0]
        row = D.sum(dim=1, keepdim=True)          # (n,1)
        col = D.sum(dim=0, keepdim=True)          # (1,n)
        total = D.sum()
        A = D - row / (n - 2) - col / (n - 2) + total / ((n - 1) * (n - 2))
        A.fill_diagonal_(0)
        return A

    n = D.shape[-1]
    row = D.sum(dim=-1, keepdim=True)             # (C,n,1)
    col = D.sum(dim=-2, keepdim=True)             # (C,1,n)
    total = D.sum(dim=(-2, -1), keepdim=True)     # (C,1,1)
    A = D - row / (n - 2) - col / (n - 2) + total / ((n - 1) * (n - 2))
    A.diagonal(dim1=-2, dim2=-1).zero_()
    return A

def _median_bandwidth_1d(dist_block: torch.Tensor) -> torch.Tensor:
    out = _median_bandwidth_gpu(dist_block)
    return out.to(dist_block.dtype) if out.dtype != dist_block.dtype else out

def _midranks(Xt: torch.Tensor) -> torch.Tensor:
    r = _midranks_gpu(Xt)
    return r.to(Xt.dtype) if Xt.dtype != torch.float64 else r

class TrackAScreening:

    # ---- 1. Pearson / SIS （§1） ----
    @staticmethod
    def pearson_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        Y, is_cat, classes = _prepare_y_torch(y, n, dtype=torch.float64)

        Xc = Xt - Xt.mean(dim=0, keepdim=True)
        Xstd = Xt.std(dim=0, unbiased=False)
        if is_cat:


            Yc = Y - Y.mean(dim=0, keepdim=True)
            Ystd = Y.std(dim=0, unbiased=False)
            # (p, R)
            corr = (Xc.T @ Yc) / (n * Xstd.unsqueeze(1) * Ystd.unsqueeze(0) + 1e-30)
            corr = corr.abs()
            mask = (Xstd.unsqueeze(1) < 1e-12) | (Ystd.unsqueeze(0) < 1e-12)
            corr = corr.masked_fill(mask, 0.0)
            scores = corr.max(dim=1).values
        else:
            yv = Y.squeeze(1)
            yc = yv - yv.mean()
            ystd = yv.std(unbiased=False)
            scores = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
            scores = scores.abs()
            scores[Xstd < 1e-12] = 0.0
        return scores.cpu().numpy()

    # ---- 2. Spearman （§2） ----
    @staticmethod
    def spearman_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        RX = _midranks(Xt)                               # (n, p) mid-rank
        yt = _to_tensor(y, dtype=torch.float64).reshape(-1)
        Xc = RX - RX.mean(dim=0, keepdim=True)
        Xstd = RX.std(dim=0, unbiased=False) + 1e-30

        uniq = torch.unique(yt)
        if uniq.numel() > 2:
            scores = torch.zeros(p, device=Xt.device, dtype=Xt.dtype)
            for c in uniq:

                binv = (yt == c).to(torch.float64)
                if binv.std() < 1e-12: continue
                RYc = _midranks(binv.unsqueeze(1)).squeeze(1)
                yc = RYc - RYc.mean()
                ystd = RYc.std(unbiased=False) + 1e-30
                sc = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
                sc = sc.abs()
                sc = sc.masked_fill(Xstd < 1e-12, 0.0)
                scores = torch.maximum(scores, sc)
            return scores.cpu().numpy()

        RY_all = _midranks(yt.unsqueeze(1))              # (n, 1)
        yv = RY_all.squeeze(1)
        yc = yv - yv.mean()
        ystd = yv.std(unbiased=False) + 1e-30
        scores = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
        scores = scores.abs()
        scores = scores.masked_fill(Xstd < 1e-12, 0.0)
        return scores.cpu().numpy()

    # ---- 3. Distance Correlation ----
    @staticmethod
    def dc_biased_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        Y, is_cat, _ = _prepare_y_torch(y, n, dtype=torch.float64)

        b = torch.cdist(Y, Y)                            # (n,n)
        B = _double_center(b)
        b_var = (B ** 2).sum()

        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)
        chunk = CHUNK_SIZE
        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)         # (C,n,1)
            a = torch.cdist(Xc, Xc)                      # (C,n,n)
            A = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(1, 2), keepdim=True)
            num = (A * B.unsqueeze(0)).sum(dim=(1, 2))
            a_var = (A ** 2).sum(dim=(1, 2))
            dc_sq = num / torch.sqrt(a_var * b_var + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(dc_sq, min=0, max=1))
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return scores.cpu().numpy()

    @staticmethod
    def dc_unbiased_accelerated(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        if n < 4:

            return TrackAScreening.dc_biased_accelerated(X, y)
        Y, is_cat, _ = _prepare_y_torch(y, n, dtype=torch.float64)

        b = torch.cdist(Y, Y)
        B_t = _ucenter(b)
        Vyy = (B_t ** 2).sum() / (n * (n - 3))

        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)
        chunk = CHUNK_SIZE
        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)
            a = torch.cdist(Xc, Xc)
            A_t = _ucenter(a)                            # (C,n,n)
            Vxy = (A_t * B_t.unsqueeze(0)).sum(dim=(1, 2)) / (n * (n - 3))
            Vxx = (A_t ** 2).sum(dim=(1, 2)) / (n * (n - 3))
            denom = torch.sqrt(torch.clamp(Vxx, min=0) * max(Vyy, 0) + 1e-30)
            s_u = torch.where(denom > 1e-30,
                             torch.clamp(Vxy, min=0) / (denom + 1e-30),
                             torch.zeros_like(Vxy))
            scores[start:end] = s_u
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return scores.cpu().numpy()

    # ---- 4. Mutual Information （§4） ----
    @staticmethod
    def mi_ksg_accelerated(X: np.ndarray, y: np.ndarray, k: int = 3, **kw) -> np.ndarray:
        Xf64 = np.ascontiguousarray(X, dtype=np.float64)

        if np.isnan(Xf64).any():
            col_med = np.nanmedian(Xf64, axis=0)
            inds = np.where(np.isnan(Xf64))
            Xf64[inds] = np.take(col_med, inds[1])
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(Xf64, y, discrete_features=False,
                                     n_neighbors=k, random_state=0, n_jobs=-1)
        except Exception as e:
            print(f"  [TrackA] mi_ksg sklearn fallback failed: {e}", flush=True)

            mi = TrackAScreening._mi_quantile_fallback(Xf64, y)
        mi = np.asarray(mi, dtype=np.float64)
        return np.clip(mi, 0.0, None)

    @staticmethod
    def _mi_quantile_fallback(X: np.ndarray, y: np.ndarray, B: Optional[int] = None) -> np.ndarray:
        n, p = X.shape
        if B is None:
            B = min(10, max(2, int(math.sqrt(n))))
        from sklearn.preprocessing import KBinsDiscretizer
        try:
            kb = KBinsDiscretizer(n_bins=B, encode='ordinal', strategy='quantile',
                                  subsample=None)
            Xb = kb.fit_transform(X)
        except Exception:

            Xb = np.zeros_like(X, dtype=np.int64)
            for j in range(p):
                qs = np.quantile(X[:, j], np.linspace(0, 1, B + 1)[1:-1])
                Xb[:, j] = np.digitize(X[:, j], qs)
        y = np.asarray(y)
        classes = np.unique(y)
        mi = np.zeros(p)
        # H(Y)
        py = np.array([(y == c).mean() for c in classes])
        Hy = -(py * np.log(py + 1e-30)).sum()
        for j in range(p):
            xv = Xb[:, j]
            vals = np.unique(xv)
            px = np.array([(xv == v).mean() for v in vals])
            Hx = -(px * np.log(px + 1e-30)).sum()

            joint = np.zeros((len(vals), len(classes)))
            for ia, v in enumerate(vals):
                mask = (xv == v)
                for ib, c in enumerate(classes):
                    joint[ia, ib] = (mask & (y == c)).sum() / n
            pxy = joint
            with np.errstate(divide='ignore', invalid='ignore'):
                term = pxy * np.log(pxy / (np.outer(px, py) + 1e-30) + 1e-30)
            term[pxy == 0] = 0
            Hxy = -term.sum()
            mi[j] = Hx + Hy - Hxy
        return mi

    # ---- 5. HSIC （§5） ----
    @staticmethod
    def _y_kernel_response(y_prep: torch.Tensor, is_cat: bool, n: int,
                           sigma_y: float = None) -> torch.Tensor:
        if is_cat:
            # delta kernel: 1(y==y')
            return (y_prep @ y_prep.t())
        else:
            yv = y_prep.squeeze(1).reshape(-1, 1)
            d = torch.cdist(yv, yv)
            if sigma_y is None:
                nz = d[d > 0]
                s = nz.median() if nz.numel() > 0 else 1.0
                if s == 0:
                    s = nz.mean() if nz.numel() > 0 else 1.0
            else:
                s = sigma_y
            return torch.exp(-d ** 2 / (2 * s * s + 1e-30))

    @staticmethod
    def hsic_accelerated(X: np.ndarray, y: np.ndarray, *,
                         normalized: bool = False, unbiased: bool = False,
                         sigma: str = 'median', **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        Y, is_cat, _ = _prepare_y_torch(y, n, dtype=torch.float64)

        L = TrackAScreening._y_kernel_response(Y, is_cat, n)     # (n,n)
        H = torch.eye(n, device=Xt.device, dtype=Xt.dtype) - 1.0 / n

        if unbiased:
            if n < 4:

                return TrackAScreening.hsic_accelerated(
                    X, y, normalized=normalized, unbiased=False, sigma=sigma)
            L_t = L.clone()
            L_t.fill_diagonal_(0)                                # \tilde L
            one = torch.ones(n, 1, device=Xt.device, dtype=Xt.dtype)
            lL1 = (one.t() @ L_t @ one).squeeze()                # scalar
        else:
            HLH = H @ L @ H
            Lc_norm = torch.sqrt((HLH ** 2).sum() + 1e-30) if normalized else None

        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)
        chunk = CHUNK_SIZE
        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)               # (C,n,1)
            dist = torch.cdist(Xc, Xc)                          # (C,n,n)
            if sigma == 'median':
                sg = _median_bandwidth_1d(dist)                # (C,)
            else:
                sg = torch.full((end - start,), float(sigma), device=Xt.device, dtype=Xt.dtype)

            sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
            K = torch.exp(-dist ** 2 / (2 * sg_safe.unsqueeze(1).unsqueeze(2) ** 2 + 1e-30))
            const_mask = sg == 0
            if const_mask.any():
                K[const_mask] = 0.0

            if unbiased:
                for j in range(end - start):
                    if const_mask[j]:
                        continue
                    Kj = K[j]
                    Kt = Kj.clone()
                    Kt.fill_diagonal_(0)
                    kK1 = (one.t() @ Kt @ one).squeeze()
                    trKL = (Kt * L_t).sum()
                    cross = (one.t() @ Kt @ L_t @ one).squeeze()
                    denom = n * (n - 3)
                    hsic_u = (trKL + kK1 * lL1 / ((n - 1) * (n - 2))
                              - 2 * cross / (n - 2)) / denom
                    scores[start + j] = torch.clamp(hsic_u, min=0)
            elif normalized:
                for j in range(end - start):
                    if const_mask[j]:
                        continue
                    Kc = H @ K[j] @ H
                    num = (Kc * HLH).sum()
                    k_norm = torch.sqrt((Kc ** 2).sum() + 1e-30)
                    scores[start + j] = num / (k_norm * Lc_norm + 1e-30)
            else:

                HKxH = torch.einsum('ij,cjk,kl->cil', H, K, H)

                HKxH = H @ K @ H
                hsic_b = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2)) / (n * n)
                scores[start:end] = torch.clamp(hsic_b, min=0)
                scores[start:end][const_mask] = 0.0
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return scores.cpu().numpy()


    @staticmethod
    def hsic_biased_accelerated(X, y, **kw):
        return TrackAScreening.hsic_accelerated(X, y, normalized=False, unbiased=False)

    @staticmethod
    def hsic_unbiased_accelerated(X, y, **kw):
        return TrackAScreening.hsic_accelerated(X, y, normalized=False, unbiased=True)

    @staticmethod
    def nhsic_accelerated(X, y, **kw):
        return TrackAScreening.hsic_accelerated(X, y, normalized=True, unbiased=False)

    # ---- 6. HSIC-Lasso （§6） ----
    @staticmethod
    def hsic_lasso_accelerated(X: np.ndarray, y: np.ndarray,
                               lam: float = 0.01, n_iter: int = 200,
                               max_features: int = 5000, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float32)
        n, p = Xt.shape
        if p > max_features:
            print(f"  [TrackA] hsic_lasso: p={p}>{max_features}, "
                  f"skipping (per §6 first-version bound)", flush=True)
            return None

        Y, is_cat, _ = _prepare_y_torch(y, n, dtype=torch.float32)
        L = TrackAScreening._y_kernel_response(Y, is_cat, n)     # (n,n)
        H = torch.eye(n, device=Xt.device, dtype=Xt.dtype) - 1.0 / n
        Lbar = H @ L @ H
        Lbar = Lbar / (torch.sqrt((Lbar ** 2).sum()) + 1e-12)


        Kbank = torch.empty(p, n, n, device=Xt.device, dtype=Xt.dtype)
        one = torch.ones(n, 1, device=Xt.device, dtype=Xt.dtype)
        chunk = CHUNK_SIZE
        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)               # (C,n,1)
            dist = torch.cdist(Xc, Xc)                          # (C,n,n)
            sg = _median_bandwidth_1d(dist)
            sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
            K = torch.exp(-dist ** 2 / (2 * sg_safe.unsqueeze(1).unsqueeze(2) ** 2 + 1e-30))
            const_mask = sg == 0
            if const_mask.any():
                K[const_mask] = 0.0
            Kc = H @ K @ H                                      # (C,n,n)
            norms = torch.sqrt((Kc ** 2).sum(dim=(1, 2)) + 1e-12)
            Kbank[start:end] = Kc / norms.unsqueeze(1).unsqueeze(2)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # relevance r_j = <K̄⁽ʲ⁾, L̄>
        r = torch.einsum('jab,ab->j', Kbank, Lbar)             # (p,)

        G = torch.einsum('jab,kab->jk', Kbank, Kbank)          # (p,p)

        lip = torch.sqrt((G ** 2).sum()) + 1e-12
        lr = 1.0 / lip

        alpha = torch.clamp(r.clone(), min=0) / (lip + 1e-12)
        for it in range(n_iter):
            grad = G @ alpha - r                               # (p,)
            alpha = torch.clamp(alpha - lr * grad - lr * lam, min=0)


        del Kbank, G
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return alpha.detach().cpu().numpy()


    @staticmethod
    def qt_nmi_accelerated(X: np.ndarray, y: np.ndarray,
                            B: int = None, **kw) -> np.ndarray:
        n, p = X.shape
        if B is None:
            B = min(10, max(2, int(math.sqrt(n))))
        y = np.asarray(y)
        classes = np.unique(y)

        Zc = np.stack([(y == c).astype(np.float64) for c in classes], axis=1)  # (n, C)

        def _binary_nmi(a, b):
            # a,b ∈ {0,1}
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            n11 = np.sum(a & b == 1) if False else np.sum((a == 1) & (b == 1))
            n10 = np.sum((a == 1) & (b == 0))
            n01 = np.sum((a == 0) & (b == 1))
            n00 = np.sum((a == 0) & (b == 0))
            ns = [n11, n10, n01, n00]
            pa = np.array([n11 + n10, n01 + n00])
            pb = np.array([n11 + n01, n10 + n00])
            Ha = -(pa / n * np.log(pa / n + 1e-30) + (pa == 0)).sum()

            Ha = 0.0
            for cnt in pa:
                if cnt > 0:
                    Ha -= (cnt / n) * np.log(cnt / n)
            Hb = 0.0
            for cnt in pb:
                if cnt > 0:
                    Hb -= (cnt / n) * np.log(cnt / n)
            # I
            I = 0.0
            for ia in [1, 0]:
                for ib in [1, 0]:
                    nj = ns[(1 - ia) * 2 + (1 - ib)]
                    if nj > 0:
                        pai = (ns[0 if ia == 1 else 1] + ns[2 if ia == 0 else 0]) if False else pa[1 - ia]
                        pbi = pb[1 - ib]
                        I += (nj / n) * np.log((nj / n) / (pai / n * pbi / n + 1e-30) + 1e-30)
            return I / (np.sqrt(Ha * Hb) + 1e-30)

        scores = np.zeros(p, dtype=np.float64)
        for j in range(p):
            xj = X[:, j]
            if np.std(xj) < 1e-12:
                continue

            qs = np.quantile(xj, [(b + 1) / (B + 1) for b in range(B)])
            qs = np.unique(qs)
            best = 0.0
            for tau in qs:
                zb = (xj > tau).astype(np.float64)
                if np.all(zb == zb[0]):
                    continue
                for c in range(len(classes)):
                    zc = Zc[:, c]
                    if np.all(zc == zc[0]):
                        continue
                    val = _binary_nmi(zb, zc)
                    if val > best:
                        best = val
            scores[j] = best
        return scores


_RawTrackA = TrackAScreening


_PaceA = sys.modules[__name__]
_HAS_PACEA_A = True

TRACK_A_DISPATCH = {


    'pearson':   lambda X, y: PAceRawA.pearson(X, y),
    'spearman':  lambda X, y: PAceRawA.spearman(X, y),
    'dc':        lambda X, y: PAceRawA.dc(X, y),
    'hsic':      lambda X, y: PAceRawA.hsic(X, y),
    'nhsic':     lambda X, y: PAceRawA.nhsic(X, y),
    'nmi':       lambda X, y: PAceRawA.nmi(X, y),
}

TRACK_A_EMB_DISPATCH = {
    'emb_pearson':  lambda emb, y: PAceEmbA.emb_pearson(emb, y),
    'emb_spearman': lambda emb, y: PAceEmbA.emb_spearman(emb, y),
    'emb_dc':       lambda emb, y: PAceEmbA.emb_dc(emb, y),
    'emb_hsic':     lambda emb, y: PAceEmbA.emb_hsic(emb, y),
    'emb_nhsic':    lambda emb, y: PAceEmbA.emb_nhsic(emb, y),
    'emb_nmi':      lambda emb, y: PAceEmbA.emb_nmi(emb, y),
}

TRACK_A_DISPATCH['dc_biased'] = TRACK_A_DISPATCH['dc']

TRACK_A_DISPATCH['hsic_biased'] = TRACK_A_DISPATCH['hsic']

def compute_trackA_scores(X, y, methods: List[str],
                          cat_indices: Optional[List[int]] = None,
                          dispatch: Optional[Dict] = None,
                          tag: str = "TrackA") -> Dict[str, np.ndarray]:
    if dispatch is None:
        dispatch = TRACK_A_DISPATCH_REG if cat_indices is not None else TRACK_A_DISPATCH
    results = {}
    for m in methods:
        try:
            if cat_indices is not None:
                sc = dispatch[m](X, y, cat_indices=cat_indices)
            else:
                sc = dispatch[m](X, y)
            if sc is not None and not np.isnan(np.asarray(sc, dtype=np.float64)).all():
                results[m] = sc
        except Exception as e:
            print(f"  [{tag}] {m} failed: {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

def compute_trackA_emb_scores(emb_tensor, y, methods: List[str]) -> Dict[str, np.ndarray]:
    results = {}
    for m in methods:
        try:
            scores = TRACK_A_EMB_DISPATCH[m](emb_tensor, y)
            if scores is not None:
                sc = np.asarray(scores, dtype=np.float64)
                if not sc.shape: continue
                if not np.isnan(sc).all():
                    results[m] = sc
        except Exception as e:
            print(f"  [TrackA] {m} failed: {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

def _tiny_meanshift(n=200, p=120, s=8, mu=0.6, seed=0):
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

def validate_gpu_scorers(methods: Optional[List[str]] = None,
                         relax_methods: Optional[List[str]] = None) -> bool:
    methods = methods or TRACK_A_METHODS
    relax_methods = relax_methods or ['qt_nmi', 'nmi']
    torch.manual_seed(0)
    np.random.seed(0)

    print("\n" + "=" * 70)
    print("Track-A GPU scorer validation")
    print("=" * 70, flush=True)


    raw_methods = [m for m in methods if not m.startswith('emb_')]
    emb_methods_list = [m for m in methods if m.startswith('emb_')]
    if emb_methods_list:
        print(f"  [VALIDATE] emb_* methods {emb_methods_list} require an embedder; use the end-to-end smoke test", flush=True)


    X, y, true_features = _tiny_meanshift(n=200, p=120, s=8, mu=0.6, seed=0)
    s = len(true_features)
    p = X.shape[1]
    failures = []
    for m in raw_methods:
        if m not in TRACK_A_DISPATCH:
            print(f"  [VALIDATE] {m}: not in raw dispatch; skipped", flush=True)
            continue
        try:
            sc = TRACK_A_DISPATCH[m](X, y)
        except Exception as e:
            print(f"  [VALIDATE] {m}: SCORER EXCEPTION -> {e}", flush=True)
            failures.append(m)
            continue
        if sc is None:
            print(f"  [VALIDATE] {m}: returned None", flush=True)
            failures.append(m)
            continue
        if not np.all(np.isfinite(sc)):
            sc = np.nan_to_num(sc, nan=-np.inf)
        top_k = min(s, p)
        top = set(np.argsort(-sc)[:top_k].tolist())
        if m in relax_methods:

            thr = min(3 * s, p)
            top_relax = set(np.argsort(-sc)[:thr].tolist())
            ok = len(set(true_features) & top_relax) >= int(0.75 * s)
            tag = f"relax(top{thr},>={int(0.75*s)})"
        else:
            ok = set(true_features).issubset(top)
            tag = f"top{s}"
        status = "OK" if ok else "FAIL"
        if not ok:
            failures.append(m)
        print(f"  [VALIDATE] {m:<14} signal recovery [{tag}]: {status} "
              f"(signal ranks: {np.argsort(np.argsort(-sc)[true_features]) + 1})", flush=True)


    if 'spearman' in methods:
        try:
            rng = np.random.RandomState(1)
            n2 = 50
            xs = rng.randn(n2)
            ys = (rng.randn(n2) > 0).astype(int)
            Xs = np.stack([xs, rng.randn(n2)], axis=1)
            sc = TrackAScreening.spearman_accelerated(Xs, ys)
            ref = abs(sp_stats.spearmanr(xs, ys).correlation)
            diff = abs(sc[0] - ref)
            ok = diff < 1e-4
            print(f"  [VALIDATE] spearman numerical parity vs scipy: "
                  f"{'OK' if ok else 'FAIL'} (diff={diff:.2e})", flush=True)
            if not ok:
                failures.append('spearman_parity')
        except Exception as e:
            print(f"  [VALIDATE] spearman parity FAILED: {e}", flush=True)
            failures.append('spear_parity')


    if any(m in methods for m in ['hsic', 'nhsic', 'hsic_unbiased']):
        try:
            rng = np.random.RandomState(2)
            n2 = 40
            xs = rng.randn(n2)
            ys = (xs + 0.5 * rng.randn(n2) > 0).astype(int)
            Xs = xs.reshape(-1, 1)

            dx = np.abs(xs[:, None] - xs[None, :])
            nz = dx[dx > 0]
            sig = np.median(nz) if nz.size else 1.0
            if sig == 0 and nz.size:
                sig = np.mean(nz)
            Kx = np.exp(-dx ** 2 / (2 * sig ** 2 + 1e-30))
            Ly = np.equal(ys[:, None], ys[None, :]).astype(np.float64)
            H = np.eye(n2) - 1.0 / n2
            HKxH = H @ Kx @ H
            HLyH = H @ Ly @ H
            ref = max(0.0, np.sum(HKxH * HLyH) / (n2 * n2))
            sc = TrackAScreening.hsic_biased_accelerated(Xs.copy(), ys.copy())[0]
            diff = abs(sc - ref)
            ok = (diff < max(1e-6, 1e-5 * max(abs(ref), 1e-12)))
            print(f"  [VALIDATE] hsic-biased parity vs numpy ref: "
                  f"{'OK' if ok else 'FAIL'} (gpu={sc:.6e} ref={ref:.6e} diff={diff:.2e})",
                  flush=True)
            if not ok:
                failures.append('hsic_parity')

            nhsic_sc = TrackAScreening.nhsic_accelerated(Xs.copy(), ys.copy())[0]
            ok2 = (nhsic_sc >= -1e-9) and (nhsic_sc <= 1.0 + 1e-6)
            print(f"  [VALIDATE] nhsic range [0,1]: {'OK' if ok2 else 'FAIL'} "
                  f"(val={nhsic_sc:.6f})", flush=True)
            if not ok2:
                failures.append('nhsic_range')
        except Exception as e:
            print(f"  [VALIDATE] hsic parity FAILED: {e}", flush=True)
            failures.append('hsic_parity')

    print("-" * 70)
    if failures:
        print(f"[VALIDATE] FAILED methods: {failures}", flush=True)
        print("[VALIDATE] Aborting to avoid garbage results. "
              "Use --skip-validation to override (at your own risk).", flush=True)
        return False
    print("[VALIDATE] All checks passed.", flush=True)
    return True


_HAS_PACEA_REG_A = True

TRACK_A_RAW_METHODS_REG = ['pearson', 'spearman', 'dc', 'hsic', 'nhsic', 'nmi']

TRACK_A_EMB_BASE_METHODS_REG = ['pearson', 'spearman', 'dc', 'hsic', 'nhsic', 'nmi']

TRACK_A_METHODS_REG = list(TRACK_A_RAW_METHODS_REG) + [
    f'emb_{m}_L{L}' for m in TRACK_A_EMB_BASE_METHODS_REG for L in EMB_LAYERS]

_EMB_LEGACY_TO_L3 = {f'emb_{m}': f'emb_{m}_L3' for m in TRACK_A_EMB_BASE_METHODS_REG}

_EMB_ALL_LEGACY = set(_EMB_LEGACY_TO_L3.keys())

def _parse_emb_methods_reg(track_methods):
    out = []
    for m in track_methods:
        if m in _EMB_ALL_LEGACY:
            out.append((m.replace('emb_', ''), 3))
        if m.startswith('emb_') and m.endswith(('_L1', '_L2', '_L3')):
            L = int(m.split('_L')[-1]); base = m[4:].rsplit('_L', 1)[0]
            out.append((base, L))
    return out


_ucenter_reg = _ucenter
_median_bandwidth_1d_reg = _median_bandwidth_1d
_midranks_reg = _midranks

def _median_bandwidth_general_reg(V: torch.Tensor) -> float:
    d = torch.cdist(V.unsqueeze(1), V.unsqueeze(1))
    nz = d[d > 0]
    if nz.numel() == 0:
        return 0.0
    s = nz.median().item()
    if s == 0:
        s = nz.mean().item()
    return s

class TrackAScreeningReg:


    @staticmethod
    def pearson_accelerated(X: np.ndarray, y: np.ndarray, *,
                            cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        Y = _to_tensor(y, dtype=torch.float64).reshape(-1)

        cont_mask = torch.tensor([j not in (cat_indices or []) for j in range(p)],
                                 device=Xt.device, dtype=torch.bool)
        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)
        Xc = Xt - Xt.mean(dim=0, keepdim=True)
        Xstd = Xt.std(dim=0, unbiased=False)
        yc = Y - Y.mean()
        ystd = Y.std(unbiased=False)
        s_cont = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
        s_cont = s_cont.abs()
        s_cont[Xstd < 1e-12] = 0.0
        scores[cont_mask] = s_cont[cont_mask]

        for j in (cat_indices or []):
            v = Xt[:, j]
            uniq = torch.unique(v)
            if uniq.numel() < 2:
                continue
            best = torch.tensor(0.0, device=Xt.device, dtype=torch.float64)
            for c in uniq:
                ind = (v == c).float()
                if ind.sum() < 2 or (1 - ind).sum() < 2:
                    continue
                # corr(ind, y)
                ic = ind - ind.mean()
                cc = (ic * yc).sum() / (n * ind.std(unbiased=False) * ystd + 1e-30)
                best = torch.maximum(best, cc.abs())
            scores[j] = best
        return scores.cpu().numpy()


    @staticmethod
    def spearman_accelerated(X: np.ndarray, y: np.ndarray, *,
                             cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        cont_mask = torch.tensor([j not in (cat_indices or []) for j in range(p)],
                                 device=Xt.device, dtype=torch.bool)
        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)


        RX = _midranks_reg(Xt[:, torch.where(cont_mask)[0]] if cont_mask.any() else Xt)
        yt = _to_tensor(y, dtype=torch.float64).reshape(-1)
        RY = _midranks_reg(yt).reshape(-1, 1)
        Xc = RX - RX.mean(dim=0, keepdim=True) if RX.numel() else RX
        Xstd = RX.std(dim=0, unbiased=False) if RX.numel() else RX
        yc = RY.squeeze(1) - RY.mean()
        ystd = RY.std(unbiased=False)

        if cont_mask.any():
            s = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
            s = s.abs()
            s[Xstd < 1e-12] = 0.0
            scores[torch.where(cont_mask)[0]] = s

        ys = yt.cpu().numpy()
        for j in (cat_indices or []):
            v = Xt[:, j].cpu().numpy()
            uniq = np.unique(v)
            best = 0.0
            for c in uniq:
                ind = (v == c).astype(np.float64)
                if ind.sum() < 2 or (1 - ind).sum() < 2:
                    continue
                r = abs(float(np.corrcoef(ind, ys)[0, 1]))
                if not np.isnan(r) and r > best:
                    best = r
            scores[j] = best
        return scores.cpu().numpy()


    @staticmethod
    def dc_biased_accelerated(X: np.ndarray, y: np.ndarray, *,
                              cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        Y = _to_tensor(y, dtype=torch.float64).reshape(-1, 1)
        b = torch.cdist(Y, Y)
        B = _double_center(b)
        b_var = (B ** 2).sum()
        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)

        for start in range(0, p, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)
            if cat_indices:

                a = torch.cdist(Xc, Xc)
                for k, j in enumerate(range(start, end)):
                    if j in cat_indices:
                        v = Xc[k, :, 0]
                        a[k] = (v.unsqueeze(0) != v.unsqueeze(1)).double()
            else:
                a = torch.cdist(Xc, Xc)
            A = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(1, 2), keepdim=True)
            num = (A * B.unsqueeze(0)).sum(dim=(1, 2))
            a_var = (A ** 2).sum(dim=(1, 2))
            dc_sq = num / torch.sqrt(a_var * b_var + 1e-30)
            scores[start:end] = torch.sqrt(torch.clamp(dc_sq, min=0, max=1))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores.cpu().numpy()

    @staticmethod
    def dc_unbiased_accelerated(X: np.ndarray, y: np.ndarray, *,
                                cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        if n < 4:
            return TrackAScreeningReg.dc_biased_accelerated(X, y, cat_indices=cat_indices)
        Y = _to_tensor(y, dtype=torch.float64).reshape(-1, 1)
        b = torch.cdist(Y, Y)
        B_t = _ucenter_reg(b)
        Vyy = (B_t ** 2).sum() / (n * (n - 3))
        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)
        for start in range(0, p, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)
            if cat_indices:
                a = torch.cdist(Xc, Xc)
                for k, j in enumerate(range(start, end)):
                    if j in cat_indices:
                        v = Xc[k, :, 0]
                        a[k] = (v.unsqueeze(0) != v.unsqueeze(1)).double()
            else:
                a = torch.cdist(Xc, Xc)
            A_t = _ucenter_reg(a)
            Vxy = (A_t * B_t.unsqueeze(0)).sum(dim=(1, 2)) / (n * (n - 3))
            Vxx = (A_t ** 2).sum(dim=(1, 2)) / (n * (n - 3))
            denom = torch.sqrt(torch.clamp(Vxx, min=0) * max(Vyy, 0) + 1e-30)
            s_u = torch.where(denom > 1e-30,
                             torch.clamp(Vxy, min=0) / (denom + 1e-30),
                             torch.zeros_like(Vxy))
            scores[start:end] = s_u
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores.cpu().numpy()


    @staticmethod
    def mi_ksg_accelerated(X: np.ndarray, y: np.ndarray, k: int = 3, *,
                           cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        Xf = np.ascontiguousarray(X, dtype=np.float64)
        if np.isnan(Xf).any():
            col_med = np.nanmedian(Xf, axis=0)
            inds = np.where(np.isnan(Xf))
            Xf[inds] = np.take(col_med, inds[1])
        n, p = Xf.shape
        y_cont = np.ascontiguousarray(y, dtype=np.float64)

        scores = np.zeros(p, dtype=np.float64)

        cont_idx = [j for j in range(p) if not cat_indices or j not in cat_indices]
        cat_idx = list(cat_indices) if cat_indices else []
        if cont_idx:
            Xc = np.ascontiguousarray(Xf[:, cont_idx])

            mi = mutual_info_regression(Xc, y_cont, n_neighbors=k,
                                        random_state=0, n_jobs=-1)
            for a, j in enumerate(cont_idx):
                scores[j] = max(0.0, mi[a])
        if cat_idx:
            Xg = np.ascontiguousarray(Xf[:, cat_idx])
            mic = mutual_info_classif(Xg, y_cont, discrete_features=True,
                                      n_neighbors=k, random_state=0, n_jobs=-1)
            for a, j in enumerate(cat_idx):
                scores[j] = max(0.0, mic[a])
        return scores


    @staticmethod
    def _y_kernel_regression(y: np.ndarray, device, dtype, n: int) -> torch.Tensor:
        Y = _to_tensor(y, dtype=dtype).reshape(-1, 1)
        d = torch.cdist(Y, Y)
        sig = _median_bandwidth_general_reg(_to_tensor(y, dtype=dtype).reshape(-1))
        sig = sig if sig > 0 else 1.0
        L = torch.exp(-d ** 2 / (2 * sig * sig + 1e-30))
        return L

    @staticmethod
    def hsic_accelerated(X: np.ndarray, y: np.ndarray, *,
                         normalized: bool = False, unbiased: bool = False,
                         cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        L = TrackAScreeningReg._y_kernel_regression(y, Xt.device, Xt.dtype, n)
        H = torch.eye(n, device=Xt.device, dtype=Xt.dtype) - 1.0 / n
        if unbiased:
            if n < 4:
                return TrackAScreeningReg.hsic_accelerated(
                    X, y, normalized=normalized, unbiased=False, cat_indices=cat_indices)
            L_t = L.clone(); L_t.fill_diagonal_(0)
            one = torch.ones(n, 1, device=Xt.device, dtype=Xt.dtype)
            lL1 = (one.t() @ L_t @ one).squeeze()
        else:
            HLH = H @ L @ H
            Lc_norm = torch.sqrt((HLH ** 2).sum() + 1e-30) if normalized else None

        scores = torch.zeros(p, device=Xt.device, dtype=torch.float64)
        for start in range(0, p, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)
            if cat_indices:
                dist = torch.cdist(Xc, Xc)
                for k, j in enumerate(range(start, end)):
                    if j in cat_indices:
                        v = Xc[k, :, 0]
                        dist[k] = (v.unsqueeze(0) != v.unsqueeze(1)).double()
                sg = _median_bandwidth_gpu(dist)
                sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
                K = torch.exp(-dist ** 2 / (2 * sg_safe.unsqueeze(1).unsqueeze(2) ** 2 + 1e-30))
                const_mask = sg == 0
                if const_mask.any():
                    K[const_mask] = 0.0
            else:
                dist = torch.cdist(Xc, Xc)
                sg = _median_bandwidth_gpu(dist)
                sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
                K = torch.exp(-dist ** 2 / (2 * sg_safe.unsqueeze(1).unsqueeze(2) ** 2 + 1e-30))
                const_mask = sg == 0
                if const_mask.any():
                    K[const_mask] = 0.0
            if unbiased:
                for j in range(end - start):
                    if const_mask[j]:
                        continue
                    Kt = K[j].clone(); Kt.fill_diagonal_(0)
                    kK1 = (one.t() @ Kt @ one).squeeze()
                    trKL = (Kt * L_t).sum()
                    cross = (one.t() @ Kt @ L_t @ one).squeeze()
                    hsic_u = (trKL + kK1 * lL1 / ((n - 1) * (n - 2))
                              - 2 * cross / (n - 2)) / (n * (n - 3))
                    scores[start + j] = torch.clamp(hsic_u, min=0)
            elif normalized:


                HKxH = H @ K @ H                                       # (C, n, n)
                num = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2))       # (C,)
                knorm = torch.sqrt((HKxH ** 2).sum(dim=(1, 2)) + 1e-30)
                sc = num / (knorm * Lc_norm + 1e-30)
                sc = sc.masked_fill(const_mask, 0.0)
                scores[start:end] = sc
            else:
                HKxH = H @ K @ H
                hsic_b = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2)) / (n * n)
                scores[start:end] = torch.clamp(hsic_b, min=0)
                scores[start:end][const_mask] = 0.0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores.cpu().numpy()

    @staticmethod
    def hsic_biased_accelerated(X, y, cat_indices=None, **kw):
        return TrackAScreeningReg.hsic_accelerated(X, y, normalized=False, unbiased=False,
                                                   cat_indices=cat_indices)

    @staticmethod
    def hsic_unbiased_accelerated(X, y, cat_indices=None, **kw):
        return TrackAScreeningReg.hsic_accelerated(X, y, normalized=False, unbiased=True,
                                                   cat_indices=cat_indices)

    @staticmethod
    def nhsic_accelerated(X, y, cat_indices=None, **kw):
        return TrackAScreeningReg.hsic_accelerated(X, y, normalized=True, unbiased=False,
                                                   cat_indices=cat_indices)


    @staticmethod
    def hsic_lasso_accelerated(X: np.ndarray, y: np.ndarray,
                               lam: float = 0.01, n_iter: int = 200,
                               max_features: int = 5000,
                               cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float32)
        n, p = Xt.shape
        if p > max_features:
            print(f"  [TrackA] hsic_lasso: p={p}>{max_features}, skipping per §6 (timeout/OOM)",
                  flush=True)
            return None
        L = TrackAScreeningReg._y_kernel_regression(y, Xt.device, Xt.dtype, n)
        H = torch.eye(n, device=Xt.device, dtype=Xt.dtype) - 1.0 / n
        Lbar = H @ L @ H
        Lbar = Lbar / (torch.sqrt((Lbar ** 2).sum()) + 1e-12)

        Kbank = torch.empty(p, n, n, device=Xt.device, dtype=Xt.dtype)
        for start in range(0, p, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, p)
            Xc = Xt[:, start:end].T.unsqueeze(2)
            if cat_indices:
                dist = torch.cdist(Xc, Xc)
                for k, j in enumerate(range(start, end)):
                    if j in cat_indices:
                        v = Xc[k, :, 0]
                        dist[k] = (v.unsqueeze(0) != v.unsqueeze(1)).double()
            else:
                dist = torch.cdist(Xc, Xc)
            sg = _median_bandwidth_1d_reg(dist)
            sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
            K = torch.exp(-dist ** 2 / (2 * sg_safe.unsqueeze(1).unsqueeze(2) ** 2 + 1e-30))
            cm = sg == 0
            if cm.any():
                K[cm] = 0.0
            Kc = H @ K @ H
            norms = torch.sqrt((Kc ** 2).sum(dim=(1, 2)) + 1e-12)
            Kbank[start:end] = Kc / norms.unsqueeze(1).unsqueeze(2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        r = torch.einsum('jab,ab->j', Kbank, Lbar)
        G = torch.einsum('jab,kab->jk', Kbank, Kbank)
        lip = torch.sqrt((G ** 2).sum()) + 1e-12
        lr = 1.0 / lip
        alpha = torch.clamp(r.clone(), min=0) / (lip + 1e-12)
        for it in range(n_iter):
            grad = G @ alpha - r
            alpha = torch.clamp(alpha - lr * grad - lr * lam, min=0)
        del Kbank, G
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return alpha.detach().cpu().numpy()


    @staticmethod
    def qt_nmi_accelerated(X: np.ndarray, y: np.ndarray, B: int = None,
                            cat_indices: Optional[List[int]] = None, **kw) -> np.ndarray:
        n, p = X.shape
        if B is None:
            B = min(10, max(2, int(math.sqrt(n))))
        y = np.asarray(y, dtype=np.float64)

        y_qs = np.quantile(y, [(b + 1) / (B + 1) for b in range(B)])
        y_qs = np.unique(y_qs)

        def _binary_nmi(a, b):
            a = a.astype(np.int64); b = b.astype(np.int64)
            n11 = np.sum((a == 1) & (b == 1))
            n10 = np.sum((a == 1) & (b == 0))
            n01 = np.sum((a == 0) & (b == 1))
            n00 = np.sum((a == 0) & (b == 0))
            ns = [n11, n10, n01, n00]
            pa = [n11 + n10, n01 + n00]
            pb = [n11 + n01, n10 + n00]
            Hb = sum(-(c / n) * np.log(c / n) for c in pb if c > 0)
            Ha = sum(-(c / n) * np.log(c / n) for c in pa if c > 0)
            I = 0.0
            for ia in [1, 0]:
                for ib in [1, 0]:
                    nj = ns[(1 - ia) * 2 + (1 - ib)]
                    if nj > 0:
                        pa_i = pa[1 - ia]; pb_i = pb[1 - ib]
                        I += (nj / n) * np.log((nj / n) / (pa_i / n * pb_i / n + 1e-30) + 1e-30)
            return I / (np.sqrt(Ha * Hb) + 1e-30)

        scores = np.zeros(p, dtype=np.float64)
        cat_set = set(cat_indices or [])
        for j in range(p):
            if j in cat_set:

                xv = X[:, j].astype(np.int64)
                uniq = np.unique(xv)
                if uniq.size < 2:
                    continue
                best = 0.0
                for tau in y_qs:
                    zy = (y > tau).astype(np.int64)
                    if np.all(zy == zy[0]):
                        continue
                    for c in uniq:
                        zb = (xv == c).astype(np.int64)
                        if np.all(zb == zb[0]):
                            continue
                        v = _binary_nmi(zb, zy)
                        if v > best:
                            best = v
                scores[j] = best
                continue
            xj = X[:, j]
            if np.std(xj) < 1e-12:
                continue
            x_qs = np.quantile(xj, [(b + 1) / (B + 1) for b in range(B)])
            x_qs = np.unique(x_qs)
            best = 0.0
            for tau in x_qs:
                zb = (xj > tau).astype(np.int64)
                if np.all(zb == zb[0]):
                    continue
                for ty in y_qs:
                    zy = (y > ty).astype(np.int64)
                    if np.all(zy == zy[0]):
                        continue
                    v = _binary_nmi(zb, zy)
                    if v > best:
                        best = v
            scores[j] = best
        return scores

TRACK_A_DISPATCH_REG = {

    'pearson':      lambda X, y, cat_indices=None: TrackAScreeningReg.pearson_accelerated(X, y, cat_indices=cat_indices),
    'spearman':     lambda X, y, cat_indices=None: TrackAScreeningReg.spearman_accelerated(X, y, cat_indices=cat_indices),
    'dc':           lambda X, y, cat_indices=None: TrackAScreeningReg.dc_biased_accelerated(X, y, cat_indices=cat_indices),
    'hsic':         lambda X, y, cat_indices=None: TrackAScreeningReg.hsic_biased_accelerated(X, y, cat_indices=cat_indices),
    'nhsic':        lambda X, y, cat_indices=None: TrackAScreeningReg.nhsic_accelerated(X, y, cat_indices=cat_indices),
    'nmi':          lambda X, y, cat_indices=None: PAceRawA.nmi(X, y),

    'dc_biased':    lambda X, y, cat_indices=None: TrackAScreeningReg.dc_biased_accelerated(X, y, cat_indices=cat_indices),
    'hsic_biased':  lambda X, y, cat_indices=None: TrackAScreeningReg.hsic_biased_accelerated(X, y, cat_indices=cat_indices),
}

TRACK_A_EMB_DISPATCH_REG = {
    'emb_pearson':  lambda emb, y: PAceEmbA.emb_pearson(emb, y),
    'emb_spearman': lambda emb, y: PAceEmbA.emb_spearman(emb, y),
    'emb_dc':       lambda emb, y: PAceEmbA.emb_dc(emb, y),
    'emb_hsic':     lambda emb, y: PAceEmbA.emb_hsic(emb, y),
    'emb_nhsic':    lambda emb, y: PAceEmbA.emb_nhsic(emb, y),
    'emb_nmi':      lambda emb, y: PAceEmbA.emb_nmi(emb, y),
}

def compute_trackA_scores_reg(X, y, methods: List[str],
                          cat_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    return compute_trackA_scores(X, y, methods, cat_indices=cat_indices,
                                 dispatch=TRACK_A_DISPATCH_REG, tag="TrackA-reg")

def _tiny_reg_nonlinear_A(n=200, p=120, s=4, c=4.0, seed=0, t3=False):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    eps = rng.standard_t(3, n) if t3 else rng.randn(n)
    Y = (2.0 * X[:, 0] + 2.0 * X[:, 1] ** 2 + 2.0 * np.exp(X[:, 2])
         + 2.0 * np.log(np.abs(X[:, 3]) + 1e-8) + c * eps)
    return X, Y, [0, 1, 2, 3]

def validate_gpu_scorers_reg(methods: Optional[List[str]] = None,
                              relax_methods: Optional[List[str]] = None) -> bool:
    methods = methods or TRACK_A_METHODS_REG
    relax_methods = relax_methods or ['qt_nmi', 'nmi']
    torch.manual_seed(0); np.random.seed(0)

    print("\n" + "=" * 70)
    print("Track-A (regression) GPU scorer validation")
    print("=" * 70, flush=True)


    raw_methods = [m for m in methods if not m.startswith('emb_')]
    emb_methods_list = [m for m in methods if m.startswith('emb_')]
    if emb_methods_list:
        print(f"  [VALIDATE] emb_* methods {emb_methods_list} require an embedder; use the end-to-end smoke test", flush=True)

    X, Y, tf = _tiny_reg_nonlinear_A(n=200, p=120, s=4, c=4.0, seed=0, t3=False)
    s = len(tf); p = X.shape[1]
    failures = []


    linear_ok_methods = {'dc', 'dc_biased', 'hsic', 'nhsic'}
    for m in raw_methods:
        if m not in TRACK_A_DISPATCH_REG:
            print(f"  [VALIDATE] {m}: not in raw dispatch; skipped", flush=True)
            continue
        try:
            sc = TRACK_A_DISPATCH_REG[m](X, Y, cat_indices=None)
        except Exception as e:
            print(f"  [VALIDATE] {m}: SCORER EXCEPTION -> {e}", flush=True)
            failures.append(m); continue
        if sc is None:
            print(f"  [VALIDATE] {m}: returned None", flush=True)
            failures.append(m); continue
        sc = np.nan_to_num(np.asarray(sc, dtype=np.float64), nan=-np.inf)
        ranks = np.argsort(np.argsort(-sc)) + 1            # rank 1..p
        sig_ranks = ranks[tf]
        null_ranks = np.delete(ranks, tf)
        sig_mean = float(sig_ranks.mean())
        null_mean = float(null_ranks.mean())
        thr = min(2 * s, p)
        top2s = set(np.argsort(-sc)[:thr].tolist())
        n_in = len(set(tf) & top2s)

        ok = (sig_mean < null_mean) and (n_in >= int(np.ceil(s / 2)))

        if m in linear_ok_methods:
            ok = ok and set(tf).issubset(set(np.argsort(-sc)[:s].tolist()))
            tag = f"top{s}"
        else:
            tag = f"top{thr},>={int(np.ceil(s/2))}"
        if not ok:
            failures.append(m)
        print(f"  [VALIDATE] {m:<14} signal recovery [{tag}]: {'OK' if ok else 'FAIL'} "
              f"(signal ranks: {sig_ranks.tolist()}, sig mean rank={sig_mean:.0f} "
              f"< null mean rank={null_mean:.0f})", flush=True)


    if 'pearson' in methods:
        try:
            rng = np.random.RandomState(1); n2 = 60
            xs = rng.randn(n2)
            ys = rng.randn(n2) * 2 + 0.5 * xs
            Xs = np.stack([xs, rng.randn(n2)], axis=1)
            sc = TrackAScreeningReg.pearson_accelerated(Xs, ys)
            ref = abs(float(np.corrcoef(xs, ys)[0, 1]))
            diff = abs(sc[0] - ref)
            ok = diff < 1e-6
            print(f"  [VALIDATE] pearson parity vs numpy.corrcoef: "
                  f"{'OK' if ok else 'FAIL'} (diff={diff:.2e})", flush=True)
            if not ok:
                failures.append('pearson_parity')
        except Exception as e:
            print(f"  [VALIDATE] pearson parity FAILED: {e}", flush=True)
            failures.append('pearson_parity')


    if any(m in methods for m in ['hsic', 'nhsic', 'hsic_unbiased']):
        try:
            rng = np.random.RandomState(2); n2 = 40
            xs = rng.randn(n2)
            ys = xs ** 2 + 0.3 * rng.randn(n2)
            Xs = xs.reshape(-1, 1)
            dx = np.abs(xs[:, None] - xs[None, :])
            nz = dx[dx > 0]; sig = np.median(nz) if nz.size else 1.0
            Kx = np.exp(-dx ** 2 / (2 * sig ** 2 + 1e-30))
            dy = np.abs(ys[:, None] - ys[None, :])
            nzy = dy[dy > 0]; sigy = np.median(nzy) if nzy.size else 1.0
            Ly = np.exp(-dy ** 2 / (2 * sigy ** 2 + 1e-30))
            H = np.eye(n2) - 1.0 / n2
            ref = max(0.0, np.sum((H @ Kx @ H) * (H @ Ly @ H)) / (n2 * n2))
            sc = TrackAScreeningReg.hsic_biased_accelerated(Xs.copy(), ys.copy())[0]
            diff = abs(sc - ref)

            ok = diff < max(1e-6, 5e-3 * max(abs(ref), 1e-12))
            print(f"  [VALIDATE] hsic-biased parity vs numpy ref (continuous-y): "
                  f"{'OK' if ok else 'FAIL'} (gpu={sc:.6e} ref={ref:.6e} diff={diff:.2e} "
                  f"rel={diff/max(abs(ref),1e-12):.2e})",
                  flush=True)
            if not ok:
                failures.append('hsic_parity')
            nhsic_sc = TrackAScreeningReg.nhsic_accelerated(Xs.copy(), ys.copy())[0]
            ok2 = (-1e-9 <= nhsic_sc <= 1.0 + 1e-6)
            print(f"  [VALIDATE] nhsic range [0,1]: {'OK' if ok2 else 'FAIL'} (val={nhsic_sc:.6f})",
                  flush=True)
            if not ok2:
                failures.append('nhsic_range')
        except Exception as e:
            print(f"  [VALIDATE] hsic parity FAILED: {e}", flush=True)
            failures.append('hsic_parity')

    print("-" * 70)
    if failures:
        print(f"[VALIDATE] FAILED: {failures}", flush=True)
        print("[VALIDATE] Aborting. Use --skip-validation to override.", flush=True)
        return False
    print("[VALIDATE] All checks passed.", flush=True)
    return True


_A_median_bw_gpu = _median_bandwidth_gpu


_HAS_TORCH_BASELINE = True

class PAceRawA:


    pearson = staticmethod(_RawTrackA.pearson_accelerated)


    @staticmethod
    def _spearman_gpu(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        Xt = _to_tensor(X, dtype=torch.float64)
        n, p = Xt.shape
        RX = _midranks_gpu(Xt)                                  # (n, p) GPU mid-rank
        yt = _to_tensor(y, dtype=torch.float64).reshape(-1)
        Xc = RX - RX.mean(dim=0, keepdim=True)
        Xstd = RX.std(dim=0, unbiased=False) + 1e-30

        uniq = torch.unique(yt)
        if uniq.numel() > 2:
            scores = torch.zeros(p, device=Xt.device, dtype=Xt.dtype)
            for c in uniq:
                binv = (yt == c).to(torch.float64)
                if binv.std() < 1e-12:
                    continue
                RYc = _midranks_gpu(binv.unsqueeze(1)).squeeze(1)   # (n,)
                yc = RYc - RYc.mean()
                ystd = RYc.std(unbiased=False) + 1e-30
                sc = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
                sc = sc.abs()
                sc = sc.masked_fill(Xstd < 1e-12, 0.0)
                scores = torch.maximum(scores, sc)
            return scores.cpu().numpy()

        RY_all = _midranks_gpu(yt.unsqueeze(1))                 # (n, 1)
        yv = RY_all.squeeze(1)
        yc = yv - yv.mean()
        ystd = yv.std(unbiased=False) + 1e-30
        scores = (Xc.T @ yc).squeeze() / (n * Xstd * ystd + 1e-30)
        scores = scores.abs()
        scores = scores.masked_fill(Xstd < 1e-12, 0.0)
        return scores.cpu().numpy()

    spearman = staticmethod(_spearman_gpu.__func__)
    dc = staticmethod(_RawTrackA.dc_biased_accelerated)


    hsic = staticmethod(lambda X, y, **kw: _hsic_gpu(X, y, normalized=False))
    nhsic = staticmethod(lambda X, y, **kw: _hsic_gpu(X, y, normalized=True))


    @staticmethod
    def nmi(X: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
        y = np.asarray(y)
        n, p = X.shape
        uniq = np.unique(y)
        if len(uniq) < 2:
            return np.zeros(p, dtype=np.float64)
        is_cat = (len(uniq) < n / 2)
        Xt = _to_tensor(X, dtype=torch.float64, device=DEVICE)
        Yt, _, _ = _prepare_y_torch(y, n, device=DEVICE, dtype=torch.float64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        scores_t = _gpu_hist_nmi(Xt, Yt, is_cat_y=is_cat)
        return scores_t.cpu().numpy()


# ===========================================================================

# ===========================================================================

class PAceEmbA:


    @staticmethod
    def emb_pearson(embeddings, y, **kw) -> np.ndarray:

        emb = _to_tensor(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
        return PAceEmbA._with_dev(emb.device,
                                  lambda: TorchAcceleratedScreening.emb_sis_accelerated(emb, y))

    @staticmethod
    def emb_dc(embeddings, y, **kw) -> np.ndarray:
        emb = _to_tensor(embeddings) if isinstance(embeddings, np.ndarray) else embeddings
        return PAceEmbA._with_dev(emb.device,
                                  lambda: TorchAcceleratedScreening.emb_dc_accelerated(emb, y))

    @staticmethod
    def emb_hsic(embeddings, y, **kw) -> np.ndarray:


        return TorchAcceleratedScreening.emb_hsic_accelerated(embeddings, y, **kw)

    @staticmethod
    def emb_hsic_medianbw(embeddings, y, **kw) -> np.ndarray:
        emb = _to_tensor(embeddings, dtype=torch.float64) if isinstance(embeddings, np.ndarray) else embeddings
        if emb.dtype != torch.float64:
            emb = emb.to(torch.float64)
        return PAceEmbA._with_dev(emb.device, lambda: PAceEmbA._emb_hsic_medianbw_run(emb, y))

    @staticmethod
    def _emb_hsic_medianbw_run(emb, y):
        n, p, d = emb.shape
        dev = emb.device
        dt = emb.dtype

        Y, is_cat, _ = _prepare_y_torch(y, n, device=dev, dtype=dt)
        Y = Y.to(dev)
        L = _RawTrackA._y_kernel_response(Y, is_cat, n)            # (n,n)
        H = torch.eye(n, device=dev, dtype=dt) - 1.0 / n
        HLH = H @ L @ H                                             # (n,n)
        scores = torch.zeros(p, device=dev, dtype=dt)
        chunk = CHUNK_SIZE
        mbw = _median_bandwidth_gpu
        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            C = end - start
            emb_chunk = emb[:, start:end, :].permute(1, 0, 2)       # (C, n, d) whole-128-d
            dist_x = torch.cdist(emb_chunk, emb_chunk)             # (C, n, n)
            sg = mbw(dist_x)                                        # (C,) per-feature median σ
            sg_safe = torch.where(sg > 0, sg, torch.ones_like(sg))
            Kx = torch.exp(-dist_x ** 2 / (2 * sg_safe.view(C, 1, 1) ** 2 + 1e-30))
            const_mask = sg == 0
            if const_mask.any():
                Kx[const_mask] = 0.0

            HKxH = H @ Kx @ H                                        # (C, n, n)
            hsic_b = (HKxH * HLH.unsqueeze(0)).sum(dim=(1, 2)) / (n * n)
            hsic_b = torch.clamp(hsic_b, min=0)
            hsic_b = hsic_b.masked_fill(const_mask, 0.0)
            scores[start:end] = hsic_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores.cpu().numpy()

    @staticmethod
    def _with_dev(dev, fn, *args, **kw):
        from src import screening_methods_torch as _smt
        orig = _smt.DEVICE
        _smt.DEVICE = dev
        try:
            return fn()
        finally:
            _smt.DEVICE = orig


    @staticmethod
    def emb_spearman(embeddings, y, **kw) -> np.ndarray:

        emb = _to_tensor(embeddings, dtype=torch.float64) if isinstance(embeddings, np.ndarray) else embeddings
        if emb.dtype != torch.float64:
            emb = emb.to(torch.float64)
        def _run():
            n, p, d = emb.shape
            Y, is_cat, _ = _prepare_y_torch(y, n, dtype=torch.float64)
            Y = Y.to(emb.device)
            scores = torch.zeros(p, device=emb.device, dtype=torch.float64)
            for dim in range(d):
                Z = emb[:, :, dim]                               # (n, p)
                R = _midranks_gpu(Z)
                Rc = R - R.mean(dim=0, keepdim=True)
                Rstd = Rc.std(dim=0, unbiased=False) + 1e-30
                if is_cat:
                    Yc = Y - Y.mean(dim=0, keepdim=True)
                    Ystd = Y.std(dim=0, unbiased=False) + 1e-30
                    corr = (Rc.T.double() @ Yc.double()) / (n * Rstd.unsqueeze(1) * Ystd.unsqueeze(0) + 1e-30)
                    corr = corr.abs()
                    mask = (Rstd.unsqueeze(1) < 1e-12) | (Ystd.unsqueeze(0) < 1e-12)
                    corr = corr.masked_fill(mask, 0.0)

                    scores = torch.maximum(scores, corr.max(dim=1).values)
                else:
                    yv = Y.squeeze(1).double()
                    yc = yv - yv.mean()
                    ystd = yv.std(unbiased=False) + 1e-30
                    sc = (Rc.T.double() @ yc) / (n * Rstd * ystd + 1e-30)
                    sc = sc.abs()
                    sc = sc.masked_fill(Rstd < 1e-12, 0.0)
                    scores += sc.double()
            scores /= d
            return scores.cpu().numpy()
        return PAceEmbA._with_dev(emb.device, _run)


    @staticmethod
    def emb_nhsic(embeddings, y, **kw) -> np.ndarray:
        return PAceEmbA.emb_hsic_medianbw(embeddings, y)


    @staticmethod
    def emb_nmi(embeddings, y, **kw) -> np.ndarray:
        emb = _to_tensor(embeddings, dtype=torch.float64) if isinstance(embeddings, np.ndarray) else embeddings
        if emb.dtype != torch.float64:
            emb = emb.to(torch.float64)
        n, p, d = emb.shape
        y = np.asarray(y)
        uniq = np.unique(y)
        if len(uniq) < 2:
            return np.zeros(p, dtype=np.float64)
        is_cat = (len(uniq) < n / 2)
        Yt, _, _ = _prepare_y_torch(y, n, device=emb.device, dtype=torch.float64)
        Yt = Yt.to(emb.device)


        emb_flat = emb.reshape(n, p * d).contiguous()
        nmi_flat = _gpu_hist_nmi(emb_flat, Yt, is_cat_y=is_cat)   # (p*d,)
        nmi_pd = nmi_flat.reshape(p, d).mean(dim=1)
        return nmi_pd.cpu().numpy()


# ===========================================================================

# ===========================================================================

def get_merged_embeddings(embedder: TorchAcceleratedEmbedder,
                          X: np.ndarray,
                          layers: List[int] = (3,),
                          batch_size: int = 32,
                          device=None) -> torch.Tensor:
    if device is None:
        device = getattr(embedder, 'device', DEVICE)
    if not embedder.loaded:
        raise RuntimeError("embedder not loaded")
    res = embedder.get_embeddings_batch_all_layers(X, layers=list(layers),
                                                    batch_size=batch_size)
    merged = None
    for l in layers:
        t = res[l].to(device)
        merged = t if merged is None else (merged + t)
    merged = merged / max(1, len(layers))
    return merged


def get_embeddings_incremental(embedder, X, layer_idx=3, feat_chunk=5000,
                              device=None):
    n, p = X.shape
    if device is None:
        device = DEVICE
    if not embedder.loaded:
        return None

    raise NotImplementedError("Use the experiment runner for chunked ultrawide embeddings")


# ===========================================================================

# ===========================================================================

PACE_TRACK_A_RAW = {
    'pearson':  PAceRawA.pearson,
    'spearman': PAceRawA.spearman,
    'dc':       PAceRawA.dc,
    'hsic':     PAceRawA.hsic,
    'nhsic':    PAceRawA.nhsic,
    'nmi':      PAceRawA.nmi,
}


PACE_TRACK_A_EMB_BASE = {
    'pearson':  PAceEmbA.emb_pearson,
    'spearman': PAceEmbA.emb_spearman,
    'dc':       PAceEmbA.emb_dc,
    'hsic':     PAceEmbA.emb_hsic,
    'nhsic':    PAceEmbA.emb_nhsic,
    'nmi':      PAceEmbA.emb_nmi,
}


PACE_TRACK_A_EMB = {
    f'emb_{k}': v for k, v in PACE_TRACK_A_EMB_BASE.items()
}


def compute_paceA_raw_scores(X, y, methods=None) -> Dict[str, np.ndarray]:
    if methods is None:
        methods = list(PACE_TRACK_A_RAW.keys())
    out = {}
    for m in methods:
        try:
            sc = PACE_TRACK_A_RAW[m](X, y)
            if sc is not None and len(sc) == X.shape[1] and not np.isnan(sc).all():
                out[m] = np.asarray(sc, dtype=np.float64)
        except Exception as e:
            print(f"  [PACE-A] raw {m} failed: {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def compute_paceA_emb_scores(emb_tensor, y, methods=None) -> Dict[str, np.ndarray]:
    if methods is None:
        methods = list(PACE_TRACK_A_EMB.keys())
    out = {}
    for m in methods:
        try:
            sc = PACE_TRACK_A_EMB[m](emb_tensor, y)
            if sc is not None and len(sc) == (emb_tensor.shape[1] if hasattr(emb_tensor, 'shape') else 0) \
                    and not np.isnan(np.asarray(sc, dtype=np.float64)).all():
                out[m] = np.asarray(sc, dtype=np.float64)
        except Exception as e:
            print(f"  [PACE-A] emb {m} failed: {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def compute_paceA_emb_scores_multilayer(embedder, X, y,
                                        emb_layers=(1, 2, 3),
                                        methods=None,
                                        batch_size: int = 64,
                                        device=None) -> Dict[str, np.ndarray]:
    if not getattr(embedder, 'loaded', False):
        return {}
    if methods is None:
        methods = list(PACE_TRACK_A_EMB_BASE.keys())
    n, p = X.shape
    layers = list(emb_layers)
    n_layers = len(layers)

    out: Dict[str, np.ndarray] = {}
    try:
        all_emb = embedder.get_embeddings_batch_all_layers(X, layers=layers,
                                                           batch_size=batch_size)
    except Exception as e:
        print(f"  [PACE-A] multi-layer embed extraction failed: {e}", flush=True)
        return out
    for layer in layers:
        emb_t = all_emb.get(layer)
        if emb_t is None:
            continue
        for m in methods:
            key = f"emb_{m}_L{layer}"
            try:
                sc = PACE_TRACK_A_EMB_BASE[m](emb_t, y)
                if sc is not None:
                    sc = np.asarray(sc, dtype=np.float64)
                    if sc.shape and sc.shape[0] == p and not np.isnan(sc).all():
                        out[key] = sc
            except Exception as e:
                print(f"  [PACE-A] emb {m}_L{layer} failed: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_emb[layer] = None
    del all_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def compute_paceA_emb_scores_incremental(embedder, X, y,
                                          emb_layers=(1, 2, 3),
                                          methods=None,
                                          feat_chunk: int = 4096,
                                          batch_size: int = 32,
                                          device=None) -> Dict[str, np.ndarray]:
    if not getattr(embedder, 'loaded', False):
        return {}
    if methods is None:
        methods = list(PACE_TRACK_A_EMB_BASE.keys())
    n, p = X.shape
    layers = list(emb_layers)
    out: Dict[str, np.ndarray] = {}


    try:
        dev = device if device is not None else getattr(embedder, 'device', DEVICE)
        avail = _get_available_gpu_mem(dev) * 0.5
    except Exception:
        avail = 8 * 1024 ** 3
    embed_dim = getattr(embedder, 'embed_dim', 128)
    auto_chunk = max(256, int(avail / max(n * embed_dim * 8, 1)))
    feat_chunk = min(feat_chunk, auto_chunk, EMB_FEAT_CHUNK, p)
    print(f"  [PACE-A-incr] p={p}, feat_chunk={feat_chunk}, layers={layers}, "
          f"methods={methods}", flush=True)

    for layer in layers:
        layer_scores: Dict[str, np.ndarray] = {m: np.zeros(p, dtype=np.float64) for m in methods}
        feat_start = 0
        while feat_start < p:
            feat_end = min(feat_start + feat_chunk, p)
            chunk_p = feat_end - feat_start
            try:
                chunk_emb = embedder.get_embeddings_for_range(
                    X, layer_idx=layer, feat_start=feat_start,
                    feat_end=feat_end, batch_size=batch_size)
            except Exception as e:
                print(f"  [PACE-A] incremental L{layer} [{feat_start}:{feat_end}] extraction failed: {e}", flush=True)
                feat_start = feat_end
                continue
            if chunk_emb is None:
                feat_start = feat_end
                continue
            for m in methods:
                try:
                    sc = PACE_TRACK_A_EMB_BASE[m](chunk_emb, y)
                    if sc is not None:
                        sc = np.asarray(sc, dtype=np.float64)
                        if sc.shape and sc.shape[0] >= chunk_p and not np.isnan(sc).all():
                            layer_scores[m][feat_start:feat_end] = sc[:chunk_p]
                except Exception as e:
                    print(f"  [PACE-A] incremental emb {m}_L{layer} failed: {e}", flush=True)
            del chunk_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            feat_start = feat_end
        for m in methods:
            key = f"emb_{m}_L{layer}"
            out[key] = layer_scores[m]
    return out



COORDINATE_AGGREGATION_METHODS = frozenset(
    ('sis', 'fkf', 'cru', 'cavs', 'mv_sis', 'lr_ffs', 'psis', 'hd', 'ids'))


def compute_scores(X, y, method, *, task='bin', device=None, aggregation='native',
                   embeddings=None, embedder=None, model_root=None, model_path=None,
                   layer=2, feature_chunk=256, batch_size=64, max_screen_samples=None,
                   impute=True, seed=None, backend='torch', **options):
    """Compute raw or PACE utility for each input feature."""
    from src.screening_utils import impute_mean, sample_rows, validate_scores
    global DEVICE
    if device is not None:
        DEVICE = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
    if backend not in ('torch', 'numpy'):
        raise ValueError('backend must be torch or numpy')
    if method.startswith('raw_a_'):
        base = method[6:]
        dispatch = TRACK_A_DISPATCH_REG if task == 'reg' else TRACK_A_DISPATCH
        return validate_scores(dispatch[base](X, y), X.shape[1], method)
    if not method.startswith('pace_'):
        if aggregation != 'native':
            raise ValueError('aggregation applies to PACE coordinate utilities only')
        if impute:
            X = impute_mean(X)
        X, y = sample_rows(X, y, task, max_screen_samples)
        if method == 'hsic':
            return TRACK_A_DISPATCH['hsic'](X, y)
        cls = TorchAcceleratedScreening
        if backend == 'numpy':
            from src.screening_methods import AcceleratedScreening as cls
        result = cls.compute_all_scores(X, y, methods=[method])
        return validate_scores(result.get(method), X.shape[1], method)

    base = method[5:]
    track_a = base.startswith('a_')
    if track_a:
        base = base[2:]
    coordinate = (base in ('pearson', 'spearman', 'nmi') if track_a
                  else base in COORDINATE_AGGREGATION_METHODS)
    if aggregation != 'native' and not coordinate:
        raise ValueError(f'{method} uses the whole embedding vector; use aggregation="native"')

    def reduce_embedding(E):
        if track_a:
            fn = PACE_TRACK_A_EMB_BASE[base]
        else:
            cls = TorchAcceleratedScreening
            if backend == 'numpy':
                from src.screening_methods import AcceleratedScreening as cls
            fn = getattr(cls, 'emb_' + base + '_accelerated', None)
            if fn is None:
                fn = lambda tensor, target: cls.emb_compute_all_scores(
                    tensor, target, methods=[base]).get('emb_' + base)
        if base == 'hsic':
            return validate_scores(fn(E, y, task=task), E.shape[1], method)
        if aggregation != 'max':
            return validate_scores(fn(E, y), E.shape[1], method)
        maximum = None
        for dim in range(E.shape[2]):
            value = validate_scores(fn(E[:, :, dim:dim+1], y), E.shape[1], method)
            maximum = value if maximum is None else np.maximum(maximum, value)
        return maximum

    if embeddings is not None:
        if embeddings.shape[:2] != X.shape:
            raise ValueError('Embedding row/feature dimensions must match X')
        return reduce_embedding(embeddings)
    X, y = sample_rows(X, y, task, max_screen_samples)
    if impute:
        X = impute_mean(X)
    if embedder is None:
        checkpoint = Path(model_path) if model_path else mp.checkpoint('tabicl_v11', model_root)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        embedder = TorchAcceleratedEmbedder(device=str(DEVICE))
        if not embedder.load_model(str(checkpoint), seed=42, device=str(DEVICE)):
            raise RuntimeError('Could not load the PACE encoder')
    if not embedder.loaded:
        raise RuntimeError('PACE encoder is not loaded')
    if feature_chunk < 1:
        raise ValueError('feature_chunk must be positive')
    pieces = []
    for start in range(0, X.shape[1], feature_chunk):
        end = min(start+feature_chunk, X.shape[1])
        E = embedder.get_embeddings_for_range(X, layer, start, end, batch_size=batch_size)
        pieces.append(reduce_embedding(E))
        del E
    return np.concatenate(pieces)
