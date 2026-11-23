from src import model_path as mp
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy import stats
from typing import List, Tuple, Dict, Optional


try:
    from joblib import Parallel, delayed
    from multiprocessing import cpu_count
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TABICL_PATH = mp.TABICL_PATH


# =============================================================================

# =============================================================================

def _compute_A_matrix(z: np.ndarray, sigma: float) -> np.ndarray:
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    zz = z @ z.T
    z_sq = np.sum(z ** 2, axis=1)
    num = sigma ** 2 + zz
    denom = np.sqrt((sigma ** 2 + z_sq[:, np.newaxis]) *
                    (sigma ** 2 + z_sq[np.newaxis, :]))
    cos_val = np.clip(num / (denom + 1e-10), -1, 1)
    return np.arccos(cos_val)


class CorrelationMeasures:

    @staticmethod
    def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        if n < 2:
            return 0.0

        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y


        a = cdist(x, x, 'euclidean')
        b = cdist(y, y, 'euclidean')


        a_centered = a - a.mean(-1, keepdims=True) - a.mean(-2, keepdims=True) + a.mean(axis=(-2, -1), keepdims=True)
        b_centered = b - b.mean(-1, keepdims=True) - b.mean(-2, keepdims=True) + b.mean(axis=(-2, -1), keepdims=True)


        dcov2 = (a_centered * b_centered).sum() / (n * n)
        dvar_x2 = (a_centered ** 2).sum() / (n * n)
        dvar_y2 = (b_centered ** 2).sum() / (n * n)

        if dvar_x2 * dvar_y2 <= 0:
            return 0.0

        return np.sqrt(abs(dcov2) / np.sqrt(dvar_x2 * dvar_y2))

    @staticmethod
    def projection_correlation(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
        n = len(x)
        if n < 4:
            return 0.0

        try:
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-10)
            y_scaled = (y - y.mean()) / (y.std() + 1e-10)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            def compute_A_matrix(z, sigma):
                if z.ndim == 1:
                    z = z.reshape(-1, 1)
                zz = z @ z.T
                z_sq = np.sum(z ** 2, axis=1)
                num = sigma ** 2 + zz
                denom = np.sqrt((sigma ** 2 + z_sq[:, np.newaxis]) *
                               (sigma ** 2 + z_sq[np.newaxis, :]))
                cos_val = np.clip(num / (denom + 1e-10), -1, 1)
                return np.arccos(cos_val)

            Ax = compute_A_matrix(x, sigma)
            Ay = compute_A_matrix(y_scaled.reshape(-1, 1), sigma)

            n_pairs = n * (n - 1)
            term1 = (Ax * Ay).sum() / n_pairs
            term2 = 2 * (Ax.mean(axis=1) * Ay.mean(axis=1)).mean()
            term3 = Ax.mean() * Ay.mean()

            pcov_xy = term1 - term2 + term3
            pcov_xx = (Ax ** 2).sum() / n_pairs - 2 * (Ax.mean(axis=1) ** 2).mean() + Ax.mean() ** 2
            pcov_yy = (Ay ** 2).sum() / n_pairs - 2 * (Ay.mean(axis=1) ** 2).mean() + Ay.mean() ** 2

            if pcov_xx < 1e-10 or pcov_yy < 1e-10:
                return 0.0

            pc_sq = pcov_xy / np.sqrt(pcov_xx * pcov_yy)
            return max(0, min(1, pc_sq))
        except:
            return 0.0

    @staticmethod
    def improved_projection_correlation(x: np.ndarray, y: np.ndarray, sigma: float = None) -> float:
        n = len(x)
        if n < 4:
            return 0.0

        try:
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-10)
            y_scaled = (y - y.mean()) / (y.std() + 1e-10)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            def compute_G_matrix(z, sigma_z):
                if z.ndim == 1:
                    z = z.reshape(-1, 1)
                z_sq = np.sum(z ** 2, axis=1)
                z_norm = np.sqrt(sigma_z ** 2 + z_sq)
                return z / (z_norm[:, np.newaxis] * np.sqrt(sigma_z ** 2))


            if sigma is None:
                sigma_x = np.sqrt(np.median(np.sum(x ** 2, axis=1)))
                sigma_y = np.sqrt(np.median(np.sum(y_scaled.reshape(-1, 1) ** 2, axis=1)))
            else:
                sigma_x = sigma
                sigma_y = sigma

            Gx = compute_G_matrix(x, sigma_x)
            Gy = compute_G_matrix(y_scaled.reshape(-1, 1), sigma_y)

            M = Gx @ Gx.T
            N = Gy @ Gy.T

            H = np.eye(n) - np.ones((n, n)) / n
            MH = M @ H
            NH = N @ H

            ipc = np.trace(MH @ NH) / np.sqrt(np.trace(MH @ MH) * np.trace(NH @ NH))
            return max(0, min(1, ipc))
        except:
            return 0.0

    @staticmethod
    def hsic(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
        n = len(x)
        if n < 4:
            return 0.0

        try:
            def rbf_kernel(a, b, sigma):
                if a.ndim == 1:
                    a = a.reshape(-1, 1)
                if b.ndim == 1:
                    b = b.reshape(-1, 1)
                dist = cdist(a, b, 'euclidean')
                return np.exp(-dist ** 2 / (2 * sigma ** 2))

            Kx = rbf_kernel(x, x, sigma)
            Ky = rbf_kernel(y.reshape(-1, 1), y.reshape(-1, 1), sigma)

            H = np.eye(n) - np.ones((n, n)) / n
            Kx_centered = H @ Kx @ H
            Ky_centered = H @ Ky @ H

            hsic_val = np.trace(Kx_centered @ Ky_centered) / (n ** 2)
            return max(0, hsic_val)
        except:
            return 0.0

    @staticmethod
    def stable_correlation(x: np.ndarray, y: np.ndarray, a: float = 0.5) -> float:
        n = len(x)
        if n < 3:
            return 0.0

        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        try:

            dist_x = cdist(x, x, 'euclidean')
            dist_y = cdist(y, y, 'euclidean')


            Kx = np.exp(-np.power(dist_x, a))
            Ky = np.exp(-np.power(dist_y, a))


            mask = ~np.eye(n, dtype=bool)

            # E1 = 1/[n(n-1)] * sum_{i neq j} Kx[i,j] * Ky[i,j]
            E1 = (Kx * Ky)[mask].sum() / (n * (n - 1))

            # E2
            E2_x = Kx[mask].sum() / (n * (n - 1))
            E2_y = Ky[mask].sum() / (n * (n - 1))
            E2 = E2_x * E2_y

            # E3 = 1/[n(n-1)(n-2)] * sum_i sum_{j neq i} sum_{l neq i, j neq l} Kx[i,j] * Ky[i,l]
            Kx_off = Kx.copy()
            Ky_off = Ky.copy()
            np.fill_diagonal(Kx_off, 0)
            np.fill_diagonal(Ky_off, 0)

            # E3 = 1/[n(n-1)(n-2)] * sum_i [ (sum_j Kx_off[i,j]) * (sum_l Ky_off[i,l]) - sum_j Kx[i,j]*Ky[i,j] (j neq i) ]
            row_sum_Kx = Kx_off.sum(axis=1)
            row_sum_Ky = Ky_off.sum(axis=1)
            row_sum_Kxy = (Kx * Ky)[mask].reshape(n, n-1).sum(axis=1)

            E3 = (row_sum_Kx * row_sum_Ky - row_sum_Kxy).sum() / (n * (n-1) * (n-2))


            Scov2 = E1 + E2 - 2 * E3


            Kxx = np.exp(-np.power(dist_x, a) * 2)
            np.fill_diagonal(Kxx, 0)
            E1_xx = Kxx.sum() / (n * (n - 1))
            E2_xx = E2_x ** 2
            # E3_xx
            row_sum_Kxx = Kxx.sum(axis=1)
            E3_xx = (row_sum_Kx ** 2 - row_sum_Kxx).sum() / (n * (n-1) * (n-2))
            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx


            Kyy = np.exp(-np.power(dist_y, a) * 2)
            np.fill_diagonal(Kyy, 0)
            E1_yy = Kyy.sum() / (n * (n - 1))
            E2_yy = E2_y ** 2
            row_sum_Kyy = Kyy.sum(axis=1)
            E3_yy = (row_sum_Ky ** 2 - row_sum_Kyy).sum() / (n * (n-1) * (n-2))
            Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

            if Svar_x2 <= 0 or Svar_y2 <= 0:
                return 0.0

            SC2 = Scov2 / np.sqrt(Svar_x2 * Svar_y2)
            return max(0.0, min(1.0, np.sqrt(abs(SC2))))

        except:
            return 0.0

    @staticmethod
    def _uniform2sphere_qmc(vec: np.ndarray) -> np.ndarray:
        from scipy.stats import beta as beta_dist

        d = len(vec) + 1
        vec_temp = vec.copy()
        vec_temp[0] = 2 * np.pi * vec_temp[0]

        x_temp = np.array([np.cos(vec_temp[0]), np.sin(vec_temp[0])])

        if d == 1:
            return x_temp

        for i in range(2, d):
            y = vec_temp[i - 1]
            if y > 0.5:
                y = np.sqrt(beta_dist.ppf(2 * (y - 0.5), 0.5, i / 2))
            elif y < 0.5:
                y = -np.sqrt(beta_dist.ppf(2 * (0.5 - y), 0.5, i / 2))
            else:
                y = 0.0
            x_temp = np.concatenate([[y], np.sqrt(1 - y ** 2) * x_temp])

        return x_temp

    @staticmethod
    def _uniform2sphere_mc(n: int, d: int, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        vec_temp = rng.randn(n, d)
        norms = np.sqrt(np.sum(vec_temp ** 2, axis=1, keepdims=True))
        return vec_temp / norms

    @staticmethod
    def sliced_wasserstein_distance(xs: np.ndarray, ys: np.ndarray,
                                     L: int = 20, p: int = 1,
                                     qmc: bool = True, seed: int = 42) -> float:
        n, d = xs.shape

        if qmc:
            try:
                from scipy.stats.qmc import Sobol
                sobol_engine = Sobol(d=d - 1, scramble=True, seed=seed)
                vec_meta = sobol_engine.random(L * d)
                directions = np.array([
                    CorrelationMeasures._uniform2sphere_qmc(vec_meta[i])
                    for i in range(L * d)
                ])
            except ImportError:
                qmc = False

        if not qmc:
            directions = CorrelationMeasures._uniform2sphere_mc(L * d, d, seed)

        total = 0.0
        for i in range(L * d):
            dvec = directions[i]
            xt = xs @ dvec
            yt = ys @ dvec
            if p == 1:
                total += np.mean(np.abs(np.sort(xt) - np.sort(yt)))
            else:
                total += np.mean(np.abs(np.sort(xt) - np.sort(yt)) ** p)

        return (total / (L * d)) ** (1 / p)

    @staticmethod
    def sliced_wasserstein_dependency(v1: np.ndarray, v2: np.ndarray,
                                       L: int = 20, p: int = 1,
                                       qmc: bool = True, seed: int = 42) -> float:
        n = len(v1)
        v1 = v1.reshape(n, -1) if v1.ndim == 1 else v1
        v2 = v2.reshape(n, -1) if v2.ndim == 1 else v2

        rng = np.random.RandomState(seed)
        shuf = rng.permutation(n)

        xy = np.hstack([v1, v2])
        xys = np.hstack([v1[shuf], v2])

        return CorrelationMeasures.sliced_wasserstein_distance(xy, xys, L, p, qmc, seed)

    @staticmethod
    def sliced_wasserstein_correlation(x: np.ndarray, y: np.ndarray,
                                        L: int = 20, p: int = 1,
                                        qmc: bool = True, seed: int = 42) -> float:
        n = len(x)
        if n < 4:
            return 0.0

        try:
            x = x.reshape(n, -1) if x.ndim == 1 else x
            y = y.reshape(n, -1) if y.ndim == 1 else y

            swd_xy = CorrelationMeasures.sliced_wasserstein_dependency(x, y, L, p, qmc, seed)
            swd_xx = CorrelationMeasures.sliced_wasserstein_dependency(x, x, L, p, qmc, seed)
            swd_yy = CorrelationMeasures.sliced_wasserstein_dependency(y, y, L, p, qmc, seed)

            denom = np.sqrt(swd_xx * swd_yy)
            if denom < 1e-15:
                return 0.0

            return swd_xy / denom
        except:
            return 0.0

    @staticmethod
    def hellinger_distance_index(x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        if n < 4:
            return 0.0

        try:
            classes = np.unique(y)
            R = len(classes)
            if R < 2:
                return 0.0


            sorted_idx = np.argsort(x)
            x_sorted = x[sorted_idx]


            ranks = np.zeros(n, dtype=int)
            ranks[sorted_idx] = np.arange(1, n + 1)
            F_hat = ranks / n  # ˆF(x_i) for each sample

            hd_total = 0.0
            for r in classes:
                mask_r = (y == r)
                n_r = mask_r.sum()
                if n_r == 0:
                    continue
                p_r = n_r / n


                y_at_sorted = y[sorted_idx]
                cum_count_r = np.cumsum(y_at_sorted == r)
                # F_r(x_sorted[k]) = cum_count_r[k] / n_r

                F_r_hat = np.zeros(n)
                F_r_hat[sorted_idx] = cum_count_r / n_r


                # dF(x) ≈ 1/n at each observation
                sqrt_Fr = np.sqrt(np.maximum(F_r_hat, 0))
                sqrt_F = np.sqrt(np.maximum(F_hat, 0))
                sqrt_1_Fr = np.sqrt(np.maximum(1 - F_r_hat, 0))
                sqrt_1_F = np.sqrt(np.maximum(1 - F_hat, 0))

                integrand = (sqrt_Fr - sqrt_F) ** 2 + (sqrt_1_Fr - sqrt_1_F) ** 2
                hd_total += p_r * np.mean(integrand)

            return hd_total
        except:
            return 0.0

    @staticmethod
    def avg_hellinger_distance_index(x: np.ndarray, y: np.ndarray,
                                      K: int = None, n_slices: int = None) -> float:
        n = len(x)
        if n < 8:
            return 0.0

        try:

            unique_vals = np.unique(y)
            is_categorical = len(unique_vals) < n / 2 or not np.issubdtype(y.dtype, np.number)

            if is_categorical:

                return CorrelationMeasures.hellinger_distance_index(x, y)


            if n_slices is not None:

                R = n_slices
                quantiles = np.quantile(y, np.arange(1, R) / R)
                y_sliced = np.digitize(y, quantiles)  # 0 to R-1
                return CorrelationMeasures.hellinger_distance_index(x, y_sliced)

            if K is None:
                K = max(1, int(np.ceil(np.log(n))) - 2)

            ahd_total = 0.0
            for k in range(1, K + 1):
                R_k = k + 3
                quantiles = np.quantile(y, np.arange(1, R_k) / R_k)
                y_sliced = np.digitize(y, quantiles)
                ahd_total += CorrelationMeasures.hellinger_distance_index(x, y_sliced)

            return ahd_total / K
        except:
            return 0.0

    @staticmethod
    def grothendieck_correlation(x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        if n < 4:
            return 0.0

        try:
            x = x.reshape(n, -1) if x.ndim == 1 else x
            y = y.reshape(n, -1) if y.ndim == 1 else y

            def compute_kappa_matrix(Z):
                # z_i^T z_j
                dot_ZZ = Z @ Z.T
                # ||z_i||²
                norm_sq = np.sum(Z ** 2, axis=1)
                # (1 + z_i^T z_j) / sqrt((1 + ||z_i||²)(1 + ||z_j||²))
                num = 1 + dot_ZZ
                denom = np.sqrt((1 + norm_sq[:, np.newaxis]) * (1 + norm_sq[np.newaxis, :]))
                cos_val = np.clip(num / (denom + 1e-15), -1, 1)
                return np.arcsin(cos_val)

            Kx = compute_kappa_matrix(x)
            Ky = compute_kappa_matrix(y)

            # GCov²(X,Y) = 1/(n(n-3)) Σ_{i≠j} (ã_ij)(b̃_ij)
            # ã_ij = a_ij - a_i. - a_.j + a_.., b̃_ij = b_ij - b_i. - b_.j + b_..
            def double_center(K_matrix):
                n_sq = K_matrix.shape[0]
                K_off = K_matrix.copy()
                np.fill_diagonal(K_off, 0)
                row_sums = K_off.sum(axis=1)
                col_sums = K_off.sum(axis=0)
                total_sum = K_off.sum()
                a_row = row_sums / (n_sq - 2)
                a_col = col_sums / (n_sq - 2)
                a_total = total_sum / ((n_sq - 1) * (n_sq - 2))
                K_centered = K_matrix - a_row[:, np.newaxis] - a_col[np.newaxis, :] + a_total
                np.fill_diagonal(K_centered, 0)
                return K_centered

            Kx_c = double_center(Kx)
            Ky_c = double_center(Ky)

            gcov2_xy = np.sum(Kx_c * Ky_c) / (n * (n - 3))
            gcov2_xx = np.sum(Kx_c * Kx_c) / (n * (n - 3))
            gcov2_yy = np.sum(Ky_c * Ky_c) / (n * (n - 3))

            if gcov2_xx * gcov2_yy < 1e-15:
                return 0.0

            gcor2 = gcov2_xy / np.sqrt(gcov2_xx * gcov2_yy)
            return max(0.0, min(1.0, np.sqrt(abs(gcor2))))
        except:
            return 0.0


# =============================================================================

# =============================================================================

class TraditionalScreening:

    @staticmethod
    def sis(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.sis_accelerated(X, y, n_jobs=n_jobs)

    @staticmethod
    def dc_sis(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.dc_sis_accelerated(X, y, n_jobs=n_jobs)

    @staticmethod
    def pc_screen(X: np.ndarray, y: np.ndarray, sigma: float = 1.0, n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.pc_sis_accelerated(X, y, sigma, n_jobs=n_jobs)

    @staticmethod
    def ipc_screen(X: np.ndarray, y: np.ndarray, sigma: float = None, n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.ipc_sis_accelerated(X, y, sigma, n_jobs=n_jobs)

    @staticmethod
    def hsic_screen(X: np.ndarray, y: np.ndarray, sigma: float = 1.0,
                    n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)

        scores = np.zeros(p)
        for j in range(p):
            scores[j] = CorrelationMeasures.hsic(X[:, j].reshape(-1, 1), y_prepared, sigma)
        return scores

    @staticmethod
    def sc_screen(X: np.ndarray, y: np.ndarray, a: float = 0.5,
                  fast: bool = True, n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.sc_sis_accelerated(X, y, a, n_jobs=n_jobs)

    @staticmethod
    def swc_screen(X: np.ndarray, y: np.ndarray, L: int = 20, p: int = 1,
                   n_shuf: int = 11, qmc: bool = True, seed: int = 42,
                   n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.swc_accelerated(X, y, L, p, n_shuf, qmc, seed, n_jobs=n_jobs)

    @staticmethod
    def hd_sis(X: np.ndarray, y: np.ndarray, K: int = None,
               n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.hd_sis_accelerated(X, y, K, n_jobs=n_jobs)

    @staticmethod
    def gcor_sis(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        return AcceleratedScreening.gcor_sis_accelerated(X, y, n_jobs=n_jobs)


# =============================================================================

# =============================================================================

class CLSEmbedder:

    def __init__(self, model_path: str = TABICL_PATH, device: torch.device = DEVICE):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tabicl = None
        self.col_embedder = None
        self.row_interactor = None
        self.embed_dim = 128
        self.num_cls = 4
        self.loaded = False

    def load(self) -> bool:
        if self.loaded:
            return True

        try:
            from tabicl import TabICLClassifier

            self.model = TabICLClassifier(
                model_path=self.model_path,
                n_estimators=1,
                random_state=42,
                device=self.device
            )
            self.model.fit([[0, 1], [1, 0]], [0, 1])

            self.tabicl = self.model.model_
            self.col_embedder = self.tabicl.col_embedder
            self.row_interactor = self.tabicl.row_interactor
            self.embed_dim = self.tabicl.embed_dim
            self.num_cls = self.tabicl.row_num_cls
            self.loaded = True

            print(f"TabICL loaded: embed_dim={self.embed_dim}, num_cls={self.num_cls}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def get_col_embedding_single_feature(self, x_1d: np.ndarray) -> np.ndarray:
        if not self.loaded:
            if not self.load():
                return None

        n = len(x_1d)


        sorted_idx = np.argsort(x_1d)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(n)
        normalized = (ranks + 0.5) / n

        x_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            self.col_embedder.eval()
            x_proj = self.col_embedder.in_linear(x_tensor)

            if hasattr(self.col_embedder, 'cls_tokens'):
                cls_tokens = self.col_embedder.cls_tokens.expand(1, n, -1, -1)
                x_with_cls = torch.cat([cls_tokens, x_proj], dim=2)
            else:
                x_with_cls = x_proj

            for block in self.col_embedder.tf_col.blocks:
                x_with_cls = block(x_with_cls)

            if hasattr(self.col_embedder, 'cls_tokens'):
                emb = x_with_cls[:, :, 1:, :]
            else:
                emb = x_with_cls

            if hasattr(self.col_embedder, 'out_ln'):
                emb = self.col_embedder.out_ln(emb)

        return emb.squeeze(2).squeeze(0).cpu().numpy()

    def get_cls_embedding_for_group(self, X_group: np.ndarray) -> np.ndarray:
        if not self.loaded:
            if not self.load():
                return None

        n, k = X_group.shape


        col_emb = np.zeros((n, k, self.embed_dim))
        for j in range(k):
            col_emb[:, j, :] = self.get_col_embedding_single_feature(X_group[:, j])


        with torch.no_grad():
            self.row_interactor.eval()
            col_tensor = torch.from_numpy(col_emb).float().unsqueeze(0).to(self.device)

            B, T, H, E = col_tensor.shape
            cls_tokens = self.row_interactor.cls_tokens.expand(B, T, self.num_cls, E).to(self.device)
            full_emb = torch.cat([cls_tokens, col_tensor], dim=2)

            row_outputs = self.row_interactor.tf_row(full_emb)

            if self.row_interactor.norm_first:
                row_outputs = self.row_interactor.out_ln(row_outputs)

            cls_emb = row_outputs[:, :, :self.num_cls, :].flatten(-2)

        return cls_emb.squeeze(0).cpu().numpy()

    def get_all_feature_embeddings(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        embeddings = np.zeros((n, p, self.embed_dim))

        for j in range(p):
            embeddings[:, j, :] = self.get_col_embedding_single_feature(X[:, j])

        return embeddings


# =============================================================================

# =============================================================================

class GroupScreening:

    @staticmethod
    def _combine_embeddings(embeddings: np.ndarray, group: Tuple,
                            strategy: str = 'hadamard') -> np.ndarray:
        n = embeddings.shape[0]
        d = embeddings.shape[2]

        if strategy == 'add':
            result = np.zeros((n, d))
            for idx in group:
                result += embeddings[:, idx, :]
            return result
        elif strategy == 'hadamard':
            result = np.ones((n, d))
            for idx in group:
                result *= embeddings[:, idx, :]
            return result
        else:  # concat
            return np.concatenate([embeddings[:, idx, :] for idx in group], axis=1)

    @staticmethod
    def compute_raw_scores(X: np.ndarray, y: np.ndarray, groups: List[Tuple],
                           metric: str = 'dc', a: float = 0.5) -> np.ndarray:
        n_groups = len(groups)
        scores = np.zeros(n_groups)

        if metric == 'sc':

            for i, group in enumerate(groups):
                X_group = X[:, list(group)]
                scores[i] = CorrelationMeasures.stable_correlation(X_group, y, a)
        else:
            metric_fn = {
                'dc': CorrelationMeasures.distance_correlation,
                'pc': CorrelationMeasures.projection_correlation,
                'ipc': CorrelationMeasures.improved_projection_correlation,
                'hsic': CorrelationMeasures.hsic
            }[metric]

            for i, group in enumerate(groups):
                X_group = X[:, list(group)]
                scores[i] = metric_fn(X_group, y)

        return scores

    @staticmethod
    def compute_cls_scores(embedder: CLSEmbedder, X: np.ndarray, y: np.ndarray,
                           groups: List[Tuple], metric: str = 'dc') -> np.ndarray:
        n_groups = len(groups)
        scores = np.zeros(n_groups)

        metric_fn = {
            'dc': CorrelationMeasures.distance_correlation,
            'pc': CorrelationMeasures.projection_correlation,
            'ipc': CorrelationMeasures.improved_projection_correlation,
            'sc': lambda x, y: CorrelationMeasures.stable_correlation(x, y, a=0.5)
        }[metric]

        for i, group in enumerate(groups):
            X_group = X[:, list(group)]
            cls_emb = embedder.get_cls_embedding_for_group(X_group)
            scores[i] = metric_fn(cls_emb, y)

        return scores

# =============================================================================

# =============================================================================

class ScreeningEvaluator:

    @staticmethod
    def compute_mms(scores: np.ndarray, true_features: List[int]) -> int:
        if scores is None or len(scores) == 0:
            return len(scores)


        ranked = np.argsort(-scores)
        rank_of = np.empty(ranked.shape[0], dtype=np.int64)
        rank_of[ranked] = np.arange(ranked.shape[0])
        tf = np.asarray(list(true_features), dtype=np.int64)
        tf = tf[(tf >= 0) & (tf < ranked.shape[0])]
        if tf.size == 0:
            return int(ranked.shape[0])

        return int(rank_of[tf].max()) + 1

    @staticmethod
    def compute_mms_for_groups(scores: np.ndarray, active_groups: List[Tuple],
                                groups: List[Tuple]) -> int:
        if scores is None or len(scores) == 0:
            return len(groups)

        ranked = np.argsort(-scores)
        active_set = set(active_groups)

        for k in range(len(ranked)):
            selected = [groups[ranked[i]] for i in range(k + 1)]
            if set(selected) >= active_set:
                return k + 1

        return len(groups)

    @staticmethod
    def compute_power_at_k(scores: np.ndarray, true_features: List[int],
                           k: int = 5) -> float:
        ranked = np.argsort(-scores)
        top_k = set(ranked[:min(k, len(ranked))])
        true_set = set(true_features)

        return len(top_k & true_set) / len(true_features) if true_features else 0.0

    @staticmethod
    def compute_fdr(scores: np.ndarray, true_features: List[int],
                    threshold: float) -> float:
        selected = set(np.where(scores >= threshold)[0])
        true_set = set(true_features)

        false_discoveries = len(selected - true_set)
        total_discoveries = len(selected)

        return false_discoveries / total_discoveries if total_discoveries > 0 else 0.0

    @staticmethod
    def compute_tpr_at_k(scores: np.ndarray, true_features: List[int],
                         k: int) -> float:
        if scores is None or len(scores) == 0:
            return 0.0

        ranked = np.argsort(-scores)
        top_k = set(ranked[:min(k, len(ranked))])
        true_set = set(true_features)

        tp = len(top_k & true_set)
        return tp / len(true_set) if true_set else 0.0

    @staticmethod
    def compute_fpr_at_k(scores: np.ndarray, true_features: List[int],
                         k: int, p: int) -> float:
        if scores is None or len(scores) == 0:
            return 1.0

        ranked = np.argsort(-scores)
        top_k = set(ranked[:min(k, len(ranked))])
        true_set = set(true_features)

        fp = len(top_k - true_set)
        n_negative = p - len(true_set)
        return fp / n_negative if n_negative > 0 else 0.0

    @staticmethod
    def evaluate_screening(scores: np.ndarray, true_features: List[int],
                           s: int, p: int, n: int) -> Dict:
        if scores is None:
            return {
                'mms': p,
                'tpr_n1': 0.0,
                'fpr_n1': 1.0 if (p - s) > 0 else 0.0,
                'tpr_nlogn': 0.0,
                'fpr_nlogn': 1.0 if (p - s) > 0 else 0.0,
            }

        # MMS
        mms = ScreeningEvaluator.compute_mms(scores, true_features)

        # k = n-1
        k_n1 = min(n - 1, p)
        tpr_n1 = ScreeningEvaluator.compute_tpr_at_k(scores, true_features, k_n1)
        fpr_n1 = ScreeningEvaluator.compute_fpr_at_k(scores, true_features, k_n1, p)

        # k = ⌊n / log(n)⌋
        k_nlogn = min(max(1, int(n / np.log(n))), p)
        tpr_nlogn = ScreeningEvaluator.compute_tpr_at_k(scores, true_features, k_nlogn)
        fpr_nlogn = ScreeningEvaluator.compute_fpr_at_k(scores, true_features, k_nlogn, p)

        return {
            'mms': mms,
            'tpr_n1': tpr_n1,
            'fpr_n1': fpr_n1,
            'tpr_nlogn': tpr_nlogn,
            'fpr_nlogn': fpr_nlogn,
        }

    @staticmethod
    def evaluate_for_groups(scores: np.ndarray, active_groups: List[Tuple],
                            groups: List[Tuple], n: int) -> Dict:
        n_groups = len(groups)
        s_groups = len(active_groups)

        if scores is None:
            return {
                'mms': n_groups,
                'tpr_n1': 0.0,
                'fpr_n1': 1.0 if (n_groups - s_groups) > 0 else 0.0,
                'tpr_nlogn': 0.0,
                'fpr_nlogn': 1.0 if (n_groups - s_groups) > 0 else 0.0,
            }


        active_indices = [i for i, g in enumerate(groups) if g in set(active_groups)]

        mms = ScreeningEvaluator.compute_mms_for_groups(scores, active_groups, groups)

        k_n1 = min(n - 1, n_groups)
        tpr_n1 = ScreeningEvaluator.compute_tpr_at_k(scores, active_indices, k_n1)
        fpr_n1 = ScreeningEvaluator.compute_fpr_at_k(scores, active_indices, k_n1, n_groups)

        k_nlogn = min(max(1, int(n / np.log(n))), n_groups)
        tpr_nlogn = ScreeningEvaluator.compute_tpr_at_k(scores, active_indices, k_nlogn)
        fpr_nlogn = ScreeningEvaluator.compute_fpr_at_k(scores, active_indices, k_nlogn, n_groups)

        return {
            'mms': mms,
            'tpr_n1': tpr_n1,
            'fpr_n1': fpr_n1,
            'tpr_nlogn': tpr_nlogn,
            'fpr_nlogn': fpr_nlogn,
        }

# =============================================================================

# =============================================================================

def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return CorrelationMeasures.distance_correlation(x, y)


def compute_pc(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return CorrelationMeasures.projection_correlation(x, y, sigma)


def compute_ipc(x: np.ndarray, y: np.ndarray, sigma: float = None) -> float:
    return CorrelationMeasures.improved_projection_correlation(x, y, sigma)


def compute_hsic(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return CorrelationMeasures.hsic(x, y, sigma)


def compute_sc(x: np.ndarray, y: np.ndarray, a: float = 0.5) -> float:
    return CorrelationMeasures.stable_correlation(x, y, a)


def compute_swc(x: np.ndarray, y: np.ndarray, L: int = 20, p: int = 1,
                qmc: bool = True, seed: int = 42) -> float:
    return CorrelationMeasures.sliced_wasserstein_correlation(x, y, L, p, qmc, seed)


def compute_hd(x: np.ndarray, y: np.ndarray) -> float:
    return CorrelationMeasures.hellinger_distance_index(x, y)


def compute_ahd(x: np.ndarray, y: np.ndarray, K: int = None) -> float:
    return CorrelationMeasures.avg_hellinger_distance_index(x, y, K=K)


def compute_gcor(x: np.ndarray, y: np.ndarray) -> float:
    return CorrelationMeasures.grothendieck_correlation(x, y)


def compute_mms(scores: np.ndarray, active_groups: List[Tuple],
                groups: List[Tuple]) -> int:
    return ScreeningEvaluator.compute_mms_for_groups(scores, active_groups, groups)


# =============================================================================

# =============================================================================

class MixedFeatureScreening:

    @staticmethod
    def one_hot_encode_feature(x: np.ndarray) -> np.ndarray:
        categories = np.unique(x)
        n = len(x)
        n_categories = len(categories)

        X_onehot = np.zeros((n, n_categories))
        for i, cat in enumerate(categories):
            X_onehot[x == cat, i] = 1

        return X_onehot

    @staticmethod
    def encode_group_for_distance(
        X_group: np.ndarray,
        is_discrete: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        n, k = X_group.shape
        encoded_features = []

        for j in range(k):
            if is_discrete[j]:

                onehot = MixedFeatureScreening.one_hot_encode_feature(X_group[:, j])
                if normalize:

                    onehot = onehot - onehot.mean(axis=0)
                encoded_features.append(onehot)
            else:

                col = X_group[:, j:j+1]
                if normalize:
                    col = (col - col.mean()) / (col.std() + 1e-10)
                encoded_features.append(col)

        return np.hstack(encoded_features)

    @staticmethod
    def gower_distance(X_group: np.ndarray, is_discrete: np.ndarray) -> np.ndarray:
        n, k = X_group.shape
        D = np.zeros((n, n))

        for j in range(k):
            if is_discrete[j]:

                col = X_group[:, j]
                for i in range(n):
                    for l in range(i + 1, n):
                        if col[i] != col[l]:
                            D[i, l] += 1
                            D[l, i] += 1
            else:

                col = X_group[:, j]
                col_range = col.max() - col.min()
                if col_range > 0:
                    col_normalized = (col - col.min()) / col_range
                    for i in range(n):
                        for l in range(i + 1, n):
                            dist = abs(col_normalized[i] - col_normalized[l])
                            D[i, l] += dist
                            D[l, i] += dist


        return D / k

    @staticmethod
    def distance_correlation_mixed(
        X_group: np.ndarray,
        y: np.ndarray,
        is_discrete: np.ndarray,
        method: str = 'onehot'
    ) -> float:
        n = len(y)

        if method == 'onehot':

            X_encoded = MixedFeatureScreening.encode_group_for_distance(
                X_group, is_discrete, normalize=True)
            return CorrelationMeasures.distance_correlation(X_encoded, y)

        elif method == 'gower':

            D_X = MixedFeatureScreening.gower_distance(X_group, is_discrete)
            D_y = cdist(y.reshape(-1, 1), y.reshape(-1, 1), 'euclidean')


            D_X_centered = D_X - D_X.mean(1, keepdims=True) - D_X.mean(0, keepdims=True) + D_X.mean()
            D_y_centered = D_y - D_y.mean(1, keepdims=True) - D_y.mean(0, keepdims=True) + D_y.mean()


            dcov2 = (D_X_centered * D_y_centered).sum() / (n * n)
            dvar_x2 = (D_X_centered ** 2).sum() / (n * n)
            dvar_y2 = (D_y_centered ** 2).sum() / (n * n)

            if dvar_x2 * dvar_y2 <= 0:
                return 0.0
            return np.sqrt(abs(dcov2) / np.sqrt(dvar_x2 * dvar_y2))

        elif method == 'separate':

            discrete_cols = [j for j in range(X_group.shape[1]) if is_discrete[j]]
            continuous_cols = [j for j in range(X_group.shape[1]) if not is_discrete[j]]

            dc_values = []

            if discrete_cols:
                X_disc = X_group[:, discrete_cols]
                is_disc_true = is_discrete[discrete_cols]
                X_encoded = MixedFeatureScreening.encode_group_for_distance(X_disc, is_disc_true)
                dc_disc = CorrelationMeasures.distance_correlation(X_encoded, y)
                dc_values.append(dc_disc)

            if continuous_cols:
                X_cont = X_group[:, continuous_cols]
                dc_cont = CorrelationMeasures.distance_correlation(X_cont, y)
                dc_values.append(dc_cont)


            return np.mean(dc_values) if dc_values else 0.0

        return 0.0

    @staticmethod
    def compute_raw_scores_mixed(
        X: np.ndarray,
        y: np.ndarray,
        groups: List[Tuple],
        is_discrete: np.ndarray,
        metric: str = 'dc',
        encoding: str = 'onehot'
    ) -> np.ndarray:
        n_groups = len(groups)
        scores = np.zeros(n_groups)

        for i, group in enumerate(groups):
            group = list(group)
            X_group = X[:, group]
            is_discrete_group = is_discrete[group]

            if metric == 'dc':
                scores[i] = MixedFeatureScreening.distance_correlation_mixed(
                    X_group, y, is_discrete_group, encoding)
            elif metric == 'pc':

                X_encoded = MixedFeatureScreening.encode_group_for_distance(
                    X_group, is_discrete_group)
                scores[i] = CorrelationMeasures.projection_correlation(X_encoded, y)
            elif metric == 'ipc':

                X_encoded = MixedFeatureScreening.encode_group_for_distance(
                    X_group, is_discrete_group)
                scores[i] = CorrelationMeasures.improved_projection_correlation(X_encoded, y)

        return scores


# =============================================================================

# =============================================================================

class MissingValueScreening:

    @staticmethod
    def get_complete_case_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        complete_mask = ~np.isnan(X).any(axis=1)
        return X[complete_mask], y[complete_mask]

    @staticmethod
    def mean_imputation(X: np.ndarray) -> np.ndarray:
        X_imputed = X.copy()
        for j in range(X.shape[1]):
            col_mean = np.nanmean(X[:, j])
            X_imputed[np.isnan(X[:, j]), j] = col_mean
        return X_imputed

    @staticmethod
    def group_imputation(
        X: np.ndarray,
        groups: List[Tuple],
        missing_mask: np.ndarray
    ) -> np.ndarray:
        X_imputed = X.copy()

        for group in groups:
            group = list(group)
            for j in group:
                missing_rows = np.where(missing_mask[:, j])[0]
                for i in missing_rows:
                    other_features = [f for f in group if f != j and not missing_mask[i, f]]
                    if other_features:
                        X_imputed[i, j] = np.nanmean(X_imputed[i, other_features])
                    else:
                        X_imputed[i, j] = np.nanmean(X[:, j])

        return X_imputed

    @staticmethod
    def compute_raw_scores_missing(
        X: np.ndarray,
        y: np.ndarray,
        groups: List[Tuple],
        missing_mask: np.ndarray,
        metric: str = 'dc',
        handling: str = 'complete_case'
    ) -> Tuple[np.ndarray, float]:
        n_groups = len(groups)
        scores = np.zeros(n_groups)
        effective_ratios = []

        if handling == 'complete_case':

            X_complete, y_complete = MissingValueScreening.get_complete_case_data(X, y)
            effective_ratio = len(X_complete) / len(X)

            for i, group in enumerate(groups):
                group = list(group)
                X_group = X_complete[:, group]

                if metric == 'dc':
                    scores[i] = CorrelationMeasures.distance_correlation(X_group, y_complete)
                elif metric == 'pc':
                    scores[i] = CorrelationMeasures.projection_correlation(X_group, y_complete)
                elif metric == 'ipc':
                    scores[i] = CorrelationMeasures.improved_projection_correlation(X_group, y_complete)

            return scores, effective_ratio

        elif handling == 'mean_imputation':

            X_imputed = MissingValueScreening.mean_imputation(X)
            effective_ratio = 1.0

            for i, group in enumerate(groups):
                group = list(group)
                X_group = X_imputed[:, group]

                if metric == 'dc':
                    scores[i] = CorrelationMeasures.distance_correlation(X_group, y)
                elif metric == 'pc':
                    scores[i] = CorrelationMeasures.projection_correlation(X_group, y)
                elif metric == 'ipc':
                    scores[i] = CorrelationMeasures.improved_projection_correlation(X_group, y)

            return scores, effective_ratio

        elif handling == 'group_imputation':

            X_imputed = MissingValueScreening.group_imputation(X, groups, missing_mask)
            effective_ratio = 1.0

            for i, group in enumerate(groups):
                group = list(group)
                X_group = X_imputed[:, group]

                if metric == 'dc':
                    scores[i] = CorrelationMeasures.distance_correlation(X_group, y)
                elif metric == 'pc':
                    scores[i] = CorrelationMeasures.projection_correlation(X_group, y)
                elif metric == 'ipc':
                    scores[i] = CorrelationMeasures.improved_projection_correlation(X_group, y)

            return scores, effective_ratio

        return scores, 1.0


# =============================================================================

# =============================================================================

class MissingResponseScreening:


    @staticmethod
    def _gaussian_kernel(x, h):
        if x.ndim == 1:
            z = -0.5 * (x / h) ** 2
            z = np.clip(z, -500, 0)
            return np.exp(z) / (h * np.sqrt(2 * np.pi))
        else:
            d = x.shape[1] if x.ndim > 1 else 1
            z = -0.5 * np.sum((x / h) ** 2, axis=1)
            z = np.clip(z, -500, 0)
            # log-sum trick to avoid division underflow in high-d product kernel
            log_norm = np.sum(np.log(h)) + 0.5 * d * np.log(2 * np.pi)
            log_kernel = z - log_norm
            return np.exp(np.clip(log_kernel, -500, 500))

    @staticmethod
    def _ecdf_1d(x, data):
        return np.mean(data <= x)

    @staticmethod
    def _ecdf_vector(x_vals, data):
        data_sorted = np.sort(data)
        return np.searchsorted(data_sorted, x_vals, side='right') / len(data)

    @staticmethod
    def _nadaraya_watson(x_new, y_resp, x_obs, h, weights=None):
        n = len(y_resp)
        if x_obs.ndim == 1:
            diffs = x_obs - x_new
            kernel_vals = MissingResponseScreening._gaussian_kernel(diffs, h)
        else:
            diffs = x_obs - x_new.reshape(1, -1)
            kernel_vals = MissingResponseScreening._gaussian_kernel(diffs, h)

        if weights is not None:
            kernel_vals = kernel_vals * weights

        denom = np.sum(kernel_vals)
        if denom < 1e-15:
            return np.mean(y_resp)
        return np.sum(kernel_vals * y_resp) / denom

    # =====================================================================
    # ASRC: Adjusted Spearman Rank Correlation (MAR)
    # =====================================================================

    @staticmethod
    def asrc_screening(X: np.ndarray, y: np.ndarray, delta: np.ndarray,
                       h_factor: float = 1.0) -> np.ndarray:
        n, p = X.shape
        y_obs = y.copy()

        y_obs[delta == 0] = 0.0

        obs_idx = np.where(delta == 1)[0]
        n_obs = len(obs_idx)


        F_y_vals = np.zeros(n)
        F_y_obs_sorted = np.sort(y[delta == 1])
        n_y_obs = len(F_y_obs_sorted)
        for i in range(n):
            if delta[i] == 1:
                F_y_vals[i] = np.searchsorted(F_y_obs_sorted, y[i], side='right') / n_y_obs


        omega = np.zeros(p)

        for k in range(p):
            x_k = X[:, k]


            F_x_k = MissingResponseScreening._ecdf_vector(x_k, x_k)


            sigma_k = np.std(x_k, ddof=1) if np.std(x_k) > 1e-10 else 1.0
            h = h_factor * sigma_k * n_obs ** (-1.0 / 3)


            x_obs_k = x_k[obs_idx]
            F_y_obs = F_y_vals[obs_idx]

            for i in range(n):
                if delta[i] == 0:

                    F_y_vals[i] = MissingResponseScreening._nadaraya_watson(
                        x_k[i], F_y_obs, x_obs_k, h)


            omega_k = np.mean(delta * F_x_k * F_y_vals +
                              (1 - delta) * F_x_k * F_y_vals) - 0.25
            omega[k] = omega_k

        return omega

    # =====================================================================
    # DCNI: Distance Correlation with Nonparametric Imputation (MAR, two-step)
    # =====================================================================

    @staticmethod
    def _mv_sis_delta(X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        n, p = X.shape
        P_0 = np.mean(delta == 0)
        P_1 = np.mean(delta == 1)

        if P_0 < 1e-10 or P_1 < 1e-10:
            return np.zeros(p)

        omega_delta = np.zeros(p)

        for k in range(p):
            x_k = X[:, k]

            x_sorted = np.sort(x_k)


            x_0 = x_k[delta == 0]
            x_1 = x_k[delta == 1]


            n_grid = min(200, n)
            grid = np.linspace(x_sorted[0], x_sorted[-1], n_grid)

            F_k = np.searchsorted(x_sorted, grid, side='right') / n
            F_k0 = np.searchsorted(np.sort(x_0), grid, side='right') / len(x_0) if len(x_0) > 0 else F_k
            F_k1 = np.searchsorted(np.sort(x_1), grid, side='right') / len(x_1) if len(x_1) > 0 else F_k


            dx = np.diff(grid, prepend=grid[0] - (grid[1] - grid[0]))
            dF = np.diff(F_k, prepend=0)

            omega_delta[k] = P_0 * np.sum((F_k0 - F_k) ** 2 * dF) + \
                             P_1 * np.sum((F_k1 - F_k) ** 2 * dF)

        return omega_delta

    @staticmethod
    def dcni_screening(X: np.ndarray, y: np.ndarray, delta: np.ndarray,
                       h_factor: float = 1.0,
                       d_delta: int = None,
                       d_n: int = None) -> np.ndarray:
        n, p = X.shape

        if d_delta is None:
            d_delta = max(1, int(n * np.log(n) / 10))
        d_delta = min(d_delta, p)

        # ---- Step 1: MV-SIS for delta ----
        omega_delta = MissingResponseScreening._mv_sis_delta(X, delta)
        z_hat_idx = np.argsort(-np.abs(omega_delta))[:d_delta]


        obs_idx = np.where(delta == 1)[0]
        miss_idx = np.where(delta == 0)[0]
        n_obs = len(obs_idx)


        Z = X[:, z_hat_idx]

        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)


        F_y_vals = np.zeros(n)
        F_y_obs_sorted = np.sort(y[delta == 1])
        n_y_obs = len(F_y_obs_sorted)

        for i in obs_idx:
            F_y_vals[i] = np.searchsorted(F_y_obs_sorted, y[i], side='right') / n_y_obs


        if len(miss_idx) > 0:
            Z_obs = Z[obs_idx]
            F_y_obs = F_y_vals[obs_idx]


            sigma_Z = np.std(Z_obs, axis=0, ddof=1)
            sigma_Z[sigma_Z < 1e-10] = 1.0
            h_Z = h_factor * sigma_Z * n_obs ** (-1.0 / (4 + Z.shape[1]))

            for i in miss_idx:
                F_y_vals[i] = MissingResponseScreening._nadaraya_watson(
                    Z[i], F_y_obs, Z_obs, h_Z)


        omega = np.zeros(p)
        for k in range(p):
            F_x_k = MissingResponseScreening._ecdf_vector(X[:, k], X[:, k])

            omega[k] = CorrelationMeasures.distance_correlation(
                F_x_k.reshape(-1, 1), F_y_vals.reshape(-1, 1))

        return omega

    # =====================================================================
    # MNAR: Missing-Not-At-Random adjusted Spearman (MNAR)
    # =====================================================================

    @staticmethod
    def mnar_screening(X: np.ndarray, y: np.ndarray, delta: np.ndarray,
                       Z: np.ndarray = None,
                       gamma: float = None,
                       q_func: callable = None,
                       h_factor: float = 1.0) -> np.ndarray:
        n, p = X.shape

        if Z is None:

            r = min(2, p)
            Z = X[:, :r]
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if q_func is None:
            q_func = lambda gamma, y: np.exp(-gamma * y)

        obs_idx = np.where(delta == 1)[0]
        miss_idx = np.where(delta == 0)[0]
        n_obs = len(obs_idx)


        if gamma is None:
            gamma = MissingResponseScreening._estimate_gamma_mnar(
                y, delta, Z, q_func, h_factor)


        omega = np.zeros(p)

        for k in range(p):
            x_k = X[:, k]


            F_x_k = MissingResponseScreening._ecdf_vector(x_k, x_k)


            XZ_k = np.column_stack([x_k.reshape(-1, 1), Z])


            sigma_XZ = np.std(XZ_k, axis=0, ddof=1)
            sigma_XZ[sigma_XZ < 1e-10] = 1.0
            d_XZ = XZ_k.shape[1]
            h_XZ = h_factor * sigma_XZ * n_obs ** (-1.0 / (4 + d_XZ))

            XZ_obs = XZ_k[obs_idx]
            y_obs = y[obs_idx]


            F_y_k = np.zeros(n)


            y_obs_sorted = np.sort(y_obs)
            n_y_obs = len(y_obs_sorted)

            for i in obs_idx:
                F_y_k[i] = np.searchsorted(y_obs_sorted, y[i], side='right') / n_y_obs


            if len(miss_idx) > 0:
                q_obs = q_func(gamma, y_obs)

                for i in miss_idx:

                    diffs = XZ_obs - XZ_k[i].reshape(1, -1)
                    K_vals = MissingResponseScreening._gaussian_kernel(diffs, h_XZ)


                    weighted_K = K_vals * q_obs
                    denom = np.sum(weighted_K)

                    if denom > 1e-15:


                        # P_hat_k = E{delta*F(Y)*q(gamma,Y)|X_k,Z} / E{delta*q(gamma,Y)|X_k,Z}

                        F_y_obs_vals = F_y_k[obs_idx]
                        F_y_k[i] = np.sum(weighted_K * F_y_obs_vals) / denom
                    else:
                        F_y_k[i] = np.mean(F_y_k[obs_idx])


            P_hat = np.zeros(n)
            for i in range(n):
                if delta[i] == 0:
                    diffs = XZ_obs - XZ_k[i].reshape(1, -1)
                    K_vals = MissingResponseScreening._gaussian_kernel(diffs, h_XZ)
                    weighted_K = K_vals * q_obs

                    denom_q = np.sum(weighted_K)
                    if denom_q > 1e-15:
                        P_hat[i] = np.sum(weighted_K * F_y_k[obs_idx]) / denom_q
                    else:
                        P_hat[i] = np.mean(F_y_k[obs_idx])

            # omega_k = mean{delta * F_k * F^(k)_Y + (1-delta) * F_k * P_hat} - 1/4
            omega_k = np.mean(delta * F_x_k * F_y_k +
                              (1 - delta) * F_x_k * P_hat) - 0.25
            omega[k] = omega_k

        return omega

    @staticmethod
    def _estimate_gamma_mnar(y: np.ndarray, delta: np.ndarray,
                             Z: np.ndarray, q_func: callable,
                             h_factor: float = 1.0,
                             gamma_init: float = 0.0) -> float:
        obs_idx = np.where(delta == 1)[0]
        n_obs = len(obs_idx)
        Z_obs = Z[obs_idx]
        y_obs = y[obs_idx]


        sigma_Z = np.std(Z_obs, axis=0, ddof=1)
        sigma_Z[sigma_Z < 1e-10] = 1.0
        d_Z = Z.shape[1]
        h_Z = h_factor * sigma_Z * n_obs ** (-1.0 / (4 + d_Z))

        def _profile_loglik(gamma_val):
            q_y = q_func(gamma_val, y_obs)

            # g_hat(Z_j, gamma) = sum_i (1-delta_i)*K(Z_j-Z_i) / sum_i delta_i*q(gamma,Y_i)*K(Z_j-Z_i)
            loglik = 0.0
            for j in range(n_obs):
                diffs = Z_obs - Z_obs[j].reshape(1, -1)
                K_vals = MissingResponseScreening._gaussian_kernel(diffs, h_Z)
                numer = np.sum(K_vals * q_y) + 1e-15
                pi_j = 1.0 / (1.0 + np.sum(K_vals * (1 - delta[obs_idx])) / numer + 1e-15)
                pi_j = np.clip(pi_j, 1e-6, 1 - 1e-6)
                loglik += np.log(pi_j)
            return loglik


        gamma_grid = np.linspace(-2.0, 2.0, 21)
        best_gamma = gamma_init
        best_ll = -np.inf

        for g in gamma_grid:
            try:
                ll = _profile_loglik(g)
                if ll > best_ll:
                    best_ll = ll
                    best_gamma = g
            except Exception:
                continue

        return best_gamma
# =============================================================================

class AcceleratedScreening:

    # =========================================================================

    # =========================================================================

    @staticmethod
    def _prepare_y_for_screening(y: np.ndarray, n: int) -> np.ndarray:
        unique_vals = np.unique(y)
        is_categorical = len(unique_vals) < n / 2 or not np.issubdtype(y.dtype, np.number)

        if is_categorical:

            classes = unique_vals
            Y_encoded = np.zeros((n, len(classes)))
            for i, c in enumerate(classes):
                Y_encoded[:, i] = (y == c).astype(float)
            return Y_encoded
        else:

            return y.reshape(-1, 1) if y.ndim == 1 else y

    @staticmethod
    def dc_sis_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        b = cdist(y_prepared, y_prepared, 'euclidean')
        b_centered = b - b.mean(-1, keepdims=True) - b.mean(-2, keepdims=True) + b.mean(axis=(-2, -1), keepdims=True)
        b_var = np.sum(b_centered ** 2)

        def compute_dc_for_feature(j):
            x_j = X[:, j].reshape(-1, 1)
            a = cdist(x_j, x_j, 'euclidean')
            a_centered = a - a.mean(-1, keepdims=True) - a.mean(-2, keepdims=True) + a.mean(axis=(-2, -1), keepdims=True)

            num = np.sum(a_centered * b_centered)
            a_var = np.sum(a_centered ** 2)

            if a_var * b_var < 1e-15:
                return 0.0

            dc_sq = num / np.sqrt(a_var * b_var)
            return max(0, min(1, np.sqrt(abs(dc_sq))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_dc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_dc_for_feature(j) for j in range(p)])

    @staticmethod
    def sc_sis_accelerated(X: np.ndarray, y: np.ndarray, a: float = 0.5,
                           n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        dist_y = cdist(y_prepared, y_prepared, 'euclidean')
        Ky = np.exp(-np.power(dist_y, a))

        E2_y = (Ky.sum() - np.trace(Ky)) / (n * (n - 1))
        Kyy = np.exp(-np.power(dist_y, a) * 2)
        E1_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        E2_yy = E2_y ** 2

        E3_yy = 0.0
        for i in range(n):
            Ky_row_i = np.delete(Ky[i, :], i)
            E3_yy += Ky_row_i.sum() ** 2
        E3_yy = E3_yy / (n * (n - 1) * (n - 2))
        Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

        def compute_sc_for_feature(j):
            x_j = X[:, j].reshape(-1, 1)
            dist_x = cdist(x_j, x_j, 'euclidean')
            Kx = np.exp(-np.power(dist_x, a))

            Kxy = Kx * Ky
            E1 = (Kxy.sum() - np.trace(Kxy)) / (n * (n - 1))

            E2_x = (Kx.sum() - np.trace(Kx)) / (n * (n - 1))
            E2 = E2_x * E2_y

            E3 = 0.0
            for i in range(n):
                Kx_row_i = np.delete(Kx[i, :], i)
                Ky_row_i = np.delete(Ky[i, :], i)
                E3 += Kx_row_i.sum() * Ky_row_i.sum()
            E3 = E3 / (n * (n - 1) * (n - 2))

            Scov2 = E1 + E2 - 2 * E3

            Kxx = np.exp(-np.power(dist_x, a) * 2)
            E1_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            E2_xx = E2_x ** 2

            E3_xx = 0.0
            for i in range(n):
                Kx_row_i = np.delete(Kx[i, :], i)
                E3_xx += Kx_row_i.sum() ** 2
            E3_xx = E3_xx / (n * (n - 1) * (n - 2))

            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx

            if Svar_x2 <= 0 or Svar_y2 <= 0:
                return 0.0

            SC2 = Scov2 / np.sqrt(Svar_x2 * Svar_y2)
            return max(0.0, min(1.0, np.sqrt(abs(SC2))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_sc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_sc_for_feature(j) for j in range(p)])

    @staticmethod
    def ipc_sis_accelerated(X: np.ndarray, y: np.ndarray, sigma: float = None,
                            n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        y_scaled = (y_prepared - y_prepared.mean(axis=0)) / (y_prepared.std(axis=0) + 1e-10)


        if sigma is None:
            sigma_y = np.sqrt(np.median(np.sum(y_scaled ** 2, axis=1)))
        else:
            sigma_y = sigma


        y_sq = np.sum(y_scaled ** 2, axis=1)
        y_norm = np.sqrt(sigma_y ** 2 + y_sq)
        Gy = y_scaled / (y_norm[:, np.newaxis] * np.sqrt(sigma_y ** 2))
        N = Gy @ Gy.T
        H = np.eye(n) - np.ones((n, n)) / n
        NH = N @ H

        def compute_ipc_for_feature(j):
            Z = X[:, j].reshape(-1, 1)
            Z_centered = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-10)


            if sigma is None:
                sigma_j = np.sqrt(np.median(np.sum(Z_centered ** 2, axis=1)))
            else:
                sigma_j = sigma

            z_sq = np.sum(Z_centered ** 2, axis=1)
            z_norm = np.sqrt(sigma_j ** 2 + z_sq)
            Gz = Z_centered / (z_norm[:, np.newaxis] * np.sqrt(sigma_j ** 2))

            M = Gz @ Gz.T
            MH = M @ H

            ipc = np.trace(MH @ NH) / np.sqrt(np.trace(MH @ MH) * np.trace(NH @ NH))
            return max(0, min(1, ipc))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_ipc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_ipc_for_feature(j) for j in range(p)])

    @staticmethod
    def pc_sis_accelerated(X: np.ndarray, y: np.ndarray, sigma: float = 1.0,
                           n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        y_scaled = (y_prepared - y_prepared.mean(axis=0)) / (y_prepared.std(axis=0) + 1e-10)


        y_sq = np.sum(y_scaled ** 2, axis=1)
        dot_y = y_scaled @ y_scaled.T
        denom_y = np.sqrt((sigma**2 + y_sq[:, np.newaxis]) * (sigma**2 + y_sq[np.newaxis, :]))
        cos_y = np.clip((sigma**2 + dot_y) / (denom_y + 1e-10), -1, 1)
        Ay = np.arccos(cos_y)

        def compute_pc_for_feature(j):
            x_j = X[:, j].reshape(-1, 1)
            x_centered = (x_j - x_j.mean(axis=0)) / (x_j.std(axis=0) + 1e-10)

            x_sq = np.sum(x_centered ** 2, axis=1)
            dot_x = x_centered @ x_centered.T
            denom_x = np.sqrt((sigma**2 + x_sq[:, np.newaxis]) * (sigma**2 + x_sq[np.newaxis, :]))
            cos_x = np.clip((sigma**2 + dot_x) / (denom_x + 1e-10), -1, 1)
            Ax = np.arccos(cos_x)

            n_pairs = n * (n - 1)
            term1 = (Ax * Ay).sum() / n_pairs
            term2 = 2 * (Ax.mean(axis=1) * Ay.mean(axis=1)).mean()
            term3 = Ax.mean() * Ay.mean()

            pcov_xy = term1 - term2 + term3
            pcov_xx = (Ax ** 2).sum() / n_pairs - 2 * (Ax.mean(axis=1) ** 2).mean() + Ax.mean() ** 2
            pcov_yy = (Ay ** 2).sum() / n_pairs - 2 * (Ay.mean(axis=1) ** 2).mean() + Ay.mean() ** 2

            if pcov_xx < 1e-10 or pcov_yy < 1e-10:
                return 0.0

            pc_sq = pcov_xy / np.sqrt(pcov_xx * pcov_yy)
            return max(0, min(1, pc_sq))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_pc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_pc_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def _mann_whitney_statistic(x: np.ndarray, mask_r: np.ndarray) -> float:
        mask_not_r = ~mask_r
        if mask_r.sum() == 0 or mask_not_r.sum() == 0:
            return 0.5
        x_r = x[mask_r]
        x_not_r = x[mask_not_r]
        count = sum(np.sum(x_not_r < val) for val in x_r)
        return count / (len(x_r) * len(x_not_r))

    @staticmethod
    def fkf_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}

        def compute_fkf_for_feature(j):
            max_ks = 0
            for r in classes:
                mask_r = class_masks[r]
                mask_not_r = ~mask_r
                if mask_r.sum() > 0 and mask_not_r.sum() > 0:
                    ks_stat, _ = stats.ks_2samp(X[mask_r, j], X[mask_not_r, j])
                    max_ks = max(max_ks, ks_stat)
            return max_ks

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_fkf_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_fkf_for_feature(j) for j in range(p)])

    @staticmethod
    def cru_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}
        pi_r = {r: mask.mean() for r, mask in class_masks.items()}

        def compute_cru_for_feature(j):
            total = 0
            x_j = X[:, j]
            for r in classes:
                mask_r = class_masks[r]
                mask_not_r = ~mask_r

                if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                    continue


                x_r = x_j[mask_r]
                x_not_r = x_j[mask_not_r]
                count = sum(np.sum(x_not_r < val) for val in x_r)
                gamma = count / (len(x_r) * len(x_not_r))


                weight = (pi_r[r] * (1 - pi_r[r])) ** 2
                total += weight * (gamma - 0.5) ** 2
            return total

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_cru_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_cru_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def sis_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        unique_vals = np.unique(y)
        is_categorical = len(unique_vals) < n / 2 or not np.issubdtype(y.dtype, np.number)

        if is_categorical:

            classes = unique_vals
            Y_encoded = np.zeros((n, len(classes)))
            for i, c in enumerate(classes):
                Y_encoded[:, i] = (y == c).astype(float)

            def compute_sis_for_feature(j):
                if np.std(X[:, j]) < 1e-10:
                    return 0.0
                max_corr = 0.0
                for k in range(Y_encoded.shape[1]):
                    if np.std(Y_encoded[:, k]) > 1e-10:
                        corr = abs(np.corrcoef(X[:, j], Y_encoded[:, k])[0, 1])
                        if not np.isnan(corr):
                            max_corr = max(max_corr, corr)
                return max_corr

            if HAS_JOBLIB and n_jobs != 1:
                actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
                scores = Parallel(n_jobs=actual_jobs)(
                    delayed(compute_sis_for_feature)(j) for j in range(p)
                )
                return np.array(scores)
            else:
                return np.array([compute_sis_for_feature(j) for j in range(p)])
        else:

            def compute_sis_for_feature(j):
                if np.std(X[:, j]) < 1e-10:
                    return 0.0
                corr = abs(np.corrcoef(X[:, j], y)[0, 1])
                return corr if not np.isnan(corr) else 0.0

            if HAS_JOBLIB and n_jobs != 1:
                actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
                scores = Parallel(n_jobs=actual_jobs)(
                    delayed(compute_sis_for_feature)(j) for j in range(p)
                )
                return np.array(scores)
            else:
                return np.array([compute_sis_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def cavs_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}
        pi_r = {r: mask.mean() for r, mask in class_masks.items()}

        def compute_cavs_for_feature(j):
            x_j = X[:, j]
            max_val = 0.0
            for r in classes:
                mask_r = class_masks[r]
                mask_not_r = ~mask_r
                if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                    continue


                x_r = x_j[mask_r]
                x_not_r = x_j[mask_not_r]
                count = sum(np.sum(x_not_r < val) for val in x_r)
                gamma = count / (len(x_r) * len(x_not_r))

                weight = 1 - pi_r[r]
                max_val = max(max_val, weight * abs(gamma - 0.5))
            return max_val

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_cavs_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_cavs_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def mv_sis_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}
        pi_r = {r: mask.mean() for r, mask in class_masks.items()}

        def compute_mv_sis_for_feature(j):
            x_j = X[:, j]
            total = 0.0
            for r in classes:
                mask_r = class_masks[r]
                mask_not_r = ~mask_r
                if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                    continue


                x_r = x_j[mask_r]
                x_not_r = x_j[mask_not_r]
                count = sum(np.sum(x_not_r < val) for val in x_r)
                gamma = count / (len(x_r) * len(x_not_r))

                weight = pi_r[r] * (1 - pi_r[r]) ** 2
                total += weight * (gamma - 0.5) ** 2
            return total

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_mv_sis_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_mv_sis_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def lr_ffs_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}

        def compute_lr_ffs_for_feature(j):
            x_j = X[:, j]
            max_dev = 0.0
            for r in classes:
                mask_r = class_masks[r]
                mask_not_r = ~mask_r
                if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                    continue


                x_r = x_j[mask_r]
                x_not_r = x_j[mask_not_r]
                count = sum(np.sum(x_not_r < val) for val in x_r)
                gamma = count / (len(x_r) * len(x_not_r))

                max_dev = max(max_dev, abs(gamma - 0.5))
            return max_dev

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_lr_ffs_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_lr_ffs_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def psis_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)
        n_classes = len(classes)


        class_masks = {c: (y == c) for c in classes}

        def compute_psis_for_feature(j):
            x_j = X[:, j]


            class_medians = {}
            for c in classes:
                mask_c = class_masks[c]
                if mask_c.sum() > 0:
                    class_medians[c] = np.median(x_j[mask_c])
                else:
                    class_medians[c] = np.nan


            max_diff = 0.0
            for i in range(n_classes):
                for k in range(i + 1, n_classes):
                    c_i = classes[i]
                    c_k = classes[k]
                    if c_i in class_medians and c_k in class_medians:
                        diff = abs(class_medians[c_i] - class_medians[c_k])
                        max_diff = max(max_diff, diff)

            return max_diff

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_psis_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_psis_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def swc_accelerated(X: np.ndarray, y: np.ndarray, L: int = 20, p: int = 1,
                        n_shuf: int = 11, qmc: bool = True, seed: int = 42,
                        n_jobs: int = -1) -> np.ndarray:
        n, n_features = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)
        dy = y_prepared.shape[1] if y_prepared.ndim > 1 else 1
        if y_prepared.ndim == 1:
            y_prepared = y_prepared.reshape(-1, 1)


        gaussian_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        def rank_gaussian_transform(x_1d):
            ranks = stats.rankdata(x_1d, method='average')
            return gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]


        gry = np.zeros_like(y_prepared)
        for k in range(dy):
            gry[:, k] = rank_gaussian_transform(y_prepared[:, k])


        X_transformed = np.zeros((n, n_features))
        for j in range(n_features):
            X_transformed[:, j] = rank_gaussian_transform(X[:, j])


        total_dir = (dy + 1) * L
        if qmc:
            try:
                from scipy.stats.qmc import Sobol
                sobol_engine = Sobol(d=dy, scramble=True, seed=seed)
                vec_meta = sobol_engine.random(total_dir)
                dir_xy = np.array([
                    CorrelationMeasures._uniform2sphere_qmc(vec_meta[i])
                    for i in range(total_dir)
                ])
            except ImportError:
                dir_xy = CorrelationMeasures._uniform2sphere_mc(total_dir, dy + 1, seed)
        else:
            dir_xy = CorrelationMeasures._uniform2sphere_mc(total_dir, dy + 1, seed)

        def compute_swc_for_feature(j):
            x_j = X_transformed[:, j]  # shape (n,)
            scores_shuf = np.zeros(n_shuf)

            for s in range(n_shuf):
                rng = np.random.RandomState(seed + s + 1)
                shuf_idx = rng.permutation(n)

                x_shuf = x_j[shuf_idx]

                L_mat = np.zeros(total_dir)
                for i in range(total_dir):

                    # joint = X*θ_1 + Y*θ_{2:}, independent = X_shuf*θ_1 + Y*θ_{2:}

                    proj_joint = x_j * dir_xy[i, 0] + np.sum(gry * dir_xy[i, 1:dy + 1], axis=1)
                    proj_indep = x_shuf * dir_xy[i, 0] + np.sum(gry * dir_xy[i, 1:dy + 1], axis=1)

                    if p == 1:
                        L_mat[i] = np.mean(np.abs(np.sort(proj_joint) - np.sort(proj_indep)))
                    else:
                        L_mat[i] = np.mean(np.abs(np.sort(proj_joint) - np.sort(proj_indep)) ** p)

                scores_shuf[s] = np.mean(L_mat) ** (1 / p)

            return np.median(scores_shuf)

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_swc_for_feature)(j) for j in range(n_features)
            )
            return np.array(scores)
        else:
            return np.array([compute_swc_for_feature(j) for j in range(n_features)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def hd_sis_accelerated(X: np.ndarray, y: np.ndarray, K: int = None,
                           n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape
        classes = np.unique(y)
        is_categorical = len(classes) < n / 2 or not np.issubdtype(y.dtype, np.number)

        def compute_hd_for_feature(j):
            if is_categorical:
                return CorrelationMeasures.hellinger_distance_index(X[:, j], y)
            else:
                return CorrelationMeasures.avg_hellinger_distance_index(X[:, j], y, K=K)

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_hd_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_hd_for_feature(j) for j in range(p)])

    # =========================================================================

    # =========================================================================

    @staticmethod
    def gcor_sis_accelerated(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p = X.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)

        def compute_kappa_matrix(Z):
            dot_ZZ = Z @ Z.T
            norm_sq = np.sum(Z ** 2, axis=1)
            num = 1 + dot_ZZ
            denom = np.sqrt((1 + norm_sq[:, np.newaxis]) * (1 + norm_sq[np.newaxis, :]))
            cos_val = np.clip(num / (denom + 1e-15), -1, 1)
            return np.arcsin(cos_val)

        def double_center(K_matrix):
            n_sq = K_matrix.shape[0]
            K_off = K_matrix.copy()
            np.fill_diagonal(K_off, 0)
            row_sums = K_off.sum(axis=1)
            col_sums = K_off.sum(axis=0)
            total_sum = K_off.sum()
            a_row = row_sums / (n_sq - 2)
            a_col = col_sums / (n_sq - 2)
            a_total = total_sum / ((n_sq - 1) * (n_sq - 2))
            K_centered = K_matrix - a_row[:, np.newaxis] - a_col[np.newaxis, :] + a_total
            np.fill_diagonal(K_centered, 0)
            return K_centered


        Ky = compute_kappa_matrix(y_prepared)
        Ky_c = double_center(Ky)
        gcov2_yy = np.sum(Ky_c * Ky_c) / (n * (n - 3))

        def compute_gcor_for_feature(j):
            x_j = X[:, j].reshape(-1, 1)

            Kx = compute_kappa_matrix(x_j)
            Kx_c = double_center(Kx)

            gcov2_xy = np.sum(Kx_c * Ky_c) / (n * (n - 3))
            gcov2_xx = np.sum(Kx_c * Kx_c) / (n * (n - 3))

            if gcov2_xx * gcov2_yy < 1e-15:
                return 0.0

            gcor2 = gcov2_xy / np.sqrt(gcov2_xx * gcov2_yy)
            return max(0.0, min(1.0, np.sqrt(abs(gcor2))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_gcor_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_gcor_for_feature(j) for j in range(p)])

    # =========================================================================


    @staticmethod
    def emb_dc_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        b = cdist(y_prepared, y_prepared, 'euclidean')
        b_centered = b - b.mean(-1, keepdims=True) - b.mean(-2, keepdims=True) + b.mean(axis=(-2, -1), keepdims=True)
        b_var = np.sum(b_centered ** 2)

        def compute_dc_for_feature(j):

            a = cdist(embeddings[:, j, :], embeddings[:, j, :], 'euclidean')
            a_centered = a - a.mean(-1, keepdims=True) - a.mean(-2, keepdims=True) + a.mean(axis=(-2, -1), keepdims=True)

            num = np.sum(a_centered * b_centered)
            a_var = np.sum(a_centered ** 2)

            if a_var * b_var < 1e-15:
                return 0.0

            dc_sq = num / np.sqrt(a_var * b_var)
            return max(0, min(1, np.sqrt(abs(dc_sq))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_dc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_dc_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_sc_accelerated(embeddings: np.ndarray, y: np.ndarray, a: float = 0.5,
                           n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        dist_y = cdist(y_prepared, y_prepared, 'euclidean')
        Ky = np.exp(-np.power(dist_y, a))

        E2_y = (Ky.sum() - np.trace(Ky)) / (n * (n - 1))
        Kyy = np.exp(-np.power(dist_y, a) * 2)
        E1_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        E2_yy = E2_y ** 2

        E3_yy = 0.0
        for i in range(n):
            Ky_row_i = np.delete(Ky[i, :], i)
            E3_yy += Ky_row_i.sum() ** 2
        E3_yy = E3_yy / (n * (n - 1) * (n - 2))
        Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

        def compute_sc_for_feature(j):

            dist_x = cdist(embeddings[:, j, :], embeddings[:, j, :], 'euclidean')
            Kx = np.exp(-np.power(dist_x, a))

            Kxy = Kx * Ky
            E1 = (Kxy.sum() - np.trace(Kxy)) / (n * (n - 1))

            E2_x = (Kx.sum() - np.trace(Kx)) / (n * (n - 1))
            E2 = E2_x * E2_y

            E3 = 0.0
            for i in range(n):
                Kx_row_i = np.delete(Kx[i, :], i)
                Ky_row_i = np.delete(Ky[i, :], i)
                E3 += Kx_row_i.sum() * Ky_row_i.sum()
            E3 = E3 / (n * (n - 1) * (n - 2))

            Scov2 = E1 + E2 - 2 * E3

            Kxx = np.exp(-np.power(dist_x, a) * 2)
            E1_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            E2_xx = E2_x ** 2

            E3_xx = 0.0
            for i in range(n):
                Kx_row_i = np.delete(Kx[i, :], i)
                E3_xx += Kx_row_i.sum() ** 2
            E3_xx = E3_xx / (n * (n - 1) * (n - 2))

            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx

            if Svar_x2 <= 0 or Svar_y2 <= 0:
                return 0.0

            SC2 = Scov2 / np.sqrt(Svar_x2 * Svar_y2)
            return max(0.0, min(1.0, np.sqrt(abs(SC2))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_sc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_sc_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_ipc_accelerated(embeddings: np.ndarray, y: np.ndarray, sigma: float = None,
                            n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        y_scaled = (y_prepared - y_prepared.mean(axis=0)) / (y_prepared.std(axis=0) + 1e-10)


        if sigma is None:
            sigma_y = np.sqrt(np.median(np.sum(y_scaled ** 2, axis=1)))
        else:
            sigma_y = sigma


        y_sq = np.sum(y_scaled ** 2, axis=1)
        y_norm = np.sqrt(sigma_y ** 2 + y_sq)
        Gy = y_scaled / (y_norm[:, np.newaxis] * np.sqrt(sigma_y ** 2))
        N = Gy @ Gy.T
        H = np.eye(n) - np.ones((n, n)) / n
        NH = N @ H

        def compute_ipc_for_feature(j):

            Z = embeddings[:, j, :]
            Z_centered = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-10)


            if sigma is None:
                sigma_j = np.sqrt(np.median(np.sum(Z_centered ** 2, axis=1)))
            else:
                sigma_j = sigma

            z_sq = np.sum(Z_centered ** 2, axis=1)
            z_norm = np.sqrt(sigma_j ** 2 + z_sq)
            Gz = Z_centered / (z_norm[:, np.newaxis] * np.sqrt(sigma_j ** 2))

            M = Gz @ Gz.T
            MH = M @ H

            denom = np.trace(MH @ MH) * np.trace(NH @ NH)
            if denom < 1e-15:
                return 0.0

            ipc = np.trace(MH @ NH) / np.sqrt(denom)
            return max(0, min(1, ipc))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_ipc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_ipc_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_pc_accelerated(embeddings: np.ndarray, y: np.ndarray, sigma: float = 1.0,
                           n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)


        y_scaled = (y_prepared - y_prepared.mean(axis=0)) / (y_prepared.std(axis=0) + 1e-10)


        y_sq = np.sum(y_scaled ** 2, axis=1)
        dot_y = y_scaled @ y_scaled.T
        denom_y = np.sqrt((sigma**2 + y_sq[:, np.newaxis]) * (sigma**2 + y_sq[np.newaxis, :]))
        cos_y = np.clip((sigma**2 + dot_y) / (denom_y + 1e-10), -1, 1)
        Ay = np.arccos(cos_y)

        def compute_pc_for_feature(j):
            Z = embeddings[:, j, :]
            Z_centered = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-10)

            z_sq = np.sum(Z_centered ** 2, axis=1)
            dot_z = Z_centered @ Z_centered.T
            denom_z = np.sqrt((sigma**2 + z_sq[:, np.newaxis]) * (sigma**2 + z_sq[np.newaxis, :]))
            cos_z = np.clip((sigma**2 + dot_z) / (denom_z + 1e-10), -1, 1)
            Az = np.arccos(cos_z)

            n_pairs = n * (n - 1)
            term1 = (Az * Ay).sum() / n_pairs
            term2 = 2 * (Az.mean(axis=1) * Ay.mean(axis=1)).mean()
            term3 = Az.mean() * Ay.mean()

            pcov_xy = term1 - term2 + term3
            pcov_xx = (Az ** 2).sum() / n_pairs - 2 * (Az.mean(axis=1) ** 2).mean() + Az.mean() ** 2
            pcov_yy = (Ay ** 2).sum() / n_pairs - 2 * (Ay.mean(axis=1) ** 2).mean() + Ay.mean() ** 2

            if pcov_xx < 1e-10 or pcov_yy < 1e-10:
                return 0.0

            pc_sq = pcov_xy / np.sqrt(pcov_xx * pcov_yy)
            return max(0, min(1, pc_sq))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_pc_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_pc_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_fkf_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}

        def compute_fkf_for_feature(j):
            total_ks = 0.0
            for dim in range(d):
                max_ks = 0.0
                for r in classes:
                    mask_r = class_masks[r]
                    mask_not_r = ~mask_r
                    if mask_r.sum() > 0 and mask_not_r.sum() > 0:
                        ks_stat, _ = stats.ks_2samp(embeddings[mask_r, j, dim],
                                                     embeddings[mask_not_r, j, dim])
                        max_ks = max(max_ks, ks_stat)
                total_ks += max_ks
            return total_ks / d

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_fkf_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_fkf_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_cru_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}
        pi_r = {r: mask.mean() for r, mask in class_masks.items()}

        def compute_cru_for_feature(j):
            total_cru = 0.0
            for dim in range(d):
                dim_total = 0.0
                x_dim = embeddings[:, j, dim]
                for r in classes:
                    mask_r = class_masks[r]
                    mask_not_r = ~mask_r
                    if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                        continue


                    x_r = x_dim[mask_r]
                    x_not_r = x_dim[mask_not_r]
                    count = sum(np.sum(x_not_r < val) for val in x_r)
                    gamma = count / (len(x_r) * len(x_not_r))

                    weight = (pi_r[r] * (1 - pi_r[r])) ** 2
                    dim_total += weight * (gamma - 0.5) ** 2
                total_cru += dim_total
            return total_cru / d

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_cru_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_cru_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_cavs_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}
        pi_r = {r: mask.mean() for r, mask in class_masks.items()}

        def compute_cavs_for_feature(j):
            total_cavs = 0.0
            for dim in range(d):
                x_dim = embeddings[:, j, dim]
                max_val = 0.0
                for r in classes:
                    mask_r = class_masks[r]
                    mask_not_r = ~mask_r
                    if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                        continue


                    x_r = x_dim[mask_r]
                    x_not_r = x_dim[mask_not_r]
                    count = sum(np.sum(x_not_r < val) for val in x_r)
                    gamma = count / (len(x_r) * len(x_not_r))

                    weight = 1 - pi_r[r]
                    max_val = max(max_val, weight * abs(gamma - 0.5))
                total_cavs += max_val
            return total_cavs / d

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_cavs_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_cavs_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_mv_sis_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}
        pi_r = {r: mask.mean() for r, mask in class_masks.items()}

        def compute_mv_sis_for_feature(j):
            total_mv = 0.0
            for dim in range(d):
                x_dim = embeddings[:, j, dim]
                dim_total = 0.0
                for r in classes:
                    mask_r = class_masks[r]
                    mask_not_r = ~mask_r
                    if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                        continue


                    x_r = x_dim[mask_r]
                    x_not_r = x_dim[mask_not_r]
                    count = sum(np.sum(x_not_r < val) for val in x_r)
                    gamma = count / (len(x_r) * len(x_not_r))

                    weight = pi_r[r] * (1 - pi_r[r]) ** 2
                    dim_total += weight * (gamma - 0.5) ** 2
                total_mv += dim_total
            return total_mv / d

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_mv_sis_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_mv_sis_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_lr_ffs_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape
        classes = np.unique(y)


        class_masks = {r: (y == r) for r in classes}

        def compute_lr_ffs_for_feature(j):
            total_lr = 0.0
            for dim in range(d):
                x_dim = embeddings[:, j, dim]
                max_dev = 0.0
                for r in classes:
                    mask_r = class_masks[r]
                    mask_not_r = ~mask_r
                    if mask_r.sum() == 0 or mask_not_r.sum() == 0:
                        continue


                    x_r = x_dim[mask_r]
                    x_not_r = x_dim[mask_not_r]
                    count = sum(np.sum(x_not_r < val) for val in x_r)
                    gamma = count / (len(x_r) * len(x_not_r))

                    max_dev = max(max_dev, abs(gamma - 0.5))
                total_lr += max_dev
            return total_lr / d

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_lr_ffs_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_lr_ffs_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_psis_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape
        classes = np.unique(y)
        n_classes = len(classes)


        class_masks = {c: (y == c) for c in classes}

        def compute_psis_for_feature(j):
            total_psis = 0.0
            for dim in range(d):
                x_dim = embeddings[:, j, dim]


                class_medians = {}
                for c in classes:
                    mask_c = class_masks[c]
                    if mask_c.sum() > 0:
                        class_medians[c] = np.median(x_dim[mask_c])


                max_diff = 0.0
                for i in range(n_classes):
                    for k in range(i + 1, n_classes):
                        c_i = classes[i]
                        c_k = classes[k]
                        if c_i in class_medians and c_k in class_medians:
                            diff = abs(class_medians[c_i] - class_medians[c_k])
                            max_diff = max(max_diff, diff)
                total_psis += max_diff
            return total_psis / d

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_psis_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_psis_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_sis_accelerated(embeddings: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        n, p, d = embeddings.shape


        unique_vals = np.unique(y)
        is_categorical = len(unique_vals) < n / 2 or not np.issubdtype(y.dtype, np.number)

        if is_categorical:

            classes = unique_vals
            Y_encoded = np.zeros((n, len(classes)))
            for i, c in enumerate(classes):
                Y_encoded[:, i] = (y == c).astype(float)

            def compute_sis_for_feature(j):
                total_corr = 0.0
                count = 0
                for dim in range(d):
                    x_dim = embeddings[:, j, dim]
                    if np.std(x_dim) < 1e-10:
                        continue

                    max_corr_dim = 0.0
                    for k in range(Y_encoded.shape[1]):
                        if np.std(Y_encoded[:, k]) > 1e-10:
                            corr = abs(np.corrcoef(x_dim, Y_encoded[:, k])[0, 1])
                            if not np.isnan(corr):
                                max_corr_dim = max(max_corr_dim, corr)
                    total_corr += max_corr_dim
                    count += 1
                return total_corr / count if count > 0 else 0.0

            if HAS_JOBLIB and n_jobs != 1:
                actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
                scores = Parallel(n_jobs=actual_jobs)(
                    delayed(compute_sis_for_feature)(j) for j in range(p)
                )
                return np.array(scores)
            else:
                return np.array([compute_sis_for_feature(j) for j in range(p)])
        else:

            y_centered = (y - y.mean()) / (y.std() + 1e-10)

            def compute_sis_for_feature(j):
                total_corr = 0.0
                count = 0
                for dim in range(d):
                    x_dim = embeddings[:, j, dim]
                    if np.std(x_dim) < 1e-10:
                        continue
                    corr = abs(np.corrcoef(x_dim, y_centered)[0, 1])
                    if not np.isnan(corr):
                        total_corr += corr
                        count += 1
                return total_corr / count if count > 0 else 0.0

            if HAS_JOBLIB and n_jobs != 1:
                actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
                scores = Parallel(n_jobs=actual_jobs)(
                    delayed(compute_sis_for_feature)(j) for j in range(p)
                )
                return np.array(scores)
            else:
                return np.array([compute_sis_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_hsic_accelerated(embeddings: np.ndarray, y: np.ndarray, sigma=None,
                             n_jobs: int = -1, *, task=None) -> np.ndarray:
        from src.screening_utils import pace_hsic_target

        n, p, _ = embeddings.shape
        target, categorical = pace_hsic_target(y, task)

        def centered_kernel(values, width):
            distances = cdist(values, values, 'euclidean')
            if width is None:
                pairs = distances[np.triu_indices(n, 1)]
                positive = pairs[pairs > 0]
                mid = (positive.size - 1) // 2
                width = np.partition(positive, mid)[mid] if positive.size else 1.0
            kernel = np.exp(-0.5 * (distances / width) ** 2)
            return kernel - kernel.mean(0) - kernel.mean(1, keepdims=True) + kernel.mean()

        response = centered_kernel(target, 1.0 if categorical and sigma is None else sigma)

        def compute_hsic_for_feature(j):
            kernel = centered_kernel(embeddings[:, j, :], sigma)
            return max(0.0, (kernel * response).mean())

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_hsic_for_feature)(j) for j in range(p)
            )
            return np.array(scores)
        else:
            return np.array([compute_hsic_for_feature(j) for j in range(p)])

    @staticmethod
    def emb_swc_accelerated(embeddings: np.ndarray, y: np.ndarray, L: int = 20, p_order: int = 1,
                            n_shuf: int = 11, qmc: bool = True, seed: int = 42,
                            n_jobs: int = -1) -> np.ndarray:
        n, n_features, d = embeddings.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)
        dy = y_prepared.shape[1] if y_prepared.ndim > 1 else 1
        if y_prepared.ndim == 1:
            y_prepared = y_prepared.reshape(-1, 1)


        gaussian_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        def rank_gaussian_transform(x_1d):
            ranks = stats.rankdata(x_1d, method='average')
            return gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]


        gry = np.zeros_like(y_prepared)
        for k in range(dy):
            gry[:, k] = rank_gaussian_transform(y_prepared[:, k])


        emb_transformed = np.zeros_like(embeddings)
        for j in range(n_features):
            for dim in range(d):
                emb_transformed[:, j, dim] = rank_gaussian_transform(embeddings[:, j, dim])


        total_dir = (d + dy) * L
        if qmc:
            try:
                from scipy.stats.qmc import Sobol
                sobol_engine = Sobol(d=d + dy - 1, scramble=True, seed=seed)
                vec_meta = sobol_engine.random(total_dir)
                dir_xy = np.array([
                    CorrelationMeasures._uniform2sphere_qmc(vec_meta[i])
                    for i in range(total_dir)
                ])
            except ImportError:
                dir_xy = CorrelationMeasures._uniform2sphere_mc(total_dir, d + dy, seed)
        else:
            dir_xy = CorrelationMeasures._uniform2sphere_mc(total_dir, d + dy, seed)

        def compute_swc_for_feature(j):
            emb_j = emb_transformed[:, j, :]  # shape (n, d)
            scores_shuf = np.zeros(n_shuf)

            for s in range(n_shuf):
                rng_shuf = np.random.RandomState(seed + s + 1)
                shuf_idx = rng_shuf.permutation(n)
                emb_shuf = emb_j[shuf_idx]

                L_mat = np.zeros(total_dir)
                for i in range(total_dir):
                    proj_joint = emb_j @ dir_xy[i, :d] + gry @ dir_xy[i, d:d + dy]
                    proj_indep = emb_shuf @ dir_xy[i, :d] + gry @ dir_xy[i, d:d + dy]

                    if p_order == 1:
                        L_mat[i] = np.mean(np.abs(np.sort(proj_joint) - np.sort(proj_indep)))
                    else:
                        L_mat[i] = np.mean(np.abs(np.sort(proj_joint) - np.sort(proj_indep)) ** p_order)

                scores_shuf[s] = np.mean(L_mat) ** (1 / p_order)

            return np.median(scores_shuf)

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_swc_for_feature)(j) for j in range(n_features)
            )
            return np.array(scores)
        else:
            return np.array([compute_swc_for_feature(j) for j in range(n_features)])

    @staticmethod
    def emb_hd_accelerated(embeddings: np.ndarray, y: np.ndarray, K: int = None,
                           n_jobs: int = -1) -> np.ndarray:
        n, n_features, d = embeddings.shape
        classes = np.unique(y)
        is_categorical = len(classes) < n / 2 or not np.issubdtype(y.dtype, np.number)

        def compute_hd_for_feature(j):
            total_hd = 0.0
            count = 0
            for dim in range(d):
                x_dim = embeddings[:, j, dim]
                if is_categorical:
                    total_hd += CorrelationMeasures.hellinger_distance_index(x_dim, y)
                else:
                    total_hd += CorrelationMeasures.avg_hellinger_distance_index(x_dim, y, K=K)
                count += 1
            return total_hd / count if count > 0 else 0.0

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_hd_for_feature)(j) for j in range(n_features)
            )
            return np.array(scores)
        else:
            return np.array([compute_hd_for_feature(j) for j in range(n_features)])

    @staticmethod
    def emb_gcor_accelerated(embeddings: np.ndarray, y: np.ndarray,
                             n_jobs: int = -1) -> np.ndarray:
        n, n_features, d = embeddings.shape


        y_prepared = AcceleratedScreening._prepare_y_for_screening(y, n)

        def compute_kappa_matrix(Z):
            dot_ZZ = Z @ Z.T
            norm_sq = np.sum(Z ** 2, axis=1)
            num = 1 + dot_ZZ
            denom = np.sqrt((1 + norm_sq[:, np.newaxis]) * (1 + norm_sq[np.newaxis, :]))
            cos_val = np.clip(num / (denom + 1e-15), -1, 1)
            return np.arcsin(cos_val)

        def double_center(K_matrix):
            n_sq = K_matrix.shape[0]
            K_off = K_matrix.copy()
            np.fill_diagonal(K_off, 0)
            row_sums = K_off.sum(axis=1)
            col_sums = K_off.sum(axis=0)
            total_sum = K_off.sum()
            a_row = row_sums / (n_sq - 2)
            a_col = col_sums / (n_sq - 2)
            a_total = total_sum / ((n_sq - 1) * (n_sq - 2))
            K_centered = K_matrix - a_row[:, np.newaxis] - a_col[np.newaxis, :] + a_total
            np.fill_diagonal(K_centered, 0)
            return K_centered


        Ky = compute_kappa_matrix(y_prepared)
        Ky_c = double_center(Ky)
        gcov2_yy = np.sum(Ky_c * Ky_c) / (n * (n - 3))

        def compute_gcor_for_feature(j):

            Z = embeddings[:, j, :]

            Kx = compute_kappa_matrix(Z)
            Kx_c = double_center(Kx)

            gcov2_xy = np.sum(Kx_c * Ky_c) / (n * (n - 3))
            gcov2_xx = np.sum(Kx_c * Kx_c) / (n * (n - 3))

            if gcov2_xx * gcov2_yy < 1e-15:
                return 0.0

            gcor2 = gcov2_xy / np.sqrt(gcov2_xx * gcov2_yy)
            return max(0.0, min(1.0, np.sqrt(abs(gcor2))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_gcor_for_feature)(j) for j in range(n_features)
            )
            return np.array(scores)
        else:
            return np.array([compute_gcor_for_feature(j) for j in range(n_features)])

    @staticmethod
    def emb_fisher_accelerated(embeddings: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p, d = embeddings.shape
        scores = np.zeros(p)
        classes = np.unique(y)

        for j in range(p):
            Z = embeddings[:, j, :]
            class_means = []
            class_vars = []

            for c in classes:
                mask = (y == c)
                Z_c = Z[mask]
                class_means.append(Z_c.mean(axis=0))
                class_vars.append(Z_c.var(axis=0).sum())

            if len(classes) == 2:
                distance = np.linalg.norm(class_means[0] - class_means[1])
            else:
                max_dist = 0
                for k in range(len(classes)):
                    for l in range(k + 1, len(classes)):
                        dist = np.linalg.norm(class_means[k] - class_means[l])
                        max_dist = max(max_dist, dist)
                distance = max_dist

            pooled_var = sum(class_vars) + 1e-8
            scores[j] = distance / np.sqrt(pooled_var)

        return scores

    # =========================================================================

    # =========================================================================

    @staticmethod
    def compute_all_scores(X: np.ndarray, y: np.ndarray,
                           methods: List[str] = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru',
                                                  'cavs', 'mv_sis', 'lr_ffs', 'psis', 'swc',
                                                  'hd', 'gcor'],
                           n_jobs: int = -1) -> Dict[str, np.ndarray]:
        results = {}

        if 'sis' in methods:
            results['sis'] = AcceleratedScreening.sis_accelerated(X, y, n_jobs)

        if 'dc' in methods:
            results['dc'] = AcceleratedScreening.dc_sis_accelerated(X, y, n_jobs)

        if 'sc' in methods:
            results['sc'] = AcceleratedScreening.sc_sis_accelerated(X, y, n_jobs=n_jobs)

        if 'ipc' in methods:
            results['ipc'] = AcceleratedScreening.ipc_sis_accelerated(X, y, n_jobs=n_jobs)

        if 'pc' in methods:
            results['pc'] = AcceleratedScreening.pc_sis_accelerated(X, y, n_jobs=n_jobs)

        if 'fkf' in methods:
            results['fkf'] = AcceleratedScreening.fkf_accelerated(X, y, n_jobs)

        if 'cru' in methods:
            results['cru'] = AcceleratedScreening.cru_accelerated(X, y, n_jobs)

        if 'cavs' in methods:
            results['cavs'] = AcceleratedScreening.cavs_accelerated(X, y, n_jobs)

        if 'mv_sis' in methods:
            results['mv_sis'] = AcceleratedScreening.mv_sis_accelerated(X, y, n_jobs)

        if 'lr_ffs' in methods:
            results['lr_ffs'] = AcceleratedScreening.lr_ffs_accelerated(X, y, n_jobs)

        if 'psis' in methods:
            results['psis'] = AcceleratedScreening.psis_accelerated(X, y, n_jobs)

        if 'swc' in methods:
            results['swc'] = AcceleratedScreening.swc_accelerated(X, y, n_jobs=n_jobs)

        if 'hd' in methods:
            results['hd'] = AcceleratedScreening.hd_sis_accelerated(X, y, n_jobs=n_jobs)

        if 'gcor' in methods:
            results['gcor'] = AcceleratedScreening.gcor_sis_accelerated(X, y, n_jobs=n_jobs)

        return results

    @staticmethod
    def emb_compute_all_scores(embeddings: np.ndarray, y: np.ndarray,
                                methods: List[str] = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hsic',
                                                       'fkf', 'cru', 'cavs', 'mv_sis', 'lr_ffs', 'psis',
                                                       'swc', 'hd', 'gcor'],
                                n_jobs: int = -1) -> Dict[str, np.ndarray]:
        results = {}

        if 'sis' in methods:
            results['emb_sis'] = AcceleratedScreening.emb_sis_accelerated(embeddings, y, n_jobs)

        if 'dc' in methods:
            results['emb_dc'] = AcceleratedScreening.emb_dc_accelerated(embeddings, y, n_jobs)

        if 'sc' in methods:
            results['emb_sc'] = AcceleratedScreening.emb_sc_accelerated(embeddings, y, n_jobs=n_jobs)

        if 'ipc' in methods:
            results['emb_ipc'] = AcceleratedScreening.emb_ipc_accelerated(embeddings, y, n_jobs=n_jobs)

        if 'pc' in methods:
            results['emb_pc'] = AcceleratedScreening.emb_pc_accelerated(embeddings, y, n_jobs)

        if 'hsic' in methods:
            results['emb_hsic'] = AcceleratedScreening.emb_hsic_accelerated(embeddings, y, n_jobs=n_jobs)

        if 'swc' in methods:
            results['emb_swc'] = AcceleratedScreening.emb_swc_accelerated(embeddings, y, n_jobs=n_jobs)

        if 'hd' in methods:
            results['emb_hd'] = AcceleratedScreening.emb_hd_accelerated(embeddings, y, n_jobs=n_jobs)

        if 'gcor' in methods:
            results['emb_gcor'] = AcceleratedScreening.emb_gcor_accelerated(embeddings, y, n_jobs=n_jobs)

        if 'fkf' in methods:
            results['emb_fkf'] = AcceleratedScreening.emb_fkf_accelerated(embeddings, y, n_jobs)

        if 'cru' in methods:
            results['emb_cru'] = AcceleratedScreening.emb_cru_accelerated(embeddings, y, n_jobs)

        if 'cavs' in methods:
            results['emb_cavs'] = AcceleratedScreening.emb_cavs_accelerated(embeddings, y, n_jobs)

        if 'mv_sis' in methods:
            results['emb_mv_sis'] = AcceleratedScreening.emb_mv_sis_accelerated(embeddings, y, n_jobs)

        if 'lr_ffs' in methods:
            results['emb_lr_ffs'] = AcceleratedScreening.emb_lr_ffs_accelerated(embeddings, y, n_jobs)

        if 'psis' in methods:
            results['emb_psis'] = AcceleratedScreening.emb_psis_accelerated(embeddings, y, n_jobs)

        return results

    # =========================================================================

    # =========================================================================

    @staticmethod
    def group_dc_accelerated(X: np.ndarray, y: np.ndarray, groups: List[Tuple],
                              n_jobs: int = -1) -> np.ndarray:
        n_groups = len(groups)

        if y.ndim == 1:
            y = y.reshape(-1, 1)


        b = cdist(y, y, 'euclidean')
        b_centered = b - b.mean(-1, keepdims=True) - b.mean(-2, keepdims=True) + b.mean(axis=(-2, -1), keepdims=True)
        b_var = np.sum(b_centered ** 2)

        def compute_dc_for_group(i):
            group = groups[i]
            X_group = X[:, list(group)]  # shape: (n, k)


            a = cdist(X_group, X_group, 'euclidean')
            a_centered = a - a.mean(-1, keepdims=True) - a.mean(-2, keepdims=True) + a.mean(axis=(-2, -1), keepdims=True)

            num = np.sum(a_centered * b_centered)
            a_var = np.sum(a_centered ** 2)

            if a_var * b_var < 1e-15:
                return 0.0

            dc_sq = num / np.sqrt(a_var * b_var)
            return max(0, min(1, np.sqrt(abs(dc_sq))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_dc_for_group)(i) for i in range(n_groups)
            )
            return np.array(scores)
        else:
            return np.array([compute_dc_for_group(i) for i in range(n_groups)])

    @staticmethod
    def group_sc_accelerated(X: np.ndarray, y: np.ndarray, groups: List[Tuple],
                              a: float = 0.5, n_jobs: int = -1) -> np.ndarray:
        n_groups = len(groups)
        n = X.shape[0]

        if y.ndim == 1:
            y = y.reshape(-1, 1)


        dist_y = cdist(y, y, 'euclidean')
        Ky = np.exp(-np.power(dist_y, a))

        E2_y = (Ky.sum() - np.trace(Ky)) / (n * (n - 1))
        Kyy = np.exp(-np.power(dist_y, a) * 2)
        E1_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
        E2_yy = E2_y ** 2

        E3_yy = 0.0
        for i in range(n):
            Ky_row_i = np.delete(Ky[i, :], i)
            E3_yy += Ky_row_i.sum() ** 2
        E3_yy = E3_yy / (n * (n - 1) * (n - 2))
        Svar_y2 = E1_yy + E2_yy - 2 * E3_yy

        def compute_sc_for_group(idx):
            group = groups[idx]
            X_group = X[:, list(group)]

            dist_x = cdist(X_group, X_group, 'euclidean')
            Kx = np.exp(-np.power(dist_x, a))

            Kxy = Kx * Ky
            E1 = (Kxy.sum() - np.trace(Kxy)) / (n * (n - 1))

            E2_x = (Kx.sum() - np.trace(Kx)) / (n * (n - 1))
            E2 = E2_x * E2_y

            E3 = 0.0
            for i in range(n):
                Kx_row_i = np.delete(Kx[i, :], i)
                Ky_row_i = np.delete(Ky[i, :], i)
                E3 += Kx_row_i.sum() * Ky_row_i.sum()
            E3 = E3 / (n * (n - 1) * (n - 2))

            Scov2 = E1 + E2 - 2 * E3

            Kxx = np.exp(-np.power(dist_x, a) * 2)
            E1_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            E2_xx = E2_x ** 2

            E3_xx = 0.0
            for i in range(n):
                Kx_row_i = np.delete(Kx[i, :], i)
                E3_xx += Kx_row_i.sum() ** 2
            E3_xx = E3_xx / (n * (n - 1) * (n - 2))

            Svar_x2 = E1_xx + E2_xx - 2 * E3_xx

            if Svar_x2 <= 0 or Svar_y2 <= 0:
                return 0.0

            SC2 = Scov2 / np.sqrt(Svar_x2 * Svar_y2)
            return max(0.0, min(1.0, np.sqrt(abs(SC2))))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_sc_for_group)(i) for i in range(n_groups)
            )
            return np.array(scores)
        else:
            return np.array([compute_sc_for_group(i) for i in range(n_groups)])

    @staticmethod
    def group_ipc_accelerated(X: np.ndarray, y: np.ndarray, groups: List[Tuple],
                               sigma: float = None, n_jobs: int = -1) -> np.ndarray:
        n_groups = len(groups)
        n = X.shape[0]


        y_scaled = (y - y.mean()) / (y.std() + 1e-10)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)


        if sigma is None:
            sigma_y = np.sqrt(np.median(np.sum(y_scaled ** 2, axis=1)))
        else:
            sigma_y = sigma


        y_sq = np.sum(y_scaled ** 2, axis=1)
        y_norm = np.sqrt(sigma_y ** 2 + y_sq)
        Gy = y_scaled / (y_norm[:, np.newaxis] * np.sqrt(sigma_y ** 2))
        N = Gy @ Gy.T
        H = np.eye(n) - np.ones((n, n)) / n
        NH = N @ H

        def compute_ipc_for_group(idx):
            group = groups[idx]
            X_group = X[:, list(group)]


            Z_centered = (X_group - X_group.mean(axis=0)) / (X_group.std(axis=0) + 1e-10)


            if sigma is None:
                sigma_g = np.sqrt(np.median(np.sum(Z_centered ** 2, axis=1)))
            else:
                sigma_g = sigma

            z_sq = np.sum(Z_centered ** 2, axis=1)
            z_norm = np.sqrt(sigma_g ** 2 + z_sq)
            Gz = Z_centered / (z_norm[:, np.newaxis] * np.sqrt(sigma_g ** 2))

            M = Gz @ Gz.T
            MH = M @ H

            ipc = np.trace(MH @ NH) / np.sqrt(np.trace(MH @ MH) * np.trace(NH @ NH))
            return max(0, min(1, ipc))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_ipc_for_group)(i) for i in range(n_groups)
            )
            return np.array(scores)
        else:
            return np.array([compute_ipc_for_group(i) for i in range(n_groups)])

    @staticmethod
    def group_pc_accelerated(X: np.ndarray, y: np.ndarray, groups: List[Tuple],
                              sigma: float = None, n_jobs: int = -1) -> np.ndarray:
        n_groups = len(groups)
        n = X.shape[0]


        y_scaled = (y - y.mean()) / (y.std() + 1e-10)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)


        if sigma is None:
            sigma_y = np.sqrt(np.median(np.sum(y_scaled ** 2, axis=1)))
        else:
            sigma_y = sigma


        Ay = _compute_A_matrix(y_scaled, sigma_y)
        n_pairs = n * (n - 1)
        E2_y = 2 * (Ay.mean(axis=1) ** 2).mean()
        E3_y = Ay.mean() ** 2
        pcov_yy = (Ay ** 2).sum() / n_pairs - E2_y + E3_y

        def compute_pc_for_group(idx):
            group = groups[idx]
            X_group = X[:, list(group)]


            Z_centered = (X_group - X_group.mean(axis=0)) / (X_group.std(axis=0) + 1e-10)


            if sigma is None:
                sigma_g = np.sqrt(np.median(np.sum(Z_centered ** 2, axis=1)))
            else:
                sigma_g = sigma

            Az = _compute_A_matrix(Z_centered, sigma_g)

            term1 = (Az * Ay).sum() / n_pairs
            term2 = 2 * (Az.mean(axis=1) * Ay.mean(axis=1)).mean()
            term3 = Az.mean() * Ay.mean()

            pcov_xy = term1 - term2 + term3
            pcov_xx = (Az ** 2).sum() / n_pairs - 2 * (Az.mean(axis=1) ** 2).mean() + Az.mean() ** 2

            if pcov_xx < 1e-10 or pcov_yy < 1e-10:
                return 0.0

            pc_sq = pcov_xy / np.sqrt(pcov_xx * pcov_yy)
            return max(0, min(1, pc_sq))

        if HAS_JOBLIB and n_jobs != 1:
            actual_jobs = cpu_count() if n_jobs == -1 else n_jobs
            scores = Parallel(n_jobs=actual_jobs)(
                delayed(compute_pc_for_group)(i) for i in range(n_groups)
            )
            return np.array(scores)
        else:
            return np.array([compute_pc_for_group(i) for i in range(n_groups)])

    @staticmethod
    def group_compute_all_scores(X: np.ndarray, y: np.ndarray, groups: List[Tuple],
                                  metrics: List[str] = ['dc', 'sc', 'ipc', 'pc'],
                                  n_jobs: int = -1) -> Dict[str, np.ndarray]:
        results = {}

        if 'dc' in metrics:
            results['dc'] = AcceleratedScreening.group_dc_accelerated(X, y, groups, n_jobs)

        if 'sc' in metrics:
            results['sc'] = AcceleratedScreening.group_sc_accelerated(X, y, groups, n_jobs=n_jobs)

        if 'ipc' in metrics:
            results['ipc'] = AcceleratedScreening.group_ipc_accelerated(X, y, groups, n_jobs=n_jobs)

        if 'pc' in metrics:
            results['pc'] = AcceleratedScreening.group_pc_accelerated(X, y, groups, n_jobs=n_jobs)

        return results


# =============================================================================

# =============================================================================

class AcceleratedEmbedder:

    def __init__(self, col_embedder=None, device='cuda:0'):
        self.col_embedder = col_embedder
        self.device = device
        self.loaded = False
        if col_embedder is not None:
            self.embed_dim = col_embedder.embed_dim
            self.n_cls_tokens = getattr(col_embedder, 'reserve_cls_tokens', 4)
            self.n_blocks = len(col_embedder.tf_col.blocks)
            self.loaded = True

    def load_model(self, model_path: str, seed: int = 42, device: str = None) -> bool:
        if self.loaded:
            return True
        try:
            from tabicl import TabICLClassifier

            if device is not None:
                self.device = device

            model = TabICLClassifier(
                model_path=model_path,
                n_estimators=1,
                random_state=seed,
                device=self.device
            )


            model.fit([[0, 1], [1, 0]], [0, 1])

            self.col_embedder = model.model_.col_embedder
            self.embed_dim = self.col_embedder.embed_dim
            self.n_cls_tokens = getattr(self.col_embedder, 'reserve_cls_tokens', 4)
            self.n_blocks = len(self.col_embedder.tf_col.blocks)
            self.loaded = True

            print(f"TabICL loaded: embed_dim={self.embed_dim}, blocks={self.n_blocks}", flush=True)
            return True
        except Exception as e:
            print(f"Failed to load TabICL: {e}", flush=True)
            return False

    def preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        X_norm = np.zeros_like(X)

        for j in range(p):
            x = X[:, j]
            if np.std(x) < 1e-10:
                X_norm[:, j] = 0
            else:
                ranked = stats.rankdata(x)
                x_j_norm = (ranked - 0.5) / n
                X_norm[:, j] = stats.norm.ppf(np.clip(x_j_norm, 1e-6, 1 - 1e-6))

        return X_norm

    def get_embeddings_batch(self, X: np.ndarray, layer_idx: int = 2,
                              batch_size: int = 32) -> np.ndarray:
        n, p = X.shape
        embeddings = np.zeros((n, p, self.embed_dim))


        X_norm = self.preprocess_batch(X)

        n_cls = self.n_cls_tokens


        for start in range(0, p, batch_size):
            end = min(start + batch_size, p)
            batch_p = end - start


            x_input = np.zeros((batch_p, 1 + n_cls, n, 1), dtype=np.float32)
            for i, j in enumerate(range(start, end)):
                x_input[i, n_cls:, :, 0] = X_norm[:, j]

            x_tensor = torch.tensor(x_input, device=self.device)


            self.col_embedder.eval()
            with torch.no_grad():
                src = self.col_embedder.in_linear(x_tensor)
                out = src
                for i, block in enumerate(self.col_embedder.tf_col.blocks):
                    out = block(out)
                    if i + 1 == layer_idx:
                        break


            batch_embeddings = out[:, n_cls:, :, :].cpu().numpy()  # (batch_p, 1, n, embed_dim)

            for i, j in enumerate(range(start, end)):
                embeddings[:, j, :] = batch_embeddings[i, 0, :, :]

        return embeddings

    def get_embeddings_batch_all_layers(self, X: np.ndarray,
                                          layers: List[int] = [1, 2, 3],
                                          batch_size: int = 32) -> Dict[int, np.ndarray]:
        n, p = X.shape
        layer_set = set(layers)
        max_layer = max(layers)
        results = {l: np.zeros((n, p, self.embed_dim)) for l in layers}

        X_norm = self.preprocess_batch(X)
        n_cls = self.n_cls_tokens

        for start in range(0, p, batch_size):
            end = min(start + batch_size, p)
            batch_p = end - start

            x_input = np.zeros((batch_p, 1 + n_cls, n, 1), dtype=np.float32)
            for i, j in enumerate(range(start, end)):
                x_input[i, n_cls:, :, 0] = X_norm[:, j]

            x_tensor = torch.tensor(x_input, device=self.device)

            self.col_embedder.eval()
            with torch.no_grad():
                src = self.col_embedder.in_linear(x_tensor)
                out = src
                for i, block in enumerate(self.col_embedder.tf_col.blocks):
                    out = block(out)
                    layer_num = i + 1
                    if layer_num in layer_set:
                        batch_emb = out[:, n_cls:, :, :].cpu().numpy()
                        for ii, j in enumerate(range(start, end)):
                            results[layer_num][:, j, :] = batch_emb[ii, 0, :, :]
                    if layer_num >= max_layer:
                        break

        return results

    def get_all_embeddings(self, X: np.ndarray, layer_idx: int = 2,
                           show_progress: bool = False,
                           batch_size: int = 32) -> Optional[np.ndarray]:
        if not self.loaded:
            return None

        n, p = X.shape
        embeddings = np.zeros((n, p, self.embed_dim))

        X_norm = self.preprocess_batch(X)
        n_cls = self.n_cls_tokens

        if show_progress:
            if p <= 20:
                progress_step = 1
            elif p <= 100:
                progress_step = max(1, p // 10)
            else:
                progress_step = max(1, p // 20)

        processed = 0
        for start in range(0, p, batch_size):
            end = min(start + batch_size, p)
            batch_p = end - start

            x_input = np.zeros((batch_p, 1 + n_cls, n, 1), dtype=np.float32)
            for i, j in enumerate(range(start, end)):
                x_input[i, n_cls:, :, 0] = X_norm[:, j]

            x_tensor = torch.tensor(x_input, device=self.device)

            self.col_embedder.eval()
            with torch.no_grad():
                src = self.col_embedder.in_linear(x_tensor)
                out = src
                for i, block in enumerate(self.col_embedder.tf_col.blocks):
                    out = block(out)
                    if i + 1 == layer_idx:
                        break

            batch_embeddings = out[:, n_cls:, :, :].cpu().numpy()
            for i, j in enumerate(range(start, end)):
                embeddings[:, j, :] = batch_embeddings[i, 0, :, :]

            processed = end
            if show_progress and processed % progress_step < batch_size:
                pct = 100 * processed / p
                bar_len = 20
                filled_len = int(bar_len * processed / p)
                bar = '█' * filled_len + '░' * (bar_len - filled_len)
                print(f"    TabICL Embedding: [{bar}] {processed:4d}/{p:4d} ({pct:5.1f}%)", flush=True)

        if show_progress:
            bar = '█' * 20
            print(f"    TabICL Embedding: [{bar}] {p:4d}/{p:4d} (100.0%) ✓", flush=True)

        return embeddings


if __name__ == "__main__":

    print("Testing Screening Methods...")


    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    y = X[:, 0] + X[:, 1] * X[:, 2] + np.random.randn(n) * 0.5
    true_features = [0, 1, 2]


    print(f"DC(X0, Y): {CorrelationMeasures.distance_correlation(X[:, 0], y):.3f}")
    print(f"PC(X0, Y): {CorrelationMeasures.projection_correlation(X[:, 0], y):.3f}")


    sis_scores = TraditionalScreening.sis(X, y)
    dc_scores = TraditionalScreening.dc_sis(X, y)

    print(f"\nSIS top-5 features: {np.argsort(-sis_scores)[:5]}")
    print(f"DC-SIS top-5 features: {np.argsort(-dc_scores)[:5]}")


    mms = ScreeningEvaluator.compute_mms(dc_scores, true_features)
    print(f"\nMMS: {mms}")


    print("\nTesting MixedFeatureScreening...")
    is_discrete = np.zeros(p, dtype=bool)
    is_discrete[0:5] = True
    X_mixed = X.copy()
    for j in range(5):
        X_mixed[:, j] = np.digitize(X[:, j], np.quantile(X[:, j], [0.33, 0.67]))

    groups = [tuple(range(i, i+3)) for i in range(0, p-3, 3)]
    scores = MixedFeatureScreening.compute_raw_scores_mixed(
        X_mixed, y, groups[:5], is_discrete, 'dc', 'onehot')
    print(f"Mixed feature scores: {scores}")


    print("\nTesting MissingValueScreening...")
    X_missing = X.copy()
    missing_mask = np.random.rand(n, p) < 0.2
    X_missing[missing_mask] = np.nan

    scores, ratio = MissingValueScreening.compute_raw_scores_missing(
        X_missing, y, groups[:5], missing_mask, 'dc', 'complete_case')
    print(f"Complete case scores (ratio={ratio:.2f}): {scores}")

    scores, ratio = MissingValueScreening.compute_raw_scores_missing(
        X_missing, y, groups[:5], missing_mask, 'dc', 'mean_imputation')
    print(f"Mean imputation scores: {scores}")


    print("\nTesting Embedding-based methods...")


    embed_dim = 128
    embeddings = np.random.randn(n, p, embed_dim)


    print("\nFor continuous response:")
    emb_dc_scores = AcceleratedScreening.emb_dc_accelerated(embeddings, y)
    emb_sc_scores = AcceleratedScreening.emb_sc_accelerated(embeddings, y)
    emb_ipc_scores = AcceleratedScreening.emb_ipc_accelerated(embeddings, y)
    emb_pc_scores = AcceleratedScreening.emb_pc_accelerated(embeddings, y)
    emb_sis_scores = AcceleratedScreening.emb_sis_accelerated(embeddings, y)

    print(f"emb_dc top-5: {np.argsort(-emb_dc_scores)[:5]}")
    print(f"emb_sc top-5: {np.argsort(-emb_sc_scores)[:5]}")
    print(f"emb_ipc top-5: {np.argsort(-emb_ipc_scores)[:5]}")


    print("\nFor classification response:")
    y_class = (y > y.mean()).astype(int)
    emb_cru_scores = AcceleratedScreening.emb_cru_accelerated(embeddings, y_class)
    emb_cavs_scores = AcceleratedScreening.emb_cavs_accelerated(embeddings, y_class)
    emb_fkf_scores = AcceleratedScreening.emb_fkf_accelerated(embeddings, y_class)
    emb_mv_sis_scores = AcceleratedScreening.emb_mv_sis_accelerated(embeddings, y_class)
    emb_lr_ffs_scores = AcceleratedScreening.emb_lr_ffs_accelerated(embeddings, y_class)
    emb_psis_scores = AcceleratedScreening.emb_psis_accelerated(embeddings, y_class)

    print(f"emb_cru top-5: {np.argsort(-emb_cru_scores)[:5]}")
    print(f"emb_cavs top-5: {np.argsort(-emb_cavs_scores)[:5]}")
    print(f"emb_fkf top-5: {np.argsort(-emb_fkf_scores)[:5]}")


    print("\nTesting emb_compute_all_scores...")
    all_emb_scores = AcceleratedScreening.emb_compute_all_scores(
        embeddings, y, methods=['dc', 'sc', 'ipc', 'pc'], n_jobs=1)
    print(f"Computed methods: {list(all_emb_scores.keys())}")

    print("\nAll tests passed!")
