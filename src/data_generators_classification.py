"""Controlled classification datasets retained in the current manuscript."""
import numpy as np
from typing import List, Tuple, Optional

class ClassificationDataGenerator:

    @staticmethod
    def exp1_mean_shift(param_set: int=1, seed: Optional[int]=None):
        configs = {1: {'mu': 0.6, 'base_dist': 'normal'}, 2: {'mu': 0.6, 'base_dist': 't3'}, 3: {'mu': 0.6, 'base_dist': 'lognormal'}}
        cfg = configs[param_set]
        n, p, s = (200, 3000, 8)
        if seed is not None:
            np.random.seed(seed)
        y = np.random.randint(0, 2, n)
        X = np.zeros((n, p))
        for j in range(p):
            if cfg['base_dist'] == 't3':
                Z = np.random.standard_t(3, n)
            elif cfg['base_dist'] == 'lognormal':
                Z = np.random.lognormal(0, 1, n)
                Z = (Z - np.exp(0.5)) / np.sqrt(np.exp(2) - np.exp(1))
            else:
                Z = np.random.randn(n)
            if j < s:
                X[y == 0, j] = Z[y == 0] - cfg['mu']
                X[y == 1, j] = Z[y == 1] + cfg['mu']
            else:
                X[:, j] = Z
        return (X, y, list(range(s)), n, p, s)

    @staticmethod
    def exp2_latent_additive(param_set: int=1, seed: Optional[int]=None):
        configs = {1: {'cov_type': 'identity', 'noise_dist': 'normal', 'rho': 0.0, 'noise_scale': 3}, 2: {'cov_type': 'ar1', 'noise_dist': 't3', 'rho': 0.3, 'noise_scale': 3}, 3: {'cov_type': 'banded', 'noise_dist': 'normal', 'rho': 0.3, 'noise_scale': 3}}
        cfg = configs[param_set]
        n, p, s = (300, 3000, 4)
        if seed is not None:
            np.random.seed(seed)
        if cfg['cov_type'] == 'ar1':
            cov_matrix = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    cov_matrix[i, j] = cfg['rho'] ** abs(i - j)
            L = np.linalg.cholesky(cov_matrix + 1e-08 * np.eye(p))
            X = np.random.randn(n, p) @ L.T
        elif cfg['cov_type'] == 'banded':
            cov_matrix = np.eye(p)
            for i in range(p):
                for j in range(max(0, i - 2), min(p, i + 3)):
                    if i != j:
                        cov_matrix[i, j] = cfg['rho'] ** abs(i - j)
            eigvals = np.linalg.eigvalsh(cov_matrix)
            if np.min(eigvals) < 1e-06:
                cov_matrix += (1e-06 - np.min(eigvals)) * np.eye(p)
            L = np.linalg.cholesky(cov_matrix)
            X = np.random.randn(n, p) @ L.T
        else:
            X = np.random.randn(n, p)
        Y_star = 2.0 * X[:, 0] + 2.0 * X[:, 1] + 2.5 * (X[:, 2] ** 2 - 1) + 2.5 * (X[:, 3] ** 2 - 1)
        if cfg['noise_dist'] == 't3':
            eps = np.random.standard_t(3, n) * cfg['noise_scale']
        else:
            eps = np.random.randn(n) * cfg['noise_scale']
        Y_star += eps
        y = (Y_star > 0).astype(int)
        return (X, y, [0, 1, 2, 3], n, p, s)

    @staticmethod
    def exp3_imbalanced_bimodal_multiclass(param_set: int=1, seed: Optional[int]=None):
        configs = {1: {'class_probs': [0.7, 0.2, 0.1], 'sep': 2.0}, 2: {'class_probs': [0.85, 0.1, 0.05], 'sep': 2.0}, 3: {'class_probs': [0.33, 0.34, 0.33], 'sep': 2.0}}
        cfg = configs[param_set]
        n, p, s = (300, 2000, 10)
        if seed is not None:
            np.random.seed(seed)
        y = np.random.choice(3, n, p=cfg['class_probs'])
        X = np.random.randn(n, p)
        sep = cfg['sep']
        bimodal_var = 0.25
        for j in range(p):
            if j < 5:
                mask1 = y == 1
                n1 = mask1.sum()
                if n1 > 0:
                    Z = np.random.binomial(1, 0.5, n1)
                    X[mask1, j] = Z * np.random.normal(sep, np.sqrt(bimodal_var), n1) + (1 - Z) * np.random.normal(-sep, np.sqrt(bimodal_var), n1)
            elif j < 10:
                mask2 = y == 2
                n2 = mask2.sum()
                if n2 > 0:
                    Z = np.random.binomial(1, 0.5, n2)
                    X[mask2, j] = Z * np.random.normal(sep, np.sqrt(bimodal_var), n2) + (1 - Z) * np.random.normal(-sep, np.sqrt(bimodal_var), n2)
        return (X, y, list(range(s)), n, p, s)

    @staticmethod
    def exp4_ultra_high_dimension(param_set: int=1, seed: Optional[int]=None):
        """Ultrahigh-dimensional classification; manuscript settings [1, 2, 3]."""
        configs = {1: {'p': 10000}, 2: {'p': 50000}, 3: {'p': 100000}}
        cfg = configs[param_set]
        n, p, s = (400, cfg['p'], 15)
        signal = 4.5
        var_ratio = 2.5
        if seed is not None:
            np.random.seed(seed)
        y = np.random.randint(0, 2, n)
        X = np.zeros((n, p))
        s_mean = 12
        s_var = 3
        for j in range(s_mean):
            Z = np.random.randn(n)
            X[y == 0, j] = Z[y == 0] - signal
            X[y == 1, j] = Z[y == 1] + signal
        for j in range(s_mean, s_mean + s_var):
            n0 = (y == 0).sum()
            n1 = (y == 1).sum()
            Z0 = np.random.randn(n0)
            Z1 = np.random.randn(n1)
            X[y == 0, j] = Z0
            X[y == 1, j] = var_ratio * Z1
        for j in range(s, p):
            X[:, j] = np.random.randn(n)
        return (X, y, list(range(s)), n, p, s)

    @staticmethod
    def exp5_mixtype_classification(param_set: int=2, seed: Optional[int]=None):
        """Mixed continuous-categorical classification; manuscript settings [2, 3]."""
        configs = {2: {'K': 25, 'mu': 0.8, 'cat_concentration': 4.0}, 3: {'K': 40, 'mu': 0.6, 'cat_concentration': 5.0}}
        cfg = configs[param_set]
        n, p = (300, 2000)
        s_cont = 3
        s_cat = 3
        s = s_cont + s_cat
        K = cfg['K']
        if seed is not None:
            np.random.seed(seed)
        y = np.random.randint(0, 2, n)
        X = np.zeros((n, p), dtype=np.float64)
        for j in range(s_cont):
            Z = np.random.randn(n)
            X[y == 0, j] = Z[y == 0] - cfg['mu']
            X[y == 1, j] = Z[y == 1] + cfg['mu']
        for j_cat_idx in range(s_cat):
            j = s_cont + j_cat_idx
            cat_0 = np.random.randint(0, K, (y == 0).sum()).astype(np.float64)
            rng_j = np.random.RandomState(seed if seed is not None else 42)
            boosted_cats = rng_j.choice(K, size=max(1, K // 3), replace=False)
            cat_probs_1 = np.ones(K) / (K * cfg['cat_concentration'])
            cat_probs_1[boosted_cats] = (cfg['cat_concentration'] - 1) / (K // 3)
            cat_probs_1 /= cat_probs_1.sum()
            cat_1 = np.random.choice(K, size=(y == 1).sum(), p=cat_probs_1).astype(np.float64)
            X[y == 0, j] = cat_0
            X[y == 1, j] = cat_1
        for j in range(s, p):
            if j % 2 == 0:
                X[:, j] = np.random.randn(n)
            else:
                X[:, j] = np.random.randint(0, K, n).astype(np.float64)
        true_features = list(range(s))
        cat_indices = list(range(s_cont, s_cont + s_cat))
        cat_info = {'cat_indices': cat_indices, 'n_categories': {j: K for j in cat_indices}}
        return (X, y, true_features, n, p, s, cat_info)

    @staticmethod
    def exp6_missing_classification(param_set: int=1, seed: Optional[int]=None):
        """Missing feature values; manuscript settings [1, 2]."""
        configs = {1: {'p_miss': 0.15, 'noise_scale': 3.0, 'cov_type': 'ar1', 'rho': 0.3}, 2: {'p_miss': 0.3, 'noise_scale': 3.0, 'cov_type': 'ar1', 'rho': 0.3}}
        cfg = configs[param_set]
        n, p, s = (300, 3000, 4)
        if seed is not None:
            np.random.seed(seed)
        if cfg['cov_type'] == 'ar1':
            cov_matrix = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    cov_matrix[i, j] = cfg['rho'] ** abs(i - j)
            L = np.linalg.cholesky(cov_matrix + 1e-08 * np.eye(p))
            X = np.random.randn(n, p) @ L.T
        else:
            X = np.random.randn(n, p)
        Y_star = 2.0 * X[:, 0] + 2.0 * X[:, 1] + 2.5 * (X[:, 2] ** 2 - 1) + 2.5 * (X[:, 3] ** 2 - 1)
        eps = np.random.randn(n) * cfg['noise_scale']
        Y_star += eps
        y = (Y_star > 0).astype(int)
        missing_mask = np.random.rand(n, p) < cfg['p_miss']
        X_with_nan = X.copy()
        X_with_nan[missing_mask] = np.nan
        return (X_with_nan, y, list(range(s)), n, p, s, missing_mask)

EXPERIMENTS = {
    'exp1': ClassificationDataGenerator.exp1_mean_shift,
    'exp2': ClassificationDataGenerator.exp2_latent_additive,
    'exp3': ClassificationDataGenerator.exp3_imbalanced_bimodal_multiclass,
    'exp4': ClassificationDataGenerator.exp4_ultra_high_dimension,
    'exp5': ClassificationDataGenerator.exp5_mixtype_classification,
    'exp6': ClassificationDataGenerator.exp6_missing_classification,
}
EXPERIMENT_NAMES = {'exp1': 'Strong additive classification', 'exp2': 'Nonlinear latent classification', 'exp3': 'Three-class bimodal signal', 'exp4': 'Ultrahigh-dimensional classification', 'exp5': 'Mixed continuous-categorical classification', 'exp6': 'Missing feature values'}
PARAMETER_SETS = {'exp1': [1, 2, 3], 'exp2': [1, 2, 3], 'exp3': [1, 2, 3], 'exp4': [1, 2, 3], 'exp5': [2, 3], 'exp6': [1, 2]}
