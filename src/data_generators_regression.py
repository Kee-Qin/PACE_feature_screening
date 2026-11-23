"""Controlled regression datasets retained in the current manuscript."""
import numpy as np
from typing import List, Tuple, Optional

class RegressionDataGenerator:

    @staticmethod
    def exp1_mixed_nonlinear(param_set: int=2, seed: Optional[int]=None):
        """Nonlinear regression; manuscript settings [2, 3]."""
        configs = {2: {'c': 4.0, 'eps_type': 'normal'}, 3: {'c': 4.0, 'eps_type': 't3'}}
        cfg = configs[param_set]
        n, p, s = (200, 2000, 4)
        if seed is not None:
            np.random.seed(seed)
        X = np.random.randn(n, p)
        if cfg['eps_type'] == 't3':
            eps = np.random.standard_t(df=3, size=n)
        else:
            eps = np.random.randn(n)
        Y = 2.0 * X[:, 0] + 2.0 * X[:, 1] ** 2 + 2.0 * np.exp(X[:, 2]) + 2.0 * np.log(np.abs(X[:, 3]) + 1e-08) + cfg['c'] * eps
        return (X, Y, [0, 1, 2, 3], n, p, s)

    @staticmethod
    def exp2_ultra_high_dimension(param_set: int=1, seed: Optional[int]=None):
        """Ultrahigh-dimensional regression; manuscript settings [1, 2, 3]."""
        configs = {1: {'p': 10000}, 2: {'p': 50000}, 3: {'p': 100000}}
        cfg = configs[param_set]
        n, p, s = (400, cfg['p'], 8)
        coef = 2.0
        noise_scale = 3.0
        if seed is not None:
            np.random.seed(seed)
        X = np.random.randn(n, p)
        eps = np.random.randn(n)
        Y = coef * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3])
        Y += coef * (X[:, 4] ** 2 - 1)
        Y += coef * (X[:, 5] ** 2 - 1)
        Y += coef * (X[:, 6] ** 2 - 1)
        Y += coef * (X[:, 7] ** 2 - 1)
        Y += noise_scale * eps
        return (X, Y, list(range(s)), n, p, s)

    @staticmethod
    def exp3_mixtype_regression(param_set: int=2, seed: Optional[int]=None):
        """Mixed continuous-categorical regression; manuscript settings [2, 3]."""
        configs = {2: {'tau': 1.5, 'K': 5, 'coef': 5.0, 'beta_cat': 5.0, 'noise_scale': 5.0}, 3: {'tau': 1.5, 'K': 7, 'coef': 5.0, 'beta_cat': 5.0, 'noise_scale': 5.0}}
        cfg = configs[param_set]
        n, p = (400, 2000)
        s_cont = 2
        s_cat = 2
        s = s_cont + s_cat
        K = cfg['K']
        if seed is not None:
            np.random.seed(seed)
        X = np.zeros((n, p), dtype=np.float64)
        X[:, 0] = np.random.randn(n)
        X[:, 1] = np.random.randn(n)
        cat_effects_list = []
        for j_cat_idx in range(s_cat):
            j = s_cont + j_cat_idx
            rng_j = np.random.RandomState((seed if seed is not None else 42) + j_cat_idx * 1000 + 7)
            cat_effects = np.linspace(-1.5, 1.5, K)
            rng_j.shuffle(cat_effects)
            cat_effects_list.append(cat_effects)
            X[:, j] = np.random.randint(0, K, n).astype(np.float64)
        for j in range(s, p):
            if j % 3 == 0:
                X[:, j] = np.random.randn(n)
            else:
                X[:, j] = np.random.randint(0, K, n).astype(np.float64)
        eps = np.random.randn(n)
        Y = cfg['coef'] * X[:, 0] * (X[:, 0] > cfg['tau'])
        Y += cfg['coef'] * X[:, 1]
        for j_cat_idx in range(s_cat):
            j = s_cont + j_cat_idx
            cat_vals = X[:, j].astype(int)
            cat_vals = np.clip(cat_vals, 0, K - 1)
            Y += cfg['beta_cat'] * cat_effects_list[j_cat_idx][cat_vals]
        Y += cfg['noise_scale'] * eps
        true_features = list(range(s))
        cat_indices = list(range(s_cont, s_cont + s_cat))
        cat_info = {'cat_indices': cat_indices, 'n_categories': {j: K for j in cat_indices}}
        return (X, Y, true_features, n, p, s, cat_info)

    @staticmethod
    def exp4_missing_regression(param_set: int=2, seed: Optional[int]=None):
        """Missing feature values under collinearity; manuscript settings [2, 3]."""
        configs = {2: {'rho': 0.98, 'cov_type': 'block_ar1', 'coef': 0.3, 'noise_scale': 3.0, 'p_miss': 0.15}, 3: {'rho': 0.95, 'cov_type': 'ar1', 'coef': 0.3, 'noise_scale': 3.0, 'p_miss': 0.2}}
        cfg = configs[param_set]
        n, p, s = (300, 3000, 3)
        if seed is not None:
            np.random.seed(seed)
        if cfg['cov_type'] == 'ar1':
            cov_matrix = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    cov_matrix[i, j] = cfg['rho'] ** abs(i - j)
        elif cfg['cov_type'] == 'block_ar1':
            block_size = 50
            cov_matrix = np.eye(p)
            n_blocks = p // block_size
            for b in range(n_blocks):
                start = b * block_size
                end = min((b + 1) * block_size, p)
                for i in range(start, end):
                    for j in range(start, end):
                        cov_matrix[i, j] = cfg['rho'] ** abs(i - j)
            remaining = p - n_blocks * block_size
            if remaining > 0:
                start = n_blocks * block_size
                for i in range(start, p):
                    for j in range(start, p):
                        cov_matrix[i, j] = cfg['rho'] ** abs(i - j)
        else:
            cov_matrix = np.eye(p)
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if np.min(eigvals) < 1e-06:
            cov_matrix += (1e-06 - np.min(eigvals)) * np.eye(p)
        L = np.linalg.cholesky(cov_matrix)
        X = np.random.randn(n, p) @ L.T
        eps = np.random.randn(n)
        Y = cfg['coef'] * X[:, 0] + cfg['coef'] * X[:, 1] ** 2 + cfg['coef'] * np.exp(X[:, 2]) + cfg['noise_scale'] * eps
        missing_mask = np.random.rand(n, p) < cfg['p_miss']
        X_with_nan = X.copy()
        X_with_nan[missing_mask] = np.nan
        return (X_with_nan, Y, [0, 1, 2], n, p, s, missing_mask)

EXPERIMENTS = {
    'exp1': RegressionDataGenerator.exp1_mixed_nonlinear,
    'exp2': RegressionDataGenerator.exp2_ultra_high_dimension,
    'exp3': RegressionDataGenerator.exp3_mixtype_regression,
    'exp4': RegressionDataGenerator.exp4_missing_regression,
}
EXPERIMENT_NAMES = {'exp1': 'Nonlinear regression', 'exp2': 'Ultrahigh-dimensional regression', 'exp3': 'Mixed continuous-categorical regression', 'exp4': 'Missing feature values under collinearity'}
PARAMETER_SETS = {'exp1': [2, 3], 'exp2': [1, 2, 3], 'exp3': [2, 3], 'exp4': [2, 3]}
