"""Shared input validation, preprocessing, and score validation."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import numpy as np


def pace_hsic_target(y, task=None):
    y = np.asarray(y).reshape(-1)
    if task is None:
        categorical = len(np.unique(y)) < len(y) / 2 or not np.issubdtype(y.dtype, np.number)
    else:
        categorical = normalize_task(task, y) != 'reg'
    if categorical:
        labels, codes = np.unique(y, return_inverse=True)
        return np.eye(len(labels))[codes], True
    return y[:, None], False


def normalize_task(task, y=None):
    if task in ('reg', 'regression'):
        return 'reg'
    if task in ('multi', 'multiclass'):
        return 'multi'
    if task in ('bin', 'binary', 'binclass'):
        return 'bin'
    if task == 'classification':
        return 'multi' if y is not None and len(np.unique(y)) > 2 else 'bin'
    raise ValueError('task must specify classification or regression')


def validate_input(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or min(X.shape) < 1 or len(y) != len(X):
        raise ValueError('Expected X with shape (n, p) and y with length n')
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError('Encode categorical columns numerically before screening')
    if np.isinf(X).any():
        raise ValueError('X contains infinite values')
    return X, y


def validate_scores(scores, p, method):
    if scores is None:
        raise RuntimeError(f'{method} did not return scores')
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (p,) or not np.isfinite(values).all():
        raise RuntimeError(f'{method} must return {p} finite feature scores')
    return values


def impute_mean(X):
    X = np.asarray(X).copy()
    if np.isnan(X).any():
        means = np.nanmean(X, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        row, col = np.where(np.isnan(X))
        X[row, col] = means[col]
    return X


def sample_rows(X, y, task, max_samples=None, seed=42):
    if max_samples is None or len(X) <= max_samples:
        return X, y
    if max_samples < 2:
        raise ValueError('max_samples must be at least two')
    if task != 'reg' and len(np.unique(y)) > 1:
        from sklearn.model_selection import StratifiedShuffleSplit
        idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=max_samples,
                      random_state=seed).split(X, y))
    else:
        idx = np.random.RandomState(seed).choice(len(X), max_samples, replace=False)
    return X[idx], y[idx]


def screen_split(X, y, task_type, max_samples, test_size=0.35, allow_tiny_overlap=True):
    from sklearn.model_selection import train_test_split
    X, y = sample_rows(X, y, task_type, max_samples)
    if len(X) < 12 and allow_tiny_overlap:
        return X, X, y, y
    strat = None
    if task_type != 'reg' and len(np.unique(y)) > 1:
        counts = np.bincount(np.asarray(y).astype(int))
        if len(counts) and counts.min() >= 2:
            strat = y
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)


@contextmanager
def score_workspace(path=None):
    if path is not None:
        yield Path(path)
    else:
        with tempfile.TemporaryDirectory(prefix='screening_') as folder:
            yield Path(folder) / 'scores.npz'


def atomic_save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.stem + '.', suffix='.npz', dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(name, **values)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


EMB_LAYERS = [1, 2, 3]
MAX_TRAIN_FEATURES = 5_000
LGB_PARAMS_CLS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=0, n_jobs=-1, verbose=-1)
LGB_PARAMS_REG = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=0, n_jobs=-1, verbose=-1)
CB_PARAMS_CLS = dict(iterations=300, depth=6, learning_rate=0.05,
                     rsm=0.8, random_seed=0, allow_writing_files=False, verbose=False)
CB_PARAMS_REG = dict(iterations=300, depth=6, learning_rate=0.05,
                     rsm=0.8, random_seed=0, allow_writing_files=False, verbose=False)
RF_PARAMS_CLS = dict(n_estimators=300, max_depth=6, max_features=0.8,
                      random_state=0, n_jobs=-1, oob_score=True, bootstrap=True)
RF_PARAMS_REG = dict(n_estimators=300, max_depth=6, max_features=0.8,
                      random_state=0, n_jobs=-1, oob_score=True, bootstrap=True)
PFI_REPEATS = 10
VAL_FRACTION = 0.35

# _fill_missing (B)
def _fill_missing(X, cat_indices=None):
    X = np.array(X, dtype=np.float64, copy=True)
    if not np.isnan(X).any():
        return X, cat_indices or []
    cat_set = set(cat_indices or [])
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_j = np.isnan(col)
        if nan_j.any():
            if j in cat_set:
                col[nan_j] = -1
            else:
                med = np.nanmedian(col[~nan_j])
                col[nan_j] = med if not np.isnan(med) else 0.0
            X[:, j] = col
    return X, cat_indices or []

# _train_val_split (C)
def _train_val_split(X, y, n, val_fraction=VAL_FRACTION, random_state=0):
    from sklearn.model_selection import train_test_split
    is_cls = (len(np.unique(y)) < n / 2)
    if n < 12:
        return X, y, X, y, False
    stratify = y if (is_cls and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=val_fraction, random_state=random_state, stratify=stratify)
    return Xtr, ytr, Xval, yval, True

# _model_njobs (C)
def _model_njobs(default=-1):
    v = os.environ.get('PACE_C_MODEL_NJOBS')
    return int(v) if (v and v.strip()) else default


def _metric_loss(task, model, X, y):
    y = np.asarray(y)
    try:
        if task == 'classification':
            proba = model.predict_proba(X)
            classes = np.asarray(model.classes_).astype(int)
            cmap = {c: i for i, c in enumerate(classes)}
            idx = np.array([cmap.get(int(v), 0) for v in y])
            p_true = np.clip(proba[np.arange(len(y)), idx], 1e-12, 1.0)
            return -np.mean(np.log(p_true))
        else:
            pred = model.predict(X)
            return float(np.mean((y - pred) ** 2))
    except Exception:
        if task == 'classification':
            return float(np.mean(model.predict(X) != y))
        return float(np.mean((y - model.predict(X)) ** 2))


_metric_loss_C = _metric_loss
_metric_loss_D = _metric_loss
