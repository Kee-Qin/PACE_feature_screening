#!/usr/bin/env python3
"""
Real Data Experiments — TALENT Benchmark with GAUSSIAN NOISE (LARGE)
Evaluates sis_emb, dc_emb, cru_emb on real datasets with Gaussian noise injection.

Protocol:
  - G2: Inject Gaussian noise to reach 2000 total features (p + noise = 2000)
  - G3: Select top 3p features from all 2000
  - OOM handling: Some models may OOM with 2000 features in G2

Groups:
  1. Clean baseline (original p features)
  2. Gaussian noise baseline (p + N noise features, total = 2000)
  3. Fixed-k screening (top-3p from all features)
"""
from src import model_path as mp
from src.screening_utilities import embedding_scores

import argparse
import json
import os
import sys
import traceback
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
os.environ["TABPFN_TELEMETRY"] = "0"

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = mp.DATA_ROOT
TABPFN_CLF_CKPT = str(mp.checkpoint('tabpfn_v25_clf'))
TABPFN_REG_CKPT = str(mp.checkpoint('tabpfn_v25_reg'))
TABICL_CKPT = str(mp.checkpoint('tabicl_v2_clf'))



from src.talent_benchmark.dataset_name import BINCLASS, MULTICLASS, REGRESSION

DATASET_MAP = {"bin": BINCLASS, "multi": MULTICLASS, "reg": REGRESSION}
TASK_DIR_MAP = {"bin": "binclass", "multi": "multiclass", "reg": "regression"}

# ── Seeds ──
SEEDS = [42, 123, 456, 789, 1000]

# ── Screening methods per task ──
SCREENING_METHODS = {
    "bin": ["sis_emb", "dc_emb", "cru_emb"],
    "multi": ["sis_emb", "dc_emb", "cru_emb"],
    "reg": ["sis_emb", "dc_emb"],
}

# ── Downstream models ──
MODELS = ["lgb", "catboost", "tabpfn"]

# ── Gaussian noise config ──
TARGET_TOTAL_FEATURES = 2000  # Expand to 2000 total features (p + noise = 2000)

# G3: select top 3p features
G3_TOP_K_MULTIPLIER = 3



# =============================================================================
# Data Loading
# =============================================================================
def load_dataset(task_type, dataset_name):
    """Load a TALENT dataset and return train/val/test splits."""
    task_dir = TASK_DIR_MAP[task_type]
    path = DATA_PATH / task_dir / f"{dataset_name}.pkl"
    df = pd.read_pickle(path)

    train = df[df["flag"] == "train"].reset_index(drop=True)
    val = df[df["flag"] == "vali"].reset_index(drop=True)
    test = df[df["flag"] == "test"].reset_index(drop=True)

    X_train = train.drop(columns=["flag", "target"])
    y_train = train["target"].values
    X_val = val.drop(columns=["flag", "target"])
    y_val = val["target"].values
    X_test = test.drop(columns=["flag", "target"])
    y_test = test["target"].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def identify_categorical(X):
    """Return list of categorical column indices and names."""
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    cat_indices = [list(X.columns).index(c) for c in cat_cols]
    return cat_indices, cat_cols


def encode_categorical(X_train, X_val, X_test, cat_indices):
    """Label-encode categorical columns. Returns numpy arrays."""
    X_tr = X_train.copy()
    X_va = X_val.copy()
    X_te = X_test.copy()

    encoders = {}
    for idx in cat_indices:
        col = X_tr.columns[idx]
        le = LabelEncoder()
        combined = pd.concat([X_tr[col].astype(str), X_va[col].astype(str), X_te[col].astype(str)], axis=0)
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_va[col] = le.transform(X_va[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
        encoders[col] = le

    return X_tr.values.astype(np.float32), X_va.values.astype(np.float32), X_te.values.astype(np.float32), encoders


def encode_y(y, task_type):
    """Encode labels. Returns (encoded_y, label_encoder_or_None)."""
    if task_type == "reg":
        return y.astype(np.float64), None
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


# =============================================================================
# Gaussian Noise Generation (Group 2)
# =============================================================================
def generate_gaussian_noise_features(X_train, X_val, X_test, seed, n_noise=None):
    """
    Generate pure Gaussian noise features N(0,1).

    Args:
        X_train, X_val, X_test: Original feature arrays
        seed: Random seed for reproducibility
        n_noise: Number of Gaussian noise features to generate

    Returns:
        fake_train, fake_val, fake_test: Gaussian noise arrays
    """
    rng = np.random.RandomState(seed)
    n_tr = X_train.shape[0]
    n_va = X_val.shape[0]
    n_te = X_test.shape[0]

    if n_noise is None:
        n_noise = TARGET_TOTAL_FEATURES

    fake_train = rng.randn(n_tr, n_noise).astype(np.float32)
    fake_val = rng.randn(n_va, n_noise).astype(np.float32)
    fake_test = rng.randn(n_te, n_noise).astype(np.float32)

    return fake_train, fake_val, fake_test


# =============================================================================
# OOM Detection
# =============================================================================
def is_oom_error(exc):
    """Check if an exception is likely an OOM error."""
    err_str = str(exc).lower()
    oom_keywords = ["out of memory", "oom", "cuda out of memory", "memoryerror",
                     "cannot allocate", "insufficient memory", "killed"]
    return any(kw in err_str for kw in oom_keywords)


# =============================================================================
# Screening
# =============================================================================
def compute_screening_scores(X_train, y_train, cat_indices, task_type, device="cuda:0"):
    """Compute emb utility scores for all screening methods."""
    from src.screening_methods_torch import TorchAcceleratedEmbedder, TorchAcceleratedScreening, TABICL_PATH

    methods = SCREENING_METHODS[task_type]
    n, p = X_train.shape

    embedder = TorchAcceleratedEmbedder(device=device)
    embedder.load_model(TABICL_PATH, device=device)

    MAX_EMBED_SAMPLES = 3000
    if n > MAX_EMBED_SAMPLES:
        if task_type != "reg" and len(np.unique(y_train)) > 1:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_EMBED_SAMPLES, random_state=42)
            sample_idx, _ = next(sss.split(X_train, y_train))
        else:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n, MAX_EMBED_SAMPLES, replace=False)
        X_embed = X_train[sample_idx]
        y_embed = y_train[sample_idx]
    else:
        X_embed = X_train
        y_embed = y_train

    X_clean = X_embed.copy()
    if np.any(np.isnan(X_clean)):
        col_means = np.nanmean(X_clean, axis=0)
        for j in range(X_clean.shape[1]):
            X_clean[np.isnan(X_clean[:, j]), j] = col_means[j]

    embeddings = embedder.get_all_embeddings(X_clean, layer_idx=2, batch_size=64)

    emb_methods_list = []
    for m in methods:
        if m == "sis_emb":
            emb_methods_list.append("sis")
        elif m == "dc_emb":
            emb_methods_list.append("dc")
        elif m == "cru_emb":
            emb_methods_list.append("cru")

    results = embedding_scores(
        embeddings, y_embed, methods=emb_methods_list, task=task_type
    )

    scores = {}
    for m in methods:
        key = m.replace("_emb", "")
        if f"emb_{key}" in results:
            scores[m] = results[f"emb_{key}"]
        else:
            raise RuntimeError(f"Missing screening scores for {m}")

    del embeddings
    torch = __import__("torch")
    torch.cuda.empty_cache()

    return scores


# =============================================================================
# Downstream Models
# =============================================================================
def train_lgb(X_train, y_train, X_val, y_val, task_type, seed, cat_indices=None):
    """Train LightGBM with early stopping."""
    import lightgbm as lgb

    is_clf = task_type != "reg"

    categorical_feature = None
    if cat_indices:
        categorical_feature = [f"col_{i}" for i in cat_indices]
        X_train = X_train.copy()
        X_val = X_val.copy()
        for i in cat_indices:
            X_train[:, i] = X_train[:, i].astype(int)
            X_val[:, i] = X_val[:, i].astype(int)

    col_names = [f"col_{i}" for i in range(X_train.shape[1])]
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=col_names,
                         categorical_feature=categorical_feature, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=col_names,
                       categorical_feature=categorical_feature, free_raw_data=False)

    if is_clf:
        params = {
            "objective": "binary" if task_type == "bin" else "multiclass",
            "metric": "binary_logloss" if task_type == "bin" else "multi_logloss",
            "verbose": -1,
            "seed": seed,
            "num_threads": 4,
        }
        if task_type == "multi":
            params["num_class"] = len(np.unique(y_train))
    else:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "seed": seed,
            "num_threads": 4,
        }

    model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dval], callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    return model


def predict_lgb(model, X_test, task_type):
    """Predict with LightGBM."""
    pred = model.predict(X_test)
    if task_type == "bin":
        return {"proba": pred, "pred": (pred >= 0.5).astype(int)}
    elif task_type == "multi":
        return {"proba": pred, "pred": np.argmax(pred, axis=1)}
    else:
        return {"pred": pred}


def train_catboost(X_train, y_train, X_val, y_val, task_type, seed, cat_indices=None):
    """Train CatBoost with early stopping."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    is_clf = task_type != "reg"
    cat_features = None

    dtrain = Pool(X_train, y_train, cat_features=cat_features)
    dval = Pool(X_val, y_val, cat_features=cat_features)

    if is_clf:
        model = CatBoostClassifier(
            iterations=500, early_stopping_rounds=20,
            random_seed=seed, verbose=0, thread_count=4,
            auto_class_weights="Balanced" if task_type == "bin" else None,
        )
    else:
        model = CatBoostRegressor(
            iterations=500, early_stopping_rounds=20,
            random_seed=seed, verbose=0, thread_count=4,
        )

    model.fit(dtrain, eval_set=dval, verbose=0)
    return model


def predict_catboost(model, X_test, task_type, cat_indices=None):
    """Predict with CatBoost."""
    from catboost import Pool
    dtest = Pool(X_test)

    if task_type == "reg":
        pred = model.predict(dtest)
        return {"pred": pred}
    else:
        proba = model.predict_proba(dtest)
        if task_type == "bin":
            return {"proba": proba[:, 1], "pred": (proba[:, 1] >= 0.5).astype(int)}
        else:
            return {"proba": proba, "pred": np.argmax(proba, axis=1)}


def train_tabpfn(X_train, y_train, task_type, seed, device="cuda:0"):
    """Train TabPFN (no validation, in-context learning). Subsample if too large."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    MAX_TABPFN_TRAIN = 30000
    n = X_train.shape[0]
    if n > MAX_TABPFN_TRAIN:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, MAX_TABPFN_TRAIN, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    if task_type == "reg":
        if not os.path.exists(TABPFN_REG_CKPT):
            raise FileNotFoundError(f"TabPFN regressor checkpoint not found: {TABPFN_REG_CKPT}")
        model = TabPFNRegressor(
            model_path=TABPFN_REG_CKPT,
            ignore_pretraining_limits=True,
            device=device,
            random_state=seed,
            n_estimators=4,
        )
    else:
        model = TabPFNClassifier(
            model_path=TABPFN_CLF_CKPT,
            ignore_pretraining_limits=True,
            device=device,
            random_state=seed,
            n_estimators=4,
        )

    model.fit(X_train, y_train)
    return model


def predict_tabpfn(model, X_test, task_type):
    """Predict with TabPFN."""
    if task_type == "reg":
        pred = model.predict(X_test)
        return {"pred": pred}
    else:
        proba = model.predict_proba(X_test)
        if task_type == "bin":
            return {"proba": proba[:, 1], "pred": (proba[:, 1] >= 0.5).astype(int)}
        else:
            return {"proba": proba, "pred": np.argmax(proba, axis=1)}


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true, y_pred, y_proba, task_type):
    """Compute evaluation metrics."""
    metrics = {}
    if task_type == "bin":
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["acc"] = float(accuracy_score(y_true, y_pred))
    elif task_type == "multi":
        metrics["acc"] = float(accuracy_score(y_true, y_pred))
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return metrics


# =============================================================================
# Train and evaluate a model (with OOM detection)
# =============================================================================
def train_and_evaluate(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                       task_type, seed, cat_indices=None, device="cuda:0"):
    """Train model and return metrics. Detects OOM errors specifically."""
    try:
        if model_name == "lgb":
            model = train_lgb(X_train, y_train, X_val, y_val, task_type, seed, cat_indices)
            result = predict_lgb(model, X_test, task_type)
        elif model_name == "catboost":
            model = train_catboost(X_train, y_train, X_val, y_val, task_type, seed, cat_indices)
            result = predict_catboost(model, X_test, task_type, cat_indices)
        elif model_name == "tabpfn":
            model = train_tabpfn(X_train, y_train, task_type, seed, device)
            result = predict_tabpfn(model, X_test, task_type)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        metrics = compute_metrics(y_test, result.get("pred"), result.get("proba"), task_type)
        return metrics

    except Exception as e:
        err_str = str(e)
        if is_oom_error(e):
            print(f"    [OOM] {model_name} OOM: {err_str[:120]}", flush=True)
            return {"error": "OOM", "detail": err_str[:200]}
        else:
            print(f"    [WARN] {model_name} failed: {err_str[:120]}", flush=True)
            return {"error": str(e)[:200]}


# =============================================================================
# Method-level checkpoint helpers
# =============================================================================
def find_missing_methods(existing_result, task_type, models):
    """Determine which screening methods are missing from an existing result.

    Args:
        existing_result: dict of existing results (loaded from JSON), or None
        task_type: 'bin', 'multi', or 'reg'
        models: list of model names

    Returns:
        set of missing screening method names (e.g. {'cru_emb'})
    """
    if existing_result is None:
        return set(SCREENING_METHODS.get(task_type, []))

    all_methods = set(SCREENING_METHODS.get(task_type, []))
    present_methods = set()

    for model_name in models:
        if model_name not in existing_result:
            continue
        model_data = existing_result[model_name]
        if not isinstance(model_data, dict):
            continue
        for method in all_methods:
            g3_key = f"g3_{method}"
            if g3_key in model_data and isinstance(model_data[g3_key], dict) and "error" not in model_data[g3_key]:
                present_methods.add(method)

    missing = all_methods - present_methods
    return missing


# =============================================================================
# Single Experiment Run
# =============================================================================
def run_single_dataset(dataset_name, task_type, seed, device="cuda:0",
                       existing_result=None):
    """Run all three groups for one dataset with one seed.

    Supports method-level resume: if existing_result is provided, only computes
    missing screening methods and downstream evaluations.

    Args:
        dataset_name: dataset name
        task_type: 'bin', 'multi', or 'reg'
        seed: random seed
        device: GPU device
        existing_result: dict of existing results (for resume), or None

    Returns:
        dict: complete results (merged with existing if resume)
    """
    scr_methods = SCREENING_METHODS.get(task_type, ["sis_emb", "dc_emb"])
    missing_methods = find_missing_methods(existing_result, task_type, MODELS)

    # Check if everything is already done
    if existing_result is not None and len(missing_methods) == 0:
        all_baseline_done = all(
            m in existing_result
            and "g1_clean" in existing_result[m]
            and isinstance(existing_result[m].get("g1_clean", {}), dict)
            and "error" not in existing_result[m].get("g1_clean", {})
            and m in existing_result
            and "g2_gaussian" in existing_result[m]
            and isinstance(existing_result[m].get("g2_gaussian", {}), dict)
            and "error" not in existing_result[m].get("g2_gaussian", {})
            for m in MODELS
        )
        if all_baseline_done:
            print(f"  {dataset_name} seed={seed} — ALL methods already completed, SKIP", flush=True)
            return existing_result

    if missing_methods:
        print(f"  Resume: missing methods = {missing_methods}", flush=True)

    print(f"  Loading {dataset_name}...", flush=True)
    try:
        X_train_df, y_train_raw, X_val_df, y_val_raw, X_test_df, y_test_raw = load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERROR] Failed to load {dataset_name}: {e}", flush=True)
        return None

    cat_indices, cat_cols = identify_categorical(X_train_df)
    p = X_train_df.shape[1]
    n = X_train_df.shape[0]

    print(f"  n={n}, p={p}, cat={len(cat_indices)}", flush=True)

    X_train, X_val, X_test, _ = encode_categorical(X_train_df, X_val_df, X_test_df, cat_indices)
    y_train, le_y = encode_y(y_train_raw, task_type)
    y_val, _ = encode_y(y_val_raw, task_type)
    y_test, _ = encode_y(y_test_raw, task_type)

    # Determine number of noise features
    # Expand to TARGET_TOTAL_FEATURES total (2000)
    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    print(f"  Injecting {n_noise} Gaussian noise features (total: {p} + {n_noise} = {p + n_noise})", flush=True)

    if n_noise == 0:
        print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES}, no noise to inject", flush=True)
        return None

    # ── Group 2: Generate GAUSSIAN noise features ──
    fake_train, fake_val, fake_test = generate_gaussian_noise_features(
        X_train, X_val, X_test, seed, n_noise=n_noise
    )

    # Concatenate: [real features | Gaussian noise]
    X_comb_train = np.hstack([X_train, fake_train])
    X_comb_val = np.hstack([X_val, fake_val])
    X_comb_test = np.hstack([X_test, fake_test])

    # Real features are indices 0..p-1, noise features are p..p+n_noise-1
    n_comb = p + n_noise

    # ── Group 3: Screening on combined features ──
    screening_scores = {}
    screening_error = None
    try:
        screening_scores = compute_screening_scores(X_comb_train, y_train, cat_indices, task_type, device)
    except Exception as e:
        if is_oom_error(e):
            screening_error = "OOM"
            print(f"  [OOM] Screening OOM with {n_comb} features: {str(e)[:120]}", flush=True)
        else:
            screening_error = str(e)
            print(f"  [WARN] Screening failed: {str(e)[:120]}", flush=True)

        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    # ── Run all groups ──
    results = {"dataset": dataset_name, "task_type": task_type, "seed": seed,
               "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
               "total_features": n_comb}

    for model_name in MODELS:
        # Skip tabpfn for regression if checkpoint not available
        if model_name == "tabpfn" and task_type == "reg" and not os.path.exists(TABPFN_REG_CKPT):
            continue

        model_results = {}

        # Group 1: Clean baseline (original p features)
        if existing_result is not None and model_name in existing_result:
            existing_m = existing_result[model_name]
            if "g1_clean" in existing_m and isinstance(existing_m["g1_clean"], dict) and "error" not in existing_m["g1_clean"]:
                model_results["g1_clean"] = existing_m["g1_clean"]
            else:
                m1 = train_and_evaluate(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                                        task_type, seed, cat_indices, device)
                model_results["g1_clean"] = m1
        else:
            m1 = train_and_evaluate(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                                    task_type, seed, cat_indices, device)
            model_results["g1_clean"] = m1

        # Group 2: Gaussian noise baseline (p + n_noise features)
        # With 2000 features, some models may OOM — handled by try-except
        if existing_result is not None and model_name in existing_result:
            existing_m = existing_result[model_name]
            if "g2_gaussian" in existing_m and isinstance(existing_m["g2_gaussian"], dict) and "error" not in existing_m["g2_gaussian"]:
                model_results["g2_gaussian"] = existing_m["g2_gaussian"]
            else:
                m2 = train_and_evaluate(model_name, X_comb_train, y_train, X_comb_val, y_val, X_comb_test, y_test,
                                        task_type, seed, cat_indices, device)
                model_results["g2_gaussian"] = m2
        else:
            m2 = train_and_evaluate(model_name, X_comb_train, y_train, X_comb_val, y_val, X_comb_test, y_test,
                                    task_type, seed, cat_indices, device)
            model_results["g2_gaussian"] = m2

        # Group 3: Per screening method (skip if screening failed)
        if screening_error is not None:
            for scr_method in SCREENING_METHODS[task_type]:
                model_results[f"g3_{scr_method}"] = {"error": f"screening_{screening_error}"}
            results[model_name] = model_results
            continue

        for scr_method in SCREENING_METHODS[task_type]:
            # Skip already-computed methods during resume
            if scr_method not in missing_methods:
                if existing_result is not None and model_name in existing_result:
                    existing_m = existing_result[model_name]
                    g3_key = f"g3_{scr_method}"
                    if g3_key in existing_m:
                        model_results[g3_key] = existing_m[g3_key]
                continue

            if scr_method not in screening_scores:
                continue

            scores = screening_scores[scr_method]

            # Group 3: Fixed-k (top 3p features from all features)
            g3_k = G3_TOP_K_MULTIPLIER * p
            top_g3_idx = np.argsort(scores)[::-1][:g3_k]
            true_in_top_g3 = int(np.sum(top_g3_idx < p))
            recall = true_in_top_g3

            X_g3_train = X_comb_train[:, top_g3_idx]
            X_g3_val = X_comb_val[:, top_g3_idx]
            X_g3_test = X_comb_test[:, top_g3_idx]

            if cat_indices:
                orig_to_new = {idx: i for i, idx in enumerate(top_g3_idx)}
                g3_cat = [orig_to_new[idx] for idx in cat_indices if idx in orig_to_new]
            else:
                g3_cat = None

            m3 = train_and_evaluate(model_name, X_g3_train, y_train, X_g3_val, y_val, X_g3_test, y_test,
                                    task_type, seed, g3_cat, device)
            m3["recall"] = recall
            m3["n_features"] = g3_k
            model_results[f"g3_{scr_method}"] = m3


        results[model_name] = model_results

    # Free GPU memory
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Real Data Experiments with Gaussian Noise (Large: 2000 features)")
    parser.add_argument("--task_type", type=str, required=True, choices=["bin", "multi", "reg"])
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Single seed to run (default: run all 5)")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--output", type=str, default="results_real_gaussian_large")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific dataset names (default: all in task)")
    parser.add_argument("--resume", action="store_true",
                        help="Method-level resume: load existing results and only compute missing methods")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}"
    datasets = args.datasets if args.datasets else DATASET_MAP[args.task_type]
    seeds = [args.random_seed] if args.random_seed else SEEDS
    output_dir = BASE_DIR / args.output
    output_dir.mkdir(exist_ok=True)

    task_label = TASK_DIR_MAP[args.task_type]
    print("=" * 70)
    print(f"Real Data Experiment (GAUSSIAN NOISE — LARGE) — {task_label}")
    print(f"Task: {args.task_type}, Datasets: {len(datasets)}, Seeds: {seeds}")
    print(f"Device: {device}, Output: {output_dir}")
    print(f"Resume: {args.resume}")
    print(f"Screening methods: {SCREENING_METHODS[args.task_type]}")
    print(f"Models: {MODELS}")
    print(f"Gaussian noise: expand to {TARGET_TOTAL_FEATURES} total features")
    print(f"G3: select top {G3_TOP_K_MULTIPLIER}p features")
    print("=" * 70)

    for ds_idx, dataset_name in enumerate(datasets):
        for seed in seeds:
            result_file = output_dir / f"{task_label}_{dataset_name}_seed{seed}.json"

            # Load existing result for method-level resume
            existing_result = None
            if args.resume and result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        existing_result = json.load(f)
                    missing = find_missing_methods(existing_result, args.task_type, MODELS)
                    if len(missing) == 0:
                        print(f"\n[{ds_idx+1}/{len(datasets)}] {dataset_name} seed={seed} — ALL methods done, SKIP", flush=True)
                        continue
                    else:
                        print(f"\n[{ds_idx+1}/{len(datasets)}] {dataset_name} seed={seed} — RESUME missing: {missing}", flush=True)
                except Exception as e:
                    print(f"  [WARN] Failed to load existing result: {e}, recomputing", flush=True)
                    existing_result = None
            else:
                print(f"\n[{ds_idx+1}/{len(datasets)}] {dataset_name} seed={seed}", flush=True)

            t0 = time.time()

            result = run_single_dataset(dataset_name, args.task_type, seed, device,
                                        existing_result=existing_result)

            if result is not None:
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                elapsed = time.time() - t0
                print(f"  Done ({elapsed:.1f}s) → {result_file.name}", flush=True)
            else:
                print(f"  SKIPPED (error)", flush=True)

    print("\n" + "=" * 70)
    print(f"All experiments completed! Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
