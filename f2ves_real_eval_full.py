"""TALENT scoring and downstream evaluation.

Separate scoring and prediction: --stage score, then --stage predict.
Other screening families retain --method {full,cpu,E} and --predictor.
Run --stage score --help or --stage predict --help for the split-stage CLI.
"""
from src import model_path as mp

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split

from src.pace_real.common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SCORE_ROOT,
    SCHEMA,
    TARGET_TOTAL_FEATURES,
    TABICL_SHAP_SAMPLING_ID,
    TABICL_SHAP_MAX_SAMPLES,
    TABICL_SHAP_VALIDATION_FRACTION,
    SAGE_METHODS,
    SAGE_SAMPLING_ID,
    SAGE_MAX_SAMPLES,
    SAGE_VALIDATION_FRACTION,
    TASK_FULL,
    atomic_save_npz,
    iter_units,
    prepare_full2000,
    read_score,
    score_file,
    validate_score,
    add_gpu_arguments,
    dispatch_gpu_workers,
    COHORT_ID,
    print_dataset_cohort,
    cohort_id,
)


DEFAULT_SCORE_METHODS = ("tabicl_shap", "sage_lgb", "sage_cb", "loco_simple", "loco_cv")
from src.screening_utilities import available_methods
SCORE_METHODS = tuple(available_methods()) + ("sis_emb_max",)
from src.screening_utilities import compute_scores, compute_many, embedding_scores, selected_method_id, aggregation_for_method
from src.screening_methods_torch_A import _method_sample_cap, _validation_probe, _partial_path

def _run_method(method, prepared, args, final_path):
    from src.screening_methods_torch_A import attribution_options, _method_screen_split
    if args.validate_only and method in DEFAULT_SCORE_METHODS:
        split = _method_screen_split(method, prepared, args)
        return _validation_probe(method, *split, args.current_task, args, final_path)
    options = attribution_options(**vars(args))
    settings = vars(options).copy()
    settings.pop("device", None)
    settings.pop("aggregation", None)
    scores = compute_scores(prepared["Xc_train"], prepared["y_train"], method,
        task=args.current_task, device=args.device,
        aggregation=aggregation_for_method(method, getattr(args, "aggregation", "native")),
        work_path=final_path, **settings)
    config = {"aggregation": aggregation_for_method(method, getattr(args, "aggregation", "native"))}
    if method == "sis_emb_max":
        config.update(protocol="sis_max_tabicl_v1.1_cumulative_L2_d128_mean_impute_rows2000_v1", aggregation="max")
    elif method in SAGE_METHODS:
        config["sampling_protocol"] = SAGE_SAMPLING_ID
    elif method == "tabicl_shap":
        config["sampling_protocol"] = TABICL_SHAP_SAMPLING_ID
    if args.validate_only:
        return {"probe": "complete feature scores", "n_features": len(scores), **config}
    return scores, config



def score_matrix(method, X, y, task, device, model_root, work_root, options=None):
    """Time the shared table-to-scores interface for experiment bookkeeping."""
    t0 = time.perf_counter()
    values = compute_scores(X, y, method, task=task, device=device,
                            model_root=model_root, **(options or {}))
    return values, time.perf_counter() - t0



def _metadata(args, prepared, task, dataset, seed, method, elapsed, config):
    return {
        "schema": SCHEMA,
        "dataset_cohort": cohort_id(getattr(args, "cohort", "all")),
        "strict_full_p": True,
        "noise_scores_computed": True,
        "score_scope": "all_augmented_features",
        "n_features_scored": TARGET_TOTAL_FEATURES,
        "task_type": task, "dataset": dataset, "seed": int(seed), "method": method,
        "n": prepared["n"], "p_original": prepared["p_original"],
        "n_noise": prepared["n_noise"],
        "elapsed_seconds": float(elapsed),
        "run_parameters": _run_parameters(args, method),
        "configuration": config,
    }


def _run_parameters(args, method):
    if method == "sis_emb_max":
        PROTOCOL = "sis_max_tabicl_v1.1_cumulative_L2_d128_mean_impute_rows2000_v1"
        return {"protocol": PROTOCOL, "max_screen_samples": 2000}
    keys = ["max_screen_samples", "aggregation"]
    if method.startswith("loco"):
        keys += ["rf_trees", "rf_max_depth", "rf_max_features"]
        if method == "loco_cv":
            keys += ["loco_cv_folds"]
    elif method.startswith("sage"):
        keys += ["sage_permutations", "sage_imputations", "sage_background", "sage_eval_max"]
    elif method == "tabicl_shap":
        keys += ["tabicl_estimators", "tabicl_eval_max", "tabicl_eval_batch", "tabicl_predict_batch"]
    parameters = {key: getattr(args, key) for key in keys}
    parameters["aggregation"] = aggregation_for_method(method, args.aggregation)
    if method == "tabicl_shap":
        parameters["max_screen_samples"] = _method_sample_cap(method, args)
        parameters["screen_validation_fraction"] = TABICL_SHAP_VALIDATION_FRACTION
        parameters["sampling_protocol"] = TABICL_SHAP_SAMPLING_ID
    elif method in SAGE_METHODS:
        parameters["max_screen_samples"] = _method_sample_cap(method, args)
        parameters["screen_validation_fraction"] = SAGE_VALIDATION_FRACTION
        parameters["sampling_protocol"] = SAGE_SAMPLING_ID
    return parameters


def build_score_parser():
    p = argparse.ArgumentParser(description="TALENT feature-importance scoring")
    p.add_argument("--task_type", default="all", choices=("bin", "multi", "reg", "all"))
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Optional dataset names; defaults to all available datasets")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--methods", nargs="+", choices=SCORE_METHODS, default=list(DEFAULT_SCORE_METHODS))
    p.add_argument("--cohort", choices=("all",), default="all", help=argparse.SUPPRESS)
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--output-root", default=str(DEFAULT_SCORE_ROOT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--aggregation", choices=("native", "mean", "max"), default="native")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-screen-samples", type=int, default=2000,
                   help="Screening row cap; TabICL-SHAP and both SAGE methods cap at 500 (80/20 split)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true",
                   help="Short full-P model/masking probes, not a complete production importance run")
    p.add_argument("--checkpoint-every", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--rf-trees", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=6)
    p.add_argument("--rf-max-features", type=float, default=0.8)
    p.add_argument("--rf-n-jobs", type=int, default=-1)
    p.add_argument("--loco-feature-workers", type=int, default=1,
                   help="Parallel leave-one-column fits; use rf-n-jobs=1 when >1")
    p.add_argument("--loco-cv-folds", type=int, default=5)
    p.add_argument("--sage-permutations", type=int, default=32)
    p.add_argument("--sage-imputations", type=int, default=32)
    p.add_argument("--sage-background", type=int, default=200)
    p.add_argument("--sage-eval-max", type=int, default=0,
                   help="0 keeps the complete internal holdout (100 rows at the default 500-row sample)")
    p.add_argument("--tabicl-estimators", type=int, default=2)
    p.add_argument("--tabicl-eval-max", type=int, default=100,
                   help="Maximum explained rows (default 100, from the internal 20%% holdout)")
    p.add_argument("--tabicl-eval-batch", type=int, default=1)
    p.add_argument("--tabicl-predict-batch", type=int, default=16)
    add_gpu_arguments(p)
    return p


def score_main():
    global _elapsed_base, _elapsed_start
    parser = build_score_parser()
    args = parser.parse_args()
    args.methods = [selected_method_id(m, args.aggregation) for m in args.methods]
    print_dataset_cohort(args.cohort)
    if dispatch_gpu_workers(args, parser, Path(__file__).resolve(), stage="score"):
        return
    if args.loco_feature_workers > 1 and args.rf_n_jobs != 1:
        args.rf_n_jobs = 1
    for option in ("max_screen_samples", "checkpoint_every", "rf_trees", "loco_cv_folds",
                   "sage_permutations", "sage_imputations", "sage_background",
                   "tabicl_estimators", "tabicl_eval_batch", "tabicl_predict_batch"):
        if getattr(args, option) <= 0:
            raise ValueError(f"{option} must be positive")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    units = list(iter_units(args.task_type, args.datasets, args.seeds, args.cohort))
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    units = units[args.shard_index::args.num_shards]
    print(f"strict schema={SCHEMA}; units={len(units)}; methods={args.methods}", flush=True)
    if args.dry_run:
        for task, dataset, seed in units:
            for method in args.methods:
                print(task, dataset, seed, method)
        return
    failures = []
    for task, dataset, seed in units:
        print(f"\n[{task}] {dataset} seed={seed}: preparing exactly P=2000", flush=True)
        try:
            prepared = prepare_full2000(task, dataset, seed, args.data_root)
        except Exception as exc:
            failures.append((task, dataset, seed, "data", repr(exc)))
            print(f"  [FAIL] data: {exc}", flush=True)
            continue
        args.current_task = task
        args.run_identity = {"dataset": dataset, "seed": int(seed),
                             "n_rows": len(prepared["Xc_train"]),
                             "n_features": prepared["Xc_train"].shape[1],
                             "max_screen_samples": args.max_screen_samples,
                             "task_type": task, "schema": SCHEMA,
                             "validation": args.validate_only}
        for method in args.methods:
            args.run_identity["max_screen_samples"] = _method_sample_cap(method, args)
            path = score_file(args.output_root, task, dataset, seed, method)
            if args.validate_only:
                path = path.with_name(path.stem + ".validation.npz")
            if args.resume and not args.validate_only and path.exists():
                try:
                    sc, meta = read_score(path)
                    validate_score(sc, meta, prepared, task, dataset, seed, method)
                    if meta.get("run_parameters") != _run_parameters(args, method):
                        raise ValueError("run parameters changed; choose a new output-root")
                    print(f"  [resume] {method}: valid artifact exists", flush=True)
                    continue
                except Exception as exc:
                    print(f"  [resume] {method}: existing artifact rejected: {exc}", flush=True)
                    failures.append((task, dataset, seed, method, repr(exc)))
                    continue
            print(f"  [score] {method}: scorer input={prepared['Xc_train'].shape}", flush=True)
            t0 = time.perf_counter()
            _elapsed_base, _elapsed_start = 0.0, t0
            try:
                result = _run_method(method, prepared, args, path)
                if args.validate_only:
                    print(f"  [VALID] {method}: {json.dumps(result, ensure_ascii=False)}", flush=True)
                    continue
                scores, config = result
                elapsed = _elapsed_base + time.perf_counter() - t0
                scores = np.asarray(scores, dtype=np.float64)
                if scores.shape != (TARGET_TOTAL_FEATURES,) or not np.isfinite(scores).all():
                    raise ValueError(f"invalid score vector {scores.shape}")
                meta = _metadata(args, prepared, task, dataset, seed, method, elapsed, config)
                atomic_save_npz(path, scores=scores.astype(np.float32),
                                metadata_json=np.array(json.dumps(meta, sort_keys=True)))
                validate_score(scores, meta, prepared, task, dataset, seed, method)
                partial = _partial_path(path)
                if partial.exists():
                    partial.unlink()
                print(f"  [DONE] {method}: {elapsed:.1f}s -> {path}", flush=True)
            except Exception as exc:
                failures.append((task, dataset, seed, method, repr(exc)))
                print(f"  [FAIL] {method}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
            finally:
                import gc
                gc.collect()
                if "torch" in __import__("sys").modules:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    if failures:
        print("\nFailures:", flush=True)
        for row in failures:
            print(row, flush=True)
        raise SystemExit(2)



if __name__ == "__main__":
    import sys as _stage_sys
    _stage_parser = argparse.ArgumentParser(add_help=False)
    _stage_parser.add_argument("--stage", choices=("score", "predict"))
    _stage_args, _stage_rest = _stage_parser.parse_known_args()
    if _stage_args.stage:
        _stage_sys.argv = [_stage_sys.argv[0]] + _stage_rest
        if _stage_args.stage == "score":
            score_main()
        else:
            from src.pace_real.prediction import main as predict_main
            predict_main()
        raise SystemExit(0)

import sys as _sys
_sys.modules.setdefault('f2ves_real_eval_full', _sys.modules.get('__main__'))
import sys as _sys
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import f2ves_real_gaussian_large_experiments as T
import shutil
import copy
import torch
import importlib.resources as pkg_resources


try:
    from TALENT.model.utils import get_method, set_seeds
    import TALENT as _TALENT_PKG
    _HAS_TALENT = True
except Exception:
    get_method = set_seeds = _TALENT_PKG = None
    _HAS_TALENT = False
from src import screening_methods_torch_A as E
import shlex
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit




# ======================================================================

# ======================================================================

os.environ["TABPFN_TELEMETRY"] = "0"


os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

BASE_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT")
                    or mp.OUTPUT_ROOT)

# AutoDL-local model/code roots.  Every path remains overridable so that score
# artifacts can be produced on several machines and consumed on another one.
MODEL_ROOT = mp.MODEL_ROOT
TABDPT_CODE_ROOT = mp.TABDPT_CODE_ROOT
if TABDPT_CODE_ROOT.exists() and str(TABDPT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(TABDPT_CODE_ROOT))

TABPFN_V3_CLF = str(mp.checkpoint('tabpfn_v3_clf'))

TABPFN_V3_REG = str(mp.checkpoint('tabpfn_v3_reg'))

T.TABPFN_CLF_CKPT = TABPFN_V3_CLF

T.TABPFN_REG_CKPT = TABPFN_V3_REG

TABICL_V2_CLF = str(mp.checkpoint('tabicl_v2_clf'))

TABICL_V2_REG = str(mp.checkpoint('tabicl_v2_reg'))

TABPFN_V26_CLF = str(mp.checkpoint('tabpfn_v26_clf'))

TABPFN_V26_REG = str(mp.checkpoint('tabpfn_v26_reg'))

TABDPT_CKPT = str(mp.checkpoint('tabdpt'))

TABDPT_ENSEMBLES = 4

DATASET_MAP = {"bin": T.BINCLASS, "multi": T.MULTICLASS, "reg": T.REGRESSION}

TASK_DIR_MAP = {"bin": "binclass", "multi": "multiclass", "reg": "regression"}

SEEDS = [42, 123, 456]

TASK_FULL = {"bin": "classification", "multi": "classification", "reg": "regression"}

STREAM_RAW_TRAD = ["sis", "dc", "cru", "ids"]

RAW_HSIC = "hsic"

EMB_METHODS = ["sis", "dc", "cru", "hsic"]

RFF_METHODS = ["rff_dc", "rff_sis", "rff_hsic"]

CFAST_METHODS = ["rf_mdi", "lgb_split", "lgb_gain", "cb_pvc"]

DSHAP_METHODS = ["lgb_shap", "cb_shap", "loco_simple", "loco_cv", "sage_lgb", "sage_cb", "tabicl_shap"]


# Strict TALENT reruns use f2ves_real_eval_full.py --stage score and f2ves_real_eval_full.py --stage predict.
LOCO_METHODS = ["loco_simple", "loco_cv"]
SAGE_METHODS = ["sage_lgb", "sage_cb"]
# Attribution methods score the complete candidate matrix.
STAGED_METHODS = LOCO_METHODS + SAGE_METHODS + ["tabicl_shap"]

CPU_SCORER_METHODS = CFAST_METHODS + DSHAP_METHODS

SCREEN_METHODS = (
    STREAM_RAW_TRAD + [RAW_HSIC]
    + [f"{m}_emb" for m in EMB_METHODS]
    + RFF_METHODS + CFAST_METHODS + DSHAP_METHODS
)

MODELS = ["lgb", "xgb", "catboost", "tabpfn", "tabicl", "tabpfn26", "tabdpt"]

TARGET_TOTAL_FEATURES = 2000

G3_TOP_K_MULTIPLIER = 3


MAX_EMBED_SAMPLES = 2000

MAX_SCREEN_SAMPLES = 2000

def compute_metrics(y_true, y_pred, y_proba, task_type):
    m = {}
    try:
        if task_type == "bin":
            m["auc"] = float(roc_auc_score(y_true, y_proba))

            if y_pred is not None:
                m["acc"] = float(accuracy_score(y_true, y_pred))
            elif y_proba is not None:
                m["acc"] = float(accuracy_score(y_true,
                                                (np.asarray(y_proba) >= 0.5).astype(int)))
        elif task_type == "multi":
            # ovr macro AUC; multi_class needs probabilities of shape (n, K)
            if y_proba is not None and getattr(y_proba, "ndim", 0) == 2 and y_proba.shape[1] > 1:
                m["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                              average="macro"))

            if y_pred is not None:
                m["acc"] = float(accuracy_score(y_true, y_pred))
        else:  # reg
            m["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception as e:
        m["metric_error"] = str(e)[:200]
    return m

def train_tabpfn_full(X_train, y_train, task_type, seed, device):
    """TabPN-v3 in-context predictor using the FULL training set (no subsample)."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    if task_type == "reg":
        if not os.path.exists(TABPFN_V3_REG):
            raise FileNotFoundError(f"TabPN-v3 regressor ckpt not found: {TABPFN_V3_REG}")
        model = TabPFNRegressor(model_path=TABPFN_V3_REG, ignore_pretraining_limits=True,
                                device=device, random_state=seed, n_estimators=4)
    else:
        model = TabPFNClassifier(model_path=TABPFN_V3_CLF, ignore_pretraining_limits=True,
                                device=device, random_state=seed, n_estimators=4)
    model.fit(X_train, y_train)
    return model

def predict_tabpfn(model, X_test, task_type):
    if task_type == "reg":
        return {"pred": np.asarray(model.predict(X_test)).reshape(-1)}
    proba = np.asarray(model.predict_proba(X_test))
    if task_type == "bin":
        return {"proba": proba[:, 1], "pred": (proba[:, 1] >= 0.5).astype(int)}
    return {"proba": proba, "pred": np.argmax(proba, axis=1)}

def train_xgb(X_train, y_train, X_val, y_val, task_type, seed, cat_indices=None):
    import xgboost as xgb
    is_clf = task_type != "reg"
    if is_clf:
        params = dict(n_estimators=500, early_stopping_rounds=20, random_state=seed,
                      n_jobs=16, tree_method="hist", verbosity=0)
        if task_type == "bin":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
            params["num_class"] = len(np.unique(y_train))
        model = xgb.XGBClassifier(**params)
    else:
        model = xgb.XGBRegressor(n_estimators=500, early_stopping_rounds=20,
                                 random_state=seed, n_jobs=16, tree_method="hist",
                                 objective="reg:squarederror", eval_metric="rmse", verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def predict_xgb(model, X_test, task_type):
    if task_type == "reg":
        return {"pred": model.predict(X_test)}
    proba = model.predict_proba(X_test)
    if task_type == "bin":
        return {"proba": proba[:, 1], "pred": (proba[:, 1] >= 0.5).astype(int)}
    return {"proba": proba, "pred": np.argmax(proba, axis=1)}

def train_tabicl(X_train, y_train, task_type, seed, device):
    from tabicl import TabICLClassifier, TabICLRegressor
    if task_type == "reg":
        model = TabICLRegressor(
            model_path=TABICL_V2_REG,
            checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
            n_estimators=8, random_state=seed, batch_size=2, device=device)
    else:
        model = TabICLClassifier(
            model_path=TABICL_V2_CLF,
            checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
            n_estimators=8, random_state=seed, batch_size=2, device=device)
    model.fit(X_train, y_train)
    return model

def predict_tabicl(model, X_test, task_type):
    if task_type == "reg":
        return {"pred": np.asarray(model.predict(X_test)).reshape(-1)}
    proba = np.asarray(model.predict_proba(X_test))
    if task_type == "bin":
        return {"proba": proba[:, 1], "pred": (proba[:, 1] >= 0.5).astype(int)}
    return {"proba": proba, "pred": np.argmax(proba, axis=1)}

def train_tabpfn26(X_train, y_train, task_type, seed, device):
    """TabPFN-v2.6 in-context predictor using the FULL training set (no subsample).
    Identical wiring to train_tabpfn_full, only the ckpt swaps v3→v2.6."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    if task_type == "reg":
        model = TabPFNRegressor(model_path=TABPFN_V26_REG, ignore_pretraining_limits=True,
                                device=device, random_state=seed, n_estimators=4)
    else:
        model = TabPFNClassifier(model_path=TABPFN_V26_CLF, ignore_pretraining_limits=True,
                                 device=device, random_state=seed, n_estimators=4)
    model.fit(X_train, y_train)
    return model

def predict_tabpfn26(model, X_test, task_type):
    if task_type == "reg":
        return {"pred": np.asarray(model.predict(X_test)).reshape(-1)}
    proba = np.asarray(model.predict_proba(X_test))
    if task_type == "bin":
        return {"proba": proba[:, 1], "pred": (proba[:, 1] >= 0.5).astype(int)}
    return {"proba": proba, "pred": np.argmax(proba, axis=1)}

def train_tabdpt(X_train, y_train, task_type, seed, device):
    """TabDPT classifier (bin/multi) or regressor (reg); both share TABDPT_CKPT and the
    same constructor (device/model_weight_path/use_flash/compile/verbose). Reg path
    uses model.predict(X, n_ensembles, seed) -> continuous preds (see predict_tabdpt)."""
    from tabdpt import TabDPTClassifier, TabDPTRegressor
    cls = TabDPTRegressor if task_type == "reg" else TabDPTClassifier
    model = cls(device=device, model_weight_path=TABDPT_CKPT,
               use_flash=False, compile=False, verbose=False)
    model.fit(X_train, y_train)
    return model

def predict_tabdpt(model, X_test, task_type, seed=42, batch_size=50000):
    """Batched ensemble predict. Binary → (N,) pos-prob; multiclass → (N,C); reg → (N,)."""
    if task_type == "reg":
        out = []
        for start in range(0, len(X_test), batch_size):
            out.append(np.asarray(model.predict(X_test[start:start + batch_size],
                                                n_ensembles=TABDPT_ENSEMBLES, seed=seed)).reshape(-1))
        return {"pred": np.concatenate(out, axis=0).astype(np.float32)}
    out = []
    for start in range(0, len(X_test), batch_size):
        out.append(model.ensemble_predict_proba(X_test[start:start + batch_size],
                                                n_ensembles=TABDPT_ENSEMBLES, seed=seed))
    proba = np.concatenate(out, axis=0).astype(np.float32)       # (N, C)
    if task_type == "bin":
        p1 = proba[:, 1] if proba.shape[1] > 1 else proba.reshape(-1)
        return {"proba": p1, "pred": (p1 >= 0.5).astype(int)}
    return {"proba": proba, "pred": np.argmax(proba, axis=1)}

def _norm_cat(cat_indices, n_cols):
    """Normalize cat_indices to a sorted python-list of in-range ints.

    T.identify_categorical / G3 reindexing may produce numpy scalars, which
    trip T.train_lgb's `X_train[:, i] = ...astype(int)` assignment (IndexError on
    a numpy.int64 RHS-assigned-into float32 view). Cast to plain int and drop OOB."""
    if cat_indices is None:
        return None
    out = []
    seen = set()
    try:
        for c in cat_indices:
            v = int(c)
            if 0 <= v < n_cols and v not in seen:
                out.append(v); seen.add(v)
    except (TypeError, ValueError):
        return None
    return sorted(out) if out else None

def train_and_evaluate(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                        task_type, seed, cat_indices=None, device="cuda:0"):
    # normalize cat_indices (python int + in-range) before handing to T.train_lgb etc.
    n_cols = X_train.shape[1]
    cat_indices = _norm_cat(cat_indices, n_cols)
    # T.train_lgb/catboost do in-place `X[:, i] = X[:, i].astype(int)` on cat cols;
    # fancy-indexed views (G3 selection) may be non-contiguous / read-only -> IndexError.

    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val) if X_val is not None else X_val
    X_test = np.ascontiguousarray(X_test) if X_test is not None else X_test
    # degenerate-class guard: if the (sub)fold has <2 classes for a classification
    # task, training is undefined -> short-circuit to NA (a G3 feature subset +
    # small split can leave a single class; otherwise lgb emits "num_class<1").
    if task_type != "reg":
        ytr_arr = np.asarray(y_train)
        if len(np.unique(ytr_arr)) < 2:
            return {"error": "single_class_in_fold"}
    try:
        if model_name == "lgb":
            m = T.train_lgb(X_train, y_train, X_val, y_val, task_type, seed, cat_indices)
            r = T.predict_lgb(m, X_test, task_type)
        elif model_name == "xgb":
            m = train_xgb(X_train, y_train, X_val, y_val, task_type, seed, cat_indices)
            r = predict_xgb(m, X_test, task_type)
        elif model_name == "catboost":
            m = T.train_catboost(X_train, y_train, X_val, y_val, task_type, seed, cat_indices)
            r = T.predict_catboost(m, X_test, task_type, cat_indices)
        elif model_name == "tabpfn":
            m = train_tabpfn_full(X_train, y_train, task_type, seed, device)
            r = predict_tabpfn(m, X_test, task_type)
        elif model_name == "tabicl":
            m = train_tabicl(X_train, y_train, task_type, seed, device)
            r = predict_tabicl(m, X_test, task_type)
        elif model_name == "tabpfn26":
            # ckpt skip parity with tabpfn-reg above
            if task_type == "reg" and not os.path.exists(TABPFN_V26_REG):
                raise FileNotFoundError(f"TabPN-v2.6 regressor ckpt missing: {TABPFN_V26_REG}")
            if not os.path.exists(TABPFN_V26_CLF):
                raise FileNotFoundError(f"TabPN-v2.6 classifier ckpt missing: {TABPFN_V26_CLF}")
            m = train_tabpfn26(X_train, y_train, task_type, seed, device)
            r = predict_tabpfn26(m, X_test, task_type)
        elif model_name == "tabdpt":
            if not os.path.exists(TABDPT_CKPT):
                raise FileNotFoundError(f"TabDPT ckpt missing: {TABDPT_CKPT}")
            m = train_tabdpt(X_train, y_train, task_type, seed, device)
            r = predict_tabdpt(m, X_test, task_type, seed=seed)
        else:
            raise ValueError(f"unknown model {model_name}")
        return compute_metrics(y_test, r.get("pred"), r.get("proba"), task_type)
    except Exception as e:
        if T.is_oom_error(e):
            print(f"    [OOM] {model_name}: {str(e)[:100]}", flush=True)
            return {"error": "OOM", "detail": str(e)[:200]}
        import os as _os, traceback as _tb
        if _os.environ.get("EVAL_FULL_TRACE"):
            _tb.print_exc()
            print(f"    [DBG] cat_indices={cat_indices!r} type={type(cat_indices)} "
                  f"Xtr={type(X_train)} shp={getattr(X_train,'shape',None)} "
                  f"dtype={getattr(X_train,'dtype',None)}", flush=True)
        print(f"    [WARN] {model_name} failed: {str(e)[:120]}", flush=True)
        return {"error": str(e)[:200]}

def _inner_split(X, y, task_type, seed=42):
    n = X.shape[0]
    if n < 12:
        return X, X, y, y, False
    is_clf = task_type != "reg"
    strat = y if (is_clf and len(np.unique(y)) > 1
                  and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=strat)
    return Xtr, Xv, ytr, yv, True

def compute_screening_scores(X_comb_train, y_train, p_real, task_type, device, embedder,
                             skip_cpu_scorers=False):
    from src.screening_methods_torch import TorchAcceleratedScreening





    scores, times, oom_methods = {}, {}, set()
    task_full = TASK_FULL[task_type]
    n_full, P = X_comb_train.shape  # P = p_real + n_noise
    attribution_X, attribution_y = X_comb_train, y_train


    N = n_full
    if n_full > MAX_SCREEN_SAMPLES:
        seed = 42
        if task_type != "reg" and len(np.unique(y_train)) > 1:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SCREEN_SAMPLES,
                                        random_state=seed)
            idx, _ = next(sss.split(X_comb_train, y_train))
        else:
            idx = np.random.RandomState(seed).choice(n_full, MAX_SCREEN_SAMPLES,
                                                      replace=False)
        X_comb_train = X_comb_train[idx]
        y_train = y_train[idx]
        N = MAX_SCREEN_SAMPLES
        print(f"  [screen] cap {n_full}→{N} samples for utility computation", flush=True)

    # ── raw trad (sis/dc/cru/ids) — one compute_all_scores call per method (per-method time) ──
    for m in STREAM_RAW_TRAD:
        t0 = time.perf_counter()
        ok = False
        try:
            r = compute_many(X_comb_train, y_train, [m], task=task_type, device=device, impute=False)
            sc = np.asarray(r.get(m), dtype=np.float64) if r and r.get(m) is not None else None
            if sc is not None and sc.shape[0] == P and not np.isnan(sc).all():
                scores[m] = sc; ok = True
        except Exception as e:
            if T.is_oom_error(e):
                oom_methods.add(m); times[m] = "OOM"
                print(f"  [screen] raw {m} OOM-recorded: {str(e)[:90]}", flush=True)
                continue
            print(f"  [screen] raw {m} err: {str(e)[:140]}", flush=True)
        times[m] = time.perf_counter() - t0 if ok else (times[m] if m in times else None)


    # HSIC is a full n×n kernel method: peak ∝ n² × P. Free the cache before/after so


    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    t0 = time.perf_counter()
    ok = False
    try:
        sc = compute_scores(X_comb_train, y_train, "raw_a_hsic", task=task_type, device=device, impute=False)
        sc = np.asarray(sc, dtype=np.float64)
        if sc.shape[0] == P and not np.isnan(sc).all():
            scores[RAW_HSIC] = sc; ok = True
    except Exception as e:
        if T.is_oom_error(e):
            oom_methods.add(RAW_HSIC); times[RAW_HSIC] = "OOM"
            print(f"  [screen] raw hsic OOM-recorded (n={X_comb_train.shape[0]}): {str(e)[:90]}", flush=True)
        else:
            print(f"  [screen] raw hsic err: {str(e)[:140]}", flush=True)
    if RAW_HSIC not in times:
        times[RAW_HSIC] = time.perf_counter() - t0 if ok else None
    elif times.get(RAW_HSIC) != "OOM":
        times[RAW_HSIC] = time.perf_counter() - t0 if ok else None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ── PACE-emb: forward once (shared, on the ≤MAX-row capped sample) + per-method scorer ──
    if embedder is not None and getattr(embedder, "loaded", False):
        t0 = time.perf_counter()
        emb_ok = False
        embeddings = None
        try:
            Xe = X_comb_train.copy()
            ye = y_train
            if np.any(np.isnan(Xe)):
                cm = np.nanmean(Xe, axis=0)
                for j in range(Xe.shape[1]):
                    Xe[np.isnan(Xe[:, j]), j] = cm[j]
            embeddings = embedder.get_all_embeddings(Xe, layer_idx=2, batch_size=64)
            emb_ok = True
        except Exception as e:
            if T.is_oom_error(e):
                times["emb_forward"] = "OOM"
                print(f"  [screen] emb_forward OOM-recorded: {str(e)[:90]}", flush=True)
            else:
                times["emb_forward"] = None
                print(f"  [screen] emb_forward err: {str(e)[:140]}", flush=True)
        else:
            times["emb_forward"] = time.perf_counter() - t0 if emb_ok else None
        if emb_ok:
            for m in EMB_METHODS:
                t0 = time.perf_counter()
                ok2 = False
                try:
                    r = embedding_scores(embeddings, ye, [m], task=task_type, device=device)
                    sc = np.asarray(r.get(f"emb_{m}"), dtype=np.float64) if r else None
                    if sc is not None and sc.shape[0] == P and not np.isnan(sc).all():
                        scores[f"{m}_emb"] = sc; ok2 = True
                except Exception as e:
                    if T.is_oom_error(e):
                        oom_methods.add(f"{m}_emb"); times[f"{m}_emb"] = "OOM"
                        print(f"  [screen] emb_{m} OOM-recorded: {str(e)[:90]}", flush=True)
                        continue
                    print(f"  [screen] emb_{m} err: {str(e)[:140]}", flush=True)
                times[f"{m}_emb"] = time.perf_counter() - t0 if ok2 else (times.get(f"{m}_emb"))
            import torch
            del embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        times["emb_forward"] = None
        for m in EMB_METHODS:
            times[f"{m}_emb"] = None

    # ── Track-B RFF / Track-C fast4 / Track-D shap — inner tr/val split ──
    Xtr_t, Xv_t, ytr_t, yv_t, _has = _inner_split(X_comb_train, y_train, task_type)


    # --skip-cpu-scorers: drop the CPU-tree groups (Tracks C+D); keep RFF (Track-B
    # GPU, fast). Those methods' G3 stay screening_NA; cache stays mergeable.
    cfg_c = [] if skip_cpu_scorers else CFAST_METHODS
    cfg_d = [] if skip_cpu_scorers else (DSHAP_METHODS)
    for grp in (RFF_METHODS, cfg_c, cfg_d):
        for m in grp:
            t0 = time.perf_counter()
            ok = False
            try:
                if m in STAGED_METHODS:
                    sc, _ = score_matrix(m, attribution_X, attribution_y, task_type,
                        device if m == "tabicl_shap" else "cpu", MODEL_ROOT,
                        RESULTS_ROOT / "attribution_work")
                else:
                    sc = compute_scores(Xtr_t, ytr_t, m, task=task_type, device=device,
                        validation_data=(Xv_t if _has else None, yv_t if _has else None))
                if sc is not None:
                    sc = np.asarray(sc, dtype=np.float64)
                    if sc.shape[0] == P and not np.isnan(sc).all():
                        scores[m] = sc; ok = True
            except Exception as e:
                if T.is_oom_error(e):
                    oom_methods.add(m); times[m] = "OOM"
                    print(f"  [screen] {m} OOM-recorded: {str(e)[:90]}", flush=True)
                    continue
                print(f"  [screen] {m} err: {str(e)[:140]}", flush=True)
            times[m] = time.perf_counter() - t0 if ok else (times.get(m))

    return scores, times, oom_methods

SCORE_CACHE_DIR = RESULTS_ROOT / "screening_score_cache_complete_v3"

def _score_cache_path(task_type, dataset_name, seed):
    task_label = TASK_DIR_MAP[task_type]
    return SCORE_CACHE_DIR / f"{task_label}_{dataset_name}_seed{seed}.npz"

def save_screening_scores(task_type, dataset_name, seed, scores, times, oom_methods):
    """Persist per-method scores (+ meta) to a sidecar .npz.

    Keys: one array per method (float32, shape (P,)); plus '__times__'
    (object/method→s or None or 'OOM'), '__oom__' (str list), '__schema__'='v1'.
    Existing methods in the cache are merged (rewrite/additive), so a partial
    rerun (e.g. emb_* recovered after --no-embedder) amends rather than wipes.
    """
    try:
        SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _score_cache_path(task_type, dataset_name, seed)
        merged = {}
        if path.exists():
            # Merge the existing arrays into memory BEFORE any write. np.load
            # returns a LAZY NpzFile whose underlying file handle stays open
            # until closed; an open handle on `path` blocks the os.replace
            # below (EIO/EBUSY on NFS) and is the read-then-truncate-same-file
            # EIO hazard. Use a context manager so the handle is closed, and
            # materialise each array inside it. A corrupt file raises here
            # (caught) → start the merge from empty.
            try:
                with np.load(path, allow_pickle=True) as old:
                    for k in old.files:
                        if not k.startswith("__"):
                            merged[k] = np.asarray(old[k])
            except Exception:
                pass
        for m, sc in scores.items():
            arr = np.asarray(sc, dtype=np.float32)
            if arr.ndim == 1:
                merged[m] = arr
        meta = {
            "__times__": np.array(times, dtype=object),
            "__oom__": np.array(sorted(oom_methods) if isinstance(oom_methods, (set, list)) else [], dtype=object),
            "__schema__": np.array("v1"),
        }
        merged.update(meta)
        # NFS-safe write. np.savez(path) does interleaved seek+write which, on
        # this NAS, corrupts the zip (BadZipFile "Bad magic number for file
        # header" on reload). Serialize to an in-memory BytesIO (one contiguous
        # buffer → no interleaved seek → no zip corruption), then persist.
        # Strategy: temp .npz in the SAME dir + fsync + atomic os.replace, with
        # retries (os.replace transiently EIOs on this NAS under load). If the
        # rename keeps failing, fall back to a direct contiguous overwrite of


        buf = io.BytesIO()
        np.savez(buf, **merged)
        data = buf.getvalue()
        saved = False
        last_err = None
        for attempt in range(4):
            fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(SCORE_CACHE_DIR))
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
                saved = True
                break
            except OSError as e:
                last_err = e
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                time.sleep(0.5 * (attempt + 1))
        if not saved:

            try:
                with open(path, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                saved = True
            except Exception as e:
                last_err = e
        if saved:
            print(f"  [score-cache] saved {len(scores)} method scores → {path.name}",
                  flush=True)
        else:
            print(f"  [score-cache] save failed after retries: {last_err}", flush=True)
    except Exception as e:
        print(f"  [score-cache] save failed: {e}", flush=True)

def load_screening_scores(task_type, dataset_name, seed, methods):
    """Read cached scores for the requested methods.

    Returns (scores:{method:(P,)float64}, times:dict, oom_methods:set).
    Missing methods are simply absent from the returned dict (caller marks
    them screening_NA). Returns ({}, {}, set()) if the cache file is absent
    or unreadable.
    """
    path = _score_cache_path(task_type, dataset_name, seed)
    if not path.exists():
        return {}, {}, set()
    scores, times, oom = {}, {}, set()
    try:
        z = np.load(path, allow_pickle=True)
        for m in methods:
            if m in z.files and z[m].ndim == 1:
                scores[m] = np.asarray(z[m], dtype=np.float64)
        if "__times__" in z.files:
            try:
                times = dict(z["__times__"].item())
                # normalize 'OOM' string → keep as-is (None float stays None)
            except Exception:
                times = {}
        if "__oom__" in z.files:
            try:
                oom = set(z["__oom__"].tolist())
            except Exception:
                oom = set()
    except Exception as e:
        print(f"  [score-cache] load failed: {e}", flush=True)
        return {}, {}, set()
    return scores, times, oom

def find_missing_methods(existing, task_type, models, scr_methods):
    if existing is None:
        return set(scr_methods)
    all_m = set(scr_methods)
    present = set()
    for model_name in models:
        if model_name not in existing or not isinstance(existing[model_name], dict):
            continue
        for m in all_m:
            g3 = existing[model_name].get(f"g3_{m}")
            if isinstance(g3, dict) and "error" not in g3:
                present.add(m)
    return all_m - present

def _entry_present(ev, key):
    return isinstance(ev.get(key), dict)

def _models_complete(existing, task_type, models, skip_cpu_scorers):
    if existing is None:
        return False
    expected = _resume_expected_methods(skip_cpu_scorers)
    for mdl in models:
        # tabpfn-v3 has no regression ckpt on some setups → model is legitimately
        # skipped, never expected to appear.
        if mdl == "tabpfn" and task_type == "reg" and not os.path.exists(TABPFN_V3_REG):
            continue
        if mdl not in existing or not isinstance(existing[mdl], dict):
            return False
        ev = existing[mdl]
        for key in ("g1_clean", "g2_gaussian"):
            if not _entry_present(ev, key):
                return False
        for m in expected:
            if not _entry_present(ev, f"g3_{m}"):
                return False
    return True

class _LazyEmbedder:
    """Defers TabICL-v1.1 embedder load until first scoring that misses the cache.

    If a (dataset,seed)'s score cache already covers every expected GPU scorer,
    run_single_dataset reuses it and NEVER asks for the embedder → no model load,
    no extra RAM, no ~30s spin-up per dataset. Only un-cached datasets pay the
    load. The model is created once per process (subsequent calls return the
    cached object), so a subprocess handling several seeds amortises one load.
    """
    def __init__(self, loader, device):
        self._loader = loader
        self.device = device
        self._emb = None
        self._tried = False
        self.loaded = False

    def __call__(self):
        if self._tried:
            return self._emb
        self._tried = True
        try:
            self._emb = self._loader()
            self.loaded = bool(self._emb is not None
                               and getattr(self._emb, "loaded", False))
            if not self.loaded:
                print("[embedder] v1.1 unavailable → emb_* will be NA", flush=True)
        except Exception as e:
            print(f"[embedder] unavailable ({e}) → emb_* NA", flush=True)
            self._emb = None
            self.loaded = False
        return self._emb

def _expected_gpu_scorers():
    return STREAM_RAW_TRAD + [RAW_HSIC] + [f"{m}_emb" for m in EMB_METHODS] + RFF_METHODS

def _resume_expected_methods(skip_cpu_scorers):
    """Which screen methods a run EXPECTS to be non-error in its result JSON.

    Under --skip-cpu-scorers the 10 CPU methods (Tracks C+D) are NEVER
    produced → they are screening_NA by design, NOT work-remaining. Counting
    them as 'missing' would make find_missing_methods report every complete
    dataset as incomplete, so --resume would pointlessly re-run it every time.
    Under --skip-cpu-scorers the resumable set = the 12 GPU scorers only."""
    return list(_expected_gpu_scorers()) if skip_cpu_scorers else list(SCREEN_METHODS)

def run_single_dataset(dataset_name, task_type, seed, device, embedder=None,
                       models_filter=None, scr_filter=None, existing=None,
                       predict_only=False, rescore=False, skip_cpu_scorers=False,
                       get_embedder=None):
    models_use = models_filter or MODELS
    scr_methods = scr_filter or SCREEN_METHODS
    missing = find_missing_methods(existing, task_type, models_use, scr_methods)

    # Early skip iff every requested model is fully present (per-model check, NOT
    # the union over models — see _models_complete). This is what lets --resume
    # reuse an already-complete JSON while still recomputing a model that is
    # wholly absent (the old5 gap left by the new2-only run).
    if existing is not None and _models_complete(existing, task_type, models_use,
                                                 skip_cpu_scorers):
        print(f"  {dataset_name} s{seed} — all done, SKIP", flush=True)
        return existing
    if missing and existing is not None:
        print(f"  resume missing methods: {sorted(missing)}", flush=True)

    print(f"  Loading {dataset_name}...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None
    cat_indices, cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]
    n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train, _ = T.encode_y(ytr_raw, task_type)
    y_val, _ = T.encode_y(yv_raw, task_type)
    y_test, _ = T.encode_y(yte_raw, task_type)

    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    if n_noise == 0:
        print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES}", flush=True)
        return None
    print(f"  +{n_noise} noise → total {p + n_noise}", flush=True)

    fk_tr, fk_v, fk_te = T.generate_gaussian_noise_features(X_train, X_val, X_test, seed, n_noise=n_noise)
    Xc_tr = np.hstack([X_train, fk_tr])
    Xc_v = np.hstack([X_val, fk_v])
    Xc_te = np.hstack([X_test, fk_te])
    P = p + n_noise

    # ── Screening (timed) ──


    #
    # Three modes:
    #  predict_only  → load scores from cache; never run scorers. Cache-miss
    #                  methods fall through to screening_NA (no scoring).
    #  rescore       → always recompute fresh (overwrite cache).
    #  default       → if a usable cache already exists AND covers all method-
    #                  producing paths, reuse it; else compute fresh + save.
    screening_scores, screen_times, oom_methods = {}, {}, set()
    scr_err = None
    cache_loaded = False
    expected_gpu = _expected_gpu_scorers()
    if predict_only:
        screening_scores, screen_times, oom_methods = load_screening_scores(
            task_type, dataset_name, seed, scr_methods)
        cache_loaded = bool(screening_scores)
        if cache_loaded:
            print(f"  [predict-only] loaded {len(screening_scores)}/{len(scr_methods)} "
                  f"scored methods from cache", flush=True)
        else:
            print(f"  [predict-only] NO cache → all methods screening_NA "
                  f"(run once without --predict-only to populate)", flush=True)
            scr_err = "predict_only_no_cache"
    else:
        # Smart cache reuse: if a prior (auto/rescore) pass already cached EVERY
        # expected GPU scorer (counting OOM-recorded scorers as covered — re-running
        # an OOM scorer deterministically re-OOMs, so retry is pure waste), reuse
        # the cache byte-for-byte and SKIP the (expensive, embedder-forward)
        # re-scoring entirely. Missing/extra/currupt cache → recompute fresh +

        # Fixed-budget feature selection uses the same scores in either mode.
        if not rescore:
            cached_scores, cached_times, cached_oom = load_screening_scores(
                task_type, dataset_name, seed, expected_gpu)
            coverage = set(cached_scores) | cached_oom
            if expected_gpu and set(expected_gpu).issubset(coverage):
                screening_scores, screen_times, oom_methods = (
                    cached_scores, cached_times, cached_oom)
                cache_loaded = True
                print(f"  [cache-reuse] {len(screening_scores)}/{len(expected_gpu)} "
                      f"GPU scores already cached → skip re-scoring (embedder not loaded)",
                      flush=True)
        if not cache_loaded:
            # Load the embedder ONLY now (lazy): un-cached datasets pay ~one load;
            # cache-reused datasets never enter this branch → no model/RAM cost.
            emb = get_embedder() if get_embedder is not None else embedder
            try:
                screening_scores, screen_times, oom_methods = compute_screening_scores(
                    Xc_tr, y_train, p, task_type, device, emb,
                    skip_cpu_scorers=skip_cpu_scorers)
                # Persist the freshly computed scores (reusable by --predict-only / cache-reuse).
                save_screening_scores(task_type, dataset_name, seed,
                                      screening_scores, screen_times, oom_methods)
            except Exception as e:
                scr_err = "OOM" if T.is_oom_error(e) else str(e)
                print(f"  [screen OOM/err] {scr_err[:120]}", flush=True)

    results = {"dataset": dataset_name, "task_type": task_type, "seed": seed,
               "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
               "total_features": P, "screen_times_s": screen_times,
               "screen_oom_methods": sorted(oom_methods)}

    for model_name in models_use:
        # skip tabpfn reg if ckpt missing
        if model_name == "tabpfn" and task_type == "reg" and not os.path.exists(TABPFN_V3_REG):
            continue
        mr = {}

        def _m_key(key):
            if existing is not None and model_name in existing:
                ev = existing[model_name].get(key)
                if isinstance(ev, dict) and "error" not in ev:
                    return ev
            return None

        for key, args in (("g1_clean", (X_train, X_val, X_test)),
                          ("g2_gaussian", (Xc_tr, Xc_v, Xc_te))):
            cached = _m_key(key)
            if cached is not None:
                mr[key] = cached
            else:
                a_tr, a_v, a_te = args
                mr[key] = train_and_evaluate(model_name, a_tr, y_train, a_v, y_val, a_te, y_test,
                                             task_type, seed, cat_indices, device)

        # G3 per method
        if scr_err is not None:
            for m in scr_methods:
                mr[f"g3_{m}"] = {"error": f"screening_{scr_err}"}
            results[model_name] = mr
            continue

        for m in scr_methods:
            # resume skip
            if existing is not None and model_name in existing:
                for key in (f"g3_{m}",):
                    ev = existing[model_name].get(key)
                    if isinstance(ev, dict) and "error" not in ev:
                        mr[key] = ev
            if m not in missing and (existing is not None) and (f"g3_{m}" in mr):
                continue
            # OOM-recorded at screening → mark G3 as screening_OOM
            if m in oom_methods:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_OOM"}
                continue
            if m not in screening_scores:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_NA"}
                continue

            sc = screening_scores[m]

            # G3: top 3p
            g3_k = G3_TOP_K_MULTIPLIER * p
            top_idx = np.argsort(sc)[::-1][:g3_k]
            recall = int(np.sum(top_idx < p))
            X3_tr, X3_v, X3_te = Xc_tr[:, top_idx], Xc_v[:, top_idx], Xc_te[:, top_idx]
            cat_g3 = None
            if cat_indices:
                mp3 = {idx: i for i, idx in enumerate(top_idx)}
                cat_g3 = [mp3[idx] for idx in cat_indices if idx in mp3]
            m3 = train_and_evaluate(model_name, X3_tr, y_train, X3_v, y_val, X3_te, y_test,
                                    task_type, seed, cat_g3, device)
            m3["recall"] = recall
            m3["n_features"] = int(g3_k)
            mr[f"g3_{m}"] = m3


        results[model_name] = mr

    # Preserve existing results for models outside the requested subset.
    if existing is not None:
        for k, v in existing.items():
            if k not in results and isinstance(v, dict) and k in MODELS:
                results[k] = v

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return results

def _write_results_atomic(rf, res):
    tmp = rf.with_suffix(rf.suffix + ".tmp")
    try:
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        os.replace(tmp, rf)
    except Exception as e:
        print(f"  [WARN] atomic write failed ({e}); retry direct", flush=True)
        try:
            json.dump(res, open(rf, "w"), indent=2, default=str)
        except Exception as e2:
            print(f"  [ERR] result write failed: {e2}", flush=True)
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

def _dataset_json_path(out_dir, task_type, dataset_name, seed):
    return out_dir / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"

def _unit_done(out_dir, task_type, dataset_name, models, seeds, skip_cpu_scorers):
    """Cheap pre-filter: is every (dataset,seed) JSON already complete (every
    requested MODEL present with g1/g2 + every expected method's g3 recorded)?
    Used by the orchestrator to skip spawning a worker for an already-finished
    dataset. Per-model (not union) so a new2-only JSON is not mistaken done."""
    for s in seeds:
        rf = _dataset_json_path(out_dir, task_type, dataset_name, s)
        if not rf.exists():
            return False
        try:
            existing = json.load(open(rf))
        except Exception:
            return False
        if not _models_complete(existing, task_type, models, skip_cpu_scorers):
            return False
    return True

def orchestrate(pairs, out_dir, gpu_ids, models, seeds, args, per_gpu=1,
                max_retries=1):
    LOGS = RESULTS_ROOT / "logs_real_eval"
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}     # running[tag] = (proc, ds, task, fh, gpu)
    tries_map = {}                          # (task, ds) -> consecutive crashes so far (PERSISTENT: survives re-spawn, else a perpetually-crashing dataset would loop forever)
    threads = max(4, min(16, 28 // per_gpu)) if per_gpu > 1 else 16
    print(f"Orchestrate: {len(pairs)} datasets across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, {threads} torch threads/worker, "
          f"max_retries={max_retries}, skip-cpu-scorers={args.skip_cpu_scorers}, "
          f"predict-only={args.predict_only}, rescore={args.rescore})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        # Skip already-complete datasets so we never spawn a worker for them
        # (cheap JSON check; matches the worker's own --resume SKIP). Only when
        # resuming — without --resume the user may want to re-run on purpose.
        while pending and args.resume and _unit_done(out_dir, pending[0][1],
                                                     pending[0][0], models, seeds,
                                                     args.skip_cpu_scorers):
            ds, task = pending.pop(0)
            print(f"  [skip-done] {ds} ({task}) already complete", flush=True)
        free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        while free and pending:
            (ds, task), gpu = pending.pop(0), free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)        # remap physical gpu -> cuda:0
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env.setdefault(k, str(threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            lf = open(LOGS / f"full_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [sys.executable, "-u", str(BASE_DIR / "f2ves_real_eval_full.py"),
                   "--method", "full",
                   "--worker", "--task_type", task, "--datasets", ds,
                   "--device_id", "0", "--output", args.output,
                   "--threads", str(threads)]
            for flag in ("--resume", "--predict-only", "--rescore",
                         "--skip-cpu-scorers", "--no-embedder"):
                attr = flag.lstrip("-").replace("-", "_")
                if getattr(args, attr, False):
                    cmd.append(flag)
            if args.models:
                cmd += ["--models", *args.models]
            tag = f"{gpu}:{task}:{ds}"
            running[tag] = (subprocess.Popen(cmd, env=env, stdout=lf,
                                             stderr=subprocess.STDOUT),
                            ds, task, lf, gpu)
            attempts = tries_map.get((task, ds), 0)
            print(f"  [launch] {ds} ({task}) → GPU {gpu} "
                  f"(load {gpu_load(gpu)}/{per_gpu}"
                  f"{f', attempt {attempts+1}' if attempts else ''})", flush=True)
            free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        # Poll completions
        done = []
        for tag, (proc, ds, task, lf, gpu) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            if rc == 0:
                # success → clear the crash counter so a later genuine re-run is fresh
                tries_map.pop((task, ds), None)
                print(f"  [done] {ds} ({task}) gpu{gpu} rc=0", flush=True)
            else:
                # Non-zero incl. -9 (SIGKILL/memcg-OOM). Retry within this pass up
                # to max_retries (transient blips); persistent crabs are left for
                # the next --resume run instead of looping forever. tries_map is
                # PERSISTENT (counts across re-spawns) so the budget actually binds.
                attempts = tries_map.get((task, ds), 0)
                if attempts < max_retries:
                    tries_map[(task, ds)] = attempts + 1
                    print(f"  [crash → retry] {ds} ({task}) gpu{gpu} rc={rc} "
                          f"(attempt {attempts+1} of {max_retries+1})", flush=True)
                    pending.append((ds, task))
                else:
                    print(f"  [CRASH] {ds} ({task}) gpu{gpu} rc={rc} after "
                          f"{max_retries} retries — left for next --resume", flush=True)
            done.append(tag)
        for tag in done:
            del running[tag]
        if running:
            time.sleep(5)
    print("=== orchestrate: all jobs settled ===", flush=True)

def _make_embedder_loader(device):
    def _loader():
        import torch
        if not torch.cuda.is_available():
            return None
        from src.screening_methods_torch import TorchAcceleratedEmbedder, TABICL_PATH
        emb = TorchAcceleratedEmbedder(device=device)
        if not emb.load_model(TABICL_PATH, device=device):
            return None
        return emb
    return _loader

def _main_full():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default="results_real_eval_full")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, choices=MODELS)
    ap.add_argument("--no-embedder", action="store_true",
                    help="skip TabICL-v1.1 embedding forward (emb_* methods → NA); for CPU debug")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--predict-only", action="store_true",
                    help="skip scoring entirely; load per-method feature scores from "
                         "screening_score_cache/ and jump to selection+prediction "
                         "(G1/G2/G3). Cache-miss methods → screening_NA.")
    ap.add_argument("--rescore", action="store_true",
                    help="ignore existing score cache and recompute scores fresh "
                         "(then re-save). Mutually exclusive with --predict-only.")
    ap.add_argument("--skip-cpu-scorers", action="store_true",
                    help="skip the CPU-tree screening scorers (Tracks C rf_mdi/"
                         "lgb_split/lgb_gain/cb_pvc + D lgb_shap/cb_shap). Keeps "
                         "[:10] = raw+emb+RFF GPU scorers; the 6 skipped methods "
                         "record screening_NA for G3. Cache stays mergeable.")
    # ── Multi-GPU orchestration (mirrors exp30_kpool_tn8) ──
    ap.add_argument("--gpus", type=str, default=None,
                    help="orchestrator mode: comma-separated PHYSICAL gpu ids, e.g. "
                         "'0,1,2'. Spawns one worker subprocess per dataset (all 3 "
                         "seeds) per GPU, round-robin; a SIGKILL/OOM on one dataset "
                         "is isolated (rc!=0 logged + retried, run continues).")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--per_gpu", type=int, default=1,
                    help="concurrent datasets per GPU (>=1). 1 = one-per-GPU "
                         "(tabpfn/tabicl each want a whole 80GB card).")
    ap.add_argument("--max_retries", type=int, default=1,
                    help="retries a crashed (rc!=0) dataset within the same pass "
                         "(transient blip cover); more crashes → left for next --resume.")
    ap.add_argument("--threads", type=int, default=16,
                    help="torch/MKL threads per worker (orchestrator overrides when per_gpu>1)")
    args = ap.parse_args()

    if args.predict_only and args.rescore:
        print("ERR: --predict-only and --rescore are mutually exclusive.", flush=True)
        sys.exit(2)

    # Resilience-drill affordance (NEVER on in production runs): a spawned
    # worker with EVAL_FORCE_FAIL=1 exits non-zero immediately, to exercise the
    # orchestrator's crash → retry → give-up-and-continue path WITHOUT a real
    # SIGKILL/OOM (those cannot be safely triggered on demand). Gated to
    # --worker so it can never fire for an orchestrator/direct invocation.
    if args.worker and os.environ.get("EVAL_FORCE_FAIL"):
        print(f"[FORCE-FAIL] worker {args.task_type} {args.datasets} "
              f"exiting 1 (resilience drill)", flush=True)
        sys.exit(1)

    # Worker mode: invoked by the orchestrator (CUDA_VISIBLE_DEVICES remaps the
    # physical gpu → cuda:0). Handles ONE dataset's seeds; per-seed try/except so
    # one bad seed never aborts the subprocess's remaining seeds (resilience on
    # top of the orchestrator's per-dataset SIGKILL isolation).
    if args.gpus and not args.worker:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        out_dir = RESULTS_ROOT / args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = [args.random_seed] if args.random_seed else SEEDS
        if args.task_type == "all":
            pairs = ([(d, "bin") for d in DATASET_MAP["bin"]]
                    + [(d, "multi") for d in DATASET_MAP["multi"]]
                    + [(d, "reg") for d in DATASET_MAP["reg"]])
        else:
            datasets = args.datasets or DATASET_MAP[args.task_type]
            pairs = [(d, args.task_type) for d in datasets]
        print(f"Orchestrator: {len(pairs)} datasets, GPUs {gpu_ids}, seeds {seeds}",
              flush=True)
        orchestrate(pairs, out_dir, gpu_ids, args.models or MODELS, seeds, args,
                    per_gpu=args.per_gpu, max_retries=args.max_retries)
        return

    # `all` only makes sense in orchestrator (--gpus) mode; a single worker /
    # direct run must target one concrete task.
    if args.task_type == "all":
        print("ERR: --task_type all requires --gpus (orchestrator mode).", flush=True)
        sys.exit(2)

    # ── single-process worker / direct run ──
    import torch
    torch.set_num_threads(args.threads)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    # Align screening_methods_torch's module-global DEVICE with the chosen GPU.
    # emb_* scorers build their y-side/constant tensors on the global DEVICE
    # (default cuda:0); the embedder puts output embeddings on `device` (above).
    # Without this, device_id != 0 raises "two devices cuda:0 and cuda:1" in
    # emb_sis/dc/cru/hsic → every emb_* column silently NA on a non-0 shard.
    from src import screening_methods_torch as _smt
    _smt.set_device(args.device_id)

    datasets = args.datasets or DATASET_MAP[args.task_type]
    seeds = [args.random_seed] if args.random_seed else SEEDS
    out_dir = RESULTS_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    task_label = TASK_DIR_MAP[args.task_type]

    # Embedder loads LAZILY (only when a dataset's cache is incomplete and
    # scoring must actually run). Cache-reused & predict-only datasets never load
    # it → no ~30s spin-up, no extra RAM per dataset.
    get_embedder = None
    if not args.no_embedder:
        get_embedder = _LazyEmbedder(_make_embedder_loader(device), device)

    print("=" * 70)
    print(f"Real eval: task={task_label} datasets={len(datasets)} seeds={seeds} "
          f"models={args.models or MODELS} device={device} resume={args.resume}")
    print(f"screen methods ({len(SCREEN_METHODS)}): {SCREEN_METHODS}")
    print(f"mode: {'predict-only' if args.predict_only else 'rescore' if args.rescore else 'auto(score+cache)'}"
          f"{' +skip-cpu-scorers' if args.skip_cpu_scorers else ''}", flush=True)
    print("=" * 70, flush=True)

    for di, ds in enumerate(datasets):
        for seed in seeds:
            rf = _dataset_json_path(out_dir, args.task_type, ds, seed)
            existing = None
            if args.resume and rf.exists():
                try:
                    existing = json.load(open(rf))
                    if _models_complete(existing, args.task_type, args.models or MODELS,
                                        args.skip_cpu_scorers):
                        print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — done, SKIP", flush=True)
                        continue
                    print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — RESUME", flush=True)
                except Exception:
                    existing = None
            else:
                print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed}", flush=True)
            t0 = time.time()
            # Per-unit try/except: a stray exception on one dataset/seed is logged
            # and the loop continues (the result JSON is simply not written). The
            # orchestrator adds the SIGKILL layer on top for multi-GPU runs.
            try:
                res = run_single_dataset(ds, args.task_type, seed, device,
                                         models_filter=args.models, existing=existing,
                                         predict_only=args.predict_only, rescore=args.rescore,
                                         skip_cpu_scorers=args.skip_cpu_scorers,
                                         get_embedder=get_embedder)
            except Exception as e:
                print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                import traceback as _tb
                if os.environ.get("EVAL_FULL_TRACE"):
                    _tb.print_exc()
                continue
            if res is not None:
                _write_results_atomic(rf, res)
                print(f"  done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
            else:
                print(f"  SKIPPED", flush=True)
    print("\n=== all done ===", flush=True)


    if getattr(args, "worker", False):
        os._exit(0)

_HERE = os.path.dirname(os.path.abspath(__file__))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

T = T


# ======================================================================

# ======================================================================

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

NEW_MODELS = ["realmlp", "tabm", "mlp"]

TASK_TALENT = {"bin": "binclass", "multi": "multiclass", "reg": "regression"}

OUT_DIR_DEFAULT = "results_real_eval_full_dl3"

GPU12 = ["sis", "dc", "cru", "ids", "hsic",
         "sis_emb", "dc_emb", "cru_emb", "hsic_emb",
         "rff_dc", "rff_sis", "rff_hsic"]

DEEP_DEFAULTS = dict(
    max_epoch=200, batch_size=1024, normalization="standard",
    num_nan_policy="mean", cat_nan_policy="new", cat_policy="ordinal",
    num_policy="none", n_bins=2, cat_min_frequency=0.0, use_float=False,
    evaluate_option="best-val", workers=0,
)

def _load_default_config(model_type):
    if not _HAS_TALENT:
        raise RuntimeError(
            "TALENT is unavailable: neural downstream learners require TALENT on PYTHONPATH. "
            "CPU scoring does not require this learner interface.")
    with pkg_resources.files(_TALENT_PKG).joinpath(
            "configs", "default", model_type + ".json").open("r") as f:
        return json.load(f)[model_type]

def build_args(model_type, seed, save_path, cat_policy, max_epoch):
    cfg = copy.deepcopy(_load_default_config(model_type))
    cfg.setdefault("training", {})["n_bins"] = DEEP_DEFAULTS["n_bins"]
    args = argparse.Namespace(
        model_type=model_type,
        seed=seed,
        save_path=str(save_path),
        config=cfg,
        max_epoch=max_epoch,
        batch_size=DEEP_DEFAULTS["batch_size"],
        normalization=DEEP_DEFAULTS["normalization"],
        num_nan_policy=DEEP_DEFAULTS["num_nan_policy"],
        cat_nan_policy=DEEP_DEFAULTS["cat_nan_policy"],
        cat_policy=cat_policy,
        num_policy=DEEP_DEFAULTS["num_policy"],
        n_bins=DEEP_DEFAULTS["n_bins"],
        cat_min_frequency=DEEP_DEFAULTS["cat_min_frequency"],
        use_float=DEEP_DEFAULTS["use_float"],
        evaluate_option=DEEP_DEFAULTS["evaluate_option"],
        workers=DEEP_DEFAULTS["workers"],

        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    return args

def _softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def train_and_evaluate_dl(model_name, X_train, y_train, X_val, y_val,
                          X_test, y_test, task_type, seed,
                          cat_indices=None, device="cuda:0", max_epoch=200):
    _TT_NORM = {"binclass": "bin", "multiclass": "multi", "regression": "reg",
                "bin": "bin", "multi": "multi", "reg": "reg"}
    task_type = _TT_NORM.get(task_type, task_type)

    if task_type != "reg":
        if len(np.unique(np.asarray(y_train))) < 2:
            return {"error": "single_class_in_fold"}

    is_reg = (task_type == "reg")
    cat_policy = "indices" if model_name in ("tabm", "realmlp") else "ohe"


    tmp_save = tempfile.mkdtemp(prefix=f"dl3_{model_name}_s{seed}_")
    args = build_args(model_name, seed, tmp_save, cat_policy, max_epoch)

    info = {"task_type": TASK_TALENT[task_type],
            "n_num_features": X_train.shape[1], "n_cat_features": 0}
    N_tr = {"train": np.ascontiguousarray(X_train, dtype=float),
            "val":   np.ascontiguousarray(X_val, dtype=float)}
    N_te = {"test": np.ascontiguousarray(X_test, dtype=float)}
    y_tr = {"train": np.asarray(y_train), "val": np.asarray(y_val)}
    y_te = {"test": np.asarray(y_test)}
    train_val_data = (N_tr, None, y_tr)
    test_data = (N_te, None, y_te)

    try:
        set_seeds(seed)
        method = get_method(model_name)(args, is_reg)
        method.fit(train_val_data, info)
        vl, vres, metric_name, test_logit = method.predict(
            test_data, info, args.evaluate_option)

        test_logit = (test_logit.cpu().numpy()
                      if torch.is_tensor(test_logit) else np.asarray(test_logit))

        if is_reg:
            pred = test_logit.reshape(-1)
            proba = None
        else:

            proba_all = test_logit if model_name == "realmlp" else _softmax_np(test_logit)
            if proba_all.ndim == 1:
                proba_all = np.vstack([1 - proba_all, proba_all]).T
            pred = proba_all.argmax(axis=1)
            if task_type == "bin":
                proba = proba_all[:, 1] if proba_all.shape[1] >= 2 else proba_all[:, 0]
            else:
                proba = proba_all

        return compute_metrics(y_test, pred, proba, task_type)
    except Exception as e:
        if T.is_oom_error(e):
            print(f"    [OOM] {model_name}: {str(e)[:100]}", flush=True)
            return {"error": "OOM", "detail": str(e)[:200]}
        print(f"    [WARN] {model_name} failed: {str(e)[:120]}", flush=True)
        return {"error": str(e)[:200]}
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(tmp_save, ignore_errors=True)

def _model_complete_dl(existing_model, task_type):
    if not isinstance(existing_model, dict):
        return False
    for key in ("g1_clean", "g2_gaussian"):
        if key not in existing_model:
            return False
    for m in GPU12:
        for key in (f"g3_{m}",):
            if key not in existing_model:
                return False
    return True

def _models_complete_dl(existing, task_type, models):
    if not isinstance(existing, dict):
        return False
    for m in models:
        if m not in existing or not _model_complete_dl(existing[m], task_type):
            return False
    return True

def _dataset_json_path_dl(out_dir, task_type, dataset_name, seed):
    return Path(out_dir) / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"

def run_single_dataset_dl(dataset_name, task_type, seed, device,
                           models_filter=None, existing=None, max_epoch=200,
                           output_dir=None):
    models_use = models_filter or NEW_MODELS

    # resume skip
    if existing is not None and _models_complete_dl(existing, task_type, models_use):
        print(f"  {dataset_name} s{seed} — all dl models done, SKIP", flush=True)
        return existing


    screening_scores, _times, _oom = load_screening_scores(
        task_type, dataset_name, seed, GPU12)
    if not screening_scores:
        print(f"  {dataset_name} s{seed} — NO screening cache → G3 all NA", flush=True)


    print(f"  Loading {dataset_name} (dl3)...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None
    cat_indices, cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]
    n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train, _ = T.encode_y(ytr_raw, task_type)
    y_val, _ = T.encode_y(yv_raw, task_type)
    y_test, _ = T.encode_y(yte_raw, task_type)

    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    if n_noise == 0:
        print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES}", flush=True)
        return None
    fk_tr, fk_v, fk_te = T.generate_gaussian_noise_features(
        X_train, X_val, X_test, seed, n_noise=n_noise)
    Xc_tr = np.hstack([X_train, fk_tr])
    Xc_v = np.hstack([X_val, fk_v])
    Xc_te = np.hstack([X_test, fk_te])

    results = {"dataset": dataset_name, "task_type": task_type, "seed": seed,
               "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
               "total_features": p + n_noise}

    for model_name in models_use:
        mr = {}

        def _m_key(key):

            if existing is not None and model_name in existing:
                ev = existing[model_name].get(key)
                if isinstance(ev, dict) and "error" not in ev:
                    return ev
            return None

        # G1 clean / G2 gaussian
        for key, args in (("g1_clean", (X_train, X_val, X_test)),
                          ("g2_gaussian", (Xc_tr, Xc_v, Xc_te))):
            cached = _m_key(key)
            if cached is not None:
                mr[key] = cached
            else:
                a_tr, a_v, a_te = args
                mr[key] = train_and_evaluate_dl(
                    model_name, a_tr, y_train, a_v, y_val, a_te, y_test,
                    task_type, seed, cat_indices, device, max_epoch)


        for m in GPU12:
            for key in (f"g3_{m}",):
                if existing is not None and model_name in existing:
                    ev = existing[model_name].get(key)
                    if isinstance(ev, dict) and "error" not in ev:
                        mr[key] = ev
            if m not in screening_scores:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_NA"}
                continue

            sc = screening_scores[m]
            # G3: top 3p
            g3_k = G3_TOP_K_MULTIPLIER * p
            top_idx = np.argsort(sc)[::-1][:g3_k]
            recall = int(np.sum(top_idx < p))
            X3_tr, X3_v, X3_te = Xc_tr[:, top_idx], Xc_v[:, top_idx], Xc_te[:, top_idx]
            if f"g3_{m}" not in mr:
                m3 = train_and_evaluate_dl(
                    model_name, X3_tr, y_train, X3_v, y_val, X3_te, y_test,
                    task_type, seed, cat_indices, device, max_epoch)
                m3["recall"] = recall
                m3["n_features"] = int(g3_k)
                mr[f"g3_{m}"] = m3


        results[model_name] = mr


    if existing is not None:
        for k, v in existing.items():
            if k not in results and isinstance(v, dict) and k in NEW_MODELS:
                results[k] = v
    return results

DL3_LOGS = mp.OUTPUT_ROOT / "logs" / "dl3_logs"

def _unit_done_dl(out_dir, ds, task, models, seeds):
    for seed in seeds:
        rf = _dataset_json_path_dl(out_dir, task, ds, seed)
        if not rf.exists():
            return False
        try:
            existing = json.load(open(rf))
        except Exception:
            return False
        if not _models_complete_dl(existing, task, models):
            return False
    return True

def orchestrate_dl(pairs, out_dir, gpu_ids, models, seeds, args, per_gpu=1,
                   max_retries=1):
    DL3_LOGS.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}     # running[tag] = (proc, ds, task, fh, gpu)
    tries_map = {}
    threads = args.threads
    print(f"[dl3] Orchestrate: {len(pairs)} (ds,task) across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, {threads} torch threads/worker, max_retries={max_retries}, "
          f"max_epoch={args.max_epoch}, models={models})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        while pending and args.resume and _unit_done_dl(
                out_dir, pending[0][0], pending[0][1], models, seeds):
            ds, task = pending.pop(0)
            print(f"  [skip-done] {ds} ({task}) already complete", flush=True)
        free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        while free and pending:
            (ds, task), gpu = pending.pop(0), free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = str(BASE_DIR / "TALENT") + ":" + env.get("PYTHONPATH", "")
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env.setdefault(k, str(threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            lf = open(DL3_LOGS / f"dl3_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [sys.executable, "-u",
                   str(Path(_HERE) / "f2ves_real_eval_full.py"),
                   "--method", "full",
                   "--worker", "--task_type", task, "--datasets", ds,
                   "--device_id", "0", "--output", args.output,
                   "--threads", str(threads), "--max_epoch", str(args.max_epoch),
                   "--resume"]
            if args.models:
                cmd += ["--models", *args.models]
            tag = f"{gpu}:{task}:{ds}"
            running[tag] = (subprocess.Popen(cmd, env=env, stdout=lf,
                                             stderr=subprocess.STDOUT),
                            ds, task, lf, gpu)
            print(f"  [launch] {ds} ({task}) → GPU {gpu} "
                  f"(load {gpu_load(gpu)}/{per_gpu})", flush=True)
            free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        done = []
        for tag, (proc, ds, task, lf, gpu) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            if rc == 0:
                tries_map.pop((task, ds), None)
                print(f"  [done] {ds} ({task}) gpu{gpu} rc=0", flush=True)
            else:
                attempts = tries_map.get((task, ds), 0)
                if attempts < max_retries:
                    tries_map[(task, ds)] = attempts + 1
                    print(f"  [crash→retry] {ds} ({task}) gpu{gpu} rc={rc} "
                          f"(attempt {attempts+1}/{max_retries+1})", flush=True)
                    pending.append((ds, task))
                else:
                    print(f"  [CRASH] {ds} ({task}) gpu{gpu} rc={rc} after "
                          f"{max_retries} retries — left for next --resume", flush=True)
            done.append(tag)
        for tag in done:
            del running[tag]
        if running:
            time.sleep(5)
    print("=== dl3 orchestrate: all jobs settled ===", flush=True)

def _main_dl3():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default=OUT_DIR_DEFAULT)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, choices=NEW_MODELS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_epoch", type=int, default=DEEP_DEFAULTS["max_epoch"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--max_retries", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(RESULTS_ROOT) / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    models_use = args.models or NEW_MODELS
    seeds = [args.random_seed] if args.random_seed else SEEDS


    if args.worker:

        torch.set_num_threads(args.threads)
        ds = args.datasets[0]
        task = args.task_type
        print(f"[dl3-worker] {ds} ({task}) seeds={seeds} models={models_use} "
              f"max_epoch={args.max_epoch}", flush=True)
        for seed in seeds:
            rf = _dataset_json_path_dl(out_dir, task, ds, seed)
            existing = None
            if args.resume and rf.exists():
                try:
                    existing = json.load(open(rf))
                    if _models_complete_dl(existing, task, models_use):
                        print(f"  {ds} s{seed} — done, SKIP", flush=True)
                        continue
                except Exception:
                    existing = None
            t0 = time.time()
            try:
                res = run_single_dataset_dl(
                    ds, task, seed, "cuda:0",
                    models_filter=models_use, existing=existing,
                    max_epoch=args.max_epoch, output_dir=out_dir)
            except Exception as e:
                print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                import traceback
                traceback.print_exc()
                continue
            if res is not None:
                _write_results_atomic(rf, res)
                print(f"  {ds} s{seed} done ({time.time()-t0:.1f}s) → {rf.name}",
                      flush=True)
        return


    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
        if args.task_type == "all":

            pairs = ([(d, "bin") for d in DATASET_MAP["bin"]]
                     + [(d, "multi") for d in DATASET_MAP["multi"]]
                     + [(d, "reg") for d in DATASET_MAP["reg"]])
        else:
            datasets = args.datasets or DATASET_MAP[args.task_type]
            pairs = [(d, args.task_type) for d in datasets]
        orchestrate_dl(pairs, out_dir, gpu_ids, models_use, seeds, args,
                       per_gpu=args.per_gpu, max_retries=args.max_retries)
        return


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.set_num_threads(args.threads)
    if args.task_type == "all":
        tasks = ["bin", "multi", "reg"]
    else:
        tasks = [args.task_type]

    print("=" * 70, flush=True)
    print(f"dl3 eval (direct): tasks={tasks} datasets={'all' if not args.datasets else args.datasets} "
          f"seeds={seeds} models={models_use} device_id={args.device_id} "
          f"max_epoch={args.max_epoch} resume={args.resume}", flush=True)
    print(f"output → {out_dir}  | scores reused from {SCORE_CACHE_DIR} (12 GPU methods, read-only)",
          flush=True)
    print("=" * 70, flush=True)

    for task in tasks:
        ds_list = args.datasets if args.datasets else DATASET_MAP[task]
        for di, ds in enumerate(ds_list):
            for seed in seeds:
                rf = _dataset_json_path_dl(out_dir, task, ds, seed)
                existing = None
                if args.resume and rf.exists():
                    try:
                        existing = json.load(open(rf))
                        if _models_complete_dl(existing, task, models_use):
                            print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed} — done, SKIP",
                                  flush=True)
                            continue
                        print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed} — RESUME", flush=True)
                    except Exception:
                        existing = None
                else:
                    print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed}", flush=True)
                t0 = time.time()
                try:
                    res = run_single_dataset_dl(
                        ds, task, seed, f"cuda:0",
                        models_filter=models_use, existing=existing,
                        max_epoch=args.max_epoch, output_dir=out_dir)
                except Exception as e:
                    print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
                if res is not None:
                    _write_results_atomic(rf, res)
                    print(f"  done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
                else:
                    print(f"  skipped (n_noise==0 or load err)", flush=True)


# ======================================================================

# ======================================================================

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

TRACK_E_METHODS = list(E.TRACK_E_METHODS)

ALL_MODELS = list(MODELS) + list(NEW_MODELS)

DL3_MODELS = set(NEW_MODELS)

OUT_DIR_DEFAULT = "results_real_eval_full_E"

SCORE_CACHE_DIR_E = RESULTS_ROOT / "screening_score_cache_E"

E_LOGS = mp.OUTPUT_ROOT / "logs" / "E_full_logs"

NATIVE_RULE = {
    'lassonet_path': 'fi>0 (CV-best lambda active set)',
    'sand_score':    f'top-k0 (k0={E.SAND_K0}, capped p) — fixed-budget gate',
    'deep_lasso':    'saliency>=1%max (effective-support proxy, continuous)',
    'gradenfs':      'saliency>=1%max (EMA proxy, cls-only; reg→na_task)',
    'xrfm_agop':     'na_native (no native sparsity; covered by G3)',
}

def _E_cache_path(task_type, dataset_name, seed):
    return SCORE_CACHE_DIR_E / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.npz"

def _save_E_scores(task_type, dataset_name, seed, scores, statuses, times, meta):
    path = _E_cache_path(task_type, dataset_name, seed)
    SCORE_CACHE_DIR_E.mkdir(parents=True, exist_ok=True)
    merged = {}

    if path.exists():
        try:
            with np.load(path, allow_pickle=True) as old:
                for k in old.files:
                    if k.startswith('__status_') and f"{k[len('__status_'):]}" in scores:
                        continue
                    merged[k] = np.asarray(old[k])

                for m, sc in scores.items():
                    if sc is not None and m in old.files:
                        pass
        except Exception:
            merged = {}
    for m, sc in scores.items():
        if sc is not None:
            merged[m] = np.asarray(sc, dtype=np.float32)
    for m, st in statuses.items():
        merged[f"__status_{m}"] = np.array(st)
    merged['__times__'] = np.array([times.get(m, 0.0) for m in TRACK_E_METHODS], dtype=np.float64)
    merged['__meta__'] = np.array(json.dumps(meta))
    buf = __import__('io').BytesIO()
    np.savez(buf, **merged)
    data = buf.getvalue()
    tmp = path.with_suffix(path.suffix + '.tmp')
    last_err = None
    for _attempt in range(5):
        try:
            with open(tmp, 'wb') as f:
                f.write(data)


            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e; time.sleep(0.5 * (_attempt + 1))

    try:
        with open(path, 'wb') as f:
            f.write(data)
        return
    except Exception as e:
        print(f"  [E-cache WRITE FAIL] {path.name}: {e} / last={last_err}", flush=True)

def _write_E_results(rf, res):
    Path(rf).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(rf) + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=str)
    os.replace(tmp, rf)

def _load_E_scores(task_type, dataset_name, seed, methods):
    path = _E_cache_path(task_type, dataset_name, seed)
    out_s, out_st = {}, {}
    if not path.exists():
        for m in methods:
            out_s[m] = None; out_st[m] = 'missing'
        return out_s, out_st, None
    try:
        with np.load(path, allow_pickle=True) as d:
            files = set(d.files)
            st_raw = {}
            if '__meta__' in files:
                try:
                    meta = json.loads(str(d['__meta__']))
                except Exception:
                    meta = None
            else:
                meta = None
            for m in methods:
                if m in files:
                    out_s[m] = np.asarray(d[m], dtype=np.float64)
                    out_st[m] = 'ok'
                else:
                    out_s[m] = None
                    sk = f"__status_{m}"
                    out_st[m] = str(d[sk]) if sk in files else 'missing'
            return out_s, out_st, meta
    except Exception as e:
        print(f"  [E-cache LOAD FAIL] {path.name}: {e} → recompute scores", flush=True)
        for m in methods:
            out_s[m] = None; out_st[m] = 'missing'
        return out_s, out_st, None

def _compute_E_scores(methods, Xc_inner, y_inner, Xv_inner, yv_inner, has_v,
                      P, task_full, seed, device):
    Xtr_arg = Xc_inner
    Xv_arg = Xv_inner if has_v else Xc_inner
    ytr_arg = y_inner
    yv_arg = yv_inner if has_v else y_inner
    scores, statuses, times = {}, {}, {}
    sel_seed = int(seed)
    for m in methods:
        t0 = time.perf_counter()
        try:
            sc = compute_scores(Xtr_arg, ytr_arg, m, task=task_full, device=device,
                validation_data=(Xv_arg, yv_arg), seed=sel_seed)
            st = "ok"
        except (ImportError, ModuleNotFoundError) as e:
            sc, st = None, 'na_dep'
            print(f"    [E-score {m}] NA dep: {str(e)[:100]}", flush=True)
        except Exception as e:
            sc, st = None, 'error'
            print(f"    [E-score {m}] EXC -> error: {type(e).__name__}: {str(e)[:140]}", flush=True)
        times[m] = time.perf_counter() - t0
        scores[m] = sc
        statuses[m] = st if st else 'ok'
        if sc is not None:
            assert sc.shape[0] == P, f"[E {m}] score len {sc.shape[0]} != P {P}"
    return scores, statuses, times

def _score_unit_E(task_type, dataset_name, seed, methods, device,
                  Xc_tr, y_train, P, p, n, task_full):
    cached_s, cached_st, meta_cached = _load_E_scores(task_type, dataset_name, seed, methods)

    need = [m for m in methods
            if cached_s[m] is None and cached_st[m] not in ('na_resource', 'na_task', 'na_dep')]
    scores, statuses, times = dict(cached_s), dict(cached_st), {}
    for m in methods:
        times[m] = 0.0
    if need:
        Xtr_i, Xv_i, ytr_i, yv_i, has_v = _inner_split(Xc_tr, y_train, task_type)
        print(f"  [E-score] computing {len(need)} method(s): {need} on Xc_inner "
              f"(P={P}, n_inner={Xtr_i.shape[0]})", flush=True)
        ns, nst, nt = _compute_E_scores(need, Xtr_i, ytr_i, Xv_i, yv_i, has_v,
                                        P, task_full, seed, device)
        scores.update(ns); statuses.update(nst)
        for m, v in nt.items():
            times[m] = v
        meta = {'task_type': task_type, 'dataset': dataset_name, 'seed': seed,
                'p': int(p), 'n': int(n), 'P': int(P),
                'selector_seed': int(seed), 'schema': 'v1_E',
                'rank_type': {m: E.RANK_TYPE.get(m, '') for m in methods}}

        _save_E_scores(task_type, dataset_name, seed, scores, statuses, times, meta)
        return scores, statuses, times, meta

    times_meta = {'task_type': task_type, 'dataset': dataset_name, 'seed': seed,
                  'p': int(p), 'n': int(n), 'P': int(P), 'selector_seed': int(seed),
                  'schema': 'v1_E',
                  'rank_type': {m: E.RANK_TYPE.get(m, '') for m in methods}}
    return scores, statuses, times, (meta_cached or times_meta)

def _native_mask(method, sc_real_p, p):
    if method == 'xrfm_agop':
        return None
    if sc_real_p is None:
        return None
    s = np.asarray(sc_real_p, dtype=np.float64)
    if s.size == 0:
        return None
    if method == 'lassonet_path':
        mask = s > 0.0
    elif method == 'sand_score':
        k0 = min(int(E.SAND_K0), p)
        idx = np.argsort(s)[::-1][:k0]
        mask = np.zeros(p, dtype=bool); mask[idx] = True
    elif method in ('deep_lasso', 'gradenfs'):
        mx = float(np.max(s))
        thr = 0.01 * mx
        mask = s >= thr if thr > 0 else (s >= 0.0)
    else:
        return None
    if not mask.any():
        mask[int(np.argmax(s))] = True
    return mask

def _is_dl_model(m):
    return m in DL3_MODELS

def _train_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                 task_type, seed, device, max_epoch, cat_indices=None):
    if _is_dl_model(model_name):
        return train_and_evaluate_dl(model_name, X_train, y_train, X_val, y_val,
                                        X_test, y_test, task_type, seed,
                                        cat_indices, device, max_epoch)
    return train_and_evaluate(model_name, X_train, y_train, X_val, y_val,
                                X_test, y_test, task_type, seed, cat_indices, device)

def _reuse_g12(model_name, task_type, dataset_name, seed, key):
    if _is_dl_model(model_name):
        jp = RESULTS_ROOT / "results_real_eval_full_dl3" / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"
    else:
        jp = RESULTS_ROOT / "results_real_eval_full" / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"
    try:
        d = json.load(open(jp))
        ev = d.get(model_name, {}).get(key)
        if isinstance(ev, dict) and 'error' not in ev:
            return ev
    except Exception:
        pass
    return None

def _E_models_complete(existing, methods, models, task_type):
    if not isinstance(existing, dict):
        return False
    for model in models:
        em = existing.get(model)
        if not isinstance(em, dict):
            return False
        for key in ('g1_clean', 'g2_gaussian'):
            if key not in em:
                return False
        for m in methods:
            for key in (f'g3_{m}', f'g_native_{m}'):
                if key not in em:
                    return False
    return True

def _dataset_json_path_E(out_dir, task_type, dataset_name, seed):
    return Path(out_dir) / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"

def run_single_dataset_E(dataset_name, task_type, seed, device,
                         models_filter=None, methods_filter=None, existing=None,
                         max_epoch=50, output_dir=None, reuse_g12=True,
                         score_only=False, downstream_only=False):
    models_use = models_filter or ALL_MODELS
    methods_use = methods_filter or TRACK_E_METHODS
    task_full = TASK_FULL[task_type]

    if existing is not None and _E_models_complete(existing, methods_use, models_use, task_type):
        print(f"  {dataset_name} s{seed} — all E models/methods done, SKIP", flush=True)
        return existing


    print(f"  Loading {dataset_name} (E)...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None
    cat_indices, cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]; n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train, _ = T.encode_y(ytr_raw, task_type)
    y_val, _ = T.encode_y(yv_raw, task_type)
    y_test, _ = T.encode_y(yte_raw, task_type)

    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    if n_noise == 0:
        print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES} (no noise pool)", flush=True)
        return None
    fk_tr, fk_v, fk_te = T.generate_gaussian_noise_features(
        X_train, X_val, X_test, seed, n_noise=n_noise)
    Xc_tr = np.hstack([X_train, fk_tr])
    Xc_v = np.hstack([X_val, fk_v])
    Xc_te = np.hstack([X_test, fk_te])
    P = p + n_noise                       # == TARGET_TOTAL_FEATURES


    scores, statuses, stimes, scmeta = _score_unit_E(
        task_type, dataset_name, seed, methods_use, device,
        Xc_tr, y_train, P, p, n, task_full)
    if score_only:
        print(f"  [score-only] {dataset_name} s{seed} score cache saved; downstream evaluation disabled", flush=True)
        return None

    results = {"dataset": dataset_name, "task_type": task_type, "seed": seed,
               "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
               "total_features": P, "schema": "v1_E", "track": "E",
               "screen_times_s": stimes,
               "screen_status": {m: statuses[m] for m in methods_use},
               "native_rule": NATIVE_RULE}

    if existing is not None:
        for k, v in existing.items():
            if k not in results and not isinstance(v, dict):
                results[k] = v

    def _train_full(model, key, Xtr_a, Xv_a, Xte_a, cat_arg):
        return _train_model(model, Xtr_a, y_train, Xv_a, y_val, Xte_a, y_test,
                            task_type, seed, device, max_epoch, cat_indices=cat_arg)

    for model_name in models_use:
        mr = {}

        def _m_key(key):
            if existing and model_name in existing:
                ev = existing[model_name].get(key)
                if isinstance(ev, dict) and 'error' not in ev:
                    return ev
            return None

        # ── G1 clean（real p）──
        cached = _m_key('g1_clean')
        if cached is not None:
            mr['g1_clean'] = cached
        else:
            r = _reuse_g12(model_name, task_type, dataset_name, seed, 'g1_clean') if reuse_g12 else None
            if r is not None:
                mr['g1_clean'] = r
            else:
                mr['g1_clean'] = _train_full(model_name, 'g1_clean',
                                             X_train, X_val, X_test, cat_indices)
        # ── G2 gaussian（combined 2000）──
        cached = _m_key('g2_gaussian')
        if cached is not None:
            mr['g2_gaussian'] = cached
        else:
            r = _reuse_g12(model_name, task_type, dataset_name, seed, 'g2_gaussian') if reuse_g12 else None
            if r is not None:
                mr['g2_gaussian'] = r
            else:
                mr['g2_gaussian'] = _train_full(model_name, 'g2_gaussian',
                                                Xc_tr, Xc_v, Xc_te, cat_indices)


        for m in methods_use:
            sc = scores.get(m)
            st = statuses.get(m, 'missing')
            na_err = {'error': st} if st not in ('ok', 'missing') else {'error': 'screening_NA'}

            # G3: top 3p of combined scores
            g3_key = f'g3_{m}'
            cached = _m_key(g3_key)
            if cached is not None:
                mr[g3_key] = cached
            elif sc is None:
                mr[g3_key] = na_err
            else:
                g3_k = G3_TOP_K_MULTIPLIER * p
                top_idx = np.argsort(sc)[::-1][:g3_k]
                m3 = _train_model(model_name, Xc_tr[:, top_idx], y_train,
                                  Xc_v[:, top_idx], y_val, Xc_te[:, top_idx], y_test,
                                  task_type, seed, device, max_epoch, cat_indices=None)
                m3['n_features'] = int(g3_k)
                m3['recall'] = int(np.sum(top_idx < p))
                mr[g3_key] = m3



            gn_key = f'g_native_{m}'
            cached = _m_key(gn_key)
            if cached is not None:
                mr[gn_key] = cached
            elif m == 'xrfm_agop':
                mr[gn_key] = {'error': 'na_native'}
            elif sc is None:
                mr[gn_key] = na_err
            elif m == 'gradenfs' and task_type == 'reg':
                mr[gn_key] = {'error': 'na_task'}
            else:
                mask = _native_mask(m, sc[:p], p)
                if mask is None:
                    mr[gn_key] = {'error': 'screening_NA'}
                else:
                    keep = np.where(mask)[0]
                    mn = _train_model(model_name, X_train[:, keep], y_train,
                                      X_val[:, keep], y_val, X_test[:, keep], y_test,
                                      task_type, seed, device, max_epoch, cat_indices=None)
                    mn['n_features'] = int(len(keep))
                    mn['native_rule'] = NATIVE_RULE.get(m, '')
                    mr[gn_key] = mn

        results[model_name] = mr


    if existing is not None:
        for k, v in existing.items():
            if k not in results and isinstance(v, dict) and k in ALL_MODELS:
                results[k] = v
    return results

def _unit_done_E(out_dir, ds, task, methods, models, seeds):
    for seed in seeds:
        rf = _dataset_json_path_E(out_dir, task, ds, seed)
        if not rf.exists():
            return False
        try:
            with open(rf) as f:
                existing = json.load(f)
        except Exception:
            return False
        if not _E_models_complete(existing, methods, models, task):
            return False
    return True

def orchestrate_E(pairs, out_dir, gpu_ids, methods, models, seeds, args,
                  per_gpu=1, max_retries=1):
    E_LOGS.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}
    tries_map = {}
    threads = args.threads
    print(f"[E-full] Orchestrate: {len(pairs)} (ds,task) across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, {threads} threads, max_retries={max_retries}, "
          f"max_epoch={args.max_epoch}, methods={methods}, models={models}, "
          f"reuse_g12={not args.no_reuse_g12})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        while pending and args.resume and _unit_done_E(
                out_dir, pending[0][0], pending[0][1], methods, models, seeds):
            ds, task = pending.pop(0)
            print(f"  [skip-done] {ds} ({task}) already complete", flush=True)
        free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        while free and pending:
            (ds, task), gpu = pending.pop(0), free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = str(BASE_DIR / "TALENT") + ":" + env.get("PYTHONPATH", "")
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS"):
                env.setdefault(k, str(threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            lf = open(E_LOGS / f"E_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [sys.executable, "-u",
                   str(Path(_HERE) / "f2ves_real_eval_full.py"),
                   "--method", "E",
                   "--worker", "--task_type", task, "--datasets", ds,
                   "--device_id", "0", "--output", args.output,
                   "--threads", str(threads), "--max_epoch", str(args.max_epoch),
                   "--resume"]
            if args.no_reuse_g12:
                cmd += ["--no-reuse-g12"]
            if args.score_only:
                cmd += ["--score-only"]
            if args.downstream_only:
                cmd += ["--downstream-only"]
            if args.methods:
                cmd += ["--methods", *args.methods]
            if args.models:
                cmd += ["--models", *args.models]
            tag = f"{gpu}:{task}:{ds}"
            running[tag] = (subprocess.Popen(cmd, env=env, stdout=lf,
                                             stderr=subprocess.STDOUT),
                            ds, task, lf, gpu)
            print(f"  [launch] {ds} ({task}) → GPU {gpu} (load {gpu_load(gpu)}/{per_gpu})",
                  flush=True)
            free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        done = []
        for tag, (proc, ds, task, lf, gpu) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            if rc == 0:
                tries_map.pop((task, ds), None)
                print(f"  [done] {ds} ({task}) gpu{gpu} rc=0", flush=True)
            else:
                attempts = tries_map.get((task, ds), 0)
                if attempts < max_retries:
                    tries_map[(task, ds)] = attempts + 1
                    print(f"  [crash→retry] {ds} ({task}) gpu{gpu} rc={rc} "
                          f"(attempt {attempts+1}/{max_retries+1})", flush=True)
                    pending.append((ds, task))
                else:
                    print(f"  [CRASH] {ds} ({task}) gpu{gpu} rc={rc} after "
                          f"{max_retries} retries — left for next --resume", flush=True)
            done.append(tag)
        for tag in done:
            del running[tag]
        if running:
            time.sleep(5)
    print("=== E-full orchestrate: all jobs settled ===", flush=True)

def _main_E():
    ap = argparse.ArgumentParser(description="Track-E real-data (talent) eval: 5 learned scorers "
                                              "→ G1/G2/G3 + g_native × 10 downstream models.")
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default=OUT_DIR_DEFAULT)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    ap.add_argument("--methods", nargs="+", default=None, choices=TRACK_E_METHODS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-reuse-g12", action="store_true",
                    help="Recompute G1/G2 instead of reading available result JSONs")
    ap.add_argument("--score-only", action="store_true", help="Save scores to cache_E without downstream evaluation")
    ap.add_argument("--downstream-only", action="store_true", help="Evaluate downstream learners from cache_E without scoring")
    ap.add_argument("--max_epoch", type=int, default=50, help="Training epochs for neural learners")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--per_gpu", type=int, default=2)
    ap.add_argument("--max_retries", type=int, default=1)
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    models_use = args.models or ALL_MODELS
    methods_use = args.methods or TRACK_E_METHODS
    seeds = [args.random_seed] if args.random_seed else SEEDS

    if args.worker:
        torch.set_num_threads(args.threads)
        ds = args.datasets[0]
        task = args.task_type
        print(f"[E-worker] {ds} ({task}) seeds={seeds} methods={methods_use} "
              f"models={models_use} max_epoch={args.max_epoch} "
              f"reuse_g12={not args.no_reuse_g12} score_only={args.score_only} "
              f"downstream_only={args.downstream_only}", flush=True)
        for seed in seeds:
            rf = _dataset_json_path_E(out_dir, task, ds, seed)
            existing = None
            if args.resume and rf.exists():
                try:
                    with open(rf) as f:
                        existing = json.load(f)
                    if not args.score_only and _E_models_complete(existing, methods_use, models_use, task):
                        print(f"  {ds} s{seed} — done, SKIP", flush=True)
                        continue
                except Exception:
                    existing = None
            t0 = time.time()
            try:
                res = run_single_dataset_E(
                    ds, task, seed, "cuda:0",
                    models_filter=models_use, methods_filter=methods_use,
                    existing=existing, max_epoch=args.max_epoch, output_dir=out_dir,
                    reuse_g12=not args.no_reuse_g12,
                    score_only=args.score_only, downstream_only=args.downstream_only)
            except Exception as e:
                print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                import traceback; traceback.print_exc()
                continue
            if res is not None:
                _write_E_results(rf, res)
                print(f"  {ds} s{seed} done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
            elif args.score_only:
                print(f"  {ds} s{seed} score-only done ({time.time()-t0:.1f}s)", flush=True)
        return

    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
        if args.task_type == "all":
            pairs = ([(d, "bin") for d in DATASET_MAP["bin"]]
                     + [(d, "multi") for d in DATASET_MAP["multi"]]
                     + [(d, "reg") for d in DATASET_MAP["reg"]])
        else:
            datasets = args.datasets or DATASET_MAP[args.task_type]
            pairs = [(d, args.task_type) for d in datasets]
        orchestrate_E(pairs, out_dir, gpu_ids, methods_use, models_use, seeds, args,
                      per_gpu=args.per_gpu, max_retries=args.max_retries)
        return


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.set_num_threads(args.threads)
    if args.task_type == "all":
        tasks = ["bin", "multi", "reg"]
    else:
        tasks = [args.task_type]
    print("=" * 70, flush=True)
    print(f"E real-data (direct): tasks={tasks} datasets={'all' if not args.datasets else args.datasets} "
          f"seeds={seeds} methods={methods_use} models={models_use} device_id={args.device_id} "
          f"max_epoch={args.max_epoch} resume={args.resume} reuse_g12={not args.no_reuse_g12} "
          f"score_only={args.score_only} downstream_only={args.downstream_only}", flush=True)
    print(f"output → {out_dir}  | scores → {SCORE_CACHE_DIR_E} (v1_E, separate namespace)", flush=True)
    print("=" * 70, flush=True)
    for task in tasks:
        ds_list = args.datasets if args.datasets else DATASET_MAP[task]
        for di, ds in enumerate(ds_list):
            for seed in seeds:
                rf = _dataset_json_path_E(out_dir, task, ds, seed)
                existing = None
                if args.resume and rf.exists():
                    try:
                        with open(rf) as f:
                            existing = json.load(f)
                        if not args.score_only and _E_models_complete(existing, methods_use, models_use, task):
                            print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed} — done, SKIP", flush=True)
                            continue
                        print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed} — RESUME", flush=True)
                    except Exception:
                        existing = None
                else:
                    print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed}", flush=True)
                t0 = time.time()
                try:
                    res = run_single_dataset_E(
                        ds, task, seed, "cuda:0",
                        models_filter=models_use, methods_filter=methods_use,
                        existing=existing, max_epoch=args.max_epoch, output_dir=out_dir,
                        reuse_g12=not args.no_reuse_g12,
                        score_only=args.score_only, downstream_only=args.downstream_only)
                except Exception as e:
                    print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                    import traceback; traceback.print_exc(); continue
                if res is not None:
                    _write_E_results(rf, res)
                    print(f"  done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
                elif args.score_only:
                    print(f"  score-only done ({time.time()-t0:.1f}s)", flush=True)
                else:
                    print(f"  skipped (n_noise==0 or load err)", flush=True)


# ======================================================================

# ======================================================================

warnings.filterwarnings("ignore")

sys.path.insert(0, str(BASE_DIR))

SCORE_CACHE_DIR_CPU = RESULTS_ROOT / "screening_score_cache_cpu_complete_v3"

DATA_PATH = Path(os.environ.get("DATA_ROOT")
                  or str(mp.DATA_ROOT))



def _prepare_Xy(dataset_name, task_type, seed, data_root=None):
    data_path = Path(data_root) if data_root else DATA_PATH
    old_dp = T.DATA_PATH
    T.DATA_PATH = data_path                       # T.load_dataset uses this module global
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
        cat_indices, _cat_cols = T.identify_categorical(Xtr_df)
        p = Xtr_df.shape[1]
        n = Xtr_df.shape[0]
        print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

        X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
        y_train, _ = T.encode_y(ytr_raw, task_type)

        n_noise = max(0, TARGET_TOTAL_FEATURES - p)
        if n_noise == 0:
            print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES} (no noise to add)", flush=True)
            return None
        print(f"  +{n_noise} noise → total {p + n_noise}", flush=True)
        fk_tr, _fk_v, _fk_te = T.generate_gaussian_noise_features(
            X_train, X_val, X_test, seed, n_noise=n_noise)
        Xc_tr = np.hstack([X_train, fk_tr])
        return Xc_tr, y_train, p + n_noise, p
    finally:
        T.DATA_PATH = old_dp

def _cap_screen_sample(X_comb, y, task_type, seed=42):
    n_full = X_comb.shape[0]
    if n_full <= MAX_SCREEN_SAMPLES:
        return X_comb, y
    if task_type != "reg" and len(np.unique(y)) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SCREEN_SAMPLES,
                                     random_state=seed)
        idx, _ = next(sss.split(X_comb, y))
    else:
        idx = np.random.RandomState(seed).choice(n_full, MAX_SCREEN_SAMPLES, replace=False)
    print(f"  [screen] cap {n_full}→{len(idx)} samples for utility computation", flush=True)
    return X_comb[idx], y[idx]

def _inner_split_cpu(X, y, task_type, seed=42):
    n = X.shape[0]
    if n < 12:
        return X, X, y, y, False
    is_clf = task_type != "reg"
    strat = y if (is_clf and len(np.unique(y)) > 1
                  and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=strat)
    return Xtr, Xv, ytr, yv, True

def compute_cpu_scores(dataset_name, task_type, seed, data_root=None, methods_filter=None):
    prep = _prepare_Xy(dataset_name, task_type, seed, data_root=data_root)
    if prep is None:
        return None
    Xc_tr, y_train, P, p_orig = prep
    X_comb, y_comb = _cap_screen_sample(Xc_tr, y_train, task_type)

    Xtr_t, Xv_t, ytr_t, yv_t, has_val = _inner_split_cpu(X_comb, y_comb, task_type)
    task_full = TASK_FULL[task_type]
    mfilter = set(methods_filter) if methods_filter is not None else None

    scores, times, oom_methods = {}, {}, set()
    for grp in (CFAST_METHODS, DSHAP_METHODS):
        for m in grp:
            if mfilter is not None and m not in mfilter:
                continue                                   # skip already-cached method
            print(f"  [cpu] {m} ...", flush=True)
            t0 = time.perf_counter()
            ok = False
            try:
                if m in STAGED_METHODS:
                    sc, _ = score_matrix(m, Xc_tr, y_train, task_type,
                        "cuda:0" if m == "tabicl_shap" else "cpu", MODEL_ROOT,
                        RESULTS_ROOT / "attribution_work")
                else:
                    sc = compute_scores(Xtr_t, ytr_t, m, task=task_type,
                        validation_data=(Xv_t if has_val else None, yv_t if has_val else None))
                if sc is not None:
                    sc = np.asarray(sc, dtype=np.float64)
                    if sc.shape[0] == P and not np.isnan(sc).all():
                        scores[m] = sc
                        ok = True
            except Exception as e:
                if T.is_oom_error(e):
                    oom_methods.add(m)
                    times[m] = "OOM"
                    print(f"  [cpu] {m} OOM-recorded: {str(e)[:90]}", flush=True)
                    continue
                print(f"  [cpu] {m} err: {str(e)[:140]}", flush=True)
            times[m] = (time.perf_counter() - t0) if ok else (times.get(m))
    return scores, times, oom_methods, P

def _cpu_cache_fname(task_type, dataset_name, seed):
    return f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.npz"

def _cpu_cache_read_paths(task_type, dataset_name, seed, out_root=None):
    root = Path(out_root) if out_root else RESULTS_ROOT
    fname = _cpu_cache_fname(task_type, dataset_name, seed)
    paths = []
    v2_dir = root / "screening_score_cache_cpu_complete_v3"
    v2 = v2_dir / fname
    if v2.exists():
        paths.append(v2)
    if v2_dir.exists():
        stem = Path(fname).stem
        amends = sorted(v2_dir.glob(f"{stem}.a*.npz"),
                        key=lambda p: p.stat().st_mtime)
        paths.extend(amends)
    return paths

def _cpu_cache_write_path(task_type, dataset_name, seed, out_root=None):
    root = Path(out_root) if out_root else RESULTS_ROOT
    return root / "screening_score_cache_cpu_complete_v3" / _cpu_cache_fname(task_type, dataset_name, seed)

def save_cpu_scores(task_type, dataset_name, seed, scores, times, oom_methods, out_root=None):
    merged, m_times, m_oom = {}, {}, set()
    for p in _cpu_cache_read_paths(task_type, dataset_name, seed, out_root):
        try:
            old = np.load(p, allow_pickle=True)
            for k in old.files:
                if not k.startswith("__"):
                    merged[k] = old[k]
            if "__times__" in old.files:
                try:
                    m_times.update(dict(old["__times__"].item()))
                except Exception:
                    pass
            if "__oom__" in old.files:
                try:
                    m_oom |= set(old["__oom__"].tolist())
                except Exception:
                    pass
        except Exception:
            pass
    for m, sc in scores.items():
        arr = np.asarray(sc, dtype=np.float32)
        if arr.ndim == 1:
            merged[m] = arr
    m_times.update(times)
    m_oom |= set(oom_methods) if isinstance(oom_methods, (set, list)) else set()
    merged["__times__"] = np.array(m_times, dtype=object)
    merged["__oom__"] = np.array(sorted(m_oom), dtype=object)
    merged["__schema__"] = np.array("v1_cpu")
    # Build the .npz fully in memory, then write to a TEMP .npz file and os.replace
    # onto the target. Two NFS pitfalls, both now avoided:
    #  (a) np.savez(path) interleaved-seek zip → corrupt local-header offsets (small
    #      files). Fix: np.savez(BytesIO) → one f.write.
    #  (b) open(path,'wb') to TRUNCATE the SAME file we just np.load()'d above (for
    #      merge) raises NFS EIO [Errno 5] under concurrent load (the GPU-downstream
    #      orchestrator writes this same NAS). Fix: write a fresh temp file + fsync +
    #      os.replace (atomic rename) — never truncate the file we just read.
    # The temp name MUST end in .npz (else numpy appends ".npz" and the rename misses).
    path = _cpu_cache_write_path(task_type, dataset_name, seed, out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **merged)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npz")
    try:
        with open(tmp, "wb") as f:
            f.write(buf.getvalue())
            os.fsync(f.fileno())
        try:
            os.replace(tmp, str(path))   # atomic; replaces the read file without truncating it
        except OSError:


            amend = path.with_name(f"{path.stem}.a{os.getpid()}.npz")
            os.replace(tmp, str(amend))
            print(f"  [cpu-cache] target read-only (EIO) → wrote amend {amend.name}",
                  flush=True)
            return amend
    except Exception:
        if tmp.exists():

            rec = path.with_name(f"{path.stem}.recovery{os.getpid()}.npz")
            try:
                os.replace(tmp, rec)
            except OSError:
                pass
        raise
    print(f"  [cpu-cache] saved {len(scores)} method scores → {path.name}", flush=True)
    return path

def load_cpu_scores(task_type, dataset_name, seed, methods, out_root=None):
    scores, times, oom = {}, {}, set()
    for path in _cpu_cache_read_paths(task_type, dataset_name, seed, out_root):
        try:
            z = np.load(path, allow_pickle=True)
            for m in methods:
                if m in z.files and z[m].ndim == 1:
                    scores[m] = np.asarray(z[m], dtype=np.float64)
            if "__times__" in z.files:
                try:
                    times.update(dict(z["__times__"].item()))
                except Exception:
                    pass
            if "__oom__" in z.files:
                try:
                    oom |= set(z["__oom__"].tolist())
                except Exception:
                    pass
        except Exception as e:
            print(f"  [cpu-cache] load failed ({path.name}): {e}", flush=True)
    return scores, times, oom

def unit_done(task_type, dataset_name, seed, out_root=None):
    """Cheap pre-filter: are all 10 CPU method scores present in the cache (counting
    OOM-recorded methods as covered — a deterministic OOM re-runs identically, so
    retry is pure waste)? Used by the orchestrator to skip-done."""
    cached, _times, cached_oom = load_cpu_scores(
        task_type, dataset_name, seed, CPU_SCORER_METHODS, out_root)
    coverage = set(cached) | cached_oom
    return set(CPU_SCORER_METHODS).issubset(coverage)

def enumerate_tasks(tasks, seeds):
    """Yield (task_type, dataset_name, seed) in bin→multi→reg order."""
    tasks = tasks or ["bin", "multi", "reg"]
    seeds = seeds if seeds else SEEDS
    for task in tasks:
        for ds in DATASET_MAP[task]:
            for seed in seeds:
                yield task, ds, seed

def _main_cpu():
    """Run portable CPU scoring with the shared staged interface."""
    score_main()


def _dataset_json_path_cpu(out_dir, task_type, dataset_name, seed):
    return out_dir / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"

def _unit_done_cpu(out_dir, task_type, dataset_name, models, seeds):
    """Cheap pre-filter: is every (dataset,seed) JSON complete for all 10 CPU
    methods' G3 (no 'error', method present)? Used by the orchestrator to
    skip spawning a worker for an already-finished dataset."""
    for s in seeds:
        rf = _dataset_json_path_cpu(out_dir, task_type, dataset_name, s)
        if not rf.exists():
            return False
        try:
            existing = json.load(open(rf))
        except Exception:
            return False
        # Per-model: each model must have a non-error g3_<m> for all 10 CPU methods.
        if not _cpu_models_complete(existing, models):
            return False
    return True

def _resume_expected_cpu():
    """Non-error expectation set = the 10 CPU scorers (vs the main experiment's
    12 GPU scorers under --skip-cpu-scorers). find_missing_methods uses this so
    an already-complete CPU-downstream dataset doesn't get pointlessly re-run."""
    return list(CPU_SCORER_METHODS)

def _cpu_models_complete(existing, models):
    if existing is None:
        return False
    for model_name in models:
        if model_name not in existing or not isinstance(existing[model_name], dict):
            return False
        for m in CPU_SCORER_METHODS:
            g3 = existing[model_name].get(f"g3_{m}")
            if not isinstance(g3, dict) or "error" in g3:
                return False
    return True

def run_single_dataset_cpu(dataset_name, task_type, seed, device, models_filter=None,
                           existing=None):
    """Downstream G3 for ONE (dataset, seed) using the 6 cached CPU scores.

    Mirrors f2ves_real_eval_full.run_single_dataset's screening+downstream body,
    with the screening stage replaced by load_cpu_scores() (no embedder/GPU scoring).
    """
    models_use = models_filter or MODELS
    scr_methods = CPU_SCORER_METHODS
    missing = find_missing_methods(existing, task_type, models_use, scr_methods)
    # Early skip iff every requested model has non-error G3 for all 10 CPU
    # methods (per-model, not union — see _cpu_models_complete).
    if existing is not None and _cpu_models_complete(existing, models_use):
        print(f"  {dataset_name} s{seed} — all CPU G3 done, SKIP", flush=True)
        return existing
    if missing and existing is not None:
        print(f"  resume missing methods: {sorted(missing)}", flush=True)

    print(f"  Loading {dataset_name}...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None
    cat_indices, _cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]
    n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train, _ = T.encode_y(ytr_raw, task_type)
    y_val, _ = T.encode_y(yv_raw, task_type)
    y_test, _ = T.encode_y(yte_raw, task_type)

    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    if n_noise == 0:
        print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES} (no noise; no G3 to do)", flush=True)
        return None
    print(f"  +{n_noise} noise → total {p + n_noise}", flush=True)

    fk_tr, fk_v, fk_te = T.generate_gaussian_noise_features(X_train, X_val, X_test, seed, n_noise=n_noise)
    Xc_tr = np.hstack([X_train, fk_tr])
    Xc_v = np.hstack([X_val, fk_v])
    Xc_te = np.hstack([X_test, fk_te])

    # ── Screening: LOAD the CPU scores from the CPU branch cache (no compute) ──
    screening_scores, screen_times, oom_methods = load_cpu_scores(
        task_type, dataset_name, seed, scr_methods)
    if screening_scores:
        print(f"  [cpu-cache] loaded {len(screening_scores)}/{len(scr_methods)} "
              f"CPU scores → {len(screening_scores)} methods will get G3; "
              f"the rest screening_NA", flush=True)
    else:
        print(f"  [cpu-cache] NO cache for {dataset_name} s{seed} → all 10 "
              f"methods screening_NA (run the score stage first to populate)",
              flush=True)

    results = {"dataset": dataset_name, "task_type": task_type, "seed": seed,
               "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
               "screen_times_s": screen_times,
               "screen_oom_methods": sorted(oom_methods)}

    for model_name in models_use:
        # skip tabpfn reg if ckpt missing (parity with main experiment)
        if model_name == "tabpfn" and task_type == "reg" and not os.path.exists(TABPFN_V3_REG):
            continue
        mr = {}

        # No G1/G2 here — those are method-agnostic, already done in the main
        # experiment. We ONLY do G3 per CPU method. Resume: reuse existing.
        if existing is not None and model_name in existing:
            for key in (f"g1_clean", "g2_gaussian"):
                ev = existing[model_name].get(key)
                if isinstance(ev, dict) and "error" not in ev:
                    mr[key] = ev

        for m in scr_methods:
            # resume skip: reuse an already-good g3_<m> if present
            if existing is not None and model_name in existing:
                for key in (f"g3_{m}",):
                    ev = existing[model_name].get(key)
                    if isinstance(ev, dict) and "error" not in ev:
                        mr[key] = ev
            if m not in missing and (existing is not None) and (f"g3_{m}" in mr):
                continue

            if m in oom_methods:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_OOM"}
                continue
            if m not in screening_scores:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_NA"}
                continue

            sc = screening_scores[m]

            # G3: top (G3_TOP_K_MULTIPLIER * p) features by score — byte-identical
            # to the main experiment.
            g3_k = G3_TOP_K_MULTIPLIER * p
            top_idx = np.argsort(sc)[::-1][:g3_k]
            recall = int(np.sum(top_idx < p))
            X3_tr, X3_v, X3_te = Xc_tr[:, top_idx], Xc_v[:, top_idx], Xc_te[:, top_idx]
            cat_g3 = None
            if cat_indices:
                mp3 = {idx: i for i, idx in enumerate(top_idx)}
                cat_g3 = [mp3[idx] for idx in cat_indices if idx in mp3]
            m3 = train_and_evaluate(model_name, X3_tr, y_train, X3_v, y_val, X3_te, y_test,
                                    task_type, seed, cat_g3, device)
            m3["recall"] = recall
            m3["n_features"] = int(g3_k)
            mr[f"g3_{m}"] = m3


        results[model_name] = mr

    # Defensive additive merge (see main-arm f2ves_real_eval_full.py): a subset
    # run (models_filter != None) must not clobber downstream-model entries it
    # did not touch — otherwise _write_results_atomic overwrites the JSON with
    # only the filtered models. Untouched existing model entries are carried

    if existing is not None:
        for k, v in existing.items():
            if k not in results and isinstance(v, dict) and k in MODELS:
                results[k] = v

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return results

def orchestrate_cpu(pairs, out_dir, gpu_ids, models, seeds, args, per_gpu=1,
                   max_retries=1):
    LOGS = RESULTS_ROOT / "logs_real_eval_cpu"
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}
    tries_map = {}
    threads = max(4, min(16, 28 // per_gpu)) if per_gpu > 1 else 16
    print(f"orchestrate_cpu: {len(pairs)} datasets across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, {threads} torch threads/worker, "
          f"max_retries={max_retries})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        while pending and args.resume and _unit_done_cpu(out_dir, pending[0][1],
                                                         pending[0][0], models, seeds):
            ds, task = pending.pop(0)
            print(f"  [skip-done] {ds} ({task}) already complete", flush=True)
        free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        while free and pending:
            (ds, task), gpu = pending.pop(0), free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env.setdefault(k, str(threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            lf = open(LOGS / f"cpu_ds_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [sys.executable, "-u", str(BASE_DIR / "f2ves_real_eval_full.py"),
                   "--method", "cpu",
                   "--worker", "--task_type", task, "--datasets", ds,
                   "--device_id", "0", "--output", args.output,
                   "--threads", str(threads)]
            # NOTE: command must stay SIMPLE (no cd/find/$()) — see load_memory §7-9:
            # a $(...) command-substitution corrupts pypai→server YAML. Here it's a
            # LOCAL subprocess (no pypai) so $() wouldn't harm, but we keep it simple

            for flag in ("--resume",):
                attr = flag.lstrip("-").replace("-", "_")
                if getattr(args, attr, False):
                    cmd.append(flag)
            if args.models:
                cmd += ["--models", *args.models]
            tag = f"{gpu}:{task}:{ds}"
            running[tag] = (subprocess.Popen(cmd, env=env, stdout=lf,
                                             stderr=subprocess.STDOUT),
                            ds, task, lf, gpu)
            attempts = tries_map.get((task, ds), 0)
            print(f"  [launch] {ds} ({task}) → GPU {gpu} "
                  f"(load {gpu_load(gpu)}/{per_gpu}"
                  f"{f', attempt {attempts+1}' if attempts else ''})", flush=True)
            free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        done = []
        for tag, (proc, ds, task, lf, gpu) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            if rc == 0:
                tries_map.pop((task, ds), None)
                print(f"  [done] {ds} ({task}) gpu{gpu} rc=0", flush=True)
            else:
                attempts = tries_map.get((task, ds), 0)
                if attempts < max_retries:
                    tries_map[(task, ds)] = attempts + 1
                    print(f"  [crash → retry] {ds} ({task}) gpu{gpu} rc={rc} "
                          f"(attempt {attempts+1} of {max_retries+1})", flush=True)
                    pending.append((ds, task))
                else:
                    print(f"  [CRASH] {ds} ({task}) gpu{gpu} rc={rc} after "
                          f"{max_retries} retries — left for next --resume", flush=True)
            done.append(tag)
        for tag in done:
            del running[tag]
        if running:
            time.sleep(5)
    print("=== orchestrate_cpu: all jobs settled ===", flush=True)

def _main_cpudn():
    ap = argparse.ArgumentParser(description="CPU-scores → G3 downstream, 4-GPU orchestrator.")
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default=OUT_REL)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, choices=MODELS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpus", type=str, default=None,
                    help="comma list of physical GPU ids; absent = single-process worker/direct run")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    # Force-FAIL drill gate (parity with main experiment): only under --worker,
    # so it can never fire for the orchestrator/direct invocation.
    if args.worker and os.environ.get("EVAL_FORCE_FAIL"):
        print(f"[FORCE-FAIL] worker {args.task_type} {args.datasets} exiting 1", flush=True)
        sys.exit(1)

    if args.gpus and not args.worker:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        out_dir = RESULTS_ROOT / args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = [args.random_seed] if args.random_seed else SEEDS_LIST
        if args.task_type == "all":
            pairs = ([(d, "bin") for d in DATASET_MAP["bin"]]
                    + [(d, "multi") for d in DATASET_MAP["multi"]]
                    + [(d, "reg") for d in DATASET_MAP["reg"]])
        else:
            datasets = args.datasets or DATASET_MAP[args.task_type]
            pairs = [(d, args.task_type) for d in datasets]
        print(f"orchestrate_cpu: {len(pairs)} datasets, GPUs {gpu_ids}, seeds {seeds}", flush=True)
        orchestrate_cpu(pairs, out_dir, gpu_ids, args.models or MODELS, seeds, args,
                        per_gpu=args.per_gpu, max_retries=args.max_retries)
        return

    if args.task_type == "all":
        print("ERR: --task_type all requires --gpus (orchestrator mode).", flush=True)
        sys.exit(2)

    # ── single-process worker / direct run ──
    import torch
    torch.set_num_threads(args.threads)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    # Align screening_methods_torch module-global DEVICE (parity with main exp:
    # emb_* scorers use it; even though WE don't run emb_*, train_and_evaluate's
    # tabpfn/tabicl paths use `device`, and importing torch sets a default).
    try:
        from src import screening_methods_torch as _smt
        _smt.set_device(args.device_id)
    except Exception:
        pass

    datasets = args.datasets or DATASET_MAP[args.task_type]
    seeds = [args.random_seed] if args.random_seed else SEEDS_LIST
    out_dir = RESULTS_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    task_label = TASK_DIR_MAP[args.task_type]

    print("======================================================================", flush=True)
    print(f"CPU-downstream: task={task_label} datasets={len(datasets)} "
          f"seeds={seeds} models={args.models or MODELS} device={device} "
          f"resume={args.resume}", flush=True)
    print("======================================================================", flush=True)

    for di, ds in enumerate(datasets):
        for seed in seeds:
            rf = _dataset_json_path_cpu(out_dir, args.task_type, ds, seed)
            existing = None
            if args.resume and rf.exists():
                try:
                    existing = json.load(open(rf))
                    if _cpu_models_complete(existing, args.models or MODELS):
                        print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — done, SKIP", flush=True)
                        continue
                    print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — RESUME", flush=True)
                except Exception:
                    existing = None
            else:
                print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed}", flush=True)
            t0 = time.time()
            try:
                res = run_single_dataset_cpu(ds, args.task_type, seed, device,
                                             models_filter=args.models, existing=existing)
            except Exception as e:
                print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                if os.environ.get("EVAL_FULL_TRACE"):
                    import traceback as _tb
                    _tb.print_exc()
                continue
            if res is not None:
                _write_results_atomic(rf, res)
                print(f"  done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
            else:
                print(f"  SKIPPED", flush=True)
    print("\n=== all done ===", flush=True)


# ======================================================================

# ======================================================================

warnings.filterwarnings("ignore")

sys.path.insert(0, str(BASE_DIR))

OUT_REL = "results_real_eval_full_dl3_cpu"

DL3_CPU_LOGS = mp.OUTPUT_ROOT / "logs" / "dl3_cpu_logs"

NEW_MODELS = NEW_MODELS

SEEDS_LIST = SEEDS

def _dataset_json_path_cpu_dl3(out_dir, task_type, dataset_name, seed):
    return out_dir / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"

def _cpu_models_complete_dl3(existing, models):
    if existing is None:
        return False
    for model_name in models:
        if model_name not in existing or not isinstance(existing[model_name], dict):
            return False
        for m in CPU_SCORER_METHODS:
            g3 = existing[model_name].get(f"g3_{m}")
            if not isinstance(g3, dict) or "error" in g3:
                return False
    return True

def _unit_done_cpu_dl3(out_dir, task_type, dataset_name, models, seeds):
    for s in seeds:
        rf = _dataset_json_path_cpu_dl3(out_dir, task_type, dataset_name, s)
        if not rf.exists():
            return False
        try:
            existing = json.load(open(rf))
        except Exception:
            return False
        if not _cpu_models_complete_dl3(existing, models):
            return False
    return True

def run_single_dataset_cpu_dl(dataset_name, task_type, seed, device,
                              models_filter=None, existing=None, max_epoch=50):
    models_use = models_filter or NEW_MODELS
    scr_methods = CPU_SCORER_METHODS

    if existing is not None and _cpu_models_complete_dl3(existing, models_use):
        print(f"  {dataset_name} s{seed} — all CPU G3 done (dl3), SKIP", flush=True)
        return existing

    print(f"  Loading {dataset_name} (dl3-cpu)...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None
    cat_indices, _cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]
    n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train, _ = T.encode_y(ytr_raw, task_type)
    y_val, _ = T.encode_y(yv_raw, task_type)
    y_test, _ = T.encode_y(yte_raw, task_type)

    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    if n_noise == 0:
        print(f"  [SKIP] p={p} >= {TARGET_TOTAL_FEATURES}", flush=True)
        return None
    fk_tr, fk_v, fk_te = T.generate_gaussian_noise_features(X_train, X_val, X_test, seed, n_noise=n_noise)
    Xc_tr = np.hstack([X_train, fk_tr])
    Xc_v = np.hstack([X_val, fk_v])
    Xc_te = np.hstack([X_test, fk_te])

    screening_scores, screen_times, oom_methods = load_cpu_scores(
        task_type, dataset_name, seed, scr_methods)
    if screening_scores:
        print(f"  [cpu-cache] {len(screening_scores)}/{len(scr_methods)} CPU scores", flush=True)
    else:
        print(f"  [cpu-cache] NO cache → all 10 screening_NA", flush=True)

    results = {"dataset": dataset_name, "task_type": task_type, "seed": seed,
               "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
               "total_features": p + n_noise, "dl3_models": True,
               "screen_times_s": screen_times, "screen_oom_methods": sorted(oom_methods)}

    for model_name in models_use:
        mr = {}

        if existing is not None and model_name in existing:
            for key in ("g1_clean", "g2_gaussian"):
                ev = existing[model_name].get(key)
                if isinstance(ev, dict) and "error" not in ev:
                    mr[key] = ev

        for m in scr_methods:
            if existing is not None and model_name in existing:
                for key in (f"g3_{m}",):
                    ev = existing[model_name].get(key)
                    if isinstance(ev, dict) and "error" not in ev:
                        mr[key] = ev
            if m in oom_methods and m not in screening_scores:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_OOM"}
                continue
            if m not in screening_scores:
                if f"g3_{m}" not in mr:
                    mr[f"g3_{m}"] = {"error": "screening_NA"}
                continue
            if f"g3_{m}" in mr:
                continue

            sc = screening_scores[m]
            # G3: top 3p
            g3_k = G3_TOP_K_MULTIPLIER * p
            top_idx = np.argsort(sc)[::-1][:g3_k]
            recall = int(np.sum(top_idx < p))
            X3_tr, X3_v, X3_te = Xc_tr[:, top_idx], Xc_v[:, top_idx], Xc_te[:, top_idx]
            if f"g3_{m}" not in mr:
                m3 = train_and_evaluate_dl(model_name, X3_tr, y_train, X3_v, y_val, X3_te, y_test,
                                        task_type, seed, cat_indices, device, max_epoch)
                m3["recall"] = recall
                m3["n_features"] = int(g3_k)
                mr[f"g3_{m}"] = m3


        results[model_name] = mr

    if existing is not None:
        for k, v in existing.items():
            if k not in results and isinstance(v, dict) and k in NEW_MODELS:
                results[k] = v
    return results

def orchestrate_cpu_dl(pairs, out_dir, gpu_ids, models, seeds, args, per_gpu=1,
                       max_retries=1):
    DL3_CPU_LOGS.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}
    tries_map = {}
    threads = args.threads
    print(f"[dl3-cpu] Orchestrate: {len(pairs)} (ds,task) across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, {threads} threads, max_retries={max_retries}, "
          f"max_epoch={args.max_epoch}, models={models})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        while pending and args.resume and _unit_done_cpu_dl3(out_dir, pending[0][1],
                                                         pending[0][0], models, seeds):
            ds, task = pending.pop(0)
            print(f"  [skip-done] {ds} ({task}) already complete", flush=True)
        free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        while free and pending:
            (ds, task), gpu = pending.pop(0), free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = str(BASE_DIR / "TALENT") + ":" + env.get("PYTHONPATH", "")
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS"):
                env.setdefault(k, str(threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            lf = open(DL3_CPU_LOGS / f"dl3cpu_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [sys.executable, "-u",
                   str(BASE_DIR / "f2ves_real_eval_full.py"),
                   "--method", "cpu",
                   "--worker", "--task_type", task, "--datasets", ds,
                   "--device_id", "0", "--output", args.output,
                   "--threads", str(threads), "--max_epoch", str(args.max_epoch),
                   "--resume"]
            if args.models:
                cmd += ["--models", *args.models]
            tag = f"{gpu}:{task}:{ds}"
            running[tag] = (subprocess.Popen(cmd, env=env, stdout=lf,
                                             stderr=subprocess.STDOUT),
                            ds, task, lf, gpu)
            print(f"  [launch] {ds} ({task}) → GPU {gpu} (load {gpu_load(gpu)}/{per_gpu})",
                  flush=True)
            free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        done = []
        for tag, (proc, ds, task, lf, gpu) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            if rc == 0:
                tries_map.pop((task, ds), None)
                print(f"  [done] {ds} ({task}) gpu{gpu} rc=0", flush=True)
            else:
                attempts = tries_map.get((task, ds), 0)
                if attempts < max_retries:
                    tries_map[(task, ds)] = attempts + 1
                    print(f"  [crash→retry] {ds} ({task}) gpu{gpu} rc={rc} "
                          f"(attempt {attempts+1}/{max_retries+1})", flush=True)
                    pending.append((ds, task))
                else:
                    print(f"  [CRASH] {ds} ({task}) gpu{gpu} rc={rc} after "
                          f"{max_retries} retries", flush=True)
            done.append(tag)
        for tag in done:
            del running[tag]
        if running:
            time.sleep(5)
    print("=== dl3-cpu orchestrate: all jobs settled ===", flush=True)

def _main_cpudl3():
    ap = argparse.ArgumentParser(description="dl3: 6-CPU-scores → G3 × 3 new DL models, 4-GPU orchestrator.")
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default=OUT_REL)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, choices=NEW_MODELS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_epoch", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--max_retries", type=int, default=1)
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    models_use = args.models or NEW_MODELS
    seeds = [args.random_seed] if args.random_seed else SEEDS_LIST

    if args.worker:
        torch.set_num_threads(args.threads)
        ds = args.datasets[0]
        task = args.task_type
        print(f"[dl3-cpu-worker] {ds} ({task}) seeds={seeds} models={models_use} "
              f"max_epoch={args.max_epoch}", flush=True)
        for seed in seeds:
            rf = _dataset_json_path_cpu_dl3(out_dir, task, ds, seed)
            existing = None
            if args.resume and rf.exists():
                try:
                    existing = json.load(open(rf))
                    if _cpu_models_complete_dl3(existing, models_use):
                        print(f"  {ds} s{seed} — done, SKIP", flush=True); continue
                except Exception:
                    existing = None
            t0 = time.time()
            try:
                res = run_single_dataset_cpu_dl(ds, task, seed, "cuda:0",
                    models_filter=models_use, existing=existing, max_epoch=args.max_epoch)
            except Exception as e:
                print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                import traceback; traceback.print_exc(); continue
            if res is not None:
                _write_results_atomic(rf, res)
                print(f"  {ds} s{seed} done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
        return

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        if args.task_type == "all":
            pairs = ([(d, "bin") for d in DATASET_MAP["bin"]]
                     + [(d, "multi") for d in DATASET_MAP["multi"]]
                     + [(d, "reg") for d in DATASET_MAP["reg"]])
        else:
            datasets = args.datasets or DATASET_MAP[args.task_type]
            pairs = [(d, args.task_type) for d in datasets]
        orchestrate_cpu_dl(pairs, out_dir, gpu_ids, models_use, seeds, args,
                           per_gpu=args.per_gpu, max_retries=args.max_retries)
        return


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.set_num_threads(args.threads)
    if args.task_type == "all":
        tasks = ["bin", "multi", "reg"]
    else:
        tasks = [args.task_type]
    print("=" * 70, flush=True)
    print(f"dl3-cpu (direct): tasks={tasks} models={models_use} max_epoch={args.max_epoch} "
          f"resume={args.resume}", flush=True)
    print("=" * 70, flush=True)
    for task in tasks:
        ds_list = args.datasets if args.datasets else DATASET_MAP[task]
        for di, ds in enumerate(ds_list):
            for seed in seeds:
                rf = _dataset_json_path_cpu_dl3(out_dir, task, ds, seed)
                existing = None
                if args.resume and rf.exists():
                    try:
                        existing = json.load(open(rf))
                        if _cpu_models_complete_dl3(existing, models_use):
                            print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed} — done, SKIP", flush=True); continue
                    except Exception:
                        existing = None
                print(f"\n[{di+1}/{len(ds_list)}] {ds} s{seed}", flush=True)
                t0 = time.time()
                try:
                    res = run_single_dataset_cpu_dl(ds, task, seed, "cuda:0",
                        models_filter=models_use, existing=existing, max_epoch=args.max_epoch)
                except Exception as e:
                    print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                    import traceback; traceback.print_exc(); continue
                if res is not None:
                    _write_results_atomic(rf, res)
                    print(f"  done ({time.time()-t0:.1f}s) → {rf.name}", flush=True)
                else:
                    print(f"  skipped", flush=True)


# ======================================================================

# ======================================================================


def _peek_models(argv):
    """Return the list after --models (predictor), or None if absent."""
    try: i=argv.index('--models')
    except ValueError: return None
    out=[]
    for w in argv[i+1:]:
        if w.startswith('--'): break
        out.append(w)
    return out

def _pick_arm(method, preds):
    """(method, predictor) -> list of arm names to run in order.
    NE_MODELS=dl3 set, MODELS=main7 set. Pure main7 / pure dl3 / E supported;
    mixed main+dl3 rejected (pick one set)."""
    dl=set(NEW_MODELS); main=set(MODELS)
    if method=='E':
        return ('E',)
    if preds is None:
        return ('full_cpu',) if method=='cpu' else ('full',)
    p=set(preds)
    if p<=dl:  return ('dl3',) if method=='full' else ('cpudl3',)
    if p<=main: return ('full',) if method=='full' else ('cpudn',)
    if p<=(main|dl):
        print("ERR: mixed main+dl3 predictor not supported (pick one set); got %s"%preds, flush=True)
        sys.exit(2)
    print("ERR: unknown predictor(s) %s (valid: %s / %s)"%(preds, MODELS, NEW_MODELS), flush=True)
    sys.exit(2)

def main():
    import argparse as _ap
    pre=_ap.ArgumentParser(add_help=False)
    pre.add_argument('--method', default='full', choices=['full','cpu','E'])
    a, rest = pre.parse_known_args()

    rest=[('--models' if w=='--predictor' else w) for w in rest]
    preds=_peek_models(rest)
    _sys.argv=[_sys.argv[0]]+rest   # strip --method(+val via parse_known); keep rest (arm parses its own)
    arms=_pick_arm(a.method, preds)
    print('='*70)
    print('unified dispatcher: method=%s predictor=%s -> arm(s)=%s'%(a.method,preds,arms), flush=True)
    print('='*70, flush=True)
    dispatch={'full':_main_full,'dl3':_main_dl3,'full_cpu':_main_cpu,
              'cpudn':_main_cpudn,'cpudl3':_main_cpudl3,'E':_main_E}
    for arm in arms:
        if arm=='full_cpu':
            # full_cpu (NAS-offload) uses --tasks/--seeds vocabulary, not --task_type/--random_seed.

            av=[]; i=0; a=_sys.argv
            while i<len(a):
                if a[i]=='--task_type': av+=['--tasks']; i+=1
                elif a[i]=='--random_seed': av+=['--seeds']; i+=1
                else: av.append(a[i]); i+=1
            _sys.argv=av
        dispatch[arm]()


    if '--worker' in _sys.argv:
        os._exit(0)

if __name__=='__main__':
    main()
