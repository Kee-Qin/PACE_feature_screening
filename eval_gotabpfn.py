from src import model_path as mp
import sys as _sys
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

# ── Inject tabpfn 6.3.x deps BEFORE any tabpfn import ──
_DEPS = str(mp.GOTABPFN_DEPS)
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

import numpy as np
import pandas as pd
import torch

# ── version gate: GoTabPFN REQUIRES tabpfn 6.3.x ──
try:
    import tabpfn as _tpf
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    _tpf_ver = _tpf.__version__
    _tpf_major_minor = tuple(int(x) for x in _tpf_ver.split(".")[:2])
    if _tpf_major_minor != (6, 3):
        print(f"FATAL: GoTabPFN requires tabpfn==6.3.x, found {_tpf_ver}. "
              f"Activate venv_gotabpfn first.", flush=True)
        _sys.exit(1)
except ImportError:
    print("FATAL: tabpfn not installed. Activate venv_gotabpfn (tabpfn==6.3.1).", flush=True)
    _sys.exit(1)

try:
    from kmeans_gpu import KMeans as KMeansGPU
    _HAVE_KMEANS_GPU = True
except ImportError:
    _HAVE_KMEANS_GPU = False

# ── GoTabPFN imports (sys.path injection) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_GOTABPFN_ROOT = str(mp.GOTABPFN_CODE_ROOT)
if _GOTABPFN_ROOT not in sys.path:
    sys.path.insert(0, _GOTABPFN_ROOT)

# ── GoTabPFN imports (GO-LR, NSC-pSP only — TabPFN head uses direct API) ──
from gotabpfn import GraphFeatureOrdering, pidf_segpca

# ── tabpfn 2.5 local checkpoints (avoid HuggingFace download behind corporate proxy) ──
TABPFN_V25_CLF = str(mp.checkpoint('tabpfn_v25_clf'))
TABPFN_V25_REG = str(mp.checkpoint('tabpfn_v25_reg'))

# ── TALENT data loading imports (path injection so worker subprocess finds them) ──
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import f2ves_real_gaussian_large_experiments as T

T.DATA_PATH = Path(os.environ.get("TALENT_DATA_PATH") or mp.DATA_ROOT)


def load_talent_dataset(task_type, dataset_name):
    legacy_path = T.DATA_PATH / TASK_DIR_MAP[task_type] / f"{dataset_name}.pkl"
    if legacy_path.exists():
        return T.load_dataset(task_type, dataset_name)

    dataset_dir = T.DATA_PATH / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"neither legacy pickle nor TALENT directory found for {dataset_name}"
        )

    def load_split(split):
        blocks = []
        numeric_path = dataset_dir / f"N_{split}.npy"
        categorical_path = dataset_dir / f"C_{split}.npy"
        if numeric_path.exists():
            numeric = np.asarray(
                np.load(numeric_path, allow_pickle=True), dtype=np.float32
            )
            if numeric.ndim == 1:
                numeric = numeric.reshape(-1, 1)
            blocks.append(
                pd.DataFrame(
                    numeric,
                    columns=[f"num_{j}" for j in range(numeric.shape[1])],
                )
            )
        if categorical_path.exists():
            categorical = np.load(categorical_path, allow_pickle=True)
            if categorical.ndim == 1:
                categorical = categorical.reshape(-1, 1)
            blocks.append(
                pd.DataFrame(
                    categorical,
                    columns=[f"cat_{j}" for j in range(categorical.shape[1])],
                    dtype="object",
                )
            )
        if not blocks:
            raise FileNotFoundError(f"no N/C arrays found in {dataset_dir}")
        features = pd.concat(blocks, axis=1)
        target = np.load(dataset_dir / f"y_{split}.npy", allow_pickle=True).reshape(-1)
        if len(features) != len(target):
            raise ValueError(
                f"{dataset_name}/{split}: {len(features)} rows but {len(target)} labels"
            )
        return features, target

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ======================================================================
# Constants (mirror f2ves_real_eval_full.py)
# ======================================================================

warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_TELEMETRY", "0")

BASE_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT")
                    or mp.OUTPUT_ROOT)

OUT_REL = "results_real_eval_gotabpfn"

DATASET_MAP = {"bin": T.BINCLASS, "multi": T.MULTICLASS, "reg": T.REGRESSION}

TASK_DIR_MAP = {"bin": "binclass", "multi": "multiclass", "reg": "regression"}

SEEDS = [42, 123, 456]

TASK_FULL = {"bin": "classification", "multi": "classification", "reg": "regression"}

MODEL_NAME = "gotabpfn"

TARGET_TOTAL_FEATURES = 2000

# ── Default hyperparameters (paper defaults; no per-dataset tuning) ──
GO_METRIC = "correlation"
GO_NUM_CLUSTERS = 7
GO_REFINE = True
GO_REFINE_PASSES = 1
GO_DIRECTION_SELECT = True

NSC_SEGMENTATION = "equal_mass"
NSC_M_RULE = "idf"
NSC_TAU = 0.99
NSC_GAMMA = 2.0
NSC_BETA = 0.5
NSC_M_MIN = 32
NSC_M_MAX = 512
NSC_L_MIN = 12

# ======================================================================
# Metrics
# ======================================================================

from sklearn.metrics import roc_auc_score, mean_squared_error


def compute_metrics(y_true, y_pred, y_proba, task_type):
    m = {}
    try:
        if task_type == "bin":
            m["auc"] = float(roc_auc_score(y_true, y_proba))
        elif task_type == "multi":
            if y_proba is not None and getattr(y_proba, "ndim", 0) == 2 and y_proba.shape[1] > 1:
                m["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                              average="macro"))
        else:
            m["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception as e:
        m["metric_error"] = str(e)[:200]
    return m


def is_oom_error(e):
    """Check if an exception is an out-of-memory error."""
    msg = str(e).lower()
    return any(kw in msg for kw in ("out of memory", "oom", "cuda out of memory",
                                     "memory", "cannot allocate"))


def _has_primary_metric(metrics, task_type):
    """Return whether a stage result contains the task's required metric."""
    if not isinstance(metrics, dict) or "error" in metrics:
        return False
    return ("rmse" in metrics) if task_type == "reg" else ("auc" in metrics)


def evaluate_configuration(X_train, y_train, X_test, y_test, task_type, seed,
                           device, stage_label):
    """Run one complete GoTabPFN configuration and return serializable metrics."""
    print(f"  {stage_label}: GO-LR → NSC-pSP → TabPFN-2.5 "
          f"(device={device}, p={X_train.shape[1]})...", flush=True)
    t0 = time.perf_counter()
    try:
        result = run_gotabpfn(
            X_train, y_train, X_test, task_type, seed, device=device
        )
    except Exception as e:
        print(f"  [ERR] {stage_label}: {str(e)[:200]}", flush=True)
        traceback.print_exc()
        return {"error": str(e)[:200]}
    elapsed = time.perf_counter() - t0

    if "error" in result:
        print(f"  [FAIL] {stage_label}: {result['error'][:120]}", flush=True)
        return result

    metrics = compute_metrics(y_test, result["pred"], result["proba"], task_type)
    metrics["n_tokens"] = result["n_tokens"]
    metrics["elapsed_stage1_s"] = result["elapsed_stage1_s"]
    metrics["elapsed_stage2_s"] = result["elapsed_stage2_s"]
    metrics["elapsed_stage3_s"] = result["elapsed_stage3_s"]
    metrics["elapsed_total_s"] = round(elapsed, 2)
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))
    metrics["n_features"] = int(X_train.shape[1])
    metrics["go_metric"] = GO_METRIC
    metrics["go_num_clusters"] = GO_NUM_CLUSTERS
    metrics["nsc_segmentation"] = NSC_SEGMENTATION

    metric_text = (
        f"auc={metrics['auc']:.4f}" if "auc" in metrics
        else f"rmse={metrics.get('rmse', float('nan')):.4f}"
    )
    print(f"  {stage_label} done ({elapsed:.1f}s) "
          f"n_tokens={result['n_tokens']} {metric_text}", flush=True)
    return metrics


# ======================================================================
# GoTabPFN pipeline (low-level, numpy-based)
# ======================================================================


def _compute_deltas_adjacent_corr(X_np, Pi_star, eps=1e-12):
    """delta[t] = 1 - |corr(feature_{Pi[t]}, feature_{Pi[t+1]})|."""
    X = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
    perm = torch.tensor(Pi_star, dtype=torch.long)
    Xp = X[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std
    corr = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    return (1.0 - corr.abs()).cpu()


def run_gotabpfn(X_train, y_train, X_test, task_type, seed, device="cuda"):
    """Full GoTabPFN pipeline on numpy arrays — GO-LR → NSC-pSP → TabPFN-2.5.

    Args:
        X_train: (n_train, D) float32 numpy array
        y_train: (n_train,) int (cls) or float32 (reg) numpy array
        X_test:  (n_test, D) float32 numpy array
        task_type: "bin" | "multi" | "reg" (short codes, matching TALENT)
        seed: random seed
        device: "cuda" or "cpu"

    Returns:
        dict with keys: "pred", "proba" (None for reg), "n_tokens", "ordering",
        "elapsed_stage1_s", "elapsed_stage2_s", "elapsed_stage3_s"
    """
    from sklearn.preprocessing import StandardScaler

    D = X_train.shape[1]
    n_classes = len(np.unique(y_train)) if task_type != "reg" else 1

    # ── NaN guard: GO-LR/NSC (corr matrix, Gram eigvalsh) cannot handle NaN.
    # Median-impute before anything else. No-op on NaN-free data. ──
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

    # ── Preprocessing: StandardScaler (required by GO-LR) ──
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train.astype(np.float64)).astype(np.float32)
    X_te_s = scaler.transform(X_test.astype(np.float64)).astype(np.float32)

    # ── Stage 1: GO-LR ──
    t0 = time.perf_counter()
    go = GraphFeatureOrdering(
        num_clusters=GO_NUM_CLUSTERS,
        metric=GO_METRIC,
        refine=GO_REFINE,
        direction_select=GO_DIRECTION_SELECT,
        refine_passes=GO_REFINE_PASSES,
    )
    try:
        use_cpu = not _HAVE_KMEANS_GPU
        Pi_star, _, _, _ = go.fit(X_tr_s, seed=seed, deterministic=True,
                                  use_cpu_kmeans=use_cpu)
    except RuntimeError as e:
        if is_oom_error(e):
            return {"error": "OOM", "detail": f"GO-LR OOM: {str(e)[:200]}"}
        # fallback to CPU KMeans
        try:
            Pi_star, _, _, _ = go.fit(X_tr_s, seed=seed, deterministic=True,
                                      use_cpu_kmeans=True)
        except Exception as e2:
            return {"error": f"GO-LR failed: {str(e2)[:200]}"}
    except Exception as e:
        return {"error": f"GO-LR failed: {str(e)[:200]}"}
    t1 = time.perf_counter()

    # ── Stage 2: NSC-pSP ──
    deltas = None if NSC_SEGMENTATION == "uniform" else _compute_deltas_adjacent_corr(X_tr_s, Pi_star)

    nsc = pidf_segpca(
        segmentation=NSC_SEGMENTATION,
        l_min=NSC_L_MIN,
        m_rule=NSC_M_RULE,
        gamma=NSC_GAMMA,
        beta=NSC_BETA,
        tau=NSC_TAU,
        M_min=NSC_M_MIN,
        M_max=NSC_M_MAX,
        assume_standardized=True,  # we already StandardScaled above
        device="cpu",
    )
    X_tr_t = torch.from_numpy(X_tr_s)
    try:
        nsc.configure(Pi_star=Pi_star, X_train=X_tr_t, tau=NSC_TAU, deltas=deltas)
    except Exception as e:
        return {"error": f"NSC configure failed: {str(e)[:200]}"}

    try:
        Z_train = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
        Z_test = nsc.compress(torch.from_numpy(X_te_s), mode="flatten").cpu().numpy()
    except Exception as e:
        if is_oom_error(e):
            return {"error": "OOM", "detail": f"NSC compress OOM: {str(e)[:200]}"}
        return {"error": f"NSC compress failed: {str(e)[:200]}"}
    n_tokens = Z_train.shape[1]
    t2 = time.perf_counter()

    # ── Stage 3: TabPFN-2.5 (direct classifier/regressor with local ckpt) ──
    try:
        if task_type == "reg":
            if not os.path.exists(TABPFN_V25_REG):
                return {"error": f"v2.5 regressor ckpt not found: {TABPFN_V25_REG}"}
            clf = TabPFNRegressor(model_path=TABPFN_V25_REG,
                                  device=device, random_state=seed, n_estimators=4,
                                  ignore_pretraining_limits=True)
            clf.fit(Z_train, y_train.astype(np.float32))
            pred = np.asarray(clf.predict(Z_test)).reshape(-1)
            proba = None
        else:
            if not os.path.exists(TABPFN_V25_CLF):
                return {"error": f"v2.5 classifier ckpt not found: {TABPFN_V25_CLF}"}
            clf = TabPFNClassifier(model_path=TABPFN_V25_CLF,
                                   device=device, random_state=seed, n_estimators=4,
                                   ignore_pretraining_limits=True)
            clf.fit(Z_train, y_train.astype(np.int64))
            proba_all = np.asarray(clf.predict_proba(Z_test))
            if task_type == "bin":
                proba = proba_all[:, 1] if proba_all.shape[1] > 1 else proba_all.reshape(-1)
                pred = (proba >= 0.5).astype(int)
            else:
                proba = proba_all
                pred = np.argmax(proba_all, axis=1)
    except Exception as e:
        if is_oom_error(e):
            return {"error": "OOM", "detail": f"TabPFN predict OOM: {str(e)[:200]}"}
        return {"error": f"TabPFN failed: {str(e)[:200]}"}
    t3 = time.perf_counter()

    return {
        "pred": pred,
        "proba": proba,
        "n_tokens": n_tokens,
        "ordering": list(map(int, Pi_star)),
        "elapsed_stage1_s": round(t1 - t0, 2),
        "elapsed_stage2_s": round(t2 - t1, 2),
        "elapsed_stage3_s": round(t3 - t2, 2),
    }


# ======================================================================
# Single-dataset evaluation
# ======================================================================


def run_single_dataset(dataset_name, task_type, seed, device, existing=None,
                       run_g2=False, g2_only=False):
    if existing is not None:
        existing_metrics = existing.get(MODEL_NAME)
        g1_done = _has_primary_metric(existing_metrics, task_type)
        existing_g2 = (
            existing_metrics.get("g2_gaussian")
            if isinstance(existing_metrics, dict)
            else None
        )
        g2_done = _has_primary_metric(existing_g2, task_type)
        unit_done = g2_done if g2_only else (g1_done and (not run_g2 or g2_done))
        if unit_done:
            print(f"  {dataset_name} s{seed} — done, SKIP", flush=True)
            return existing

    print(f"  Loading {dataset_name}...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = load_talent_dataset(
            task_type, dataset_name
        )
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None

    cat_indices, cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]
    n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)}", flush=True)

    # ── Encode data ──
    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train_enc, _ = T.encode_y(ytr_raw, task_type)
    y_val_enc, _ = T.encode_y(yv_raw, task_type)
    y_test_enc, _ = T.encode_y(yte_raw, task_type)

    # Merge train+val (GoTabPFN needs as much training data as possible)
    if X_val is not None and len(X_val) > 0:
        X_tr_all = np.vstack([X_train, X_val])
        y_tr_all = np.hstack([y_train_enc, y_val_enc])
    else:
        X_tr_all = X_train
        y_tr_all = y_train_enc

    # ── Degenerate-class guard ──
    if task_type != "reg":
        if len(np.unique(y_tr_all)) < 2:
            return {
                "dataset": dataset_name, "task_type": task_type, "seed": seed,
                "n": n, "p": p, "n_cat": len(cat_indices),
                MODEL_NAME: {"error": "single_class_in_fold"},
            }

    existing_metrics = existing.get(MODEL_NAME) if isinstance(existing, dict) else None
    if g2_only:
        metrics = dict(existing_metrics) if isinstance(existing_metrics, dict) else {}
        print("  G1 clean: skipped (--g2-only)", flush=True)
    elif _has_primary_metric(existing_metrics, task_type):
        metrics = dict(existing_metrics)
        print("  G1 clean: reuse existing result", flush=True)
    else:
        metrics = evaluate_configuration(
            X_tr_all, y_tr_all, X_test, y_test_enc,
            task_type, seed, device, "G1 clean"
        )

    n_noise = max(0, TARGET_TOTAL_FEATURES - X_train.shape[1])
    if run_g2:
        existing_g2 = metrics.get("g2_gaussian") if isinstance(metrics, dict) else None
        if _has_primary_metric(existing_g2, task_type):
            print("  G2 Gaussian: reuse existing result", flush=True)
        else:
            fake_train, fake_val, fake_test = T.generate_gaussian_noise_features(
                X_train, X_val, X_test, seed, n_noise=n_noise
            )
            Xc_train = np.hstack([X_train, fake_train]) if n_noise else X_train
            Xc_val = np.hstack([X_val, fake_val]) if n_noise else X_val
            Xc_test = np.hstack([X_test, fake_test]) if n_noise else X_test
            Xc_tr_all = (
                np.vstack([Xc_train, Xc_val])
                if Xc_val is not None and len(Xc_val) > 0
                else Xc_train
            )
            g2_metrics = evaluate_configuration(
                Xc_tr_all, y_tr_all, Xc_test, y_test_enc,
                task_type, seed, device, "G2 Gaussian"
            )
            metrics["g2_gaussian"] = g2_metrics
        metrics["g2_n_noise"] = int(n_noise)
        metrics["g2_total_features"] = int(X_train.shape[1] + n_noise)

    res = {
        "dataset": dataset_name, "task_type": task_type, "seed": seed,
        "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": int(n_noise),
        MODEL_NAME: metrics,
    }

    # Defensive merge: carry over existing model entries
    if existing is not None:
        for k, v in existing.items():
            if k not in res and k != MODEL_NAME:
                res[k] = v

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return res


# ======================================================================
# I/O helpers
# ======================================================================


def _dataset_json_path(out_dir, task_type, dataset_name, seed):
    return out_dir / f"{TASK_DIR_MAP[task_type]}_{dataset_name}_seed{seed}.json"


def _write_results_atomic(rf, res):
    """NFS-safe atomic JSON write (no fsync — pagecache async)."""
    Path(rf).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(rf) + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        os.replace(tmp, rf)
    except Exception:
        try:
            json.dump(res, open(rf, "w"), indent=2, default=str)
        except Exception:
            pass


def _unit_done(out_dir, task_type, dataset_name, seeds, require_g2=False,
               g2_only=False):
    """Cheap pre-filter: are all (dataset, seed) JSONs complete?"""
    for s in seeds:
        rf = _dataset_json_path(out_dir, task_type, dataset_name, s)
        if not rf.exists():
            return False
        try:
            with open(rf, encoding="utf-8") as stream:
                d = json.load(stream)
        except Exception:
            return False
        ev = d.get(MODEL_NAME)
        if not g2_only and not _has_primary_metric(ev, task_type):
            return False
        if require_g2 and (
            not isinstance(ev, dict)
            or not _has_primary_metric(ev.get("g2_gaussian"), task_type)
        ):
            return False
    return True


# ======================================================================
# Multi-GPU orchestrator
# ======================================================================

LOGS_DIR = RESULTS_ROOT / "logs_real_eval_gotabpfn"


def orchestrate(pairs, out_dir, gpu_ids, seeds, args, per_gpu=1, max_retries=1):
    """Single-process orchestrator: spawn subprocess workers on free GPUs."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}
    tries_map = {}
    threads = max(4, min(16, 28 // per_gpu)) if per_gpu > 1 else 16
    print(f"Orchestrate (GoTabPFN): {len(pairs)} datasets across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, max_retries={max_retries})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        while pending and args.resume and _unit_done(out_dir, pending[0][1],
                                                     pending[0][0], seeds,
                                                     require_g2=args.g2,
                                                     g2_only=args.g2_only):
            ds, task = pending.pop(0)
            print(f"  [skip-done] {ds} ({task}) already complete", flush=True)
        free = [g for g in gpu_ids if gpu_load(g) < per_gpu]
        while free and pending:
            (ds, task), gpu = pending.pop(0), free.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env.setdefault(k, str(threads))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            lf = open(LOGS_DIR / f"gotabpfn_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [
                sys.executable, "-u", __file__,
                "--worker", "--task_type", task, "--datasets", ds,
                "--device_id", "0", "--output", args.output,
                "--threads", str(threads),
            ]
            if args.resume:
                cmd.append("--resume")
            if args.g2:
                cmd.append("--g2")
            if args.g2_only:
                cmd.append("--g2-only")
            if args.random_seed is not None:
                cmd.extend(["--random_seed", str(args.random_seed)])
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
    print("=== orchestrate (GoTabPFN): all jobs settled ===", flush=True)


# ======================================================================
# Main
# ======================================================================


def main():
    ap = argparse.ArgumentParser(description="GoTabPFN evaluation on TALENT benchmark datasets")
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default=OUT_REL)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--g2", action="store_true",
                    help="also evaluate the 2,000-column Gaussian-nuisance table")
    ap.add_argument("--g2-only", action="store_true",
                    help="evaluate G2 without computing a missing G1 result")
    # Multi-GPU
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--max_retries", type=int, default=1)
    args = ap.parse_args()
    if args.g2_only:
        args.g2 = True

    if args.worker and os.environ.get("EVAL_FORCE_FAIL"):
        print(f"[FORCE-FAIL] worker {args.task_type} {args.datasets} exiting 1", flush=True)
        sys.exit(1)

    # ── Orchestrator mode ──
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
        print(f"Orchestrator (GoTabPFN): {len(pairs)} datasets, GPUs {gpu_ids}, seeds {seeds}",
              flush=True)
        orchestrate(pairs, out_dir, gpu_ids, seeds, args,
                    per_gpu=args.per_gpu, max_retries=args.max_retries)
        return

    if args.task_type == "all":
        print("ERR: --task_type all requires --gpus (orchestrator mode).", flush=True)
        sys.exit(2)

    # ── Worker / direct mode ──
    torch.set_num_threads(args.threads)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    datasets = args.datasets or DATASET_MAP[args.task_type]
    seeds = [args.random_seed] if args.random_seed else SEEDS
    out_dir = RESULTS_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    task_label = TASK_DIR_MAP[args.task_type]

    print("=" * 70)
    print(f"GoTabPFN eval: task={task_label} datasets={len(datasets)} seeds={seeds} "
          f"device={device} resume={args.resume} "
          f"kmeans_gpu={_HAVE_KMEANS_GPU}")
    print(f"GO-LR: metric={GO_METRIC} k={GO_NUM_CLUSTERS} refine={GO_REFINE}")
    print(f"NSC-pSP: seg={NSC_SEGMENTATION} rule={NSC_M_RULE} tau={NSC_TAU} "
          f"M=[{NSC_M_MIN},{NSC_M_MAX}]")
    print(f"output → {out_dir}")
    print("=" * 70, flush=True)

    for di, ds in enumerate(datasets):
        for seed in seeds:
            rf = _dataset_json_path(out_dir, args.task_type, ds, seed)
            existing = None
            if args.resume and rf.exists():
                try:
                    existing = json.load(open(rf))
                    ev = existing.get(MODEL_NAME)
                    g1_done = _has_primary_metric(ev, args.task_type)
                    g2_done = _has_primary_metric(
                        ev.get("g2_gaussian"), args.task_type
                    ) if isinstance(ev, dict) else False
                    unit_done = (
                        g2_done if args.g2_only
                        else (g1_done and (not args.g2 or g2_done))
                    )
                    if unit_done:
                        print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — done, SKIP",
                              flush=True)
                        continue
                    print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — RESUME", flush=True)
                except Exception:
                    existing = None
            else:
                print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed}", flush=True)

            try:
                res = run_single_dataset(
                    ds, args.task_type, seed, device,
                    existing=existing, run_g2=args.g2,
                    g2_only=args.g2_only
                )
            except Exception as e:
                print(f"  [UNIT-ERR] {ds} s{seed}: {str(e)[:200]}", flush=True)
                traceback.print_exc()
                continue
            if res is not None:
                _write_results_atomic(rf, res)
            else:
                print(f"  SKIPPED", flush=True)
    print("\n=== all done ===", flush=True)


if __name__ == "__main__":
    main()
