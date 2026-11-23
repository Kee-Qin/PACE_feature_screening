"""eval_tabpfnwide.py — TabPFN-Wide standalone evaluation on TALENT benchmark datasets.

Runs TabPFN-Wide (continued-pretrained TabPFN-v2 for wide data) on each
(dataset, seed) pair.  Does NOT integrate into the screening
framework — TabPFN-Wide is evaluated directly on the data, G1 (clean) and
G2 (with Gaussian noise to 2000 features) only.  Results are written as
per-dataset JSON files under RESULTS_ROOT/results_real_eval_tabpfnwide/.

Usage:
  # Single dataset, single seed (direct mode)
  python eval_tabpfnwide.py --task_type bin --datasets coloncancer

  # All binary datasets, seeds 42/123/456, GPU 0
  python eval_tabpfnwide.py --task_type bin --device_id 0

  # Multi-GPU orchestrator mode
  python eval_tabpfnwide.py --task_type all --gpus 0,1,2,3

Requirements (must use venv_tabpfnwide — tabpfn==8.0.3):
  pip install tabpfn==8.0.3
  pip install numpy scipy scikit-learn pandas torch tqdm

The script self-checks tabpfn version at startup and refuses to run if not 8.0.x.
Configure the local wide checkpoint in model_path.py.
"""
from src import model_path as mp
import sys as _sys
import argparse
import io
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

# ── Inject tabpfn 8.0.x deps BEFORE any tabpfn import ──
_DEPS = str(mp.TABPFNWIDE_DEPS)
if _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

os.environ["TABPFN_MODEL_CACHE_DIR"] = str(mp.TABPFN_CACHE_ROOT)

import numpy as np
import torch

# ── version gate: TabPFN-Wide REQUIRES tabpfn 8.0.x ──
try:
    import tabpfn as _tpf
    _tpf_ver = _tpf.__version__
    _tpf_major_minor = tuple(int(x) for x in _tpf_ver.split(".")[:2])
    if _tpf_major_minor != (8, 0):
        print(f"FATAL: TabPFN-Wide requires tabpfn==8.0.x, found {_tpf_ver}. "
              f"Activate venv_tabpfnwide first.", flush=True)
        _sys.exit(1)
except ImportError:
    print("FATAL: tabpfn not installed. Activate venv_tabpfnwide (tabpfn==8.0.3).", flush=True)
    _sys.exit(1)

# ── TabPFN-Wide import (sys.path injection) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_TABPFNWIDE_ROOT = str(mp.TABPFNWIDE_CODE_ROOT)
if _TABPFNWIDE_ROOT not in sys.path:
    sys.path.insert(0, _TABPFNWIDE_ROOT)

from tabpfnwide.classifier import TabPFNWideClassifier

# ── TALENT data loading imports ──
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import f2ves_real_gaussian_large_experiments as T

# ======================================================================
# Constants (mirror f2ves_real_eval_full.py)
# ======================================================================

warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_TELEMETRY", "0")

BASE_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT")
                    or mp.OUTPUT_ROOT)

OUT_REL = "results_real_eval_tabpfnwide"

DATASET_MAP = {"bin": T.BINCLASS, "multi": T.MULTICLASS, "reg": T.REGRESSION}

TASK_DIR_MAP = {"bin": "binclass", "multi": "multiclass", "reg": "regression"}

SEEDS = [42, 123, 456]

TARGET_TOTAL_FEATURES = 2000

MODEL_NAME = "tabpfnwide"

# ── TabPFN-Wide config ──
# Use "wide-v2-5k" which handles up to 5000 features — our datasets max at 2000.
WIDE_MODEL = "wide-v2-5k"
WIDE_N_ESTIMATORS = 4

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
    msg = str(e).lower()
    return any(kw in msg for kw in ("out of memory", "oom", "cuda out of memory",
                                     "memory", "cannot allocate"))


# ======================================================================
# TabPFN-Wide train + evaluate
# ======================================================================


def _predict_proba_adaptive(clf, X_te):
    """predict_proba with OOM-adaptive test chunking.

    TabPFN v2 test samples do not attend to each other, so chunked prediction
    is mathematically identical to full-batch — only peak memory differs.
    First tries full-batch; on CUDA OOM retries with halving chunks.
    """
    try:
        return clf.predict_proba(X_te)
    except Exception as e:
        if not is_oom_error(e):
            raise
    chunk = 2048
    while chunk >= 64:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            outs = [clf.predict_proba(X_te[i:i + chunk])
                    for i in range(0, len(X_te), chunk)]
            return np.vstack(outs)
        except Exception as e:
            if not is_oom_error(e):
                raise
            chunk //= 2
    raise RuntimeError("OOM even at chunk=64")


def _predict_adaptive(clf, X_te):
    """predict (regression) with OOM-adaptive test chunking."""
    try:
        return clf.predict(X_te)
    except Exception as e:
        if not is_oom_error(e):
            raise
    chunk = 2048
    while chunk >= 64:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            outs = [clf.predict(X_te[i:i + chunk])
                    for i in range(0, len(X_te), chunk)]
            return np.concatenate(outs)
        except Exception as e:
            if not is_oom_error(e):
                raise
            chunk //= 2
    raise RuntimeError("OOM even at chunk=64")


def train_and_evaluate_wide(X_train, y_train, X_val, y_val, X_test, y_test,
                            task_type, seed, device="cuda"):
    """Train TabPFNWideClassifier and evaluate.

    Args:
        X_train, X_val, X_test: float32 numpy arrays
        y_train, y_val, y_test: int (cls) or float32 (reg) numpy arrays
        task_type: "bin" | "multi" | "reg"
        seed: random seed
        device: "cuda" or "cpu"

    Returns:
        dict with metric keys (auc or rmse), or {"error": "..."}
    """
    from sklearn.preprocessing import LabelEncoder

    # TabPFN-Wide is classification-only: the wide-v2-* checkpoints are all
    # classifier models, the package ships no regressor.  Record N/A rather
    # than attempting an invalid API call.
    if task_type == "reg":
        return {"error": "not_supported: TabPFN-Wide is classification-only "
                         "(no wide regressor checkpoint)"}

    # Degenerate-class guard
    if task_type != "reg":
        yt_arr = np.asarray(y_train)
        if len(np.unique(yt_arr)) < 2:
            return {"error": "single_class_in_fold"}

    # Merge train+val for TabPFN-Wide training
    X_tr = np.vstack([X_train, X_val]) if X_val is not None and len(X_val) > 0 else X_train
    if task_type == "reg":
        y_tr = np.hstack([y_train, y_val]) if y_val is not None and len(y_val) > 0 else y_train
        y_tr = y_tr.astype(np.float32)
    else:
        y_tr = np.hstack([y_train, y_val]) if y_val is not None and len(y_val) > 0 else y_train
        # Ensure labels are contiguous {0, 1, ..., K-1}
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr.astype(int))
        y_test_enc = le.transform(y_test.astype(int))

    X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
    X_te = np.ascontiguousarray(X_test, dtype=np.float32)

    try:
        if task_type == "reg":
            clf = TabPFNWideClassifier(
                model_path=str(mp.checkpoint("tabpfnwide")),
                device=device,
                n_estimators=WIDE_N_ESTIMATORS,
                random_state=seed,
                ignore_pretraining_limits=True,
                task_type="regression",
            )
            clf.fit(X_tr, y_tr)
            pred = _predict_adaptive(clf, X_te)
            proba = None
        else:
            clf = TabPFNWideClassifier(
                model_path=str(mp.checkpoint("tabpfnwide")),
                device=device,
                n_estimators=WIDE_N_ESTIMATORS,
                random_state=seed,
                ignore_pretraining_limits=True,
            )
            clf.fit(X_tr, y_tr)
            proba_all = _predict_proba_adaptive(clf, X_te)
            if task_type == "bin":
                proba = proba_all[:, 1] if proba_all.shape[1] > 1 else proba_all.reshape(-1)
                pred = (proba >= 0.5).astype(int)
            else:
                proba = proba_all
                pred = np.argmax(proba_all, axis=1)
    except Exception as e:
        if is_oom_error(e):
            print(f"    [OOM] TabPFN-Wide: {str(e)[:100]}", flush=True)
            return {"error": "OOM", "detail": str(e)[:200]}
        print(f"    [WARN] TabPFN-Wide failed: {str(e)[:120]}", flush=True)
        return {"error": str(e)[:200]}

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if task_type == "reg":
        return compute_metrics(y_test, pred, proba, task_type)
    else:
        return compute_metrics(y_test_enc, pred, proba, task_type)


# ======================================================================
# Single-dataset evaluation
# ======================================================================


def run_single_dataset(dataset_name, task_type, seed, device, existing=None):
    """Load TALENT dataset → run TabPFN-Wide on G1 + G2 → return result dict.

    G1: clean real features only
    G2: real + Gaussian noise features (total TARGET_TOTAL_FEATURES)
    """
    if existing is not None:
        ev = existing.get(MODEL_NAME)
        if isinstance(ev, dict) and "g1_clean" in ev and "g2_gaussian" in ev:
            g1_ok = isinstance(ev["g1_clean"], dict) and "error" not in ev["g1_clean"]
            g2_ok = isinstance(ev["g2_gaussian"], dict) and "error" not in ev["g2_gaussian"]
            if g1_ok and g2_ok:
                print(f"  {dataset_name} s{seed} — done, SKIP", flush=True)
                return existing

    print(f"  Loading {dataset_name}...", flush=True)
    try:
        Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = T.load_dataset(task_type, dataset_name)
    except Exception as e:
        print(f"  [ERR] load {dataset_name}: {e}", flush=True)
        return None

    cat_indices, cat_cols = T.identify_categorical(Xtr_df)
    p = Xtr_df.shape[1]
    n = Xtr_df.shape[0]
    print(f"  n={n} p={p} cat={len(cat_indices)} task={task_type}", flush=True)

    # ── Encode ──
    X_train, X_val, X_test, _enc = T.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    y_train_enc, _ = T.encode_y(ytr_raw, task_type)
    y_val_enc, _ = T.encode_y(yv_raw, task_type)
    y_test_enc, _ = T.encode_y(yte_raw, task_type)

    # ── Noise features for G2 ──
    n_noise = max(0, TARGET_TOTAL_FEATURES - p)
    noise_info = f"+{n_noise} noise" if n_noise > 0 else f"p={p} >= {TARGET_TOTAL_FEATURES} (no G2 noise)"
    print(f"  {noise_info}", flush=True)
    fk_tr, fk_v, fk_te = T.generate_gaussian_noise_features(
        X_train, X_val, X_test, seed, n_noise=n_noise)
    Xc_tr = np.hstack([X_train, fk_tr]) if n_noise > 0 else X_train
    Xc_v = np.hstack([X_val, fk_v]) if n_noise > 0 else X_val
    Xc_te = np.hstack([X_test, fk_te]) if n_noise > 0 else X_test

    mr = {}

    # ── G1: clean features ──
    print(f"    G1 (clean, p={p})...", flush=True)
    t0 = time.perf_counter()
    mr["g1_clean"] = train_and_evaluate_wide(
        X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc,
        task_type, seed, device)
    mr["g1_clean"]["elapsed_s"] = round(time.perf_counter() - t0, 2)

    # ── G2: with Gaussian noise ──
    print(f"    G2 (gaussian, p_total={Xc_tr.shape[1]})...", flush=True)
    t0 = time.perf_counter()
    mr["g2_gaussian"] = train_and_evaluate_wide(
        Xc_tr, y_train_enc, Xc_v, y_val_enc, Xc_te, y_test_enc,
        task_type, seed, device)
    mr["g2_gaussian"]["elapsed_s"] = round(time.perf_counter() - t0, 2)

    res = {
        "dataset": dataset_name, "task_type": task_type, "seed": seed,
        "n": n, "p": p, "n_cat": len(cat_indices), "n_noise": n_noise,
        MODEL_NAME: mr,
    }

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


def _unit_done(out_dir, task_type, dataset_name, seeds):
    for s in seeds:
        rf = _dataset_json_path(out_dir, task_type, dataset_name, s)
        if not rf.exists():
            return False
        try:
            d = json.load(open(rf))
        except Exception:
            return False
        ev = d.get(MODEL_NAME)
        if not isinstance(ev, dict):
            return False
        for key in ("g1_clean", "g2_gaussian"):
            v = ev.get(key)
            if not isinstance(v, dict) or "error" in v:
                return False
    return True


# ======================================================================
# Multi-GPU orchestrator
# ======================================================================

LOGS_DIR = RESULTS_ROOT / "logs_real_eval_tabpfnwide"


def orchestrate(pairs, out_dir, gpu_ids, seeds, args, per_gpu=1, max_retries=1):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pending, running = list(pairs), {}
    tries_map = {}
    threads = max(4, min(16, 28 // per_gpu)) if per_gpu > 1 else 16
    print(f"Orchestrate (TabPFN-Wide): {len(pairs)} datasets across GPUs {gpu_ids} "
          f"(per_gpu={per_gpu}, max_retries={max_retries})", flush=True)

    def gpu_load(g):
        return sum(1 for v in running.values() if v[4] == g)

    while pending or running:
        while pending and args.resume and _unit_done(out_dir, pending[0][1],
                                                     pending[0][0], seeds):
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
            lf = open(LOGS_DIR / f"tabpfnwide_{task}_{ds}_gpu{gpu}.log", "w")
            cmd = [
                sys.executable, "-u", __file__,
                "--worker", "--task_type", task, "--datasets", ds,
                "--device_id", "0", "--output", args.output,
                "--threads", str(threads),
            ]
            if args.resume:
                cmd.append("--resume")
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
    print("=== orchestrate (TabPFN-Wide): all jobs settled ===", flush=True)


# ======================================================================
# Main
# ======================================================================


def main():
    ap = argparse.ArgumentParser(description="TabPFN-Wide evaluation on TALENT benchmark datasets")
    ap.add_argument("--task_type", default="all", choices=["bin", "multi", "reg", "all"])
    ap.add_argument("--random_seed", type=int, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", default=OUT_REL)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    # Multi-GPU
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--max_retries", type=int, default=1)
    args = ap.parse_args()

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
        print(f"Orchestrator (TabPFN-Wide): {len(pairs)} datasets, GPUs {gpu_ids}, seeds {seeds}",
              flush=True)
        orchestrate(pairs, out_dir, gpu_ids, seeds, args,
                    per_gpu=args.per_gpu, max_retries=args.max_retries)
        return

    if args.task_type == "all":
        print("ERR: --task_type all requires --gpus (orchestrator mode).", flush=True)
        sys.exit(2)

    # ── Worker / direct mode ──
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TabPFN-Wide internally uses device="cuda" → will auto-select cuda:0
    # after CUDA_VISIBLE_DEVICES remapping done by orchestrator

    datasets = args.datasets or DATASET_MAP[args.task_type]
    seeds = [args.random_seed] if args.random_seed else SEEDS
    out_dir = RESULTS_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    task_label = TASK_DIR_MAP[args.task_type]

    print("=" * 70)
    print(f"TabPFN-Wide eval: task={task_label} datasets={len(datasets)} seeds={seeds} "
          f"model={WIDE_MODEL} n_estimators={WIDE_N_ESTIMATORS} "
          f"device={device} resume={args.resume}")
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
                    if isinstance(ev, dict):
                        g1 = ev.get("g1_clean")
                        g2 = ev.get("g2_gaussian")
                        if (isinstance(g1, dict) and "error" not in g1 and
                                isinstance(g2, dict) and "error" not in g2):
                            print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — done, SKIP",
                                  flush=True)
                            continue
                    print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed} — RESUME", flush=True)
                except Exception:
                    existing = None
            else:
                print(f"\n[{di+1}/{len(datasets)}] {ds} s{seed}", flush=True)

            try:
                res = run_single_dataset(ds, args.task_type, seed, device, existing=existing)
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
