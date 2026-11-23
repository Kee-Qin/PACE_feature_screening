#!/usr/bin/env python3
"""Shared data/path utilities for the strict full-2000 score/predict pipeline."""

from __future__ import annotations
from src import model_path as mp

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import f2ves_real_gaussian_large_experiments as data_impl
COHORT_ID = "talent_all_v1"


SCHEMA = "pace_full2000_v1"
TARGET_TOTAL_FEATURES = 2000
TABICL_SHAP_SAMPLING_ID = "rows500_split80_20_v1"
TABICL_SHAP_MAX_SAMPLES = 500
TABICL_SHAP_VALIDATION_FRACTION = 0.2
SAGE_METHODS = ("sage_lgb", "sage_cb")
SAGE_SAMPLING_ID = "rows500_split80_20_v1"
SAGE_MAX_SAMPLES = 500
SAGE_VALIDATION_FRACTION = 0.2
TASKS = ("bin", "multi", "reg")
TASK_FULL = {"bin": "classification", "multi": "classification", "reg": "regression"}
TASK_DIR = {"bin": "binclass", "multi": "multiclass", "reg": "regression"}

DEFAULT_PROJECT_ROOT = mp.PROJECT_ROOT
DEFAULT_DATA_ROOT = mp.DATA_ROOT
DEFAULT_MODEL_ROOT = mp.MODEL_ROOT
DEFAULT_TABDPT_CODE_ROOT = mp.TABDPT_CODE_ROOT
DEFAULT_SCORE_ROOT = mp.OUTPUT_ROOT / "talent_scores"
DEFAULT_PREDICT_ROOT = mp.OUTPUT_ROOT / "talent_predictions"


def configure_data_root(data_root: str | os.PathLike[str]) -> Path:
    root = Path(data_root).expanduser().resolve()
    data_impl.DATA_PATH = root
    return root


def configure_model_paths(model_root: str | os.PathLike[str],
                          tabdpt_code_root: str | os.PathLike[str]):
    import sys
    talent_root = mp.TALENT_CODE_ROOT
    if talent_root.exists() and str(talent_root) not in sys.path:
        sys.path.insert(0, str(talent_root))
    import f2ves_real_eval_full as downstream

    model_root = Path(model_root).expanduser().resolve()
    tabdpt_code_root = Path(tabdpt_code_root).expanduser().resolve()
    if tabdpt_code_root.exists() and str(tabdpt_code_root) not in sys.path:
        sys.path.insert(0, str(tabdpt_code_root))

    downstream.MODEL_ROOT = model_root
    downstream.TABDPT_CODE_ROOT = tabdpt_code_root
    downstream.TABPFN_V3_CLF = str(mp.checkpoint('tabpfn_v3_clf', model_root))
    downstream.TABPFN_V3_REG = str(mp.checkpoint('tabpfn_v3_reg', model_root))
    downstream.TABICL_V2_CLF = str(mp.checkpoint('tabicl_v2_clf', model_root))
    downstream.TABICL_V2_REG = str(mp.checkpoint('tabicl_v2_reg', model_root))
    downstream.TABPFN_V26_CLF = str(mp.checkpoint('tabpfn_v26_clf', model_root))
    downstream.TABPFN_V26_REG = str(mp.checkpoint('tabpfn_v26_reg', model_root))
    downstream.TABDPT_CKPT = str(mp.checkpoint('tabdpt', model_root))
    downstream.T.TABPFN_CLF_CKPT = downstream.TABPFN_V3_CLF
    downstream.T.TABPFN_REG_CKPT = downstream.TABPFN_V3_REG
    return downstream


def available_datasets(task_type: str, cohort="all") -> list[str]:
    return list(data_impl.DATASET_MAP[task_type])


def iter_units(task_type: str, datasets: list[str] | None, seeds: list[int], cohort="all"):
    tasks = TASKS if task_type == "all" else (task_type,)
    wanted = set(datasets) if datasets else None
    eligible = {task: available_datasets(task, cohort) for task in tasks}
    if wanted is not None:
        missing = sorted(wanted - {d for names in eligible.values() for d in names})
        if missing:
            raise ValueError(
                f"Dataset(s) unavailable for task_type={task_type}: "
                f"{missing}.")
    for task, names in eligible.items():
        for dataset in names:
            if wanted is not None and dataset not in wanted:
                continue
            for seed in seeds:
                yield task, dataset, int(seed)


def cohort_id(cohort="all"):
    return COHORT_ID


def print_dataset_cohort(cohort="all"):
    sizes = {task: len(available_datasets(task, cohort)) for task in TASKS}
    counts = ", ".join(f"{task}={n}" for task, n in sizes.items())
    print(f"dataset_cohort={cohort_id(cohort)}; eligible datasets: {counts}; "
          f"total={sum(sizes.values())}", flush=True)


def load_talent(task_type, dataset, root):
    if (Path(root) / TASK_DIR[task_type] / f"{dataset}.pkl").exists():
        return data_impl.load_dataset(task_type, dataset)
    folder = Path(root) / dataset
    info = json.loads((folder / "info.json").read_text())
    if info.get("task_type") != TASK_DIR[task_type]:
        raise ValueError(f"task mismatch for {dataset}: {info.get('task_type')}")
    splits = []
    for split in ("train", "val", "test"):
        blocks = []
        for prefix, dtype in (("N", np.float32), ("C", object)):
            path = folder / f"{prefix}_{split}.npy"
            if path.exists():
                array = np.asarray(np.load(path, allow_pickle=True), dtype=dtype)
                if array.ndim == 1:
                    array = array[:, None]
                blocks.append(pd.DataFrame(array, dtype=dtype,
                    columns=[f"{prefix}_{j}" for j in range(array.shape[1])]))
        if not blocks:
            raise FileNotFoundError(f"No feature arrays for {folder}/{split}")
        X = pd.concat(blocks, axis=1)
        y = np.load(folder / f"y_{split}.npy", allow_pickle=True).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"row mismatch for {dataset}/{split}")
        splits.extend((X, y))
    return tuple(splits)


def prepare_full2000(task_type: str, dataset: str, seed: int,
                     data_root: str | os.PathLike[str]):
    """Load TALENT splits and append Gaussian columns to exactly 2,000 columns.

    The returned ``p_original`` is evaluation metadata.  Strict scorers receive
    only the combined arrays and their full width; they never receive this
    boundary and never receive pre-filled noise scores.
    """
    configure_data_root(data_root)
    Xtr_df, ytr_raw, Xv_df, yv_raw, Xte_df, yte_raw = load_talent(
        task_type, dataset, data_root)
    cat_indices, _ = data_impl.identify_categorical(Xtr_df)
    Xtr, Xv, Xte, _ = data_impl.encode_categorical(Xtr_df, Xv_df, Xte_df, cat_indices)
    if task_type == "reg":
        ytr, yv, yte = [np.asarray(a, dtype=np.float64) for a in (ytr_raw, yv_raw, yte_raw)]
    else:
        labels = LabelEncoder().fit(np.asarray(ytr_raw).astype(str))
        ytr, yv, yte = [labels.transform(np.asarray(a).astype(str))
                       for a in (ytr_raw, yv_raw, yte_raw)]

    p_original = int(Xtr.shape[1])
    if p_original > TARGET_TOTAL_FEATURES:
        raise ValueError(
            f"{dataset}: p_original={p_original} exceeds strict target "
            f"P={TARGET_TOTAL_FEATURES}; no columns will be silently discarded")
    n_noise = TARGET_TOTAL_FEATURES - p_original
    if n_noise:
        Ntr, Nv, Nte = data_impl.generate_gaussian_noise_features(
            Xtr, Xv, Xte, seed, n_noise=n_noise)
        Xc_tr = np.hstack((Xtr, Ntr)).astype(np.float32, copy=False)
        Xc_v = np.hstack((Xv, Nv)).astype(np.float32, copy=False)
        Xc_te = np.hstack((Xte, Nte)).astype(np.float32, copy=False)
    else:
        Xc_tr, Xc_v, Xc_te = Xtr, Xv, Xte
    if Xc_tr.shape[1] != TARGET_TOTAL_FEATURES:
        raise AssertionError(f"strict width failure: got {Xc_tr.shape[1]}")

    return {
        "X_train": Xtr, "X_val": Xv, "X_test": Xte,
        "Xc_train": Xc_tr, "Xc_val": Xc_v, "Xc_test": Xc_te,
        "y_train": np.asarray(ytr), "y_val": np.asarray(yv), "y_test": np.asarray(yte),
        "cat_indices": [int(i) for i in cat_indices],
        "n": int(len(Xtr_df) + len(Xv_df) + len(Xte_df)),
        "p_original": p_original, "n_noise": n_noise,
        "P": TARGET_TOTAL_FEATURES,
    }


def score_file(root: str | os.PathLike[str], task: str, dataset: str,
               seed: int, method: str) -> Path:
    safe_dataset = dataset.replace("/", "__").replace("\\", "__")
    protocol = (TABICL_SHAP_SAMPLING_ID if method == "tabicl_shap"
                else SAGE_SAMPLING_ID if method in SAGE_METHODS else None)
    stem = f"{method}_{protocol}" if protocol else method
    return Path(root) / TASK_DIR[task] / safe_dataset / f"seed{seed}" / f"{stem}.npz"


def prediction_file(root: str | os.PathLike[str], task: str, dataset: str,
                    seed: int) -> Path:
    safe_dataset = dataset.replace("/", "__").replace("\\", "__")
    return Path(root) / TASK_DIR[task] / safe_dataset / f"seed{seed}.json"


def atomic_save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".json", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_score(path: Path):
    with np.load(path, allow_pickle=False) as z:
        scores = np.asarray(z["scores"], dtype=np.float64)
        meta = json.loads(str(z["metadata_json"].item()))
    return scores, meta


def validate_score(scores: np.ndarray, meta: dict, prepared: dict,
                   task: str, dataset: str, seed: int, method: str):
    failures = []
    if meta.get("schema") != SCHEMA:
        failures.append(f"schema={meta.get('schema')!r}")
    if not meta.get("strict_full_p"):
        failures.append("strict_full_p is not true")
    if not meta.get("noise_scores_computed"):
        failures.append("noise_scores_computed is not true")
    if int(meta.get("n_features_scored", -1)) != TARGET_TOTAL_FEATURES:
        failures.append(f"n_features_scored={meta.get('n_features_scored')}")
    if scores.shape != (TARGET_TOTAL_FEATURES,):
        failures.append(f"score_shape={scores.shape}")
    if not np.isfinite(scores).all():
        failures.append("scores contain non-finite values")
    if method == "tabicl_shap":
        config = meta.get("configuration", {})
        if config.get("sampling_protocol") != TABICL_SHAP_SAMPLING_ID:
            failures.append("TabICL-SHAP sampling protocol mismatch (requires 500-row 80/20 protocol)")
    elif method in SAGE_METHODS:
        config = meta.get("configuration", {})
        if config.get("sampling_protocol") != SAGE_SAMPLING_ID:
            failures.append("SAGE sampling protocol mismatch (requires 500-row 80/20 protocol)")
    elif method == "sis_emb_max":
        PROTOCOL = "sis_max_tabicl_v1.1_cumulative_L2_d128_mean_impute_rows2000_v1"
        if meta.get("configuration", {}).get("protocol") != PROTOCOL:
            failures.append("PACE-SIS-max aggregation/encoder protocol mismatch")
    expected = {
        "task_type": task, "dataset": dataset, "seed": int(seed), "method": method,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            failures.append(f"{key} mismatch ({meta.get(key)!r} != {value!r})")
    if failures:
        raise ValueError(f"Rejected score artifact {method}: " + "; ".join(failures))


def add_gpu_arguments(parser):
    import argparse
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated visible GPU indices, e.g. 0 or 0,1")
    parser.add_argument("--per_gpu", "--per-gpu", type=int, default=1,
                        help="Maximum concurrent dataset/seed workers per GPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)


def dispatch_gpu_workers(args, parser, script, stage=None):
    """One dataset/seed per subprocess; never write a unit from two workers."""
    import subprocess
    import sys
    import time
    from collections import deque

    if args.per_gpu < 1:
        raise ValueError("per_gpu must be at least 1")
    if args.worker or (args.gpus is None and args.per_gpu == 1):
        return False
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    elif args.device.startswith("cuda"):
        gpu_ids = [int(args.device.split(":")[1]) if ":" in args.device else 0]
    else:
        raise ValueError("per_gpu needs a CUDA device or --gpus; omit it on CPU-only machines")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpus must contain distinct GPU indices")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    units = list(iter_units(args.task_type, args.datasets, args.seeds,
                           getattr(args, "cohort", "all")))
    units = units[args.shard_index::args.num_shards]
    print(f"GPU scheduler: gpus={gpu_ids}, per_gpu={args.per_gpu}, "
          f"units={len(units)}", flush=True)
    if args.dry_run:
        for task, ds, seed in units:
            print(f"[plan] {task} {ds} seed={seed}", flush=True)
        return True

    import torch
    count = torch.cuda.device_count()
    if any(g < 0 or g >= count for g in gpu_ids):
        raise ValueError(f"Requested {gpu_ids}; only {count} CUDA devices are visible")
    # Map logical visible indices to physical IDs/UUIDs before isolating children.
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    mapping = inherited.split(",") if inherited else [str(i) for i in range(count)]
    excluded = {"help", "gpus", "per_gpu", "worker", "task_type", "datasets",
                "seeds", "num_shards", "shard_index", "device"}
    common = []
    for action in parser._actions:
        if action.dest in excluded or not action.option_strings:
            continue
        value = getattr(args, action.dest)
        flag = action.option_strings[0]
        if isinstance(value, bool):
            if value:
                common.append(flag)
        elif value is not None:
            common.append(flag)
            common.extend(map(str, value if isinstance(value, list) else [value]))
    log_root = Path(args.output_root) / "logs" / time.strftime("%Y%m%d_%H%M%S")
    log_root.mkdir(parents=True, exist_ok=True)
    queue = deque(units)
    active = {}
    slots = [(gpu, slot) for gpu in gpu_ids for slot in range(args.per_gpu)]
    failures = []
    try:
        while queue or active:
            for slot in slots:
                if slot in active or not queue:
                    continue
                task, ds, seed = queue.popleft()
                gpu, lane = slot
                safe_ds = ds.replace("/", "__").replace("\\", "__")
                logfile = log_root / f"{task}_{safe_ds}_s{seed}_gpu{gpu}_{lane}.log"
                stream = logfile.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = mapping[gpu]
                command = [sys.executable, "-u", str(Path(script).resolve()),
                           *(["--stage", stage] if stage else []), *common,
                           "--task_type", task, "--datasets", ds, "--seeds", str(seed),
                           "--device", "cuda:0", "--worker"]
                process = subprocess.Popen(command, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT)
                active[slot] = (process, stream, task, ds, seed, logfile)
                print(f"[START gpu={gpu} lane={lane} pid={process.pid}] "
                      f"{task}/{ds}/s{seed}; log={logfile}", flush=True)
            for slot, item in list(active.items()):
                process, stream, task, ds, seed, logfile = item
                status = process.poll()
                if status is None:
                    continue
                stream.close()
                del active[slot]
                print(f"[{'DONE' if status == 0 else 'FAIL'} exit={status}] "
                      f"{task}/{ds}/s{seed}; log={logfile}", flush=True)
                if status:
                    failures.append((task, ds, seed, status, str(logfile)))
            if active:
                time.sleep(0.5)
    except BaseException:
        for process, stream, *_ in active.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        raise
    if failures:
        print(f"{len(failures)} worker(s) failed: {failures}", flush=True)
        raise SystemExit(2)
    return True
