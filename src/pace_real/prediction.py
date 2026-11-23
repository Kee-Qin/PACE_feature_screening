#!/usr/bin/env python3
"""Downstream-only G3 evaluation for strict full-2000 score artifacts.

This process never computes feature importance.  It verifies score provenance,
selects top-3p columns from the complete 2,000-vector, and evaluates requested
downstream models.  Existing G1/G2 results are intentionally not recomputed.
"""

from __future__ import annotations
from src import model_path as mp

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PREDICT_ROOT,
    DEFAULT_SCORE_ROOT,
    DEFAULT_TABDPT_CODE_ROOT,
    SCHEMA,
    TARGET_TOTAL_FEATURES,
    atomic_save_json,
    configure_model_paths,
    iter_units,
    prediction_file,
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


DEFAULT_METHODS = ("tabicl_shap", "sage_lgb", "sage_cb", "loco_simple", "loco_cv")
from src.screening_utilities import available_methods, selected_method_id
METHODS = tuple(available_methods()) + ("sis_emb_max",)
MODELS = ("lgb", "xgb", "catboost", "tabpfn", "tabicl", "tabpfn26", "tabdpt",
          "realmlp", "tabm", "mlp")
NN_MODELS = ("realmlp", "tabm", "mlp")
MODEL_GROUPS = {
    "all": MODELS,
    "cpu": ("lgb", "xgb", "catboost"),
    "gpu": ("realmlp", "tabm", "mlp", "tabpfn", "tabpfn26", "tabicl", "tabdpt"),
}
G3_RULE = "top_3p_legacy_descending_argsort"


def _find_score(roots, task, dataset, seed, method):
    candidates = [score_file(root, task, dataset, seed, method) for root in roots]
    hits = [p for p in candidates if p.exists()]
    if not hits:
        raise FileNotFoundError("score not found; checked: " + ", ".join(map(str, candidates)))
    if len(hits) > 1:
        # Multiple-machine copies are allowed only when byte-equivalent in scores/meta.
        s0, m0 = read_score(hits[0])
        for other in hits[1:]:
            sx, mx = read_score(other)
            if not np.array_equal(s0, sx) or m0 != mx:
                raise ValueError(f"conflicting score artifacts: {hits[0]} and {other}")
    return hits[0]


def _model_preflight(model, task, module, model_root, tabdpt_code_root):
    root = Path(model_root)
    required = []
    if model == "tabpfn":
        required = [mp.checkpoint("tabpfn_v3_" + ("reg" if task == "reg" else "clf"), root)]
    elif model == "tabicl":
        required = [mp.checkpoint("tabicl_v2_" + ("reg" if task == "reg" else "clf"), root)]
    elif model == "tabpfn26":
        required = [mp.checkpoint("tabpfn_v26_" + ("reg" if task == "reg" else "clf"), root)]
    elif model == "tabdpt":
        required = [mp.checkpoint("tabdpt", root)]
        code = Path(tabdpt_code_root)
        if not code.exists():
            raise FileNotFoundError(f"TabDPT code root missing: {code}")
        try:
            import tabdpt  # noqa: F401
            print(f"  TabDPT implementation: {tabdpt.__file__}", flush=True)
        except Exception as exc:
            raise ImportError(f"cannot import tabdpt from {code}: {exc}") from exc
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{model} checkpoint missing: {missing}")
    if model in NN_MODELS and not module._HAS_TALENT:
        raise ImportError("NN backbones need TALENT on PYTHONPATH")


def _evaluate(module, model, *values, max_epoch=200):
    if model == "realmlp":
        # TALENT's bundled RealMLP predates sklearn's argument rename. Keep the
        # compatibility adapter local to RealMLP, without changing site-packages.
        import inspect
        from sklearn.utils import validation
        from TALENT.model.lib.realmlp.sklearn import sklearn_base
        def adapt(fn):
            def call(*a, **kw):
                if "force_all_finite" in kw and "force_all_finite" not in inspect.signature(fn).parameters:
                    kw["ensure_all_finite"] = kw.pop("force_all_finite")
                return fn(*a, **kw)
            return call
        sklearn_base.check_X_y = adapt(validation.check_X_y)
        sklearn_base.check_array = adapt(validation.check_array)
    if model in NN_MODELS:
        return module.train_and_evaluate_dl(model, *values, max_epoch=max_epoch)
    return module.train_and_evaluate(model, *values)


def _g3_columns(scores, p_original):
    k = min(TARGET_TOTAL_FEATURES, 3 * int(p_original))
    # Match the existing experiment's selection rule, including tie ordering.
    return np.argsort(np.asarray(scores))[::-1][:k]


def _mapped_categorical(cat_indices, selected):
    position = {int(old): int(new) for new, old in enumerate(selected)}
    return [position[i] for i in cat_indices if i in position]


def _load_existing(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def build_parser():
    p = argparse.ArgumentParser(description="Downstream-only G3 evaluation from strict scores")
    p.add_argument("--task_type", default="all", choices=("bin", "multi", "reg", "all"))
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Optional dataset names; defaults to all available datasets")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    p.add_argument("--cohort", choices=("all",), default="all", help=argparse.SUPPRESS)
    p.add_argument("--models", nargs="+", choices=MODELS,
                   default=None, help="Optional subset within the chosen model-group")
    p.add_argument("--model-group", choices=("all", "gpu", "cpu"), default="all",
                   help="all=10 models; gpu=4 TFMs+3 NNs; cpu=3 tree models")
    p.add_argument("--score-roots", nargs="+", default=[str(DEFAULT_SCORE_ROOT)],
                   help="One or more copied score trees; identical duplicate artifacts are accepted")
    p.add_argument("--output-root", default=None,
                   help="Default separates cpu/gpu model-group outputs into their own directories")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--tabdpt-code-root", default=str(DEFAULT_TABDPT_CODE_ROOT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--aggregation", choices=("native", "mean", "max"), default="native")
    p.add_argument("--max-epoch", type=int, default=200)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true",
                   help="Path/import smoke test with a tiny real-data fit; does not need scores")
    add_gpu_arguments(p)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.methods = [selected_method_id(m, args.aggregation) for m in args.methods]
    print_dataset_cohort(args.cohort)
    allowed = MODEL_GROUPS[args.model_group]
    args.models = list(allowed) if args.models is None else args.models
    if any(m not in allowed for m in args.models):
        raise ValueError(f"models must belong to the {args.model_group} group: {allowed}")
    if args.output_root is None:
        args.output_root = str(DEFAULT_PREDICT_ROOT if args.model_group == "all"
                              else DEFAULT_PREDICT_ROOT / args.model_group)
    if args.model_group == "cpu":
        if args.gpus is not None or args.per_gpu != 1:
            raise ValueError("The cpu group does not use --gpus/--per_gpu; use dataset/seed shards")
        args.device = "cpu"
    if dispatch_gpu_workers(args, parser, mp.CODE_ROOT / "f2ves_real_eval_full.py", stage="predict"):
        return
    units = list(iter_units(args.task_type, args.datasets, args.seeds, args.cohort))
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    units = units[args.shard_index::args.num_shards]
    print(f"downstream-only G3; group={args.model_group}; models={args.models}; "
          f"schema={SCHEMA}; units={len(units)}", flush=True)
    if args.dry_run:
        for unit in units:
            print(*unit)
        return

    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    import tempfile
    temp_root = Path(args.output_root).resolve() / "_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_root)
    downstream = configure_model_paths(args.model_root, args.tabdpt_code_root)

    failures = []
    for task, dataset, seed in units:
        print(f"\n[{task}] {dataset} seed={seed}", flush=True)
        try:
            prepared = prepare_full2000(task, dataset, seed, args.data_root)
        except Exception as exc:
            failures.append((task, dataset, seed, "data", repr(exc)))
            print(f"  [FAIL] data: {exc}", flush=True)
            continue
        out_path = prediction_file(args.output_root, task, dataset, seed)
        result = _load_existing(out_path) if args.resume else {}
        if result and result.get("schema") != SCHEMA:
            failures.append((task, dataset, seed, "resume", "provenance mismatch"))
            print("  [FAIL] existing predictions have incompatible provenance", flush=True)
            continue
        result.update({
            "schema": SCHEMA,
            "dataset_cohort": cohort_id(args.cohort),
            "stage": "downstream_g3_only",
            "model_group": args.model_group,
            "task_type": task, "dataset": dataset, "seed": int(seed),
            "n": prepared["n"], "p_original": prepared["p_original"],
            "n_noise": prepared["n_noise"], "total_features": TARGET_TOTAL_FEATURES,
            "score_roots": [str(Path(r)) for r in args.score_roots],
        })
        result.setdefault("models", {})

        for model in args.models:
            try:
                _model_preflight(model, task, downstream, args.model_root, args.tabdpt_code_root)
            except Exception as exc:
                failures.append((task, dataset, seed, model, "preflight", repr(exc)))
                print(f"  [FAIL] {model} preflight: {exc}", flush=True)
                if not args.validate_only:
                    mr = result["models"].setdefault(model, {})
                    for method in args.methods:
                        if not isinstance(mr.get(f"g3_{method}"), dict) or "error" in mr[f"g3_{method}"]:
                            mr[f"g3_{method}"] = {"error": "preflight_failed", "detail": str(exc)}
                    atomic_save_json(out_path, result)
                continue

            if args.validate_only:
                # Test the actual downstream implementation and model path, cheaply.
                ntr = min(64, len(prepared["X_train"]))
                nv = min(32, len(prepared["X_val"]))
                nte = min(32, len(prepared["X_test"]))
                k = min(32, prepared["p_original"])
                probe = _evaluate(
                    downstream, model,
                    prepared["X_train"][:ntr, :k], prepared["y_train"][:ntr],
                    prepared["X_val"][:nv, :k], prepared["y_val"][:nv],
                    prepared["X_test"][:nte, :k], prepared["y_test"][:nte],
                    task, seed,
                    [i for i in prepared["cat_indices"] if i < k], args.device,
                    max_epoch=min(args.max_epoch, 2))
                if "error" in probe:
                    failures.append((task, dataset, seed, model, "validation", probe))
                    print(f"  [FAIL] {model} validation: {probe}", flush=True)
                else:
                    print(f"  [VALID] {model}: {probe}", flush=True)
                continue

            model_result = result["models"].setdefault(model, {})
            for method in args.methods:
                key = f"g3_{method}"
                try:
                    path = _find_score(args.score_roots, task, dataset, seed, method)
                    scores, meta = read_score(path)
                    validate_score(scores, meta, prepared, task, dataset, seed, method)
                    selected = _g3_columns(scores, prepared["p_original"])
                    if args.resume and isinstance(model_result.get(key), dict) \
                            and "error" not in model_result[key] \
                            and model_result[key].get("selected_features") == selected.tolist() \
                            and model_result[key].get("selection_rule") == G3_RULE:
                        print(f"  [resume] {model} {key}", flush=True)
                        continue
                    cat_selected = _mapped_categorical(prepared["cat_indices"], selected)
                    t0 = time.perf_counter()
                    metrics = _evaluate(
                        downstream, model,
                        prepared["Xc_train"][:, selected], prepared["y_train"],
                        prepared["Xc_val"][:, selected], prepared["y_val"],
                        prepared["Xc_test"][:, selected], prepared["y_test"],
                        task, seed, cat_selected, args.device, max_epoch=args.max_epoch)
                    if "error" in metrics:
                        raise RuntimeError(str(metrics))
                    metrics["elapsed_seconds"] = float(time.perf_counter() - t0)
                    metrics["n_features"] = int(len(selected))
                    metrics["n_original_selected"] = int(np.sum(selected < prepared["p_original"]))
                    metrics["noise_selected"] = int(np.sum(selected >= prepared["p_original"]))
                    metrics["pollution_fraction"] = float(
                        np.mean(selected >= prepared["p_original"]))
                    metrics["score_artifact"] = str(path)
                    metrics["selected_features"] = selected.tolist()
                    metrics["selection_rule"] = G3_RULE
                    model_result[key] = metrics
                    atomic_save_json(out_path, result)
                    print(f"  [DONE] {model} {key}: {metrics}", flush=True)
                except Exception as exc:
                    failures.append((task, dataset, seed, model, method, repr(exc)))
                    model_result[key] = {"error": type(exc).__name__, "detail": str(exc)}
                    atomic_save_json(out_path, result)
                    print(f"  [FAIL] {model} {method}: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        print("\nFailures:", flush=True)
        for item in failures:
            print(item, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
