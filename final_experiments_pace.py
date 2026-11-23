#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run manuscript-ordered PACE simulations (classification and regression).

Use --dry-run to inspect the paper25 plan. Default: cumulative layer 2,
50 repetitions per setting (25 for ultrahigh-dimensional settings).
"""

from src.screening_utilities import compute_scores, compute_many, embedding_scores, layered_scores, pre_split_scores, raw_a_scores, selector_call
PACE_AGGREGATION = "native"
from src import simulation_protocol as protocol
import sys
import os
import gc
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore")
gc.collect()  # Release unused host memory before a run

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generators_classification import (
    ClassificationDataGenerator,
    EXPERIMENTS as _CLS_EXPERIMENTS,
    EXPERIMENT_NAMES as _CLS_EXPERIMENT_NAMES,
)
from src.data_generators_regression import (
    RegressionDataGenerator,
    EXPERIMENTS as _REG_EXPERIMENTS,
    EXPERIMENT_NAMES as _REG_EXPERIMENT_NAMES,
)
from src.screening_methods import (
    AcceleratedScreening, ScreeningEvaluator, AcceleratedEmbedder, TABICL_PATH,
)
from src.screening_methods_torch import TorchAcceleratedScreening, TorchAcceleratedEmbedder


# =============================================================================

# =============================================================================

N_REPS = 50
N_JOBS = -1
EMB_LAYERS = [2]

_BASE = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR_CLS = str(protocol.output_root() / "classification/pace")
BASE_OUTPUT_DIR_REG = str(protocol.output_root() / "regression/pace")


TRAD_METHODS_CLS = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                    'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'ids']

EMB_METHODS_CLS = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                   'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor']


TRAD_METHODS_REG = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hd', 'gcor', 'ids']

EMB_METHODS_REG = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hd', 'gcor']


# =============================================================================

# =============================================================================

CLS_EXPERIMENTS = dict(_CLS_EXPERIMENTS)           # exp1..exp6
CLS_EXPERIMENT_NAMES = dict(_CLS_EXPERIMENT_NAMES)
REG_EXPERIMENTS = dict(_REG_EXPERIMENTS)           # exp1..exp4
REG_EXPERIMENT_NAMES = dict(_REG_EXPERIMENT_NAMES)


CLS_PLAIN_EXPS = list(protocol.CLASSIFICATION)[:4]
CLS_MIXMISS_EXPS = list(protocol.CLASSIFICATION)[4:]
REG_PLAIN_EXPS = list(protocol.REGRESSION)[:2]
REG_MIXMISS_EXPS = list(protocol.REGRESSION)[2:]


def _methods_for_task(task):
    if task == 'classification':
        return TRAD_METHODS_CLS, EMB_METHODS_CLS, BASE_OUTPUT_DIR_CLS
    return TRAD_METHODS_REG, EMB_METHODS_REG, BASE_OUTPUT_DIR_REG


def _registry_for_task(task):
    if task == 'classification':
        return CLS_EXPERIMENTS, CLS_EXPERIMENT_NAMES
    return REG_EXPERIMENTS, REG_EXPERIMENT_NAMES


# =============================================================================

# =============================================================================

class CheckpointManager:

    def __init__(self, checkpoint_path: str, manifest_path: str = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.manifest_path = Path(manifest_path) if manifest_path else self.checkpoint_path.parent / "manifest.json"
        self.data = {}
        self.load()

    def load(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def make_key(self, exp_key: str, param_set: int, rep: int) -> str:
        return f"{exp_key}_ps{param_set}_rep{rep}"

    def is_completed(self, key: str) -> bool:
        return key in self.data

    def get_result(self, key: str):
        return self.data.get(key)

    def save_result(self, key: str, result):
        clean = {}
        for k, v in result.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        self.data[key] = clean
        self.save()


    def is_experiment_complete(self, exp_key: str, param_set: int,
                               n_reps: int, emb_layers: List[int],
                               trad_methods: List[str] = None,
                               emb_methods: List[str] = None,
                               available_methods: List[str] = None) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            if manifest.get('n_reps') != n_reps:
                return False
            if sorted(manifest.get('emb_layers', [])) != sorted(emb_layers):
                return False
            if manifest.get('completed_reps', 0) < n_reps:
                return False
            cached_available = set(manifest.get('available_methods', []))
            if cached_available and available_methods is not None:
                return set(available_methods).issubset(cached_available)
            if trad_methods is not None:
                cached_trad = set(manifest.get('trad_methods', []))
                required_trad = set(trad_methods)
                if not required_trad.issubset(cached_trad):
                    return False
            if emb_methods is not None:
                cached_emb = set(manifest.get('emb_methods', []))
                required_emb = set(emb_methods)
                if not required_emb.issubset(cached_emb):
                    return False
            return True
        except Exception:
            return False

    def update_manifest(self, exp_key: str, param_set: int,
                        n_reps: int, emb_layers: List[int],
                        completed_reps: int,
                        trad_methods: List[str] = None,
                        emb_methods: List[str] = None,
                        available_methods: List[str] = None):
        manifest = {
            'exp_key': exp_key,
            'param_set': param_set,
            'n_reps': n_reps,
            'emb_layers': sorted(emb_layers),
            'completed_reps': completed_reps,
            'trad_methods': sorted(trad_methods) if trad_methods else [],
            'emb_methods': sorted(emb_methods) if emb_methods else [],
            'available_methods': sorted(available_methods) if available_methods else [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


# =============================================================================

# =============================================================================

def _unpack_data(R):
    X, y, tf, n, p, s = R[:6]
    cat_info = None
    missing_mask = None
    if len(R) >= 7:
        extra = R[6]
        if isinstance(extra, dict) and 'cat_indices' in extra:
            cat_info = extra
        elif isinstance(extra, np.ndarray):
            missing_mask = extra
    return np.asarray(X), np.asarray(y), list(tf), int(n), int(p), int(s), cat_info, missing_mask


# =============================================================================

# =============================================================================

def run_single_rep(X, y, true_features, s, p, n, embedder,
                   missing_mask=None, cat_info=None,
                   task='classification',
                   emb_layers: List[int] = EMB_LAYERS, n_jobs: int = N_JOBS,
                   use_torch: bool = True):
    trad_methods_full, emb_methods_full, _ = _methods_for_task(task)
    use_pc = n <= 500
    trad_methods = [m for m in trad_methods_full if use_pc or m not in ('pc',)]
    emb_methods = [m for m in emb_methods_full if use_pc or m not in ('pc',)]

    screening_cls = TorchAcceleratedScreening if use_torch else AcceleratedScreening
    results = {}


    if missing_mask is not None:

        X_imputed = X.copy()
        if np.any(np.isnan(X_imputed)):
            col_means = np.nanmean(X_imputed, axis=0)
            nan_cols = np.where(np.isnan(col_means))[0]
            if len(nan_cols) > 0:
                col_means[nan_cols] = 0.0
            for j in range(p):
                X_imputed[np.isnan(X_imputed[:, j]), j] = col_means[j]


        try:
            trad_scores = compute_many(X_imputed, y, methods=trad_methods, backend="torch" if use_torch else "numpy", impute=False)
        except Exception:
            trad_scores = compute_many(X_imputed, y, methods=trad_methods, backend="numpy", impute=False)

        miss_ratio = missing_mask.mean(axis=0)  # (p,)
        for method, scores in trad_scores.items():
            if scores is None:
                continue
            adjusted = scores.copy()
            for j in range(p):
                if miss_ratio[j] > 0.9:
                    adjusted[j] = 0.0
                elif miss_ratio[j] > 0:
                    adjusted[j] *= np.sqrt(1.0 - miss_ratio[j])
            eval_res = ScreeningEvaluator.evaluate_screening(adjusted, true_features, s, p, n)
            results[f'ac_{method}'] = {
                'mms': eval_res['mms'],
                'tpr_n1': eval_res['tpr_n1'],
                'fpr_n1': eval_res['fpr_n1'],
                'tpr_nlogn': eval_res['tpr_nlogn'],
                'fpr_nlogn': eval_res['fpr_nlogn'],
            }

            eval_res_imp = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[f'imp_{method}'] = {
                'mms': eval_res_imp['mms'],
                'tpr_n1': eval_res_imp['tpr_n1'],
                'fpr_n1': eval_res_imp['fpr_n1'],
                'tpr_nlogn': eval_res_imp['tpr_nlogn'],
                'fpr_nlogn': eval_res_imp['fpr_nlogn'],
            }


        if embedder is not None and embedder.loaded:
            if use_torch and p > 10000 and hasattr(TorchAcceleratedScreening, 'compute_all_scores_with_embeddings'):
                emb_scores = layered_scores(X_imputed, y, embedder, emb_layers=emb_layers, methods=emb_methods, aggregation=PACE_AGGREGATION)
                for key, scores in emb_scores.items():
                    if scores is None:
                        continue
                    eval_res = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
                    results[key] = {
                        'mms': eval_res['mms'],
                        'tpr_n1': eval_res['tpr_n1'],
                        'fpr_n1': eval_res['fpr_n1'],
                        'tpr_nlogn': eval_res['tpr_nlogn'],
                        'fpr_nlogn': eval_res['fpr_nlogn'],
                    }
            else:
                all_layer_embeddings = embedder.get_embeddings_batch_all_layers(X_imputed, emb_layers)
                for layer in emb_layers:
                    embeddings = all_layer_embeddings.get(layer)
                    if embeddings is None:
                        continue
                    if (not use_torch) and isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.cpu().numpy()
                    emb_scores = embedding_scores(embeddings, y, methods=emb_methods, backend="torch" if use_torch else "numpy", aggregation=PACE_AGGREGATION)
                    for raw_method in emb_methods:
                        scores_key = f'emb_{raw_method}'
                        if scores_key not in emb_scores or emb_scores[scores_key] is None:
                            continue
                        eval_res = ScreeningEvaluator.evaluate_screening(
                            emb_scores[scores_key], true_features, s, p, n)
                        results[f'emb_{raw_method}_L{layer}'] = {
                            'mms': eval_res['mms'],
                            'tpr_n1': eval_res['tpr_n1'],
                            'fpr_n1': eval_res['fpr_n1'],
                            'tpr_nlogn': eval_res['tpr_nlogn'],
                            'fpr_nlogn': eval_res['fpr_nlogn'],
                        }


        results['_meta_missing'] = {
            'overall_missing_rate': float(missing_mask.mean()),
            'active_feature_missing_rates': [float(missing_mask[:, j].mean()) for j in true_features],
        }
        return results


    trad_scores = compute_many(X, y, methods=trad_methods, backend="torch" if use_torch else "numpy", impute=False)
    for method, scores in trad_scores.items():
        if scores is None:
            continue
        eval_res = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
        results[method] = {
            'mms': eval_res['mms'],
            'tpr_n1': eval_res['tpr_n1'],
            'fpr_n1': eval_res['fpr_n1'],
            'tpr_nlogn': eval_res['tpr_nlogn'],
            'fpr_nlogn': eval_res['fpr_nlogn'],
        }


    if embedder is not None and embedder.loaded:

        if use_torch and p > 10000 and hasattr(TorchAcceleratedScreening, 'compute_all_scores_with_embeddings'):
            emb_scores = layered_scores(X, y, embedder, emb_layers=emb_layers, methods=emb_methods, aggregation=PACE_AGGREGATION)
            for key, scores in emb_scores.items():
                if scores is None:
                    continue
                eval_res = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
                results[key] = {
                    'mms': eval_res['mms'],
                    'tpr_n1': eval_res['tpr_n1'],
                    'fpr_n1': eval_res['fpr_n1'],
                    'tpr_nlogn': eval_res['tpr_nlogn'],
                    'fpr_nlogn': eval_res['fpr_nlogn'],
                }
        else:
            all_layer_embeddings = embedder.get_embeddings_batch_all_layers(X, emb_layers)
            for layer in emb_layers:
                embeddings = all_layer_embeddings.get(layer)
                if embeddings is None:
                    continue
                if use_torch and isinstance(embeddings, torch.Tensor):
                    pass
                elif use_torch and isinstance(embeddings, np.ndarray):
                    pass
                else:
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.cpu().numpy()
                emb_scores = embedding_scores(embeddings, y, methods=emb_methods, backend="torch" if use_torch else "numpy", aggregation=PACE_AGGREGATION)
                for raw_method in emb_methods:
                    scores_key = f'emb_{raw_method}'
                    if scores_key not in emb_scores or emb_scores[scores_key] is None:
                        continue
                    eval_res = ScreeningEvaluator.evaluate_screening(
                        emb_scores[scores_key], true_features, s, p, n)
                    results[f'emb_{raw_method}_L{layer}'] = {
                        'mms': eval_res['mms'],
                        'tpr_n1': eval_res['tpr_n1'],
                        'fpr_n1': eval_res['fpr_n1'],
                        'tpr_nlogn': eval_res['tpr_nlogn'],
                        'fpr_nlogn': eval_res['fpr_nlogn'],
                    }

    return results


# =============================================================================

# =============================================================================

def run_experiment(exp_key: str, task: str, param_sets: List[int], n_reps: int,
                   embedder, output_dir: str, emb_layers: List[int] = EMB_LAYERS,
                   n_jobs: int = N_JOBS, use_checkpoint: bool = True,
                   use_torch: bool = True):
    experiments, names = _registry_for_task(task)
    trad_methods_full, emb_methods_full, _ = _methods_for_task(task)
    gen_func = experiments[exp_key]
    exp_name = names.get(exp_key, exp_key)

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_key} — {exp_name} ({task})")
    print(f"Param sets: {param_sets}, Reps: {n_reps}, Emb layers: {emb_layers}")
    print(f"{'='*70}", flush=True)

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = ps_dir / "checkpoint.json"
        manifest_path = ps_dir / "manifest.json"
        checkpoint = CheckpointManager(str(checkpoint_path), manifest_path=str(manifest_path))


        manifest_basic_ok = False
        if use_checkpoint and checkpoint.manifest_path.exists():
            try:
                with open(checkpoint.manifest_path, 'r') as f:
                    mf = json.load(f)
                if (mf.get('n_reps') == n_reps
                        and sorted(mf.get('emb_layers', [])) == sorted(emb_layers)
                        and mf.get('completed_reps', 0) >= n_reps):
                    manifest_basic_ok = True
            except Exception:
                pass


        probe_seed = protocol.simulation_seed(task, exp_key, ps, 1)
        R_probe = gen_func(param_set=ps, seed=probe_seed)
        X_probe, y_probe, tf_probe, n_probe, p_probe, s_probe, cat_probe, mm_probe = _unpack_data(R_probe)
        probe_res = run_single_rep(
            X_probe, y_probe, tf_probe, s_probe, p_probe, n_probe, embedder,
            missing_mask=mm_probe, cat_info=cat_probe, task=task,
            emb_layers=emb_layers, n_jobs=n_jobs, use_torch=use_torch)
        available_methods = set(k for k in probe_res
                                if isinstance(k, str) and not k.startswith('_')
                                and isinstance(probe_res[k], dict) and 'mms' in probe_res[k])
        print(f"  [{exp_key}/ps{ps}] Probe: {len(available_methods)} methods available "
              f"(requested {len(trad_methods_full)} trad + {len(emb_methods_full)} emb x {len(emb_layers)} layers)",
              flush=True)


        if manifest_basic_ok:
            try:
                with open(checkpoint.manifest_path, 'r') as f:
                    mf = json.load(f)
                cached_available = set(mf.get('available_methods', []))
                if cached_available and available_methods.issubset(cached_available):
                    print(f"  [{exp_key}/ps{ps}] Already completed ({n_reps} reps, "
                          f"{len(cached_available)} methods) — SKIPPED", flush=True)
                    all_results = []
                    for rep in range(1, n_reps + 1):
                        key = checkpoint.make_key(exp_key, ps, rep)
                        all_results.append(checkpoint.get_result(key))
                    s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
                    summary = summarize_results(all_results, s)
                    save_summary(summary, ps_dir)
                    continue
                else:
                    print(f"  [{exp_key}/ps{ps}] Manifest available methods mismatch "
                          f"(cached: {len(cached_available)}, probe: {len(available_methods)}) — re-running", flush=True)
            except Exception:
                pass


        all_results = []
        newly_completed = 0
        for rep in range(1, n_reps + 1):
            key = checkpoint.make_key(exp_key, ps, rep)

            if use_checkpoint and checkpoint.is_completed(key):
                res = checkpoint.get_result(key)
                cached_methods = set(k for k in res.keys()
                                     if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                if available_methods.issubset(cached_methods):
                    all_results.append(res)
                    continue
                missing = available_methods - cached_methods
                print(f"  [{exp_key}/ps{ps}/rep{rep}] Cache missing {len(missing)} methods "
                      f"(have {len(cached_methods)}, need {len(available_methods)}) — re-running", flush=True)

            seed = protocol.simulation_seed(task, exp_key, ps, rep)
            R = gen_func(param_set=ps, seed=seed)
            X, y, true_features, n, p, s, cat_info, missing_mask = _unpack_data(R)

            res = run_single_rep(
                X, y, true_features, s, p, n, embedder,
                missing_mask=missing_mask, cat_info=cat_info, task=task,
                emb_layers=emb_layers, n_jobs=n_jobs, use_torch=use_torch)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep,
                            'n': n, 'p': p, 's': s}
            if cat_info is not None:
                res['_meta']['cat_indices'] = list(cat_info.get('cat_indices', []))
            if missing_mask is not None:
                res['_meta_missing'] = {
                    'overall_missing_rate': float(missing_mask.mean()),
                    'active_feature_missing_rates': [float(missing_mask[:, j].mean()) for j in true_features],
                }
            if use_checkpoint:
                checkpoint.save_result(key, res)
            all_results.append(res)
            newly_completed += 1

            if newly_completed % 10 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly_completed})", flush=True)


        completed_reps = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed_reps >= n_reps:
            checkpoint.update_manifest(exp_key, ps, n_reps, emb_layers, completed_reps,
                                       trad_methods=trad_methods_full, emb_methods=emb_methods_full,
                                       available_methods=sorted(available_methods))


        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
        summary = summarize_results(all_results, s)
        save_summary(summary, ps_dir)
        save_raw_results(all_results, ps_dir)

        cached = n_reps - newly_completed
        print(f"  [{exp_key}/ps{ps}] Done — cached: {cached}, new: {newly_completed}", flush=True)

    return


# =============================================================================

# =============================================================================

def summarize_results(all_results: List[Dict], s: int) -> Dict:
    if not all_results:
        return {}

    method_metrics = {}
    for res in all_results:
        for key, val in res.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict) and 'mms' in val:
                if key not in method_metrics:
                    method_metrics[key] = {k: [] for k in val.keys()}
                for k, v in val.items():
                    method_metrics[key][k].append(v)

    summary = {}
    for method, metrics in method_metrics.items():
        mms_vals = np.array(metrics['mms'])
        summary[method] = {
            'mms_median': float(np.median(mms_vals)),
            'mms_q1': float(np.percentile(mms_vals, 25)),
            'mms_q3': float(np.percentile(mms_vals, 75)),
            'mms_mean': float(np.mean(mms_vals)),
            'mms_std': float(np.std(mms_vals)),
            'ptp': float(np.mean(mms_vals <= s)),
            'tpr_n1_mean': float(np.mean(metrics.get('tpr_n1', [0]))),
            'fpr_n1_mean': float(np.mean(metrics.get('fpr_n1', [0]))),
            'tpr_nlogn_mean': float(np.mean(metrics.get('tpr_nlogn', [0]))),
            'fpr_nlogn_mean': float(np.mean(metrics.get('fpr_nlogn', [0]))),
        }
    return summary


def save_summary(summary: Dict, output_dir: Path):
    out_path = output_dir / "summary.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"{'Method':<26} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
                 f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}")
    lines.append("-" * 100)


    ac_methods = sorted([k for k in summary if k.startswith('ac_')])
    imp_methods = sorted([k for k in summary if k.startswith('imp_')])
    trad_order = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                  'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'ids']
    emb_base = ['dc', 'sis', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'hsic']
    layers_in_summary = sorted(set(int(k.split('_L')[-1]) for k in summary if '_L' in k))
    emb_methods = [f'emb_{m}_L{l}' for l in layers_in_summary for m in emb_base
                   if f'emb_{m}_L{l}' in summary]

    for method in ac_methods + imp_methods + trad_order + emb_methods:
        if method not in summary:
            continue
        d = summary[method]
        iqr_str = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(
            f"{method:<26} {d['mms_median']:>10.1f} {iqr_str:>12} {d['ptp']:>6.2f} "
            f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
            f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}"
        )

    txt_path = output_dir / "summary_table.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def save_raw_results(all_results: List[Dict], output_dir: Path):
    out_path = output_dir / "raw_results.json"
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                  float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
            elif k == '_meta' or k.startswith('_meta'):
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                  float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
        clean.append(entry)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


# =============================================================================

# =============================================================================

def get_param_sets(exp_key: str, task: str, user_param_sets=None):
    return protocol.settings(task, exp_key, user_param_sets)


# =============================================================================

# =============================================================================

def resolve_exp_keys(exp_arg: str, task: str):
    return protocol.experiments(task, exp_arg)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='final_experiments_pace: classification and regression screening experiments'
    )
    parser.add_argument('--task', type=str, required=True,
                        choices=['classification', 'regression'],
                        help='Task type (required)')
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiments: all / plain / mixtype_missing / expN / comma-separated list. '
                             'cls: exp1-4 plain, exp5-6 mixtype_missing; '
                             'reg: exp1-2 plain, exp3-4 mixtype_missing')
    parser.add_argument('--param-sets', type=int, nargs='+', default=None,
                        help='Setting IDs from simulation_protocol.py (manuscript order)')
    parser.add_argument('--n-reps', type=int, default=None,
                        help='Repetitions: default 50, or 25 for ultrahigh-dimensional settings')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU index (default: 0)')
    parser.add_argument('--emb-layers', type=int, nargs='+', default=EMB_LAYERS,
                        help=f'Embedding layer indices (default: {EMB_LAYERS})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results_simulation_paper25/<task>/pace)')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable checkpoints')
    parser.add_argument('--clear-checkpoint', action='store_true',
                        help='Remove checkpoints and manifest to restart the specified output directory')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS,
                        help=f'Parallel jobs (default: {N_JOBS} = all cores)')
    parser.add_argument('--no-torch', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without loading models or writing results')
    parser.add_argument("--aggregation", choices=("native", "mean", "max"), default="native")
    return parser.parse_args()


def main():
    global PACE_AGGREGATION
    args = parse_args()
    PACE_AGGREGATION = args.aggregation
    plan = protocol.execution_plan(args.task, args.exp, args.param_sets, args.n_reps)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    use_torch = not args.no_torch
    _, _, base_output = _methods_for_task(args.task)
    output_dir = args.output if args.output else base_output
    if args.aggregation != "native":
        output_dir = str(Path(output_dir) / ("aggregation_" + args.aggregation))
    protocol.ensure_output_protocol(output_dir, args.task, 'pace')

    print("=" * 70)
    print(f"final_experiments_pace — task={args.task} exp={args.exp}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    device = f'cuda:{args.device_id}'
    if use_torch:
        from src.screening_methods_torch import set_device
        set_device(args.device_id)
    print(f"n_reps={args.n_reps}, emb_layers={args.emb_layers}, device={device}, "
          f"use_torch={use_torch}, output={output_dir}")
    print(f"TabICL checkpoint: {TABICL_PATH}")
    print("=" * 70, flush=True)

    exp_keys = resolve_exp_keys(args.exp, args.task)
    print(f"Resolved experiments: {exp_keys}", flush=True)

    # --clear-checkpoint
    if args.clear_checkpoint:
        for ek in exp_keys:
            ps_sets = get_param_sets(ek, args.task, args.param_sets)
            for ps in ps_sets:
                ps_dir = Path(output_dir) / ek / f"param_set_{ps}"
                for fname in ['checkpoint.json', 'manifest.json']:
                    fp = ps_dir / fname
                    if fp.exists():
                        fp.unlink()
                        print(f"  Cleared: {fp}")
        print()


    embedder = None
    try:
        if use_torch:
            embedder = TorchAcceleratedEmbedder(device=device)
            loaded = embedder.load_model(TABICL_PATH, device=device)
        else:
            embedder = AcceleratedEmbedder()
            loaded = embedder.load_model(TABICL_PATH, device=device)
        if loaded:
            print(f"Embedder loaded on {device}: embed_dim={embedder.embed_dim} "
                  f"(torch={use_torch})", flush=True)
        else:
            print("Embedder failed to load, running traditional methods only", flush=True)
            embedder = None
    except Exception as e:
        print(f"Embedder not available: {e}", flush=True)
        embedder = None

    for exp_key in exp_keys:
        ps_list = get_param_sets(exp_key, args.task, args.param_sets)
        run_experiment(
            exp_key=exp_key, task=args.task,
            param_sets=ps_list, n_reps=(args.n_reps if args.n_reps is not None else protocol.repetitions(args.task, exp_key)),
            embedder=embedder, output_dir=output_dir,
            emb_layers=args.emb_layers, n_jobs=args.n_jobs,
            use_checkpoint=not args.no_checkpoint, use_torch=use_torch,
        )

    print(f"\n{'='*70}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
