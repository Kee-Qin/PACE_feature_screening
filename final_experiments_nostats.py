from src.screening_utilities import compute_scores, compute_many, embedding_scores, layered_scores, pre_split_scores, raw_a_scores, selector_call
PACE_AGGREGATION = "native"
from src import simulation_protocol as protocol
import sys
import os
import json
import warnings
import argparse
import time
import numpy as np
import torch
sys.modules.setdefault('final_experiments_nostats', sys.modules.get('__main__'))
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import io as _io
import argparse as _argparse, inspect as _inspect, sys as _sys
from src.data_generators_classification import EXPERIMENTS as CLS_EXPERIMENTS, \
    EXPERIMENT_NAMES as CLS_NAMES
from src.screening_methods import ScreeningEvaluator
from src.screening_methods_torch import DEVICE, set_device, TorchAcceleratedScreening
from src.data_generators_regression import EXPERIMENTS as REG_EXPERIMENTS, \
    EXPERIMENT_NAMES as REG_NAMES

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================================================


# ==========================================================================
from src import screening_methods_torch_A as pace_a
_PaceA = pace_a
N_REPS = 50
BASELINE_TRAD_CLS = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                      'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor', 'ids']
BASELINE_TRAD_REG = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hd', 'gcor', 'ids']
PAPER_CLS_SCENARIOS = protocol.CLASSIFICATION
PAPER_REG_SCENARIOS = protocol.REGRESSION
EXP_HASH = {k: 37 for k in list(CLS_EXPERIMENTS.keys()) + list(REG_EXPERIMENTS.keys())}


# ======================================================================

# ======================================================================


def _unpack_gen(R, exp_key=None):
    X, y, tf, n, p, s = R[:6]
    cat_indices = None; missing_mask = None
    if len(R) >= 7:
        extra = R[6]
        if isinstance(extra, dict) and 'cat_indices' in extra:
            cat_indices = list(extra['cat_indices'])
        elif isinstance(extra, np.ndarray):
            missing_mask = extra
    return np.asarray(X), np.asarray(y), list(tf), int(n), int(p), int(s), cat_indices, missing_mask


# ======================================================================

# ======================================================================


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

    def make_key(self, exp_key, ps, rep):
        return f"{exp_key}_ps{ps}_rep{rep}"

    def is_completed(self, key):
        return key in self.data

    def get_result(self, key):
        return self.data.get(key)

    def save_result(self, key, result):
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

    def is_complete(self, exp_key, ps, n_reps, methods, emb_layers=None):
        if not self.manifest_path.exists():
            return False
        try:
            with open(self.manifest_path, 'r') as f:
                mf = json.load(f)
            if mf.get('n_reps') != n_reps:
                return False
            if mf.get('completed_reps', 0) < n_reps:
                return False
            mf_methods = set(mf.get('methods', []))
            mf_methods_norm = {pace_a._EMB_LEGACY_TO_L3.get(k, k) for k in mf_methods}
            methods_norm = {pace_a._EMB_LEGACY_TO_L3.get(k, k) for k in methods}
            if not methods_norm.issubset(mf_methods_norm):
                return False
            if emb_layers is not None:
                mf_layers = set(mf.get('emb_layers', []))


                if mf_layers:
                    if not set(emb_layers).issubset(mf_layers):
                        return False
            return True
        except Exception:
            return False

    def update_manifest(self, exp_key, ps, n_reps, completed_reps, methods, emb_layers=None):

        norm = []
        for m in methods:
            mm = pace_a._EMB_LEGACY_TO_L3.get(m, m)
            if mm not in norm:
                norm.append(mm)
        mf = {
            'exp_key': exp_key, 'param_set': ps, 'n_reps': n_reps,
            'completed_reps': completed_reps,
            'methods': sorted(norm),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        if emb_layers is not None:
            mf['emb_layers'] = sorted(int(L) for L in emb_layers)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(mf, f, indent=2, ensure_ascii=False)

# CheckpointManagerE (E; B.CheckpointManager → CheckpointManager)
class CheckpointManagerE(CheckpointManager):
    def update_manifest(self, ek, ps, n_reps, completed, methods):
        mf = {'exp_key': ek, 'param_set': ps, 'n_reps': n_reps, 'completed_reps': completed,
              'methods': sorted(methods), 'track': 'E',
              'score_semantics': 'learned_score_or_budget',
              'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(mf, f, indent=2, ensure_ascii=False)

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pace_results_classification")

def _preprocess_X(X, cat_indices=None, missing_mask=None):
    X = np.array(X, dtype=np.float64, copy=True)
    cat_indices = list(cat_indices) if cat_indices else []
    if (missing_mask is not None and np.isnan(X).any()) or np.isnan(X).any():
        for j in range(X.shape[1]):
            col = X[:, j]
            nan_j = np.isnan(col)
            if not nan_j.any():
                continue
            if j in cat_indices:
                col[nan_j] = -1
            else:
                med = np.nanmedian(col[~nan_j])
                col[nan_j] = med if not np.isnan(med) else 0.0
            X[:, j] = col
    return X, cat_indices

def run_single_rep_A(X, y, true_features, s, p, n,
                     track_methods: List[str],
                     cat_indices=None, missing_mask=None,
                     include_baseline: bool = True,
                     embedder=None, cached_res=None,
                     emb_layers: List[int] = None) -> Dict:
    if emb_layers is None:
        emb_layers = pace_a.EMB_LAYERS


    emb_layers_set = set(int(L) for L in emb_layers)
    norm_methods = []
    emb_to_run_keys = set()
    for m in track_methods:
        if m in pace_a.TRACK_A_DISPATCH:                # raw
            norm_methods.append(m); continue
        if m in pace_a._EMB_ALL_LEGACY:                  # emb_<base> = L3
            base = m.replace('emb_', '')
            if 3 in emb_layers_set:
                emb_to_run_keys.add(f'emb_{base}_L3')
            continue
        if m.startswith('emb_') and m.endswith(('_L1', '_L2', '_L3')):
            L = int(m.split('_L')[-1]); base = m[4:].rsplit('_L', 1)[0]
            if L in emb_layers_set:
                emb_to_run_keys.add(f'emb_{base}_L{L}')
            continue
        norm_methods.append(m)

    results = {}
    if cached_res:

        for k, v in cached_res.items():
            if k.startswith('_') or not isinstance(v, dict) or 'mms' not in v:
                continue
            kk = pace_a._EMB_LEGACY_TO_L3.get(k, k)
            if kk in emb_to_run_keys:
                results[kk] = v
            elif kk in pace_a.TRACK_A_DISPATCH or kk in set(norm_methods):

                results[kk] = v
            elif kk not in emb_to_run_keys and not kk.startswith('emb_'):
                results[kk] = v

        for k, v in cached_res.items():
            if k in emb_to_run_keys and isinstance(v, dict) and 'mms' in v and k not in results:
                results[k] = v
    Xp, cat_idx = _preprocess_X(X, cat_indices, missing_mask)


    done = set(results.keys())
    raw_to_run = [m for m in track_methods if m in pace_a.TRACK_A_DISPATCH and m not in done]

    emb_needed = [k for k in sorted(emb_to_run_keys) if k not in done]
    ta_scores = {}
    for m in raw_to_run:
        try:
            sc = compute_scores(Xp, y, "raw_a_"+m, impute=False)
            if sc is None:
                continue
            sc = np.asarray(sc, dtype=np.float64)
            if not np.isnan(sc).all():
                ta_scores[m] = sc
        except Exception as e:
            print(f"  [TrackA] {m} failed (logged as missing): {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for method, scores in ta_scores.items():
        eval_res = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
        results[method] = {
            'mms': eval_res['mms'],
            'tpr_n1': eval_res['tpr_n1'],
            'fpr_n1': eval_res['fpr_n1'],
            'tpr_nlogn': eval_res['tpr_nlogn'],
            'fpr_nlogn': eval_res['fpr_nlogn'],
        }


    if emb_needed and pace_a._HAS_PACEA_A and embedder is not None and embedder.loaded:

        bases_needed = sorted({k[4:].rsplit('_L', 1)[0] for k in emb_needed})
        layers_needed = sorted({int(k.split('_L')[-1]) for k in emb_needed})
        try:
            if p > 10000 and hasattr(_PaceA, 'compute_paceA_emb_scores_incremental'):

                emb_scores = layered_scores(Xp, y, embedder, emb_layers=layers_needed, methods=bases_needed, profile="track_a", aggregation=PACE_AGGREGATION)
            else:
                emb_scores = layered_scores(Xp, y, embedder, emb_layers=layers_needed, methods=bases_needed, batch_size=64, profile="track_a", aggregation=PACE_AGGREGATION)
        except Exception as e:
            print(f"  [TrackA] Multi-layer embedding failed; embedding scores unavailable: {e}", flush=True)
            emb_scores = {}
        for method_key in emb_needed:
            scores = emb_scores.get(method_key)
            if scores is None:
                continue
            eval_res = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method_key] = {
                'mms': eval_res['mms'],
                'tpr_n1': eval_res['tpr_n1'],
                'fpr_n1': eval_res['fpr_n1'],
                'tpr_nlogn': eval_res['tpr_nlogn'],
                'fpr_nlogn': eval_res['fpr_nlogn'],
            }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif emb_needed and not pace_a._HAS_PACEA_A:
        print(f"  [TrackA] screening_methods_torch_A unavailable; embedding scores unavailable", flush=True)
    elif emb_needed and (embedder is None or not getattr(embedder, 'loaded', False)):
        print(f"  [TrackA] Embedder unavailable; embedding scores unavailable", flush=True)


    if include_baseline and (cached_res is None):
        use_pc = n <= 500
        trad = [m for m in BASELINE_TRAD_CLS if use_pc or m != 'pc']
        try:
            trad_scores = compute_many(Xp, y, methods=trad, impute=False)
        except Exception as e:
            print(f"  [Baseline] compute_all_scores failed: {e}", flush=True)
            trad_scores = {}
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

    return results

def run_experiment_A(exp_key: str, param_sets: List[int], n_reps: int,
                     output_dir: str, track_methods: List[str],
                     include_baseline: bool = True, use_checkpoint: bool = True,
                     n_jobs: int = -1, embedder=None, emb_layers: List[int] = None):
    if emb_layers is None:
        emb_layers = pace_a.EMB_LAYERS
    gen_func = CLS_EXPERIMENTS[exp_key]
    exp_name = CLS_NAMES.get(exp_key, exp_key)
    exp_hash = EXP_HASH.get(exp_key, 37)
    is_ultra = (exp_key == 'exp4')

    print(f"\n{'=' * 70}")
    print(f"Track-A Experiment: {exp_key} — {exp_name}")
    print(f"Param sets: {param_sets}, Reps: {n_reps}, emb_layers: {emb_layers}")
    print(f"Track-A methods: {track_methods}")
    print(f"Include baseline trad: {include_baseline}"
          + ("  [ULTRA-HIGH-DIM]" if is_ultra else "")
          + (f"  [emb] embedder loaded={getattr(embedder,'loaded',False)}" if embedder else "  [emb] embedder=None"))
    print(f"{'=' * 70}", flush=True)

    all_methods = list(track_methods)
    if include_baseline:
        all_methods = all_methods + [m for m in BASELINE_TRAD_CLS]

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointManager(str(ps_dir / "checkpoint.json"),
                                       manifest_path=str(ps_dir / "manifest.json"))

        if use_checkpoint and checkpoint.is_complete(exp_key, ps, n_reps, all_methods, emb_layers=emb_layers):
            print(f"  [{exp_key}/ps{ps}] Already complete ({n_reps} reps, "
                  f"{len(all_methods)} methods, emb_layers={emb_layers}) — SKIPPED", flush=True)
            all_results = [checkpoint.get_result(checkpoint.make_key(exp_key, ps, rep))
                           for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 8
            summary = summarize_results_A(all_results, s)
            save_summary_A(summary, ps_dir, track_methods, include_baseline)
            save_raw_results_A(all_results, ps_dir)
            continue

        all_results = []
        newly_completed = 0
        for rep in range(1, n_reps + 1):
            key = checkpoint.make_key(exp_key, ps, rep)
            cached_res = None
            run_needed = True
            if use_checkpoint and checkpoint.is_completed(key):
                res = checkpoint.get_result(key)

                cached_methods = set(k for k in res.keys()
                                     if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                cached_methods_norm = {pace_a._EMB_LEGACY_TO_L3.get(k, k) for k in cached_methods}
                all_methods_norm = {pace_a._EMB_LEGACY_TO_L3.get(k, k) for k in all_methods}
                if all_methods_norm.issubset(cached_methods_norm):
                    all_results.append(res)
                    continue

                cached_res = res

            seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=seed)
            X, y, true_features, n, p, s, cat_idx, mask = _unpack_gen(R, exp_key)

            res = run_single_rep_A(X, y, true_features, s, p, n,
                                   track_methods=track_methods,
                                   cat_indices=cat_idx, missing_mask=mask,
                                   include_baseline=include_baseline and (cached_res is None),
                                   embedder=embedder, cached_res=cached_res,
                                   emb_layers=emb_layers)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep,
                            'n': n, 'p': p, 's': s}
            if use_checkpoint:
                checkpoint.save_result(key, res)
            all_results.append(res)
            newly_completed += 1
            if newly_completed % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly_completed})"
                      + (" [ultra]" if is_ultra else ""), flush=True)

        completed_reps = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed_reps >= n_reps:


            rep_method_sets = []
            for r in all_results:
                if r is None:
                    continue
                ms = {k for k in r.keys() if not k.startswith('_')
                      and isinstance(r[k], dict) and 'mms' in r[k]}
                rep_method_sets.append(ms)
            observed = (set.intersection(*rep_method_sets)
                        if rep_method_sets else set(all_methods))
            checkpoint.update_manifest(exp_key, ps, n_reps, completed_reps,
                                       sorted(observed) if observed else list(all_methods),
                                       emb_layers=emb_layers)

        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 8
        summary = summarize_results_A(all_results, s)
        save_summary_A(summary, ps_dir, track_methods, include_baseline)
        save_raw_results_A(all_results, ps_dir)
        cached = n_reps - newly_completed
        print(f"  [{exp_key}/ps{ps}] Done — cached: {cached}, new: {newly_completed}",
              flush=True)

def summarize_results_A(all_results: List[Dict], s: int) -> Dict:
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

def save_summary_A(summary: Dict, output_dir: Path,
                   track_methods: List[str], include_baseline: bool):
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"{'Method':<18} {'Track':<6} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
                 f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}")
    lines.append("-" * 102)


    trad_order = BASELINE_TRAD_CLS
    order = (trad_order + track_methods) if include_baseline else list(track_methods)

    for method in order:
        if method not in summary:
            continue
        d = summary[method]
        tag = 'A' if method in track_methods else 'base'
        iqr_str = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(
            f"{method:<18} {tag:<6} {d['mms_median']:>10.1f} {iqr_str:>12} {d['ptp']:>6.2f} "
            f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
            f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}"
        )

    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_raw_results_A(all_results: List[Dict], output_dir: Path):
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
            elif k == '_meta':
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
        clean.append(entry)
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

BASE_OUTPUT_DIR_REG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pace_results_regression")

EXP_HASH_REG = {k: protocol.seed_offset('regression', k) for k in REG_EXPERIMENTS}

def _preprocess_X_reg(X: np.ndarray, cat_indices: Optional[List[int]] = None,
                  missing_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[int]]:
    X = np.array(X, dtype=np.float64, copy=True)
    cat_indices = list(cat_indices) if cat_indices else []
    if missing_mask is not None and np.isnan(X).any():
        for j in range(X.shape[1]):
            col = X[:, j]
            nan_j = np.isnan(col)
            if not nan_j.any():
                continue
            if j in cat_indices:

                col[nan_j] = -1
            else:
                med = np.nanmedian(col[~nan_j])
                if np.isnan(med):
                    med = 0.0
                col[nan_j] = med
            X[:, j] = col
    return X, cat_indices

def run_single_rep_A_reg(X, y, true_features, s, p, n, track_methods,
                     cat_indices=None, missing_mask=None,
                     include_baseline=True, embedder=None, cached_res=None,
                     emb_layers=None):
    if emb_layers is None:
        emb_layers = pace_a.EMB_LAYERS


    emb_layers_set = set(int(L) for L in emb_layers)
    emb_to_run_keys = set()
    for m in track_methods:
        if m in pace_a.TRACK_A_DISPATCH_REG:
            continue
        if m in pace_a._EMB_ALL_LEGACY:                  # emb_<base> = L3
            base = m.replace('emb_', '')
            if 3 in emb_layers_set:
                emb_to_run_keys.add(f'emb_{base}_L3')
            continue
        if m.startswith('emb_') and m.endswith(('_L1', '_L2', '_L3')):
            L = int(m.split('_L')[-1]); base = m[4:].rsplit('_L', 1)[0]
            if L in emb_layers_set:
                emb_to_run_keys.add(f'emb_{base}_L{L}')

    results = {}
    if cached_res:

        for k, v in cached_res.items():
            if k.startswith('_') or not isinstance(v, dict) or 'mms' not in v:
                continue
            kk = pace_a._EMB_LEGACY_TO_L3.get(k, k)
            if kk in emb_to_run_keys:
                results[kk] = v
            elif not kk.startswith('emb_'):
                results[kk] = v
        for k, v in cached_res.items():
            if k in emb_to_run_keys and isinstance(v, dict) and 'mms' in v and k not in results:
                results[k] = v

    Xp, cat_idx = _preprocess_X_reg(X, cat_indices, missing_mask)


    done = set(results.keys())
    raw_to_run = [m for m in track_methods if m in pace_a.TRACK_A_DISPATCH_REG and m not in done]
    emb_needed = [k for k in sorted(emb_to_run_keys) if k not in done]

    ta = raw_a_scores(Xp, y, methods=raw_to_run, cat_indices=cat_idx)
    for method, scores in ta.items():
        ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
        results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}


    if emb_needed and pace_a._HAS_PACEA_REG_A and embedder is not None and embedder.loaded:
        bases_needed = sorted({k[4:].rsplit('_L', 1)[0] for k in emb_needed})
        layers_needed = sorted({int(k.split('_L')[-1]) for k in emb_needed})
        try:
            if p > 10000 and hasattr(_PaceA, 'compute_paceA_emb_scores_incremental'):
                emb_scores = layered_scores(Xp, y, embedder, emb_layers=layers_needed, methods=bases_needed, profile="track_a", aggregation=PACE_AGGREGATION)
            else:
                emb_scores = layered_scores(Xp, y, embedder, emb_layers=layers_needed, methods=bases_needed, batch_size=64, profile="track_a", aggregation=PACE_AGGREGATION)
        except Exception as e:
            print(f"  [TrackA-reg] Multi-layer embedding failed; embedding scores unavailable: {e}", flush=True)
            emb_scores = {}
        for method_key in emb_needed:
            scores = emb_scores.get(method_key)
            if scores is None:
                continue
            ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method_key] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif emb_needed and not pace_a._HAS_PACEA_REG_A:
        print(f"  [TrackA-reg] screening_methods_torch_A unavailable; embedding scores unavailable", flush=True)
    elif emb_needed and (embedder is None or not getattr(embedder, 'loaded', False)):
        print(f"  [TrackA-reg] Embedder unavailable; embedding scores unavailable", flush=True)


    if include_baseline and (cached_res is None):
        use_pc = n <= 500
        trad = [m for m in BASELINE_TRAD_REG if use_pc or m != 'pc']
        try:
            ts = compute_many(Xp, y, methods=trad, impute=False)
        except Exception as e:
            print(f"  [Baseline] compute_all_scores failed: {e}", flush=True)
            ts = {}
        for method, scores in ts.items():
            if scores is None:
                continue
            ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}
    return results

def run_experiment_A_reg(exp_key, param_sets, n_reps, output_dir, track_methods,
                     include_baseline=True, use_checkpoint=True, n_jobs=-1, embedder=None,
                     emb_layers=None):
    if emb_layers is None:
        emb_layers = pace_a.EMB_LAYERS
    gen_func = REG_EXPERIMENTS[exp_key]
    exp_name = REG_NAMES.get(exp_key, exp_key)
    exp_hash = EXP_HASH_REG.get(exp_key, 37)

    print(f"\n{'=' * 70}")
    print(f"Track-A Regression Experiment: {exp_key} — {exp_name}")
    print(f"Param sets: {param_sets}, Reps: {n_reps}, emb_layers: {emb_layers}")
    print(f"Track-A methods: {track_methods}, include_baseline: {include_baseline}"
          + (f"  [emb] embedder loaded={getattr(embedder,'loaded',False)}" if embedder else "  [emb] embedder=None"))
    print(f"{'=' * 70}", flush=True)

    all_methods = list(track_methods) + (BASELINE_TRAD_REG if include_baseline else [])

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)
        ck = CheckpointManager(str(ps_dir / "checkpoint.json"),
                              manifest_path=str(ps_dir / "manifest.json"))
        if use_checkpoint and ck.is_complete(exp_key, ps, n_reps, all_methods, emb_layers=emb_layers):
            print(f"  [{exp_key}/ps{ps}] Already complete ({n_reps} reps, "
                  f"{len(all_methods)} methods, emb_layers={emb_layers}) — SKIPPED", flush=True)
            all_results = [ck.get_result(ck.make_key(exp_key, ps, rep)) for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
            summary = summarize_results_A_reg(all_results, s)
            save_summary_A_reg(summary, ps_dir, track_methods, include_baseline)
            save_raw_results_A_reg(all_results, ps_dir)
            continue

        all_results = []
        newly = 0

        is_ultra = (exp_key == 'exp2')
        for rep in range(1, n_reps + 1):
            key = ck.make_key(exp_key, ps, rep)
            cached_res = None
            if use_checkpoint and ck.is_completed(key):
                res = ck.get_result(key)
                cached = set(k for k in res.keys() if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                cached_norm = {pace_a._EMB_LEGACY_TO_L3.get(k, k) for k in cached}
                all_norm = {pace_a._EMB_LEGACY_TO_L3.get(k, k) for k in all_methods}
                if all_norm.issubset(cached_norm):
                    all_results.append(res); continue

                cached_res = res
            seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=seed)
            X, y, tf, n, p, s, cat_idx, mask = _unpack_gen(R, exp_key)
            res = run_single_rep_A_reg(X, y, tf, s, p, n, track_methods=track_methods,
                                   cat_indices=cat_idx, missing_mask=mask,
                                   include_baseline=include_baseline and (cached_res is None),
                                   embedder=embedder, cached_res=cached_res,
                                   emb_layers=emb_layers)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'n': n, 'p': p, 's': s}
            if use_checkpoint:
                ck.save_result(key, res)
            all_results.append(res); newly += 1
            if newly % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly})"
                      + (" [ultra-high-dim]" if is_ultra else ""), flush=True)
        completed = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed >= n_reps:


            rep_method_sets = []
            for r in all_results:
                if r is None:
                    continue
                ms = {k for k in r.keys() if not k.startswith('_')
                      and isinstance(r[k], dict) and 'mms' in r[k]}
                rep_method_sets.append(ms)
            observed = (set.intersection(*rep_method_sets)
                        if rep_method_sets else set(all_methods))
            ck.update_manifest(exp_key, ps, n_reps, completed,
                               sorted(observed) if observed else list(all_methods),
                               emb_layers=emb_layers)
        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
        summary = summarize_results_A_reg(all_results, s)
        save_summary_A_reg(summary, ps_dir, track_methods, include_baseline)
        save_raw_results_A_reg(all_results, ps_dir)
        print(f"  [{exp_key}/ps{ps}] Done — cached: {n_reps - newly}, new: {newly}", flush=True)

def summarize_results_A_reg(all_results, s):
    if not all_results:
        return {}
    mm = {}
    for res in all_results:
        for key, val in res.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict) and 'mms' in val:
                mm.setdefault(key, {k: [] for k in val.keys()})
                for k, v in val.items():
                    mm[key][k].append(v)
    summary = {}
    for method, metrics in mm.items():
        mms = np.array(metrics['mms'])
        summary[method] = {
            'mms_median': float(np.median(mms)),
            'mms_q1': float(np.percentile(mms, 25)),
            'mms_q3': float(np.percentile(mms, 75)),
            'mms_mean': float(np.mean(mms)),
            'mms_std': float(np.std(mms)),
            'ptp': float(np.mean(mms <= s)),
            'tpr_n1_mean': float(np.mean(metrics.get('tpr_n1', [0]))),
            'fpr_n1_mean': float(np.mean(metrics.get('fpr_n1', [0]))),
            'tpr_nlogn_mean': float(np.mean(metrics.get('tpr_nlogn', [0]))),
            'fpr_nlogn_mean': float(np.mean(metrics.get('fpr_nlogn', [0]))),
        }
    return summary

def save_summary_A_reg(summary, output_dir, track_methods, include_baseline):
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = [f"{'Method':<18} {'Track':<6} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
             f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}",
             "-" * 102]
    order = (BASELINE_TRAD_REG + track_methods) if include_baseline else list(track_methods)
    for m in order:
        if m not in summary:
            continue
        d = summary[m]
        tag = 'A' if m in track_methods else 'base'
        iqr = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(f"{m:<18} {tag:<6} {d['mms_median']:>10.1f} {iqr:>12} {d['ptp']:>6.2f} "
                      f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
                      f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}")
    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_raw_results_A_reg(all_results, output_dir):
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
            elif k == '_meta':
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
        clean.append(entry)
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

def run_single_rep_B(X, y, true_features, s, p, n, task, track_methods,
                      cat_indices=None, missing_mask=None, include_baseline=True,
                      cached_res=None):
    results = {}
    if cached_res:
        for k, v in cached_res.items():
            results[k] = v
    cat = list(cat_indices) if cat_indices else []
    Xf, _ = pace_a._fill_missing(X, cat)
    from sklearn.model_selection import train_test_split
    is_cls = (len(np.unique(y)) < n / 2)
    strat = y if (is_cls and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xval, ytr, yval = train_test_split(Xf, y, test_size=0.35, random_state=0, stratify=strat) if n >= 12 \
        else (Xf, Xf, y, y)


    track_to_run = [m for m in track_methods if m not in results]
    if track_to_run:
        ta = pre_split_scores(Xtr, ytr, Xval, yval, p, task, cat, track_to_run)
        for method, scores in ta.items():
            ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}

    if include_baseline and pace_a._HAS_TORCH_BASELINE:
        use_pc = n <= 500
        baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG

        trad = [m for m in baseline if (use_pc or m != 'pc') and m not in results]
        if trad:
            try:
                ts = compute_many(Xf, y, methods=trad, impute=False)
            except Exception as e:
                print(f"  [Baseline] failed: {e}", flush=True); ts = {}
            for method, scores in ts.items():
                if scores is None: continue
                ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
                results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}
    return results

def run_experiment_B(exp_key, task, param_sets, n_reps, output_dir, track_methods,
                      include_baseline=True, use_checkpoint=True):
    if task == 'classification':
        gen_func = CLS_EXPERIMENTS[exp_key]; name = CLS_NAMES.get(exp_key, exp_key)
    else:
        gen_func = REG_EXPERIMENTS[exp_key]; name = REG_NAMES.get(exp_key, exp_key)
    exp_hash = protocol.seed_offset(task, exp_key)
    baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
    all_methods = list(track_methods) + (baseline if include_baseline else [])

    print(f"\n{'=' * 70}")
    print(f"Track-B {task} Experiment: {exp_key} — {name}")
    print(f"Param sets: {param_sets}, Reps: {n_reps}")
    print(f"Track-B methods: {track_methods}, include_baseline: {include_baseline}")
    print(f"{'=' * 70}", flush=True)

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)
        ck = CheckpointManager(str(ps_dir / "checkpoint.json"),
                               manifest_path=str(ps_dir / "manifest.json"))
        if use_checkpoint and ck.is_complete(exp_key, ps, n_reps, all_methods):
            print(f"  [{exp_key}/ps{ps}] Already complete — SKIPPED", flush=True)
            all_results = [ck.get_result(ck.make_key(exp_key, ps, rep)) for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
            summary = summarize_results_B(all_results, s)
            save_summary_B(summary, ps_dir, track_methods, include_baseline, task)
            save_raw_results_B(all_results, ps_dir)
            continue
        all_results = []; newly = 0
        for rep in range(1, n_reps + 1):
            key = ck.make_key(exp_key, ps, rep)
            cached_res_to_use = None
            if use_checkpoint and ck.is_completed(key):
                res = ck.get_result(key)
                cached = set(k for k in res.keys() if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                if set(all_methods).issubset(cached):
                    all_results.append(res); continue


                print(f"  [{exp_key}/ps{ps}/rep{rep}] Computing missing method scores", flush=True)
                cached_res_to_use = res
            seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=seed)
            X, y, tf, n, p, s, cat_idx, mask = _unpack_gen(R)
            res = run_single_rep_B(X, y, tf, s, p, n, task, track_methods,
                                     cat_indices=cat_idx, missing_mask=mask,
                                     include_baseline=include_baseline,
                                     cached_res=cached_res_to_use)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'n': n, 'p': p, 's': s, 'task': task}
            if use_checkpoint: ck.save_result(key, res)
            all_results.append(res); newly += 1
            if newly % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly})", flush=True)
        completed = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed >= n_reps:
            ck.update_manifest(exp_key, ps, n_reps, completed, all_methods)
        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
        summary = summarize_results_B(all_results, s)
        save_summary_B(summary, ps_dir, track_methods, include_baseline, task)
        save_raw_results_B(all_results, ps_dir)
        print(f"  [{exp_key}/ps{ps}] Done — cached: {n_reps - newly}, new: {newly}", flush=True)

def summarize_results_B(all_results, s):
    if not all_results: return {}
    mm = {}
    for res in all_results:
        for key, val in res.items():
            if key.startswith('_'): continue
            if isinstance(val, dict) and 'mms' in val:
                mm.setdefault(key, {k: [] for k in val.keys()})
                for k, v in val.items(): mm[key][k].append(v)
    out = {}
    for method, M in mm.items():
        mms = np.array(M['mms'])
        out[method] = {
            'mms_median': float(np.median(mms)), 'mms_q1': float(np.percentile(mms, 25)),
            'mms_q3': float(np.percentile(mms, 75)), 'mms_mean': float(np.mean(mms)),
            'mms_std': float(np.std(mms)), 'ptp': float(np.mean(mms <= s)),
            'tpr_n1_mean': float(np.mean(M.get('tpr_n1', [0]))),
            'fpr_n1_mean': float(np.mean(M.get('fpr_n1', [0]))),
            'tpr_nlogn_mean': float(np.mean(M.get('tpr_nlogn', [0]))),
            'fpr_nlogn_mean': float(np.mean(M.get('fpr_nlogn', [0]))),
        }
    return out

def save_summary_B(summary, output_dir, track_methods, include_baseline, task):
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = [f"{'Method':<14} {'Trk':<4} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
             f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}",
             "-" * 98]
    baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
    order = (baseline + track_methods) if include_baseline else list(track_methods)
    for m in order:
        if m not in summary: continue
        d = summary[m]; tag = 'B' if m in track_methods else 'base'
        iqr = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(f"{m:<14} {tag:<4} {d['mms_median']:>10.1f} {iqr:>12} {d['ptp']:>6.2f} "
                      f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
                      f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}")
    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_raw_results_B(all_results, output_dir):
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                float(vv) if isinstance(vv, (np.floating,)) else vv) for kk, vv in v.items()}
            elif k == '_meta':
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv) for kk, vv in v.items()}
        clean.append(entry)
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

def run_single_rep_C(X, y, true_features, s, p, n, task, track_methods,
                      cat_indices=None, missing_mask=None, include_baseline=True,
                      cached_res=None):
    results = {}
    if cached_res:

        for k, v in cached_res.items():
            if (not k.startswith('_')) and isinstance(v, dict) and 'mms' in v:
                results[k] = v
    cat = list(cat_indices) if cat_indices else []
    Xf, _ = pace_a._fill_missing(X, cat)
    Xtr, ytr, Xval, yval, has_val = pace_a._train_val_split(Xf, y, n, random_state=0)


    if cached_res:
        done_methods = set(k for k in results.keys())
        methods_to_run = [m for m in track_methods if m not in done_methods]
    else:
        methods_to_run = list(track_methods)
    if methods_to_run:
        ta = pre_split_scores(Xtr, ytr, Xval if has_val else None,
                                     yval if has_val else None, p, task, cat, methods_to_run)
        for method, scores in ta.items():
            ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}

    if include_baseline and pace_a._HAS_TORCH_BASELINE:

        use_pc = n <= 500
        baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
        trad = [m for m in baseline if use_pc or m != 'pc']
        try:
            ts = compute_many(Xf, y, methods=trad, impute=False)
        except Exception as e:
            print(f"  [Baseline] compute_all_scores failed: {e}", flush=True)
            ts = {}
        for method, scores in ts.items():
            if scores is None:
                continue
            ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}
    return results

def run_experiment_C(exp_key, task, param_sets, n_reps, output_dir, track_methods,
                      include_baseline=True, use_checkpoint=True):
    if task == 'classification':
        gen_func = CLS_EXPERIMENTS[exp_key]
        name = CLS_NAMES.get(exp_key, exp_key)
    else:
        gen_func = REG_EXPERIMENTS[exp_key]
        name = REG_NAMES.get(exp_key, exp_key)
    exp_hash = protocol.seed_offset(task, exp_key)
    baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
    all_methods = list(track_methods) + (baseline if include_baseline else [])

    print(f"\n{'=' * 70}")
    print(f"Track-C {task} Experiment: {exp_key} — {name}")
    print(f"Param sets: {param_sets}, Reps: {n_reps}")
    print(f"Track-C methods: {track_methods}, include_baseline: {include_baseline}")
    print(f"{'=' * 70}", flush=True)

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)
        ck = CheckpointManager(str(ps_dir / "checkpoint.json"),
                               manifest_path=str(ps_dir / "manifest.json"))
        if use_checkpoint and ck.is_complete(exp_key, ps, n_reps, all_methods):
            print(f"  [{exp_key}/ps{ps}] Already complete ({n_reps} reps) — SKIPPED", flush=True)
            all_results = [ck.get_result(ck.make_key(exp_key, ps, rep)) for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
            summary = summarize_results_C(all_results, s)
            save_summary_C(summary, ps_dir, track_methods, include_baseline, task)
            save_raw_results_C(all_results, ps_dir)
            continue

        all_results = []
        newly = 0
        for rep in range(1, n_reps + 1):
            key = ck.make_key(exp_key, ps, rep)
            cached_res = None
            run_needed = True
            if use_checkpoint and ck.is_completed(key):
                res = ck.get_result(key)
                cached = set(k for k in res.keys() if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                if set(all_methods).issubset(cached):
                    all_results.append(res); continue

                cached_res = res

            seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=seed)
            X, y, tf, n, p, s, cat_idx, mask = _unpack_gen(R)
            res = run_single_rep_C(X, y, tf, s, p, n, task, track_methods,
                                     cat_indices=cat_idx, missing_mask=mask,
                                     include_baseline=include_baseline and not cached_res,
                                     cached_res=cached_res)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'n': n, 'p': p, 's': s,
                            'task': task}
            if use_checkpoint:
                ck.save_result(key, res)
            all_results.append(res); newly += 1
            if newly % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly})", flush=True)
        completed = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed >= n_reps:
            ck.update_manifest(exp_key, ps, n_reps, completed, all_methods)
        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
        summary = summarize_results_C(all_results, s)
        save_summary_C(summary, ps_dir, track_methods, include_baseline, task)
        save_raw_results_C(all_results, ps_dir)
        print(f"  [{exp_key}/ps{ps}] Done — cached: {n_reps - newly}, new: {newly}", flush=True)

def summarize_results_C(all_results, s):
    if not all_results:
        return {}
    mm = {}
    for res in all_results:
        for key, val in res.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict) and 'mms' in val:
                mm.setdefault(key, {k: [] for k in val.keys()})
                for k, v in val.items():
                    mm[key][k].append(v)
    out = {}
    for method, M in mm.items():
        mms = np.array(M['mms'])
        out[method] = {
            'mms_median': float(np.median(mms)), 'mms_q1': float(np.percentile(mms, 25)),
            'mms_q3': float(np.percentile(mms, 75)), 'mms_mean': float(np.mean(mms)),
            'mms_std': float(np.std(mms)), 'ptp': float(np.mean(mms <= s)),
            'tpr_n1_mean': float(np.mean(M.get('tpr_n1', [0]))),
            'fpr_n1_mean': float(np.mean(M.get('fpr_n1', [0]))),
            'tpr_nlogn_mean': float(np.mean(M.get('tpr_nlogn', [0]))),
            'fpr_nlogn_mean': float(np.mean(M.get('fpr_nlogn', [0]))),
        }
    return out

def save_summary_C(summary, output_dir, track_methods, include_baseline, task):
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = [f"{'Method':<18} {'Trk':<4} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
             f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}",
             "-" * 102]
    baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
    order = (baseline + track_methods) if include_baseline else list(track_methods)
    for m in order:
        if m not in summary:
            continue
        d = summary[m]
        tag = 'C' if m in track_methods else 'base'
        iqr = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(f"{m:<18} {tag:<4} {d['mms_median']:>10.1f} {iqr:>12} {d['ptp']:>6.2f} "
                      f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
                      f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}")
    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_raw_results_C(all_results, output_dir):
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
            elif k == '_meta':
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
        clean.append(entry)
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

def run_single_rep_D(X, y, true_features, s, p, n, task, track_methods,
                     cat_indices=None, missing_mask=None, include_baseline=True):
    results = {}
    cat = list(cat_indices) if cat_indices else []
    Xf, _ = pace_a._fill_missing(X, cat)
    Xtr, ytr, Xval, yval, has_val = pace_a._train_val_split(Xf, y, n, random_state=0)

    td = pre_split_scores(Xtr, ytr, Xval if has_val else None,
                               yval if has_val else None, p, task, cat, track_methods)
    for method, scores in td.items():
        ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
        results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}

    if include_baseline and pace_a._HAS_TORCH_BASELINE:
        use_pc = n <= 500
        baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
        trad = [m for m in baseline if use_pc or m != 'pc']
        try:
            ts = compute_many(Xf, y, methods=trad, impute=False)
        except Exception as e:
            print(f"  [Baseline] compute_all_scores failed: {e}", flush=True)
            ts = {}
        for method, scores in ts.items():
            if scores is None:
                continue
            ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
            results[method] = {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}
    return results

def run_experiment_D(exp_key, task, param_sets, n_reps, output_dir, track_methods,
                     include_baseline=True, use_checkpoint=True):
    if task == 'classification':
        gen_func = CLS_EXPERIMENTS[exp_key]
        name = CLS_NAMES.get(exp_key, exp_key)
    else:
        gen_func = REG_EXPERIMENTS[exp_key]
        name = REG_NAMES.get(exp_key, exp_key)
    exp_hash = protocol.seed_offset(task, exp_key)
    baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
    all_methods = list(track_methods) + (baseline if include_baseline else [])

    print(f"\n{'=' * 70}")
    print(f"Track-D {task} Experiment: {exp_key} — {name}")
    print(f"Param sets: {param_sets}, Reps: {n_reps}")
    print(f"Track-D methods: {track_methods}, include_baseline: {include_baseline}")
    print(f"{'=' * 70}", flush=True)

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)
        ck = CheckpointManager(str(ps_dir / "checkpoint.json"),
                               manifest_path=str(ps_dir / "manifest.json"))
        if use_checkpoint and ck.is_complete(exp_key, ps, n_reps, all_methods):
            print(f"  [{exp_key}/ps{ps}] Already complete ({n_reps} reps) — SKIPPED", flush=True)
            all_results = [ck.get_result(ck.make_key(exp_key, ps, rep)) for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
            summary = summarize_results_D(all_results, s)
            save_summary_D(summary, ps_dir, track_methods, include_baseline, task)
            save_raw_results_D(all_results, ps_dir)
            continue

        all_results = []
        newly = 0
        for rep in range(1, n_reps + 1):
            key = ck.make_key(exp_key, ps, rep)
            if use_checkpoint and ck.is_completed(key):
                res = ck.get_result(key)
                cached = set(k for k in res.keys() if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                if set(all_methods).issubset(cached):
                    all_results.append(res); continue
                print(f"  [{exp_key}/ps{ps}/rep{rep}] Cache stale — re-running", flush=True)
            seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=seed)
            X, y, tf, n, p, s, cat_idx, mask = _unpack_gen(R)
            res = run_single_rep_D(X, y, tf, s, p, n, task, track_methods,
                                   cat_indices=cat_idx, missing_mask=mask,
                                   include_baseline=include_baseline)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'n': n, 'p': p, 's': s,
                            'task': task}
            if use_checkpoint:
                ck.save_result(key, res)
            all_results.append(res); newly += 1
            if newly % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly})", flush=True)
        completed = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed >= n_reps:
            ck.update_manifest(exp_key, ps, n_reps, completed, all_methods)
        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
        summary = summarize_results_D(all_results, s)
        save_summary_D(summary, ps_dir, track_methods, include_baseline, task)
        save_raw_results_D(all_results, ps_dir)
        print(f"  [{exp_key}/ps{ps}] Done — cached: {n_reps - newly}, new: {newly}", flush=True)

def summarize_results_D(all_results, s):
    if not all_results:
        return {}
    mm = {}
    for res in all_results:
        for key, val in res.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict) and 'mms' in val:
                mm.setdefault(key, {k: [] for k in val.keys()})
                for k, v in val.items():
                    mm[key][k].append(v)
    out = {}
    for method, M in mm.items():
        mms = np.array(M['mms'])
        out[method] = {
            'mms_median': float(np.median(mms)), 'mms_q1': float(np.percentile(mms, 25)),
            'mms_q3': float(np.percentile(mms, 75)), 'mms_mean': float(np.mean(mms)),
            'mms_std': float(np.std(mms)), 'ptp': float(np.mean(mms <= s)),
            'tpr_n1_mean': float(np.mean(M.get('tpr_n1', [0]))),
            'fpr_n1_mean': float(np.mean(M.get('fpr_n1', [0]))),
            'tpr_nlogn_mean': float(np.mean(M.get('tpr_nlogn', [0]))),
            'fpr_nlogn_mean': float(np.mean(M.get('fpr_nlogn', [0]))),
        }
    return out

def save_summary_D(summary, output_dir, track_methods, include_baseline, task):
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = [f"{'Method':<18} {'Trk':<4} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
             f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}",
             "-" * 102]
    baseline = BASELINE_TRAD_CLS if task == 'classification' else BASELINE_TRAD_REG
    order = (baseline + track_methods) if include_baseline else list(track_methods)
    for m in order:
        if m not in summary:
            continue
        d = summary[m]
        tag = 'D' if m in track_methods else 'base'
        iqr = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(f"{m:<18} {tag:<4} {d['mms_median']:>10.1f} {iqr:>12} {d['ptp']:>6.2f} "
                     f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
                     f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}")
    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_raw_results_D(all_results, output_dir):
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
            elif k == '_meta':
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
        clean.append(entry)
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

def _write_atomic_bytes(path_bytes, data_bytes):
    p = Path(path_bytes)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f'.tmp{os.getpid()}')
    with open(tmp, 'wb') as f:
        f.write(data_bytes); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def save_E_scores_npz(score_cache_dir, exp_key, ps, rep, scores_dict, meta):
    buf = _io.BytesIO()
    save_kwargs = {f"score_{m}": np.asarray(v, dtype=np.float64)
                   for m, v in scores_dict.items() if v is not None}
    save_kwargs['__meta__'] = np.array(json.dumps(meta), dtype=object)
    np.savez(buf, **save_kwargs)
    out = Path(score_cache_dir) / f"{exp_key}_ps{ps}_rep{rep}.npz"
    data = buf.getvalue()
    for attempt in range(4):
        try:
            _write_atomic_bytes(str(out), data); return str(out)
        except OSError as e:
            time.sleep(0.5 * (attempt + 1))

    with open(out, 'wb') as f:
        f.write(data); f.flush()
    return str(out)

def load_E_scores_npz(score_cache_dir, exp_key, ps, rep):
    out = Path(score_cache_dir) / f"{exp_key}_ps{ps}_rep{rep}.npz"
    if not out.exists():
        return {}
    try:
        with np.load(out, allow_pickle=True) as d:
            keys = [k[len('score_'):] for k in d.files if k.startswith('score_')]
            return {k: np.asarray(d['score_' + k]) for k in keys}
    except Exception:
        return {}

def run_single_rep_E(X, y, true_features, s, p, n, task, track_methods,
                     cat_indices=None, missing_mask=None, seed=None, device=DEVICE,
                     cached_res=None):
    results = {}
    if cached_res:
        for k, v in cached_res.items():

            if isinstance(v, dict) and v.get('status') != 'error':
                results[k] = v
    cat = list(cat_indices) if cat_indices else []
    Xf, _ = pace_a._fill_missing(X, cat)
    from sklearn.model_selection import train_test_split
    is_cls = (len(np.unique(y)) < n / 2)
    strat = y if (is_cls and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
    Xtr, Xval, ytr, yval = (train_test_split(Xf, y, test_size=0.35, random_state=0, stratify=strat)
                             if n >= 12 else (Xf, Xf, y, y))

    track_to_run = [m for m in track_methods if m not in results]
    scores_cache = {}
    for m in track_to_run:
        t0 = time.time()
        sc, st = selector_call(m, Xtr, ytr, Xval, yval, p, task, cat, seed, device)
        if sc is not None:
            scores_cache[m] = sc
        ev = pace_a.evaluate_screening_E(sc, true_features, s, p, n,
                                  rank_type=pace_a.RANK_TYPE[m], status=st)
        ev['runtime_sec'] = round(time.time() - t0, 2)
        ev['selector_seed'] = int(seed) if seed is not None else None
        ev['hyperparams'] = pace_a._method_hyperparams(m)
        results[m] = ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, scores_cache

def run_experiment_E(exp_key, task, param_sets, n_reps, output_dir, track_methods,
                     use_checkpoint=True, device=DEVICE):
    if task == 'classification':
        gen_func = CLS_EXPERIMENTS[exp_key]; name = CLS_NAMES.get(exp_key, exp_key)
    else:
        gen_func = REG_EXPERIMENTS[exp_key]; name = REG_NAMES.get(exp_key, exp_key)
    exp_hash = protocol.seed_offset(task, exp_key)
    is_ultra = ((task == 'classification' and exp_key == 'exp4')
                or (task == 'regression' and exp_key == 'exp2'))

    print(f"\n{'=' * 70}\nTrack-E {task} Experiment: {exp_key} — {name}\n"
          f"Param sets: {param_sets}, Reps: {n_reps}, Methods: {track_methods}\n{'=' * 70}",
          flush=True)

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = ps_dir / "score_cache"


        if is_ultra:
            print(f"  [{exp_key}/ps{ps}] ULTRA-HIGH-DIM — running {n_reps} reps (cap=MAX_TRAIN_FEATURES_E)",
                  flush=True)

        ck = CheckpointManagerE(str(ps_dir / "checkpoint.json"),
                                 manifest_path=str(ps_dir / "manifest.json"))
        if use_checkpoint and ck.is_complete(exp_key, ps, n_reps, track_methods):
            print(f"  [{exp_key}/ps{ps}] Already complete — SKIPPED", flush=True)
            all_results = [ck.get_result(ck.make_key(exp_key, ps, rep)) for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
            save_summary_E(summarize_results_E(all_results, s), ps_dir, track_methods, task)
            continue

        all_results = []; newly = 0
        for rep in range(1, n_reps + 1):
            key = ck.make_key(exp_key, ps, rep)
            cached_res_to_use = None
            if use_checkpoint and ck.is_completed(key):
                res = ck.get_result(key)
                cached = set(k for k in res.keys() if not k.startswith('_')
                             and isinstance(res[k], dict) and 'mms' in res[k])
                if set(track_methods).issubset(cached) and all(
                        res[k].get('status') != 'error' for k in track_methods if k in cached):
                    all_results.append(res); continue
                cached_res_to_use = res
                print(f"  [{exp_key}/ps{ps}/rep{rep}] Resume: keep successful/unavailable entries and retry errors",
                      flush=True)
            rep_seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=rep_seed)
            X, y, tf, n, p, s, cat_idx, mask = _unpack_gen(R)
            res, scores_cache = run_single_rep_E(
                X, y, tf, s, p, n, task, track_methods,
                cat_indices=cat_idx, missing_mask=mask,
                seed=rep_seed, device=device, cached_res=cached_res_to_use)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep,
                            'n': n, 'p': p, 's': s, 'task': task,
                            'dgp_seed': rep_seed, 'selector_seed': rep_seed}
            if use_checkpoint:
                ck.save_result(key, res)

            if scores_cache:
                save_E_scores_npz(cache_dir, exp_key, ps, rep, scores_cache, {
                    'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'task': task,
                    'p': p, 'n': n, 's': s, 'dgp_seed': rep_seed, 'selector_seed': rep_seed,
                    'rank_type': pace_a.RANK_TYPE})
            all_results.append(res); newly += 1
            if newly % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly})", flush=True)
        completed = sum(1 for r in all_results if r is not None)
        if use_checkpoint and completed >= n_reps:
            ck.update_manifest(exp_key, ps, n_reps, completed, track_methods)
        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else 1
        save_summary_E(summarize_results_E(all_results, s), ps_dir, track_methods, task)
        save_raw_results_E(all_results, ps_dir)
        print(f"  [{exp_key}/ps{ps}] Done — cached: {n_reps - newly}, new: {newly}", flush=True)

def _write_ultra_na_summary(ps_dir, track_methods, n_reps, task):
    summary = {}
    for m in track_methods:
        summary[m] = {'mms_median': None, 'mms_q1': None, 'mms_q3': None,
                      'mms_mean': None, 'mms_std': None, 'mms_worst_tie_median': None,
                      'ptp': 0.0, 'ok_frac': 0.0, 'na_reason': 'resource (p>5K)',
                      'rank_type': pace_a.RANK_TYPE[m]}
    with open(ps_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    mf = {'n_reps': 0, 'completed_reps': 0, 'methods': sorted(track_methods), 'track': 'E',
          'ultrahigh_na': True, 'note': 'all learned selectors NA-resource (p>MAX_TRAIN_FEATURES_E=5000)',
          'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    with open(ps_dir / "manifest.json", 'w') as f:
        json.dump(mf, f, indent=2, ensure_ascii=False)
    with open(ps_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write(f"{'Method':<14} {'MMS(med)':>10} {'worst-tie':>10} {'ok%':>6} {'reason':<20}\n"
                f"{'-' * 64}\n")
        for m in track_methods:
            f.write(f"{m:<14} {'NA':>10} {'NA':>10} {0:>5.0f}% {summary[m]['na_reason']:<20}\n")

def summarize_results_E(all_results, s):
    if not all_results:
        return {}
    mm = {}
    for res in all_results:
        for key, val in res.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict) and 'mms' in val:
                mm.setdefault(key, {'mms': [], 'mms_worst_tie': [], 'status': [],
                                     'tpr_n1': [], 'fpr_n1': [], 'tpr_nlogn': [], 'fpr_nlogn': []})
                mm[key]['mms'].append(val['mms'])
                mm[key]['mms_worst_tie'].append(val.get('mms_worst_tie', val['mms']))
                mm[key]['status'].append(val.get('status', 'ok'))
                mm[key]['tpr_n1'].append(val.get('tpr_n1', 0))
                mm[key]['fpr_n1'].append(val.get('fpr_n1', 0))
                mm[key]['tpr_nlogn'].append(val.get('tpr_nlogn', 0))
                mm[key]['fpr_nlogn'].append(val.get('fpr_nlogn', 0))
    out = {}
    for method, M in mm.items():
        mms = np.array(M['mms'], dtype=float)
        mwt = np.array(M['mms_worst_tie'], dtype=float)
        n_ok = sum(1 for st in M['status'] if st == 'ok')
        all_na = (n_ok == 0)

        if not all_na:
            ok_mms = np.array([v for v, st in zip(M['mms'], M['status']) if st == 'ok'], dtype=float)
            ok_mwt = np.array([v for v, st in zip(M['mms_worst_tie'], M['status']) if st == 'ok'], dtype=float)

            base_mms, base_mwt = ok_mms, ok_mwt
        else:
            base_mms, base_mwt = mms, mwt

        from collections import Counter
        st_counter = Counter(M['status'])
        dom_st = st_counter.most_common(1)[0][0] if M['status'] else 'ok'
        na_reason = {'ok': '', 'na_task': 'task-inapplicable (cls-only)',
                     'na_oom': 'OOM (xRFM ultrahigh p>=100K preemptively skipped — strand-risk)'}.get(dom_st, '') or \
            ('resource (p>5K)' if dom_st == 'na_resource' else
             ('runtime error' if dom_st == 'error' else ''))
        if all_na:
            mms_med = mwt_med = None
        else:
            mms_med = float(np.median(base_mms)); mwt_med = float(np.median(base_mwt))
        out[method] = {
            'mms_median': mms_med,
            'mms_worst_tie_median': mwt_med,
            'mms_q1': float(np.percentile(base_mms, 25)) if not all_na else None,
            'mms_q3': float(np.percentile(base_mms, 75)) if not all_na else None,
            'mms_mean': float(np.mean(base_mms)) if not all_na else None,
            'mms_std': float(np.std(base_mms)) if not all_na else None,
            'ptp': float(np.mean(base_mms <= s)) if not all_na else 0.0,
            'ok_frac': n_ok / len(M['status']),
            'na_reason': na_reason,
            'rank_type': pace_a.RANK_TYPE.get(method),
            'tpr_n1_mean': float(np.mean(M['tpr_n1'])),
            'fpr_n1_mean': float(np.mean(M['fpr_n1'])),
            'tpr_nlogn_mean': float(np.mean(M['tpr_nlogn'])),
            'fpr_nlogn_mean': float(np.mean(M['fpr_nlogn'])),
            'n_reps_total': len(M['status']),
        }
    return out

def save_summary_E(summary, output_dir, track_methods, task):
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = [f"{'Method':<14} {'MMS(stable)':>12} {'worst-tie':>10} {'IQR':>12} "
             f"{'PTP':>6} {'ok%':>5} {'TPR_n1':>8} {'FPR_n1':>8} {'reason':<22}",
             "-" * 104]
    for m in track_methods:
        if m not in summary:
            continue
        d = summary[m]
        mm = f"{d['mms_median']:.0f}" if d['mms_median'] is not None else 'NA'
        mw = f"{d['mms_worst_tie_median']:.0f}" if d['mms_worst_tie_median'] is not None else 'NA'
        iqr = (f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
               if d['mms_q1'] is not None else '(NA)')
        lines.append(f"{m:<14} {mm:>12} {mw:>10} {iqr:>12} {d['ptp']:>6.2f} "
                     f"{d['ok_frac']*100:>4.0f}% {d['tpr_n1_mean']:>8.3f} "
                     f"{d['fpr_n1_mean']:>8.3f} {d.get('na_reason',''):<22}")
    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def save_raw_results_E(all_results, output_dir):
    clean = []
    for res in all_results:
        entry = {}
        for k, v in res.items():
            if isinstance(v, dict) and 'mms' in v:
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
            elif k == '_meta':
                entry[k] = {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                                 float(vv) if isinstance(vv, (np.floating,)) else vv)
                            for kk, vv in v.items()}
        clean.append(entry)
    with open(output_dir / "raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

def run_timing_E(task, settings, fast_reps, slow_set, slow_reps, methods, device):
    is_cls = task in ('classification', 'cls')
    task_full = 'classification' if is_cls else 'regression'
    gen = CLS_EXPERIMENTS if is_cls else REG_EXPERIMENTS
    from sklearn.model_selection import train_test_split
    out = {}
    for exp, ps_list in settings:
        for ps in ps_list:
            sk = f"{exp}/{ps}"
            cell = {'_meta': {}}
            for rep in range(1, fast_reps + 1):
                try:
                    seed = 10000 * ps + rep * 37 + 37
                    R = gen[exp](param_set=ps, seed=seed)
                    X, y, tf, n, p, s, cat_idx, mask = _unpack_gen(R)
                except Exception as e:
                    print(f"[Etime] {sk} rep{rep} gen failed: {e}", flush=True); break
                cell['_meta'].update(dict(n=n, p=p, s=s))
                try:
                    Xf, cat = pace_a._fill_missing(X, cat_idx)
                    is_cls = (len(np.unique(y)) < n / 2)
                    strat = y if (is_cls and min(np.bincount(np.asarray(y).astype(int))) >= 2) else None
                    Xtr, Xval, ytr, yval = (train_test_split(Xf, y, test_size=0.35,
                                       random_state=0, stratify=strat) if n >= 12 else (Xf, Xf, y, y))
                except Exception as e:
                    print(f"[Etime] {sk} rep{rep} prep failed: {e}", flush=True); break
                for m in methods:
                    n_rep = slow_reps if m in slow_set else fast_reps
                    if rep > n_rep:
                        continue
                    t0 = time.perf_counter(); ok = False
                    try:
                        res = selector_call(m, Xtr, ytr, Xval, yval, p, task_full, cat_idx, seed, device)
                        sc = res[0] if isinstance(res, tuple) else res
                        sc = np.asarray(sc, dtype=np.float64) if sc is not None else None
                        ok = (sc is not None and sc.shape and sc.shape[0] == p and not np.isnan(sc).all())
                    except Exception as e:
                        print(f"[Etime] {sk} rep{rep} {m} err: {e}", flush=True)
                    cell.setdefault(m, []).append((time.perf_counter() - t0) if ok else None)
            out[sk] = cell
            print(f"[Etime] {sk} done p={cell['_meta'].get('p')}", flush=True)
    return out

def _timing_stats(timing_out):
    """{setting: {'_meta':..., method:[s|None]}} → {setting: {'_meta':..., method:{median,mean,n,reps}}}"""
    stats = {}
    for sk, cell in timing_out.items():
        sc = {'_meta': cell.get('_meta', {})}
        for m, reps in cell.items():
            if m == '_meta':
                continue
            vals = [v for v in (reps or []) if v is not None]
            if not vals:
                continue
            vals = sorted(vals); n = len(vals)
            med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
            sc[m] = {'median': med, 'mean': sum(vals) / n, 'n': n, 'reps': reps}
        stats[sk] = sc
    return stats

def _fmt_sec(v):
    if v is None:
        return 'NA'
    if v < 1:
        return f"{v:.2f}"
    if v < 60:
        return f"{v:.1f}"
    return f"{int(v + 0.5)}"

def write_timing_E(out_dir, timing_out, methods, task, fast_reps, slow_reps):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = _timing_stats(timing_out)
    with open(out_dir / "timing.json", 'w', encoding='utf-8') as f:
        json.dump({'track': 'E', 'task': task, 'methods': list(methods),
                   'fast_reps': fast_reps, 'slow_reps': slow_reps, 'slow_set': list({'lassonet_path'}),
                   'stats': stats}, f, indent=2, ensure_ascii=False)
    sks = list(stats.keys())
    def _ct(sk):
        m = stats[sk].get('_meta', {})
        return f"{sk}(p{m.get('p','?')})"
    width = max(14, max((len(_ct(s)) for s in sks), default=14))
    head = f"{'Method':<14}" + ''.join(f"{_ct(s):>{width + 2}}" for s in sks) + f"{'Total(s)':>10}"
    lines = [f"Track-E runtime benchmark (median seconds; fast={fast_reps}rep / slow={slow_reps}rep lassonet)  NA=skipped/failed/width-limit",
             "-" * len(head), head, "-" * len(head)]
    for m in methods:
        tot, anv, cells = 0.0, False, []
        for sk in sks:
            st = stats[sk].get(m)
            if st is None:
                cells.append('')
            else:
                cells.append(_fmt_sec(st['median'])); tot += st['median']; anv = True
        lines.append(f"{m:<14}" + ''.join(f"{c:>{width + 2}}" for c in cells)
                      + f"{(_fmt_sec(tot) if anv else 'NA'):>10}")
    cells = []
    for sk in sks:
        t, anv2 = 0.0, False
        for m in methods:
            st = stats[sk].get(m)
            if st is not None:
                t += st['median']; anv2 = True
        cells.append(_fmt_sec(t) if anv2 else 'NA')
    lines.append("-" * len(head))
    lines.append(f"{'Setting total':<14}" + ''.join(f"{c:>{width + 2}}" for c in cells) + f"{'':>10}")
    with open(out_dir / "timing_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return stats

_TRACKS=('A','B','C','D','E')

def _common_args(p):
    p.add_argument('--exp', type=str, default='paper', help="expN, a comma-separated list, or 'paper' for the manuscript plan")
    p.add_argument('--task', type=str, default='classification', choices=['classification','regression'])
    p.add_argument('--param-sets', type=int, nargs='+', default=None)
    p.add_argument('--n-reps', type=int, default=None)
    p.add_argument('--ultra-reps', type=int, default=25)
    p.add_argument('--ultra-reps-reg', type=int, default=25)
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--methods', type=str, nargs='+', default=None)
    p.add_argument('--no-baseline', action='store_true')
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--skip-validation', action='store_true')
    p.add_argument('--validate-only', action='store_true',
                   help='Validate the selected track and methods without a production experiment')
    p.add_argument('--output', type=str, default=None)

def parse_args(argv=None):
    p=_argparse.ArgumentParser(description='final_experiments_nostats: run one track ablation (shared dispatcher).')
    p.add_argument('--track', type=str, required=True, choices=list(_TRACKS))
    _common_args(p)
    p.add_argument('--emb-layers', type=int, nargs='+', default=None)      # A only
    p.add_argument('--n-jobs', type=int, default=1)                         # A only
    p.add_argument('--clear-checkpoint', action='store_true')              # A only


    p.add_argument('--dry-run', action='store_true')
    p.add_argument("--aggregation", choices=("native", "mean", "max"), default="native")
    return p.parse_args(argv)

# -- track exp resolution: exp -> [(exp_key_with_prefix, ps_list, n_reps)] --
def _resolve_track_track(track, args):
    result = protocol.execution_plan(args.task, args.exp, args.param_sets, args.n_reps)
    if args.n_reps is None:
        ultra = args.ultra_reps if args.task == 'classification' else args.ultra_reps_reg
        if ultra < 1:
            raise ValueError('ultra repetitions must be positive')
        result = [(key, ps, ultra if protocol.scenarios(args.task)[key].get('is_ultra') else reps)
                  for key, ps, reps in result]
    return result

def _call_kv(fn, kwargs):
    sig=_inspect.signature(_call_kv if fn is None else fn)
    valid={k:v for k,v in kwargs.items() if k in sig.parameters}
    return fn(**valid)

def main(argv=None):
    global PACE_AGGREGATION
    args=parse_args(argv)
    PACE_AGGREGATION = args.aggregation
    plan = _resolve_track_track(args.track, args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    from src.screening_methods_torch import set_device
    device = set_device(args.device_id)
    use_checkpoint=not args.no_checkpoint; include_baseline=not args.no_baseline; do_cumulative=True
    plan=_resolve_track_track(args.track, args)
    run_map={'A':(run_experiment_A, run_experiment_A_reg), 'B':(run_experiment_B,None),
             'C':(run_experiment_C,None), 'D':(run_experiment_D,None), 'E':(run_experiment_E,None)}
    cls_fn, reg_fn = run_map[args.track]
    fn = reg_fn if (args.track=='A' and args.task=='regression') else cls_fn
    out_dir = args.output or str(protocol.output_root() / args.task / args.track)

    if args.track=='A':
        _tm_default = pace_a.TRACK_A_METHODS_REG if args.task=='regression' else pace_a.TRACK_A_METHODS
    else:
        _tm_default = {'B':pace_a.TRACK_B_METHODS,'C':pace_a.TRACK_C_METHODS,'D':pace_a.TRACK_D_METHODS,'E':pace_a.TRACK_E_METHODS}[args.track]
    track_methods = args.methods or _tm_default
    if args.validate_only:
        if args.track != 'D':
            raise SystemExit('--validate-only supports Track D')
        ok = pace_a.validate_trackD_scorers(track_methods, task=args.task)
        if not ok:
            raise SystemExit(1)
        print('nostats: validation passed track=%s task=%s methods=%s' %
              (args.track, args.task, track_methods))
        return
    if args.aggregation != "native":
        out_dir = str(Path(out_dir) / ("aggregation_" + args.aggregation))
    protocol.ensure_output_protocol(out_dir, args.task, f'track_{args.track}')
    for ek, ps, nreps in plan:
        kw=dict(exp_key=ek, param_sets=ps, n_reps=nreps, output_dir=out_dir,
                track_methods=track_methods, emb_layers=args.emb_layers, embedder=None,
                include_baseline=include_baseline, use_checkpoint=use_checkpoint,
                task=args.task, n_jobs=args.n_jobs, device=device)
        _call_kv(fn, kw)
    print('nostats: done track=%s exp=%s task=%s'%(args.track,args.exp,args.task))

if __name__=='__main__':
    _sys.argv # noqa
    main()
