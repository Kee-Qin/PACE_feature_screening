from src.ablation_encoders import (TabICLProjectionEmbedder, TabPFNColumnEmbedder, TabPFNV3ColumnEmbedder)
from src import model_path as mp
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src import simulation_protocol as protocol
import sys
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TABPFN_TELEMETRY"] = "0"

import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.screening_methods import AcceleratedScreening, ScreeningEvaluator, AcceleratedEmbedder, TABICL_PATH
from src.screening_methods_torch import TorchAcceleratedScreening, TorchAcceleratedEmbedder

# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
from src.data_generators_classification import (
    ClassificationDataGenerator, EXPERIMENTS as CLS_EXPERIMENTS,
    EXPERIMENT_NAMES as CLS_EXPERIMENT_NAMES
)
from src.data_generators_regression import (
    RegressionDataGenerator, EXPERIMENTS as REG_EXPERIMENTS,
    EXPERIMENT_NAMES as REG_EXPERIMENT_NAMES
)

# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

N_REPS = 50
ULTRA_REPS_CLS = 25
ULTRA_REPS_REG = 25
N_JOBS = -1
EMB_LAYERS = [1, 2, 3]
TABPFN_V26_LAYERS = [1, 4, 8, 12, 16, 20, 24]
BASE_OUTPUT_DIR = str(protocol.output_root() / "ablation")


MODEL_ROOT = mp.MODEL_ROOT

TABICL_V2_CKPT     = str(mp.checkpoint('tabicl_v2_clf'))
TABICL_V2_REG_CKPT = str(mp.checkpoint('tabicl_v2_reg'))
# TabPFN v2.6:
TABPFN_V26_CLF_CKPT = str(mp.checkpoint('tabpfn_v26_clf'))
TABPFN_V26_REG_CKPT = str(mp.checkpoint('tabpfn_v26_reg'))
# TabPFN v3:
TABPFN_V3_CLF_CKPT  = str(mp.checkpoint('tabpfn_v3_clf'))
TABPFN_V3_REG_CKPT  = str(mp.checkpoint('tabpfn_v3_reg'))


PAPER_CLS_SCENARIOS = protocol.CLASSIFICATION
PAPER_REG_SCENARIOS = protocol.REGRESSION


EXPERIMENTS = {}
EXPERIMENT_NAMES = {}
for _cls_ei in [1, 2, 3, 4, 5, 6]:
    _ek = f'cls_exp{_cls_ei}'
    EXPERIMENTS[_ek] = CLS_EXPERIMENTS[f'exp{_cls_ei}']
    EXPERIMENT_NAMES[_ek] = f'Classification Exp{_cls_ei}: {CLS_EXPERIMENT_NAMES[f"exp{_cls_ei}"]}'
for _reg_ei in [1, 2, 3, 4]:
    _ek = f'reg_exp{_reg_ei}'
    EXPERIMENTS[_ek] = REG_EXPERIMENTS[f'exp{_reg_ei}']
    EXPERIMENT_NAMES[_ek] = f'Regression Exp{_reg_ei}: {REG_EXPERIMENT_NAMES[f"exp{_reg_ei}"]}'
del _cls_ei, _reg_ei, _ek

CLS_EXP_KEYS = [k for k in EXPERIMENTS if k.startswith('cls_')]
REG_EXP_KEYS = [k for k in EXPERIMENTS if k.startswith('reg_')]

EXP_HASH = {k: protocol.seed_offset('classification' if k.startswith('cls_') else 'regression', k[4:]) for k in EXPERIMENTS}


CLS_TRAD_METHODS = ['sis', 'dc', 'sc', 'ipc', 'pc', 'fkf', 'cru', 'cavs',
                     'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor']
CLS_EMB_METHODS = list(CLS_TRAD_METHODS)
REG_TRAD_METHODS = ['sis', 'dc', 'sc', 'ipc', 'pc', 'hd', 'gcor']
REG_EMB_METHODS = list(REG_TRAD_METHODS)

BASELINE_TRAD_CLS = CLS_TRAD_METHODS + ['ids']
BASELINE_TRAD_REG = REG_TRAD_METHODS + ['ids']


# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

def _unpack_gen(R):
    X, y, tf, n, p, s = R[:6]
    cat_indices = None
    missing_mask = None
    if len(R) >= 7:
        extra = R[6]
        if isinstance(extra, dict) and 'cat_indices' in extra:
            cat_indices = list(extra['cat_indices'])
        elif isinstance(extra, np.ndarray):
            missing_mask = extra
    return np.asarray(X), np.asarray(y), list(tf), int(n), int(p), int(s), cat_indices, missing_mask


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


# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_path: str, manifest_path: str = None, extra_read_paths=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.manifest_path = Path(manifest_path) if manifest_path else self.checkpoint_path.parent / "manifest.json"
        self.data = {}
        self.aux = []
        self.extra_read_paths = [Path(p) for p in (extra_read_paths or [])]
        self.load()

    def load(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        self.aux = []
        own = self.checkpoint_path.resolve()
        for p in self.extra_read_paths:
            if p.exists() and p.resolve() != own:
                try:
                    with open(p, 'r') as f:
                        self.aux.append(json.load(f))
                except Exception:
                    pass

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def make_key(self, exp_key: str, param_set: int, rep: int) -> str:
        return f"{exp_key}_ps{param_set}_rep{rep}"

    def is_completed(self, key: str) -> bool:
        if key in self.data:
            return True
        return any(key in d for d in self.aux)

    def get_result(self, key: str):
        if key in self.data:
            return self.data[key]
        for d in self.aux:
            if key in d:
                return d[key]
        return None

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
            if methods is not None and not set(methods).issubset(mf_methods):
                return False
            if emb_layers is not None:
                mf_layers = set(mf.get('emb_layers', []))
                if mf_layers and not set(emb_layers).issubset(mf_layers):
                    return False
            return True
        except Exception:
            return False

    def update_manifest(self, exp_key, ps, n_reps, completed_reps, methods, emb_layers=None):
        mf = {
            'exp_key': exp_key, 'param_set': ps, 'n_reps': n_reps,
            'completed_reps': completed_reps, 'methods': sorted(methods),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        if emb_layers is not None:
            mf['emb_layers'] = sorted(int(L) for L in emb_layers)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(mf, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────


def _eval_slice(ev):
    return {k: ev[k] for k in ('mms', 'tpr_n1', 'fpr_n1', 'tpr_nlogn', 'fpr_nlogn')}


def run_single_rep(X, y, true_features, s, p, n,
                   proj_embedder, tabicl_v2_embedder,
                   tabpfn_v26_embedder, tabpfn_v3_embedder,
                   is_classification: bool = True,
                   emb_layers: List[int] = EMB_LAYERS,
                   n_jobs: int = N_JOBS,
                   include_baseline: bool = True,
                   cached_res: Optional[Dict] = None,
                   do_cumulative: bool = True,
                   no_tabicl: bool = False,
                   no_tabpfn: bool = False):
    results = {}
    if cached_res:
        for k, v in cached_res.items():
            results[k] = v

    use_pc = n <= 500
    if is_classification:
        emb_methods = [m for m in CLS_EMB_METHODS if use_pc or m != 'pc']
    else:
        emb_methods = [m for m in REG_EMB_METHODS if use_pc or m != 'pc']

    screening_cls = TorchAcceleratedScreening

    def _to_tensor(emb):
        if isinstance(emb, np.ndarray):
            return torch.as_tensor(emb, device='cuda:0')
        return emb

    def _score_emb(emb, key_fn, methods):
        emb = _to_tensor(emb)
        sc = screening_cls.emb_compute_all_scores(emb, y, methods=methods, n_jobs=n_jobs)
        for rm in methods:
            sk = f'emb_{rm}'
            if sk not in sc or sc[sk] is None:
                continue
            ev = ScreeningEvaluator.evaluate_screening(sc[sk], true_features, s, p, n)
            results[key_fn(rm)] = _eval_slice(ev)

    def _need(prefix_keys):
        return any(k not in results for k in prefix_keys)


    if include_baseline:
        baseline = BASELINE_TRAD_CLS if is_classification else BASELINE_TRAD_REG
        trad = [m for m in baseline if (use_pc or m != 'pc') and m not in results]
        if trad:
            try:
                ts = screening_cls.compute_all_scores(X, y, methods=trad, n_jobs=n_jobs)
            except Exception as e:
                print(f"  [WARN] Raw baseline failed: {e}", flush=True); ts = {}
            for method, scores in ts.items():
                if scores is None:
                    continue
                ev = ScreeningEvaluator.evaluate_screening(scores, true_features, s, p, n)
                results[method] = _eval_slice(ev)

    # ════════ TabICL v1.1 (proj_embedder: fpg=1) ════════
    if not no_tabicl and proj_embedder is not None and proj_embedder.loaded:
        # proj
        if _need({f'proj_{m}' for m in emb_methods}):
            try:
                emb = proj_embedder.get_projection_embeddings(X)
                _score_emb(emb, lambda rm: f'proj_{rm}', emb_methods)
                del emb; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v1.1 proj failed: {e}", flush=True)
        # independent L
        if _need({f'emb_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = proj_embedder.get_single_layer_embeddings(X, emb_layers)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'emb_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v1.1 independent L failed: {e}", flush=True)
        # cumulative L
        if do_cumulative and _need({f'emb_cum_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = proj_embedder.get_cumulative_layer_embeddings(X, emb_layers)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'emb_cum_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v1.1 cumulative L failed: {e}", flush=True)
        # rand independent L
        if _need({f'rand_emb_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = proj_embedder.get_random_transformer_embeddings(X, emb_layers, seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_emb_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v1.1 rand independent failed: {e}", flush=True)
        # rand cumulative L
        if do_cumulative and _need({f'rand_emb_cum_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = proj_embedder.get_random_cumulative_embeddings(X, emb_layers, seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_emb_cum_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v1.1 rand cumulative failed: {e}", flush=True)


    if not no_tabicl and tabicl_v2_embedder is not None and tabicl_v2_embedder.loaded:
        if _need({f'tabicl_v2_proj_{m}' for m in emb_methods}):
            try:
                emb = tabicl_v2_embedder.get_projection_embeddings(X)
                _score_emb(emb, lambda rm: f'tabicl_v2_proj_{rm}', emb_methods)
                del emb; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2 proj failed: {e}", flush=True)
        if _need({f'tabicl_v2_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = tabicl_v2_embedder.get_single_layer_embeddings(X, emb_layers)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'tabicl_v2_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2 independent L failed: {e}", flush=True)
        if do_cumulative and _need({f'tabicl_v2_cum_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = tabicl_v2_embedder.get_cumulative_layer_embeddings(X, emb_layers)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'tabicl_v2_cum_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2 cumulative L failed: {e}", flush=True)
        if _need({f'rand_emb_tabicl_v2_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = tabicl_v2_embedder.get_random_transformer_embeddings(X, emb_layers, seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_emb_tabicl_v2_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2 rand independent failed: {e}", flush=True)
        if do_cumulative and _need({f'rand_emb_tabicl_v2_cum_{m}_L{l}' for m in emb_methods for l in emb_layers}):
            try:
                embs = tabicl_v2_embedder.get_random_cumulative_embeddings(X, emb_layers, seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_emb_tabicl_v2_cum_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2 rand cumulative failed: {e}", flush=True)


    if not no_tabpfn and tabpfn_v26_embedder is not None and tabpfn_v26_embedder.loaded:
        layers26 = [l for l in TABPFN_V26_LAYERS if l <= tabpfn_v26_embedder.num_layers]
        if _need({f'tabpfn_v26_proj_{m}' for m in emb_methods}):
            try:
                emb = tabpfn_v26_embedder.get_projection_embeddings(X, y)
                _score_emb(emb, lambda rm: f'tabpfn_v26_proj_{rm}', emb_methods)
                del emb; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2.6 proj failed: {e}", flush=True)
        if _need({f'tabpfn_v26_{m}_L{l}' for m in emb_methods for l in layers26}):
            try:
                embs = tabpfn_v26_embedder.get_embeddings_multi_layer(X, y, layers=layers26)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'tabpfn_v26_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2.6 independent L failed: {e}", flush=True)
        if _need({f'rand_tabpfn_v26_{m}_L{l}' for m in emb_methods for l in layers26}):
            try:
                embs = tabpfn_v26_embedder.get_random_embeddings_multi_layer(X, y, layers=layers26, seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_tabpfn_v26_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v2.6 rand L failed: {e}", flush=True)


    if not no_tabpfn and tabpfn_v3_embedder is not None and tabpfn_v3_embedder.loaded:
        layers3 = [l for l in emb_layers if l <= tabpfn_v3_embedder.num_layers]
        if _need({f'tabpfn_v3_proj_{m}' for m in emb_methods}):
            try:
                emb = tabpfn_v3_embedder.get_projection_embeddings(X)
                _score_emb(emb, lambda rm: f'tabpfn_v3_proj_{rm}', emb_methods)
                del emb; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v3 proj failed: {e}", flush=True)
        if _need({f'tabpfn_v3_{m}_L{l}' for m in emb_methods for l in layers3}):
            try:
                embs = tabpfn_v3_embedder.get_single_layer_embeddings(X, layers3)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'tabpfn_v3_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v3 independent L failed: {e}", flush=True)
        if do_cumulative and _need({f'tabpfn_v3_cum_{m}_L{l}' for m in emb_methods for l in layers3}):
            try:
                embs = tabpfn_v3_embedder.get_cumulative_layer_embeddings(X, layers3)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'tabpfn_v3_cum_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v3 cumulative L failed: {e}", flush=True)
        if _need({f'rand_tabpfn_v3_{m}_L{l}' for m in emb_methods for l in layers3}):
            try:
                embs = tabpfn_v3_embedder.get_random_embeddings_multi_layer(X, layers3, mode='independent', seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_tabpfn_v3_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v3 rand independent failed: {e}", flush=True)
        if do_cumulative and _need({f'rand_tabpfn_v3_cum_{m}_L{l}' for m in emb_methods for l in layers3}):
            try:
                embs = tabpfn_v3_embedder.get_random_embeddings_multi_layer(X, layers3, mode='cumulative', seed=42)
                for l, emb in embs.items():
                    _score_emb(emb, lambda rm, ll=l: f'rand_tabpfn_v3_cum_{rm}_L{ll}', emb_methods)
                del embs; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [WARN] v3 rand cumulative failed: {e}", flush=True)

    return results


def save_raw_results(all_results: List[Dict], output_dir: Path):
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


def finalize_ps(exp_key: str, ps: int, n_reps: int, num_shards: int,
                output_dir: str, emb_layers: List[int]):
    ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
    if not ps_dir.exists():
        print(f"  [finalize {exp_key}/ps{ps}] dir missing — skip", flush=True)
        return False
    src_files = [ps_dir / "checkpoint.json"] + \
                [ps_dir / f"checkpoint_shard{K}.json" for K in range(num_shards)]
    reps_map = {}
    for p in src_files:
        if not p.exists():
            continue
        try:
            d = json.load(open(p, 'r'))
        except Exception:
            continue
        for k, v in d.items():
            if k not in reps_map:
                reps_map[k] = v
    all_results = [reps_map.get(f"{exp_key}_ps{ps}_rep{r}") for r in range(1, n_reps + 1)]
    missing = [r for r in range(1, n_reps + 1) if all_results[r - 1] is None]
    if missing:
        print(f"  [finalize {exp_key}/ps{ps}] ABORT — missing reps {missing} "
              f"(have {len(reps_map)}/{n_reps})", flush=True)
        return False
    rep_method_sets = [{k for k in r if not k.startswith('_') and isinstance(r[k], dict) and 'mms' in r[k]}
                       for r in all_results]
    observed = set.intersection(*rep_method_sets) if rep_method_sets else set()
    s = all_results[0].get('_meta', {}).get('s')
    if s is None:
        s = 0
    merged = {f"{exp_key}_ps{ps}_rep{r}": all_results[r - 1] for r in range(1, n_reps + 1)}
    with open(ps_dir / "checkpoint.json", 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    mf = {'exp_key': exp_key, 'param_set': ps, 'n_reps': n_reps, 'completed_reps': n_reps,
          'methods': sorted(observed) if observed else [],
          'emb_layers': sorted(int(L) for L in emb_layers),
          'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    with open(ps_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(mf, f, indent=2, ensure_ascii=False)
    summary = summarize_results(all_results, s)
    save_summary(summary, ps_dir)
    save_raw_results(all_results, ps_dir)
    print(f"  [finalize {exp_key}/ps{ps}] DONE — {n_reps} reps, {len(observed)} observed methods",
          flush=True)
    return True


def _expected_methods(is_classification, emb_layers, do_cumulative, loaded, use_pc,
                      include_baseline, v26_layers=None, v3_layers=None):
    emb_m = [m for m in (CLS_EMB_METHODS if is_classification else REG_EMB_METHODS)
             if use_pc or m != 'pc']
    ly = list(emb_layers)
    out = []
    if include_baseline:
        out += [m for m in (BASELINE_TRAD_CLS if is_classification else BASELINE_TRAD_REG)
                if use_pc or m != 'pc']
    if loaded.get('v1_1'):
        out += [f'proj_{m}' for m in emb_m]
        out += [f'emb_{m}_L{l}' for m in emb_m for l in ly]
        out += [f'rand_emb_{m}_L{l}' for m in emb_m for l in ly]
        if do_cumulative:
            out += [f'emb_cum_{m}_L{l}' for m in emb_m for l in ly]
            out += [f'rand_emb_cum_{m}_L{l}' for m in emb_m for l in ly]
    if loaded.get('v2'):
        out += [f'tabicl_v2_proj_{m}' for m in emb_m]
        out += [f'tabicl_v2_{m}_L{l}' for m in emb_m for l in ly]
        out += [f'rand_emb_tabicl_v2_{m}_L{l}' for m in emb_m for l in ly]
        if do_cumulative:
            out += [f'tabicl_v2_cum_{m}_L{l}' for m in emb_m for l in ly]
            out += [f'rand_emb_tabicl_v2_cum_{m}_L{l}' for m in emb_m for l in ly]
    if loaded.get('v2_6'):
        L26 = v26_layers or [l for l in TABPFN_V26_LAYERS]
        out += [f'tabpfn_v26_proj_{m}' for m in emb_m]
        out += [f'tabpfn_v26_{m}_L{l}' for m in emb_m for l in L26]
        out += [f'rand_tabpfn_v26_{m}_L{l}' for m in emb_m for l in L26]
    if loaded.get('v3'):
        L3 = v3_layers or ly
        out += [f'tabpfn_v3_proj_{m}' for m in emb_m]
        out += [f'tabpfn_v3_{m}_L{l}' for m in emb_m for l in L3]
        out += [f'rand_tabpfn_v3_{m}_L{l}' for m in emb_m for l in L3]
        if do_cumulative:
            out += [f'tabpfn_v3_cum_{m}_L{l}' for m in emb_m for l in L3]
            out += [f'rand_tabpfn_v3_cum_{m}_L{l}' for m in emb_m for l in L3]
    return out


def run_experiment(exp_key: str, param_sets: List[int], n_reps: int,
                   embedders: Dict, output_dir: str,
                   emb_layers: List[int] = EMB_LAYERS,
                   n_jobs: int = N_JOBS, use_checkpoint: bool = True,
                   do_cumulative: bool = True, include_baseline: bool = True,
                   shard_id: int = 0, num_shards: int = 1):
    is_classification = exp_key.startswith('cls_')
    gen_func = EXPERIMENTS[exp_key]
    exp_name = EXPERIMENT_NAMES[exp_key]
    exp_hash = EXP_HASH[exp_key]
    emb_set = embedders['clf'] if is_classification else embedders['reg']

    loaded = {
        'v1_1': getattr(emb_set.get('v1_1'), 'loaded', False),
        'v2':   getattr(emb_set.get('v2'), 'loaded', False),
        'v2_6': getattr(emb_set.get('v2_6'), 'loaded', False),
        'v3':   getattr(emb_set.get('v3'), 'loaded', False),
    }

    print(f"\n{'='*70}\nAblation: {exp_key} — {exp_name}\n"
          f"Param sets: {param_sets}, Reps: {n_reps}, "
          f"Type: {'Classification' if is_classification else 'Regression'}, "
          f"loaded={ {k: v for k, v in loaded.items() if v} }\n{'='*70}", flush=True)

    for ps in param_sets:
        ps_dir = Path(output_dir) / exp_key / f"param_set_{ps}"
        ps_dir.mkdir(parents=True, exist_ok=True)

        if num_shards > 1:


            shard_path = ps_dir / f"checkpoint_shard{shard_id}.json"
            checkpoint = CheckpointManager(str(shard_path),
                                           manifest_path=str(ps_dir / "manifest.json"),
                                           extra_read_paths=[ps_dir / "checkpoint.json"])
            own_reps = [r for r in range(1, n_reps + 1) if (r - 1) % num_shards == shard_id]
            print(f"  [{exp_key}/ps{ps}] SHARD {shard_id}/{num_shards} — own reps {own_reps}", flush=True)
            newly = 0
            for rep in own_reps:
                key = checkpoint.make_key(exp_key, ps, rep)
                if use_checkpoint and checkpoint.is_completed(key):
                    continue
                seed = 10000 * ps + rep * 37 + exp_hash
                R = gen_func(param_set=ps, seed=seed)
                X, y, true_features, n, p, s, cat_idx, mask = _unpack_gen(R)
                X, cat_idx = _preprocess_X(X, cat_idx, mask)
                res = run_single_rep(X, y, true_features, s, p, n,
                                     proj_embedder=emb_set.get('v1_1'),
                                     tabicl_v2_embedder=emb_set.get('v2'),
                                     tabpfn_v26_embedder=emb_set.get('v2_6'),
                                     tabpfn_v3_embedder=emb_set.get('v3'),
                                     is_classification=is_classification, emb_layers=emb_layers,
                                     n_jobs=n_jobs,
                                     include_baseline=include_baseline,
                                     cached_res=None, do_cumulative=do_cumulative)
                res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'n': n, 'p': p, 's': s}
                if use_checkpoint:
                    checkpoint.save_result(key, res)
                newly += 1
                if newly % 5 == 0 or rep == own_reps[-1]:
                    print(f"  [{exp_key}/ps{ps}] shard{shard_id} rep {rep}/{n_reps} (new: {newly})", flush=True)
                del X, y
                torch.cuda.empty_cache()
            print(f"  [{exp_key}/ps{ps}] SHARD {shard_id}/{num_shards} done — new: {newly}/{len(own_reps)}",
                  flush=True)
            continue

        checkpoint = CheckpointManager(str(ps_dir / "checkpoint.json"),
                                        manifest_path=str(ps_dir / "manifest.json"))


        probe_seed = 10000 * ps + 1 * 37 + exp_hash
        R = gen_func(param_set=ps, seed=probe_seed)
        Xp, yp, tfp, n, p, sp, cat_idx, mask = _unpack_gen(R)
        Xp, cat_idx = _preprocess_X(Xp, cat_idx, mask)
        use_pc = n <= 500
        v26_layers = [l for l in TABPFN_V26_LAYERS if l <= getattr(emb_set.get('v2_6'), 'num_layers', 0)]
        v3_layers = [l for l in emb_layers if l <= getattr(emb_set.get('v3'), 'num_layers', 0)]
        all_methods = _expected_methods(is_classification, emb_layers, do_cumulative, loaded,
                                        use_pc, include_baseline, v26_layers, v3_layers)
        del Xp, yp
        torch.cuda.empty_cache()

        if use_checkpoint and checkpoint.is_complete(exp_key, ps, n_reps, all_methods, emb_layers=emb_layers):
            print(f"  [{exp_key}/ps{ps}] Already complete ({n_reps} reps, {len(all_methods)} methods) — SKIPPED", flush=True)
            all_results = [checkpoint.get_result(checkpoint.make_key(exp_key, ps, rep))
                           for rep in range(1, n_reps + 1)]
            s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else sp
            summary = summarize_results(all_results, s)
            save_summary(summary, ps_dir)
            save_raw_results(all_results, ps_dir)
            continue

        all_results = []
        newly_completed = 0
        for rep in range(1, n_reps + 1):
            key = checkpoint.make_key(exp_key, ps, rep)
            cached_res = None
            if use_checkpoint and checkpoint.is_completed(key):
                res = checkpoint.get_result(key)
                cached_methods = set(k for k in res.keys()
                                     if not k.startswith('_') and isinstance(res[k], dict) and 'mms' in res[k])
                if set(all_methods).issubset(cached_methods):
                    all_results.append(res)
                    continue
                cached_res = res

            seed = 10000 * ps + rep * 37 + exp_hash
            R = gen_func(param_set=ps, seed=seed)
            X, y, true_features, n, p, s, cat_idx, mask = _unpack_gen(R)
            X, cat_idx = _preprocess_X(X, cat_idx, mask)

            res = run_single_rep(X, y, true_features, s, p, n,
                                 proj_embedder=emb_set.get('v1_1'),
                                 tabicl_v2_embedder=emb_set.get('v2'),
                                 tabpfn_v26_embedder=emb_set.get('v2_6'),
                                 tabpfn_v3_embedder=emb_set.get('v3'),
                                 is_classification=is_classification, emb_layers=emb_layers,
                                 n_jobs=n_jobs,
                                 include_baseline=include_baseline and (cached_res is None),
                                 cached_res=cached_res, do_cumulative=do_cumulative)
            res['_meta'] = {'exp_key': exp_key, 'param_set': ps, 'rep': rep, 'n': n, 'p': p, 's': s}
            if use_checkpoint:
                checkpoint.save_result(key, res)
            all_results.append(res)
            newly_completed += 1
            if newly_completed % 5 == 0 or rep == n_reps:
                print(f"  [{exp_key}/ps{ps}] rep {rep}/{n_reps} (new: {newly_completed})", flush=True)
            del X, y
            torch.cuda.empty_cache()

        completed_reps = sum(1 for r in all_results if r is not None)

        rep_method_sets = []
        for r in all_results:
            if r is None:
                continue
            rep_method_sets.append({k for k in r.keys() if not k.startswith('_')
                                    and isinstance(r[k], dict) and 'mms' in r[k]})
        observed = (set.intersection(*rep_method_sets) if rep_method_sets else set(all_methods))
        if use_checkpoint and completed_reps >= n_reps:
            checkpoint.update_manifest(exp_key, ps, n_reps, completed_reps,
                                       sorted(observed) if observed else list(all_methods),
                                       emb_layers=emb_layers)
        s = all_results[0]['_meta']['s'] if all_results and '_meta' in all_results[0] else sp
        summary = summarize_results(all_results, s)
        save_summary(summary, ps_dir)
        save_raw_results(all_results, ps_dir)
        print(f"  [{exp_key}/ps{ps}] Done — cached: {n_reps - newly_completed}, new: {newly_completed}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

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
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = []
    lines.append(f"{'Method':<34} {'MMS(med)':>10} {'IQR':>12} {'PTP':>6} "
                 f"{'TPR_n1':>8} {'FPR_n1':>8} {'TPR_nlogn':>10} {'FPR_nlogn':>10}")
    lines.append("-" * 116)

    import re
    def _layer(m):
        mm = re.search(r'_L(\d+)$', m)
        return int(mm.group(1)) if mm else 0

    def method_sort_key(m):

        if m.startswith('rand_tabpfn_v3_cum_'): return (18, _layer(m), m)
        if m.startswith('rand_tabpfn_v3_') and '_L' in m: return (17, _layer(m), m)
        if m.startswith('tabpfn_v3_cum_'): return (16, _layer(m), m)
        if m.startswith('tabpfn_v3_proj_'): return (14, 0, m)
        if m.startswith('tabpfn_v3_') and '_L' in m: return (15, _layer(m), m)
        if m.startswith('rand_tabpfn_v26_'): return (13, _layer(m), m)
        if m.startswith('tabpfn_v26_proj_'): return (11, 0, m)
        if m.startswith('tabpfn_v26_') and '_L' in m: return (12, _layer(m), m)
        if m.startswith('rand_emb_tabicl_v2_cum_'): return (10, _layer(m), m)
        if m.startswith('rand_emb_tabicl_v2_'): return (9, _layer(m), m)
        if m.startswith('tabicl_v2_cum_'): return (8, _layer(m), m)
        if m.startswith('tabicl_v2_proj_'): return (6, 0, m)
        if m.startswith('tabicl_v2_') and '_L' in m: return (7, _layer(m), m)
        if m.startswith('rand_emb_cum_'): return (5, _layer(m), m)
        if m.startswith('rand_emb_tabicl_v2'): return (9, 99, m)
        if m.startswith('rand_emb_'): return (4, _layer(m), m)
        if m.startswith('emb_cum_'): return (3, _layer(m), m)
        if m.startswith('emb_'): return (2, _layer(m), m)
        if m.startswith('proj_'): return (1, 0, m)
        return (0, 0, m)

    for method in sorted(summary.keys(), key=method_sort_key):
        d = summary[method]
        iqr_str = f"({d['mms_q1']:.0f},{d['mms_q3']:.0f})"
        lines.append(
            f"{method:<34} {d['mms_median']:>10.1f} {iqr_str:>12} {d['ptp']:>6.2f} "
            f"{d['tpr_n1_mean']:>8.3f} {d['fpr_n1_mean']:>8.3f} "
            f"{d['tpr_nlogn_mean']:>10.3f} {d['fpr_nlogn_mean']:>10.3f}")
    with open(output_dir / "summary_table.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Ablation: embedding transformation (TabICL v1.1/v2 + TabPFN v2.6/v3)')
    parser.add_argument('--task', type=str, default='all', choices=['classification', 'regression', 'all'])
    parser.add_argument('--exp', type=str, default='paper',
                        help="'paper' (PACE scenarios), 'all', or comma-separated expN (e.g. exp1,exp3)")
    parser.add_argument('--param-sets', type=int, nargs='+', default=None)
    parser.add_argument('--n-reps', type=int, default=None)
    parser.add_argument('--ultra-reps', type=int, default=ULTRA_REPS_CLS, help='cls exp4 reps (default 25)')
    parser.add_argument('--ultra-reps-reg', type=int, default=ULTRA_REPS_REG, help='reg exp2 reps (default 25)')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--emb-layers', type=int, nargs='+', default=EMB_LAYERS)
    parser.add_argument('--output', type=str, default=BASE_OUTPUT_DIR)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--clear-checkpoint', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS)
    parser.add_argument('--no-tabicl', action='store_true', help='skip TabICL v1.1/v2')
    parser.add_argument('--no-tabpfn', action='store_true', help='skip TabPFN v2.6/v3')
    parser.add_argument('--no-v3', action='store_true', help='skip TabPFN v3 only')
    parser.add_argument('--no-baseline', action='store_true')
    parser.add_argument('--no-cumulative', action='store_true', help='skip cumulative-layer sections')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='Process repetitions satisfying (rep-1)%%num_shards==shard_id')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Number of repetition shards; each writes checkpoint_shard{K}.json')
    parser.add_argument('--finalize', action='store_true',
                        help='Combine shard checkpoints into summary and manifest without model fitting')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def _resolve_plan(args):
    plan = []
    tasks = ('classification', 'regression') if args.task == 'all' else (args.task,)
    special = args.exp in ('paper', 'all', 'plain', 'mixtype_missing')
    requested = [x.strip() for x in args.exp.split(',')]
    valid_full = {p + k for t, p in [('classification', 'cls_'), ('regression', 'reg_')] for k in protocol.scenarios(t)}
    if not special:
        for item in requested:
            if item.startswith(('cls_', 'reg_')) and item not in valid_full:
                raise ValueError(f'Unknown experiment: {item}')
            if args.task != 'all' and item.startswith('reg_' if args.task == 'classification' else 'cls_'):
                raise ValueError('Experiment prefix conflicts with --task')
    for task in tasks:
        prefix = 'cls_' if task == 'classification' else 'reg_'
        wanted = args.exp
        if not special:
            items = [x[4:] if x.startswith(prefix) else x for x in requested
                     if not x.startswith(('cls_', 'reg_')) or x.startswith(prefix)]
            if not items:
                continue
            wanted = ','.join(items)
        entries = protocol.execution_plan(task, wanted, args.param_sets, args.n_reps)
        for key, ps, reps in entries:
            if args.n_reps is None and protocol.scenarios(task)[key].get('is_ultra'):
                reps = args.ultra_reps if task == 'classification' else args.ultra_reps_reg
            if reps < 1:
                raise ValueError('repetitions must be positive')
            plan.append((prefix + key, ps, reps))
    return plan


def _load_tabicl_v1_1_proj(device):
    try:
        from src.screening_methods_torch import TorchAcceleratedEmbedder, TABICL_PATH as _V11
        emb = TorchAcceleratedEmbedder(device=device)
        if not emb.load_model(_V11, device=device):
            return None
        pe = TabICLProjectionEmbedder(device=device)
        pe.col_embedder = emb.col_embedder
        pe.embed_dim = emb.embed_dim
        pe.n_cls_tokens = emb.n_cls_tokens
        col_fg = getattr(emb.col_embedder, 'feature_group', False)
        pe.feature_group_size = (emb.col_embedder.feature_group_size
                                 if col_fg and col_fg is not False else 1)
        pe.loaded = True
        print(f"TabICL v1.1 ProjectionEmbedder loaded (fpg={pe.feature_group_size})", flush=True)
        return pe
    except Exception as e:
        print(f"TabICL v1.1 not available: {e}", flush=True)
        return None


def _load_tabicl_v2(ckpt, device):
    pe = TabICLProjectionEmbedder(device=device)
    if pe.load_model(ckpt, device=device):
        return pe
    return None


def main():
    args = parse_args()
    plan = _resolve_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    protocol.ensure_output_protocol(args.output, 'all', 'ablation')
    from src.screening_methods_torch import set_device
    set_device(args.device_id)
    device = f'cuda:{args.device_id}'

    print("=" * 70)
    print("Ablation: embedding transformation (TabICL v1.1/v2 + TabPFN v2.6/v3)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"task={args.task}, exp={args.exp}, n_reps={args.n_reps}, "
          f"emb_layers={args.emb_layers}, device={device}, cumulative={not args.no_cumulative}")
    print("=" * 70, flush=True)

    plan = _resolve_plan(args)
    if not plan:
        return


    if args.finalize:
        for exp_key, ps_list, n_reps in plan:
            for ps in ps_list:
                finalize_ps(exp_key, ps, n_reps, args.num_shards, args.output, args.emb_layers)
        print(f"\n{'='*70}\nFinalize completed! Results: {args.output}\n{'='*70}", flush=True)
        return

    # Clear checkpoints
    if args.clear_checkpoint:
        for ek, ps_list, _ in plan:
            for ps in ps_list:
                ps_dir = Path(args.output) / ek / f"param_set_{ps}"
                for fn in ['checkpoint.json', 'manifest.json']:
                    fp = ps_dir / fn
                    if fp.exists():
                        fp.unlink(); print(f"  Cleared: {fp}")


    need_cls = any(ek.startswith('cls_') for ek, _, _ in plan)
    need_reg = any(ek.startswith('reg_') for ek, _, _ in plan)
    embedders = {'clf': {}, 'reg': {}}
    if not args.no_tabicl and (need_cls or need_reg):
        v11 = _load_tabicl_v1_1_proj(device)
        if need_cls: embedders['clf']['v1_1'] = v11
        if need_reg: embedders['reg']['v1_1'] = v11
        if need_cls: embedders['clf']['v2'] = _load_tabicl_v2(TABICL_V2_CKPT, device)
        if need_reg: embedders['reg']['v2'] = _load_tabicl_v2(TABICL_V2_REG_CKPT, device)
    if not args.no_tabpfn:
        for tag, need, ckpt26, ckpt3 in [('clf', need_cls, TABPFN_V26_CLF_CKPT, TABPFN_V3_CLF_CKPT),
                                         ('reg', need_reg, TABPFN_V26_REG_CKPT, TABPFN_V3_REG_CKPT)]:
            if not need:
                continue
            e26 = TabPFNColumnEmbedder(ckpt26, device=device)
            embedders[tag]['v2_6'] = e26 if e26.load_model(device=device) else None
            if not args.no_v3:
                e3 = TabPFNV3ColumnEmbedder(ckpt3, device=device)
                embedders[tag]['v3'] = e3 if e3.load_model(device=device) else None


    do_cumulative = not args.no_cumulative
    include_baseline = not args.no_baseline
    for exp_key, ps_list, n_reps in plan:
        run_experiment(exp_key=exp_key, param_sets=ps_list, n_reps=n_reps,
                       embedders=embedders, output_dir=args.output,
                       emb_layers=args.emb_layers, n_jobs=args.n_jobs,
                       use_checkpoint=not args.no_checkpoint,
                       do_cumulative=do_cumulative, include_baseline=include_baseline,
                       shard_id=args.shard_id, num_shards=args.num_shards)

    print(f"\n{'='*70}\nAblation experiments completed! Results: {args.output}\n{'='*70}", flush=True)

if __name__ == "__main__":
    main()
