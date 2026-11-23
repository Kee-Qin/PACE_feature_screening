"""Fit attribution and tree-importance methods through the shared interface."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import numpy as np
from src.screening_utilities import compute_scores
from src.screening_methods_torch_A import ATTRIBUTION_METHODS as DEFAULT_METHODS
from src.pace_real.common import DEFAULT_MODEL_ROOT, atomic_save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    args = parser.parse_args()
    rng = np.random.RandomState(42)
    X = rng.normal(size=(80, 8))
    response = X[:, -1] + X[:, 0] * 0.1
    rows = []
    for task in ('bin', 'multi', 'reg'):
        y = (response > np.median(response)).astype(int) if task == 'bin' else (
            np.digitize(response, np.quantile(response, [1/3, 2/3])) if task == 'multi' else response)
        for method in (*DEFAULT_METHODS, 'lgb_gain', 'rf_mdi', 'cb_pvc'):
            started = time.perf_counter()
            scores = compute_scores(X, y, method, task=task,
                device='cuda:0' if method == 'tabicl_shap' else 'cpu', model_root=args.model_root,
                rf_trees=3, rf_n_jobs=1, loco_cv_folds=2,
                    sage_permutations=1, sage_imputations=1, sage_background=8, sage_eval_max=4,
                    tabicl_estimators=1, tabicl_eval_max=1, threads=2)
            elapsed = time.perf_counter() - started
            rows.append(dict(task=task, method=method, n=80, p=8, status='ok',
                             scores=scores.tolist(), elapsed_seconds=elapsed))
            print(f'PASS {task}/{method}: 8/8 feature scores', flush=True)
    atomic_save_json(args.output / 'results.json',
                     dict(purpose='full_candidate_interface_smoke', results=rows))


if __name__ == '__main__':
    main()
