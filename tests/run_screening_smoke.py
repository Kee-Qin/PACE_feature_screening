"""Save reproducible SIS/DC and PACE-SIS/PACE-DC smoke-test inputs and results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import time
import numpy as np
import torch
from src import screening_methods_torch as smt
from src.screening_utilities import compute_scores
from src.data_generators_classification import EXPERIMENTS as CLASSIFICATION
from src.data_generators_regression import EXPERIMENTS as REGRESSION
from src import model_path as mp
from src.pace_real.common import (DEFAULT_DATA_ROOT, DEFAULT_MODEL_ROOT, DEFAULT_TABDPT_CODE_ROOT,
                              prepare_full2000, atomic_save_npz, atomic_save_json,
                              configure_model_paths)
from src.pace_real.prediction import _g3_columns, _mapped_categorical

METHODS = ('sis', 'dc', 'pace_sis', 'pace_dc')


def impute(X):
    X = np.array(X, dtype=np.float32, copy=True)
    rows, columns = np.where(np.isnan(X))
    if len(rows):
        means = np.nanmean(X, axis=0)
        X[rows, columns] = means[columns]
    if not np.isfinite(X).all():
        raise ValueError('The test input must be finite after mean imputation')
    return X


def utilities(X, y, embedder, task):
    embedding = embedder.get_all_embeddings(X, layer_idx=2, batch_size=32)
    scores = {method: compute_scores(X, y, method, task=task, device='cuda:0',
                  **({'embeddings': embedding} if method.startswith('pace_') else {}))
              for method in METHODS}
    del embedding
    for method, values in scores.items():
        scores[method] = np.asarray(values, dtype=np.float64)
        if scores[method].shape != (X.shape[1],) or not np.isfinite(scores[method]).all():
            raise ValueError(f'{method} did not return a finite full-width score vector')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--talent-rows', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This checkpoint-dependent smoke test requires CUDA')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory for a reproducible smoke-test record')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    smt.set_device(0)
    checkpoint = mp.checkpoint('tabicl_v11', args.model_root)
    embedder = smt.TorchAcceleratedEmbedder(device='cuda:0')
    if not embedder.load_model(str(checkpoint), seed=42, device='cuda:0'):
        raise RuntimeError('Column encoder failed to load')
    metadata = dict(purpose='smoke_test_not_paper_results', seed=42, methods=METHODS,
                    embedding_layer=2, embedding_dim=128, aggregation='mean',
                    talent_training_row_cap=args.talent_rows, downstream='LightGBM')
    atomic_save_json(args.output / 'configuration.json', metadata)
    rows = []
    for task, registry, setting in [('classification', CLASSIFICATION, 1),
                                     ('regression', REGRESSION, 2)]:
        case = registry['exp1'](param_set=setting, seed=42)
        X, y, support, n, p, s = case[:6]
        X = impute(X)
        directory = args.output / 'simulation' / task
        atomic_save_npz(directory / 'input.npz', X=X, y=y, support=np.asarray(support))
        started = time.perf_counter()
        scores = utilities(X, y, embedder, task)
        atomic_save_npz(directory / 'scores.npz', **scores)
        for method, values in scores.items():
            order = np.argsort(values)[::-1]
            ranks = np.empty(p, dtype=int)
            ranks[order] = np.arange(1, p+1)
            rows.append(dict(suite='simulation', task=task, dataset=f'exp1/setting{setting}',
                method=method, status='ok', n=n, p=p, mms=int(ranks[support].max()),
                top_s_recall=float(np.isin(order[:s], support).mean())))
        print(f'PASS simulation {task}: four score vectors; {time.perf_counter()-started:.1f}s', flush=True)
        torch.cuda.empty_cache()
    downstream = configure_model_paths(args.model_root, DEFAULT_TABDPT_CODE_ROOT)
    for task, dataset in [('bin', 'banknote_authentication'), ('multi', 'mfeat-morphological'),
                           ('reg', 'Wine_Quality_white')]:
        prepared = prepare_full2000(task, dataset, 42, args.data_root)
        X = impute(prepared['Xc_train'][:args.talent_rows])
        y = prepared['y_train'][:len(X)]
        directory = args.output / 'talent' / dataset
        atomic_save_npz(directory / 'input.npz', X_train=X, y_train=y,
                        X_val=prepared['Xc_val'][:64], y_val=prepared['y_val'][:64],
                        X_test=prepared['Xc_test'][:64], y_test=prepared['y_test'][:64],
                        p_original=np.array(prepared['p_original']))
        scores = utilities(X, y, embedder, task)
        atomic_save_npz(directory / 'scores.npz', **scores)
        details = {}
        for method, values in scores.items():
            selected = _g3_columns(values, prepared['p_original'])
            metrics = downstream.train_and_evaluate('lgb', X[:, selected], y,
                prepared['Xc_val'][:64, selected], prepared['y_val'][:64],
                prepared['Xc_test'][:64, selected], prepared['y_test'][:64], task, 42,
                _mapped_categorical(prepared['cat_indices'], selected), 'cpu')
            if 'error' in metrics:
                raise RuntimeError(f'{dataset}/{method}: {metrics}')
            details[method] = dict(metrics=metrics, selected_features=selected.tolist())
            rows.append(dict(suite='talent', task=task, dataset=dataset, method=method,
                status='ok', n=len(X), p=X.shape[1], n_selected=len(selected),
                original_retention=float(np.sum(selected < prepared['p_original']) / prepared['p_original']),
                auc=metrics.get('auc'), accuracy=metrics.get('acc'), rmse=metrics.get('rmse')))
        atomic_save_json(directory / 'downstream.json', details)
        print(f'PASS TALENT {task}/{dataset}: four full-2000 scores and four LightGBM fits', flush=True)
        torch.cuda.empty_cache()
    fields = ['suite', 'task', 'dataset', 'method', 'status', 'n', 'p', 'mms',
              'top_s_recall', 'n_selected', 'original_retention', 'auc', 'accuracy', 'rmse']
    with (args.output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_save_json(args.output / 'results.json', {'configuration': metadata, 'results': rows})
    print(f'COMPLETE: {len(rows)} method/case checks saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
