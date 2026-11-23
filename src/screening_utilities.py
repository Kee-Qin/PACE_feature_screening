"""One table-to-scores interface for simulation and TALENT experiments."""
import numpy as np
from src.screening_utils import normalize_task, validate_input, validate_scores

RAW_METHODS = ('sis', 'dc', 'hsic', 'cru', 'ids', 'sc', 'ipc', 'pc', 'fkf',
               'cavs', 'mv_sis', 'lr_ffs', 'psis', 'hd', 'gcor')


def canonical_method(method):
    name = method.lower().replace('-', '_')
    aliases = {'raw_sis': 'sis', 'raw_dc': 'dc', 'raw_hsic': 'hsic',
               'raw_cru': 'cru', 'mv': 'mv_sis', 'sage_catboost': 'sage_cb',
               'tabicl_v2_shap': 'tabicl_shap', 'sis_emb_max': 'pace_sis_max'}
    name = aliases.get(name, name)
    if name.endswith('_emb'):
        name = 'pace_' + name[:-4]
    if name.startswith('emb_'):
        name = 'pace_' + name[4:]
    return name


def available_methods():
    from src import screening_methods_torch_A as other
    from src.screening_methods_torch import COORDINATE_AGGREGATION_METHODS
    return sorted(set(RAW_METHODS) | {'pace_' + m for m in RAW_METHODS} |
                  {'pace_' + m + '_' + mode for m in COORDINATE_AGGREGATION_METHODS for mode in ('mean', 'max')} |
                  set(other.TRACK_B_METHODS + other.TRACK_C_METHODS +
                      other.TRACK_D_METHODS + other.TRACK_E_METHODS))


def compute_scores(X, y, method, *, task='classification', device=None,
                   aggregation='native', **options):
    """Return p scores, in input-column order; larger values mean more important.

    Native preserves the utility definition. Coordinate-averaged utilities
    support mean/max; vector utilities such as DC/HSIC always use the vector.
    """
    X, y = validate_input(X, y)
    task = normalize_task(task, y)
    name = canonical_method(method)
    if aggregation not in ('native', 'mean', 'max'):
        raise ValueError('aggregation must be native, mean, or max')
    if name.startswith('pace_') and name.endswith(('_mean', '_max')):
        name, aggregation = name.rsplit('_', 1)
    if name in RAW_METHODS or name.startswith(('pace_', 'raw_a_')):
        from src.screening_methods_torch import compute_scores as calculate
    else:
        if aggregation != 'native':
            raise ValueError('aggregation applies to PACE methods only')
        from src.screening_methods_torch_A import compute_scores as calculate
    return validate_scores(calculate(X, y, name, task=task, device=device,
                           aggregation=aggregation, **options), X.shape[1], name)


def compute_many(X, y, methods, **options):
    """Run methods independently; failed methods are absent and reported."""
    result = {}
    for method in methods:
        try:
            result[method] = compute_scores(X, y, method, **options)
        except Exception as exc:
            print(f'[screening] {method} failed: {type(exc).__name__}: {exc}', flush=True)
    return result


def embedding_scores(embeddings, y, methods, *, task='classification', **options):
    """Use precomputed embeddings without recomputing their representation."""
    X = np.zeros((embeddings.shape[0], embeddings.shape[1]), dtype=np.float32)
    aggregation = options.pop('aggregation', 'native')
    values = {}
    for method in methods:
        name = 'pace_' + method
        try:
            values['emb_' + method] = compute_scores(X, y, name, task=task,
                embeddings=embeddings, aggregation=aggregation_for_method(name, aggregation), **options)
        except Exception as exc:
            print(f'[screening] {name} failed: {exc}', flush=True)
    return values


def layered_scores(X, y, embedder, emb_layers, methods, *, task='classification',
                   profile='base', **options):
    values = {}
    batch_size = options.pop('batch_size', 32)
    chunk = options.pop('feat_chunk', options.pop('feature_chunk', min(X.shape[1], 4096)))
    options.pop('n_jobs', None)
    aggregation = options.pop('aggregation', 'native')
    X = np.asarray(X)
    for layer in emb_layers:
        pieces = {method: [] for method in methods}
        failed = set()
        for start in range(0, X.shape[1], chunk):
            end = min(start+chunk, X.shape[1])
            E = embedder.get_embeddings_for_range(X, layer, start, end, batch_size=batch_size)
            for method in methods:
                name = ('pace_a_' if profile == 'track_a' else 'pace_') + method
                if method in failed:
                    continue
                try:
                    pieces[method].append(compute_scores(X[:, start:end], y, name,
                        task=task, embeddings=E,
                        aggregation=aggregation_for_method(name, aggregation), **options))
                except Exception as exc:
                    failed.add(method)
                    print(f'[screening] {name}/L{layer} failed: {exc}', flush=True)
            del E
        for method in methods:
            if method not in failed:
                values[f'emb_{method}_L{layer}'] = np.concatenate(pieces[method])
    return values


def pre_split_scores(Xtr, ytr, Xval, yval, p, task, cat_indices, methods):
    """Adapt an experiment's fixed internal split to the shared interface."""
    return compute_many(Xtr, ytr, methods, task=task,
                        validation_data=(Xval, yval), cat_indices=cat_indices)


def raw_a_scores(X, y, methods, *, task='regression', cat_indices=None):
    values = compute_many(X, y, ['raw_a_' + m for m in methods], task=task, impute=False)
    return {name[6:]: value for name, value in values.items()}


def selected_method_id(method, aggregation):
    name = canonical_method(method)
    aggregation = aggregation_for_method(name, aggregation)
    if name.startswith('pace_') and aggregation in ('mean', 'max') and not name.endswith(('_mean', '_max')):
        return name + '_' + aggregation
    return method


def aggregation_for_method(method, aggregation):
    from src.screening_methods_torch import COORDINATE_AGGREGATION_METHODS
    name = canonical_method(method)
    if name.startswith('pace_') and name.endswith(('_mean', '_max')):
        name, aggregation = name.rsplit('_', 1)
    coordinate = (name[5:] in COORDINATE_AGGREGATION_METHODS if name.startswith('pace_') else False)
    coordinate = coordinate or name in ('pace_a_pearson', 'pace_a_spearman', 'pace_a_nmi')
    return aggregation if coordinate else 'native'


def selector_call(method, Xtr, ytr, Xval, yval, p, task, cat_indices, seed, device):
    """Convert an explicit selector failure into an experiment status."""
    try:
        return compute_scores(Xtr, ytr, method, task=task, device=device,
            validation_data=(Xval, yval), cat_indices=cat_indices, seed=seed), 'ok'
    except (ImportError, ModuleNotFoundError) as exc:
        print(f'[screening] {method}: {exc}', flush=True)
        return None, 'na_dep'
    except Exception as exc:
        print(f'[screening] {method}: {exc}', flush=True)
        return None, 'error'
