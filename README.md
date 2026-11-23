# PACE experiments

The repository root contains experiment entry points. Shared screening algorithms,
representations, data generators, and path configuration live in `src/`.

```
src/
  screening_utilities.py       # unified table-to-score interface
  screening_methods_torch.py   # raw and PACE utilities
  screening_methods_torch_A.py # learned and attribution-based selectors
  screening_methods.py        # statistical utilities and evaluation
  screening_utils.py          # shared scoring helpers
  ablation_encoders.py         # column encoders used by ablations
  model_path.py                # checkpoint, external-code, data, and output paths
  simulation_protocol.py      # experiment order and repetitions
  data_generators_*.py         # controlled data-generating processes
  pace_real/                  # shared TALENT data and prediction support
  talent_benchmark/           # dataset catalog
tests/                       # unit tests and executable smoke tests
f2ves_ablation.py             # representation ablations
final_experiments_pace.py     # raw/PACE controlled experiments
final_experiments_nostats.py  # other screening methods
f2ves_real_eval_full.py       # TALENT scoring and prediction
eval_*.py                    # wide-table TFM comparisons
```

## Setup

Use Python 3.10 or newer. Install the appropriate PyTorch build and the dependencies
in `requirements.txt`. Configure machine-specific locations in `src/model_path.py`.
The repository does not use an `.env` file.
Checkpoints, benchmark data, TALENT, TabDPT, GOTabPFN, and TabPFN-Wide are external
inputs; obtain them under their respective licenses. GOTabPFN and TabPFN-Wide use
separate environments with their required TabPFN versions.

## Screening API

```python
from src.screening_utilities import compute_scores

scores = compute_scores(X, y, method="sis", task="classification", device="cpu")
# X has shape (n_samples, n_features); scores has shape (n_features,).
# Larger scores indicate greater feature importance.
```

PACE methods use names such as `pace_sis`, `pace_dc`, and `pace_hsic`. Supply the
configured encoder or precomputed embeddings as supported by the experiment
runner. `aggregation="mean"` or `"max"` controls scalar-coordinate aggregation;
DC and HSIC continue to operate on the full embedding vector.

## Experiment commands

Run these commands from this directory:

```bash
python final_experiments_pace.py --task classification --dry-run
python final_experiments_pace.py --task regression --dry-run
python f2ves_ablation.py --task all --dry-run
python f2ves_real_eval_full.py --stage score --help
python f2ves_real_eval_full.py --stage predict --help
```

The simulation registry includes 16 classification and nine regression settings.
TALENT defaults to all datasets. Scoring operates on the complete candidate table,
including injected noise features. Score and prediction stages can run on separate
machines. GPU workers retain `--gpus` and `--per_gpu`; prediction supports CPU and
GPU model groups. Preserve method-specific output directories when merging runs.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/run_screening_smoke.py --output /path/to/outputs/screening_smoke
```

The CUDA smoke test uses actual encoder weights, two simulation cases, and binary,
multiclass, and regression TALENT cases. It checks SIS, DC, PACE-SIS, and PACE-DC,
including full-2,000-feature scoring and LightGBM downstream predictions. The
output directory must be empty. Full production grids and the separate-environment
wide-table models require additional runs beyond these smoke tests.

Standalone progress/summary scripts are excluded. Experiment-internal checkpoint
and result writers remain part of each reproducible experiment.

## Public-data example

`toy_example.py` uses all 569 rows and 30 features of scikit-learn's bundled
Wisconsin Diagnostic Breast Cancer dataset (UCI, CC BY 4.0;
https://doi.org/10.24432/C5DW2B). No download is required.

Run `python toy_example.py`. It scores all 569 rows and 30 features with raw
SIS, then prints the feature scores in descending order. To run PACE-SIS,
change `METHOD` to `"pace_sis"` and set `DEVICE` to `"cuda:0"` in the file
after placing the configured TabICL checkpoint in `models/`.

## Acknowledgments

The GOTabPFN evaluation code references [GOTabPFN](https://github.com/zadid6pretam/GOTabPFN),
and the TabPFN-Wide evaluation code references [TabPFN-Wide](https://github.com/pfeiferAI/TabPFN-Wide).
This code was primarily developed with assistance from ChatGPT-5.6.
