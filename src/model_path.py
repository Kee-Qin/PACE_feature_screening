"""Central model configuration. Edit this file to configure a new machine."""
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
DATA_ROOT = PROJECT_ROOT / 'data'
# Set MODEL_ROOT to the directory containing all pretrained checkpoints.
MODEL_ROOT = PROJECT_ROOT / 'models'
TABDPT_CODE_ROOT = PROJECT_ROOT / 'TabDPT-inference-main'
TALENT_CODE_ROOT = PROJECT_ROOT / 'external/TALENT-main'
GOTABPFN_CODE_ROOT = PROJECT_ROOT / 'external/GOTabPFN-main'
TABPFNWIDE_CODE_ROOT = PROJECT_ROOT / 'external/TabPFN-Wide-main'
GOTABPFN_DEPS = PROJECT_ROOT / 'external/gotabpfn_deps'
TABPFNWIDE_DEPS = PROJECT_ROOT / 'external/tabpfnwide_deps'
TABPFN_CACHE_ROOT = MODEL_ROOT

# Relative entries are resolved against MODEL_ROOT; absolute entries are allowed.
MODEL_FILES = {
    'tabicl_v11': 'tabicl-classifier-v1.1-20250506.ckpt',
    'tabicl_v2_clf': 'tabicl-classifier-v2-20260212.ckpt',
    'tabicl_v2_reg': 'tabicl-regressor-v2-20260212.ckpt',
    'tabpfn_v25_clf': 'tabpfn-v2.5-classifier-v2.5_default.ckpt',
    'tabpfn_v25_reg': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
    'tabpfn_v26_clf': 'tabpfn-v2.6-classifier-v2.6_default.ckpt',
    'tabpfn_v26_reg': 'tabpfn-v2.6-regressor-v2.6_default.ckpt',
    'tabpfn_v3_clf': 'tabpfn-v3-classifier-v3_default.ckpt',
    'tabpfn_v3_reg': 'tabpfn-v3-regressor-v3_default.ckpt',
    'tabdpt': 'tabdpt1_2.safetensors',
    'tabpfnwide': 'tabpfn-wide-v2-5k.pt',
}


def checkpoint(name, model_root=None):
    """Resolve a configured checkpoint; an explicit root overrides MODEL_ROOT."""
    entry = Path(MODEL_FILES[name]).expanduser()
    root = MODEL_ROOT if model_root is None else Path(model_root).expanduser()
    return entry if entry.is_absolute() else root / entry


TABICL_PATH = str(checkpoint('tabicl_v11'))
