"""Central path configuration and overrides."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from src import model_path as mp


class ModelPathTests(unittest.TestCase):
    def test_custom_filename_and_root(self):
        with patch.dict(mp.MODEL_FILES, {'tabicl_v11': 'custom/encoder.ckpt'}):
            self.assertEqual(mp.checkpoint('tabicl_v11', '/tmp/models'), Path('/tmp/models/custom/encoder.ckpt'))

    def test_absolute_checkpoint(self):
        absolute = Path(__file__).resolve()
        with patch.dict(mp.MODEL_FILES, {'tabdpt': str(absolute)}):
            self.assertEqual(mp.checkpoint('tabdpt', '/ignored'), absolute)

    def test_scoring_and_prediction_defaults(self):
        from src.pace_real import common, prediction
        import f2ves_real_eval_full as scoring
        from src import screening_methods_torch as base
        self.assertEqual(common.DEFAULT_MODEL_ROOT, mp.MODEL_ROOT)
        self.assertEqual(base.TABICL_PATH, str(mp.checkpoint('tabicl_v11')))
        for parser in (scoring.build_score_parser(), prediction.build_parser()):
            self.assertEqual(Path(parser.parse_args(['--task_type','all']).model_root), mp.MODEL_ROOT)
