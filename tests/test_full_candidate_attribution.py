"""Check complete candidate coverage for TALENT attribution."""
import ast
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
import f2ves_real_eval_full as scoring
from src import screening_methods_torch_A as algorithms


class CompleteCandidateTests(unittest.TestCase):
    def test_regular_talent_cpu_uses_complete_candidate_attribution(self):
        X = np.random.RandomState(19).normal(size=(600, 2000))
        y = np.arange(600) % 2
        expected = np.arange(2000, dtype=float) + 1
        methods = list(scoring.DEFAULT_SCORE_METHODS)
        with patch.object(scoring, '_prepare_Xy', return_value=(X, y, 2000, 8)), \
             patch.object(scoring, 'score_matrix', return_value=(expected, 0.01)) as backend:
            result = scoring.compute_cpu_scores('fixture', 'bin', 42, methods_filter=methods)
        self.assertEqual(backend.call_count, len(methods))
        for call in backend.call_args_list:
            np.testing.assert_array_equal(call.args[1], X)
        for method in methods:
            np.testing.assert_array_equal(result[0][method], expected)

    def test_standalone_scoring_module_is_not_required(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / 'src' / 'pace_real' / 'scoring.py').exists())
        for name in ('f2ves_real_eval_full.py',):
            source = (root / name).read_text(encoding='utf-8')
            self.assertNotIn('src.pace_real.scoring', source)
        source = (root / 'f2ves_real_eval_full.py').read_text(encoding='utf-8')
        self.assertNotIn('sha256', source)
        self.assertNotIn('fingerprint', source)

    def test_talent_dispatch_preserve_candidate_width(self):
        rng = np.random.RandomState(12)
        for width in (8, 2000, 2001):
            X = rng.normal(size=(600, width))
            y = np.arange(600) % 2
            for method in scoring.DEFAULT_SCORE_METHODS:
                seen = []
                def backend(*values):
                    arrays = [v for v in values if isinstance(v, np.ndarray) and v.ndim == 2]
                    seen.extend(a.shape[1] for a in arrays)
                    return np.arange(width, dtype=float) + 1, {"P": width}
                target = ('_score_sage' if method.startswith('sage') else
                          '_score_tabicl_shap' if method == 'tabicl_shap' else '_score_' + method)
                with tempfile.TemporaryDirectory() as root, patch.object(algorithms, target, side_effect=backend):
                    values, _ = scoring.score_matrix(method, X, y, 'bin', 'cpu', root, root)
                self.assertEqual(values.shape, (width,))
                self.assertGreater(values[-1], 0)
                self.assertTrue(seen)
                self.assertEqual(set(seen), {width})


if __name__ == '__main__':
    unittest.main()
