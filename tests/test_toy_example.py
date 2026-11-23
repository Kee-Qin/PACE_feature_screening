"""Validate the public-data example and release portability."""
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.datasets import load_breast_cancer
import toy_example


class ToyExampleTests(unittest.TestCase):
    def test_all_rows_and_columns_reach_scorer(self):
        data = load_breast_cancer(as_frame=True)
        with patch.object(toy_example, 'compute_scores', return_value=np.arange(30)) as score:
            result = toy_example.screen_table(data.data, data.target)
        self.assertEqual(score.call_args.args[0].shape, (569, 30))
        self.assertEqual(score.call_args.args[1].shape, (569,))
        self.assertEqual(result.feature.tolist(), data.feature_names.tolist())

    def test_raw_sis_is_finite(self):
        data = load_breast_cancer(as_frame=True)
        result = toy_example.screen_table(data.data, data.target, method='sis')
        self.assertEqual(len(result), 30)
        self.assertTrue(np.isfinite(result.importance).all())

    def test_vector_utilities_use_native_aggregation(self):
        data = load_breast_cancer(as_frame=True)
        for method in ('sis', 'pace_dc', 'pace_hsic'):
            with patch.object(toy_example, 'compute_scores', return_value=np.ones(30)) as score:
                toy_example.screen_table(data.data, data.target, method=method)
            self.assertEqual(score.call_args.kwargs['aggregation'], 'native')

    def test_source_has_no_machine_paths_or_credentials(self):
        root = Path(__file__).resolve().parents[1]
        pattern = re.compile(r'/root/|/mnt/|[A-Za-z]:\\\\|'
                             r'BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|'
                             r'(?i:password|api_key|access_token)\s*=\s*[\"\x27][^\"\x27]+')
        for path in root.rglob('*'):
            if path.suffix in ('.py', '.md', '.txt') and path != Path(__file__):
                self.assertIsNone(pattern.search(path.read_text(encoding='utf-8')), path.name)


if __name__ == '__main__':
    unittest.main()
