"""Contract and numerical checks for the shared screening API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src import screening_methods_torch as base
from src import screening_methods_torch_A as other
from src.screening_utilities import compute_scores, selected_method_id, available_methods


class ScreeningInterfaceTests(unittest.TestCase):
    def setUp(self):
        base.DEVICE = torch.device('cpu')
        torch.manual_seed(42)
        self.E = torch.randn(30, 5, 7)
        self.X = np.random.RandomState(42).normal(size=(30, 5))
        self.y = np.arange(30) % 3

    def test_raw_sis_and_dc_match_existing_numerics(self):
        for method in ('sis', 'dc'):
            expected = base.TorchAcceleratedScreening.compute_all_scores(self.X, self.y, [method])[method]
            actual = compute_scores(self.X, self.y, method, device='cpu')
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_coordinate_mean_and_max(self):
        for method in ('sis', 'psis', 'cru', 'hd', 'ids'):
            torch.manual_seed(31)
            expected = base.TorchAcceleratedScreening.emb_compute_all_scores(self.E, self.y, [method])['emb_'+method]
            torch.manual_seed(31)
            mean = compute_scores(self.X, self.y, 'pace_'+method, embeddings=self.E, aggregation='mean')
            np.testing.assert_allclose(mean, expected, rtol=1e-6, atol=1e-7)
            torch.manual_seed(31)
            oracle = np.stack([base.TorchAcceleratedScreening.emb_compute_all_scores(
                self.E[:, :, i:i+1], self.y, [method])['emb_'+method] for i in range(7)]).max(0)
            torch.manual_seed(31)
            maximum = compute_scores(self.X, self.y, 'pace_'+method, embeddings=self.E, aggregation='max')
            np.testing.assert_allclose(maximum, oracle, rtol=1e-6, atol=1e-7)
            self.assertTrue(np.all(maximum >= mean-1e-7))

    def test_dc_and_hsic_remain_vector_utilities(self):
        for method in ('dc', 'hsic'):
            expected = base.TorchAcceleratedScreening.emb_compute_all_scores(self.E, self.y, [method])['emb_'+method]
            actual = compute_scores(self.X, self.y, 'pace_'+method, embeddings=self.E)
            np.testing.assert_array_equal(actual, expected)
            for option in ('mean', 'max'):
                with self.assertRaisesRegex(ValueError, 'whole embedding vector'):
                    compute_scores(self.X, self.y, 'pace_'+method, embeddings=self.E, aggregation=option)

    def test_dataframe_preserves_column_order(self):
        import pandas as pd
        frame = pd.DataFrame(self.X, columns=['z', 'a', 'q', 'c', 'b'])
        np.testing.assert_array_equal(compute_scores(frame, self.y, 'sis'),
                                      compute_scores(self.X, self.y, 'sis'))

    def test_failure_is_not_zero_padding(self):
        with patch.object(other, '_score_sage', return_value=(np.ones(2), {})):
            with self.assertRaises(RuntimeError):
                compute_scores(self.X, self.y, 'sage_cb', device='cpu')

    def test_all_families_and_aggregation_artifacts(self):
        names = set(available_methods())
        self.assertTrue({'sis', 'dc', 'pace_sis', 'pace_dc', 'sage_cb', 'loco_cv',
                         'lgb_gain', 'tabicl_shap', 'xrfm_agop', 'sand_score'} <= names)
        self.assertNotEqual(selected_method_id('pace_sis', 'mean'), selected_method_id('pace_sis', 'max'))
        import f2ves_real_eval_full as runner
        for mode in ('mean', 'max'):
            name = selected_method_id('pace_sis', mode)
            parsed = runner.build_score_parser().parse_args(['--task_type', 'bin', '--methods', name])
            self.assertEqual(parsed.methods, [name])

    def test_simulation_runner_routes_raw_and_pace_through_public_api(self):
        import final_experiments_pace as runner
        from types import SimpleNamespace
        encoder = SimpleNamespace(loaded=True,
            get_embeddings_batch_all_layers=lambda X, layers: {2: self.E})
        with patch.object(runner, '_methods_for_task', return_value=(['sis', 'dc'], ['sis', 'dc'], 'unused')), \
             patch('src.screening_utilities.compute_scores', wraps=compute_scores) as called:
            result = runner.run_single_rep(self.X, self.y, [0], 1, 5, 30,
                encoder, emb_layers=[2], task='classification')
        self.assertEqual(called.call_count, 4)
        self.assertTrue({'sis', 'dc', 'emb_sis_L2', 'emb_dc_L2'} <= set(result))

    def test_simulation_other_tracks_route_through_public_api(self):
        from src.screening_utilities import pre_split_scores
        with patch('src.screening_utilities.compute_scores', return_value=np.arange(5)) as called:
            result = pre_split_scores(self.X[:20], self.y[:20], self.X[20:], self.y[20:],
                5, 'classification', [], ['sage_cb', 'lgb_gain', 'loco_cv'])
        self.assertEqual(called.call_count, 3)
        self.assertEqual(set(result), {'sage_cb', 'lgb_gain', 'loco_cv'})


if __name__ == '__main__':
    unittest.main()
