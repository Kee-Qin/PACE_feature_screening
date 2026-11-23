"""Tests for shared PACE aggregation and TALENT artifact selection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
from src import screening_methods_torch as smt
from src.pace_real import common, prediction
import f2ves_real_eval_full as scoring
from src.screening_utilities import compute_scores, selected_method_id
from src.screening_utils import sample_rows


class SISMaxTests(unittest.TestCase):
    def test_all_datasets_are_default(self):
        self.assertEqual([len(common.available_datasets(t, "all")) for t in common.TASKS],
                         [120, 80, 100])
        self.assertEqual(len(list(common.iter_units("all", None, [42, 123, 456], "all"))), 900)
        self.assertEqual(len(list(common.iter_units("all", None, [42, 123, 456]))), 900)

    def test_method_is_opt_in_and_predictable(self):
        for module in (scoring, prediction):
            parser = module.build_score_parser() if module is scoring else module.build_parser()
            self.assertNotIn("sis_emb_max", parser.parse_args(["--task_type", "all"]).methods)
            args = parser.parse_args(["--task_type", "all", "--cohort", "all",
                                      "--methods", "sis_emb_max", "--per_gpu", "2"])
            self.assertEqual(args.methods, ["sis_emb_max"])
            self.assertEqual(args.per_gpu, 2)
        self.assertEqual(len(prediction.MODELS), 10)

    def test_original_row_sampling(self):
        X = np.arange(2500 * 2).reshape(2500, 2)
        y = np.linspace(0, 1, 2500)
        a, b = sample_rows(X, y, "reg", 2000)
        expected = np.random.RandomState(42).choice(2500, 2000, replace=False)
        np.testing.assert_array_equal(a, X[expected])
        np.testing.assert_array_equal(b, y[expected])
        np.testing.assert_array_equal(sample_rows(X[:100], y[:100], "bin", 2000)[0], X[:100])

    def test_mean_matches_original_and_max_matches_coordinate_oracle(self):
        smt.DEVICE = torch.device("cpu")
        torch.manual_seed(123)
        E = torch.randn(70, 5, 128)
        E[:, 0, :] = 1
        for y in (np.arange(70) % 2, np.arange(70) % 4,
                  np.linspace(-2, 2, 70), np.ones(70)):
            X = np.zeros(E.shape[:2])
            out = {mode: compute_scores(X, y, 'pace_sis', embeddings=E,
                                        aggregation=mode) for mode in ('mean', 'max')}
            old = smt.TorchAcceleratedScreening.emb_sis_accelerated(E, y)
            np.testing.assert_allclose(out["mean"], old, rtol=1e-5, atol=1e-6)
            oracle = np.stack([smt.TorchAcceleratedScreening.emb_sis_accelerated(
                E[:, :, d:d+1], y) for d in range(128)]).max(0)
            np.testing.assert_allclose(out["max"], oracle, atol=1e-6)
            self.assertTrue(np.all(out["max"] + 1e-6 >= out["mean"]))
            self.assertEqual(out["max"][0], 0)

    def test_separate_method_artifacts_and_same_g3_budget(self):
        a = common.score_file("out", "bin", "demo", 42, "sis_emb")
        b = common.score_file("out", "bin", "demo", 42, "sis_emb_max")
        self.assertNotEqual(a, b)
        self.assertEqual(len(prediction._g3_columns(np.arange(2000), 13)), 39)

    def test_cli_aggregation_names_separate_scores(self):
        for method in ('pace_sis', 'pace_psis', 'pace_cru'):
            self.assertNotEqual(selected_method_id(method, 'mean'), selected_method_id(method, 'max'))
        for method in ('pace_dc', 'pace_hsic'):
            self.assertEqual(selected_method_id(method, 'max'), method)


if __name__ == "__main__":
    unittest.main()
