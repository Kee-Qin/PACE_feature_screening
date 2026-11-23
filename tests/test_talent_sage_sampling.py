"""SAGE 400/100 sampling and provenance tests; no model training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import src.pace_real.common as common
import src.pace_real.prediction as predict
import f2ves_real_eval_full as score
from src import screening_methods_torch_A as algorithms


class SAGESamplingTests(unittest.TestCase):
    def setUp(self):
        self.args = score.build_score_parser().parse_args(["--task_type", "bin"])
        self.args.current_task = "bin"

    def prepared(self, n, task="bin"):
        X = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 2000))
        y = np.arange(n) % 2 if task != "reg" else np.arange(n, dtype=float)
        return {"Xc_train": X, "y_train": y}

    def test_both_sage_variants_use_400_100_in_classification_and_regression(self):
        for task in ("bin", "reg"):
            self.args.current_task = task
            block = self.prepared(20972, task)
            for method in common.SAGE_METHODS:
                fit, ev, _, _ = algorithms._method_screen_split(method, block, self.args)
                self.assertEqual(fit.shape, (400, 2000))
                self.assertEqual(ev.shape, (100, 2000))
                self.assertEqual(len(set(fit[:, 0]) | set(ev[:, 0])), 500)
                self.assertFalse(set(fit[:, 0]) & set(ev[:, 0]))
                repeated = algorithms._method_screen_split(method, block, self.args)
                np.testing.assert_array_equal(fit, repeated[0])
                np.testing.assert_array_equal(ev, repeated[1])

    def test_actual_dispatch_passes_400_100_to_sage(self):
        self.args.validate_only = False
        for method in common.SAGE_METHODS:
            with patch.object(algorithms, "_score_sage", return_value=(np.ones(2000), {})) as implementation:
                self.assertEqual(len(score._run_method(method, self.prepared(900), self.args,
                                                  Path("unused.npz"))[0]), 2000)
            chosen, fit, yfit, ev, yev = implementation.call_args.args[:5]
            self.assertEqual(chosen, method)
            self.assertEqual((len(fit), len(yfit), len(ev), len(yev)), (400, 400, 100, 100))
            self.assertEqual(fit.shape[1], 2000)
            self.assertEqual(ev.shape[1], 2000)

    def test_small_training_sets_and_smaller_explicit_cap(self):
        for method in common.SAGE_METHODS:
            for n in (500, 200, 10):
                fit, ev, _, _ = algorithms._method_screen_split(method, self.prepared(n), self.args)
                self.assertEqual(len(fit) + len(ev), n)
                self.assertEqual(len(ev), int(np.ceil(n * 0.2)))
                self.assertFalse(set(fit[:, 0]) & set(ev[:, 0]))
        self.args.max_screen_samples = 64
        for method in common.SAGE_METHODS:
            fit, ev, _, _ = algorithms._method_screen_split(method, self.prepared(900), self.args)
            self.assertEqual((len(fit), len(ev)), (51, 13))

    def test_monte_carlo_budget_unchanged(self):
        self.assertEqual(self.args.sage_permutations, 32)
        self.assertEqual(self.args.sage_imputations, 32)
        self.assertEqual(self.args.sage_background, 200)
        self.assertEqual(self.args.sage_eval_max, 0)
        for method in common.SAGE_METHODS:
            params = score._run_parameters(self.args, method)
            self.assertEqual(params["max_screen_samples"], 500)
            self.assertEqual(params["screen_validation_fraction"], 0.2)
            self.assertEqual(params["sampling_protocol"], common.SAGE_SAMPLING_ID)

    def test_new_score_names_shared_with_prediction(self):
        for method in common.SAGE_METHODS:
            values = ("outputs", "bin", "Amazon_employee_access", 42, method)
            path = common.score_file(*values)
            self.assertEqual(path.name, f"{method}_rows500_split80_20_v1.npz")
            self.assertEqual(predict.score_file(*values), path)
            self.assertNotEqual(algorithms._partial_path(path).name, method + ".partial.npz")

    def test_legacy_sampling_protocol_rejected_for_both_variants(self):
        prepared = {}
        for method in common.SAGE_METHODS:
            meta = {"schema": common.SCHEMA, "strict_full_p": True,
                    "noise_scores_computed": True, "n_features_scored": 2000,
                    "task_type": "bin", "dataset": "example", "seed": 42,
                    "method": method, **prepared,
                    "configuration": {"sampling_protocol": common.SAGE_SAMPLING_ID}}
            scores = np.zeros(2000)
            common.validate_score(scores, meta, prepared, "bin", "example", 42, method)
            for config in ({}, {"sampling_protocol": "old_1300_700"}):
                with self.assertRaisesRegex(ValueError, "SAGE sampling protocol mismatch"):
                    common.validate_score(scores, {**meta, "configuration": config},
                                          prepared, "bin", "example", 42, method)


if __name__ == "__main__":
    unittest.main(verbosity=2)
