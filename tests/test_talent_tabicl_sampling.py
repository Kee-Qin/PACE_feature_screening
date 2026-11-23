"""Sampling/provenance tests without fitting models or running SHAP."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import src.pace_real.common as common
import src.pace_real.prediction as predict
import f2ves_real_eval_full as score
from src import screening_methods_torch_A as algorithms


class TabICLSamplingTests(unittest.TestCase):
    def setUp(self):
        self.args = score.build_score_parser().parse_args(["--task_type", "bin"])
        self.args.current_task = "bin"

    def prepared(self, n, task="bin"):
        # Broadcast row identifiers; no experimental outcomes or fitted model.
        X = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 2000))
        y = np.arange(n) % 2 if task != "reg" else np.arange(n, dtype=float)
        return {"Xc_train": X, "y_train": y}

    def test_default_500_rows_split_400_100_all_columns(self):
        block = self.prepared(20972)
        Xtr, Xv, ytr, yv = algorithms._method_screen_split("tabicl_shap", block, self.args)
        self.assertEqual(Xtr.shape, (400, 2000))
        self.assertEqual(Xv.shape, (100, 2000))
        self.assertEqual(len(set(Xtr[:, 0]) | set(Xv[:, 0])), 500)
        self.assertFalse(set(Xtr[:, 0]) & set(Xv[:, 0]))
        self.assertEqual(self.args.tabicl_eval_max, 100)
        self.assertEqual(set(ytr), {0, 1})
        self.assertEqual(set(yv), {0, 1})

    def test_actual_method_dispatch_uses_new_split(self):
        self.args.validate_only = False
        with patch.object(algorithms, "_score_tabicl_shap", return_value=(np.ones(2000), {})) as implementation:
            self.assertEqual(len(score._run_method("tabicl_shap", self.prepared(900),
                                              self.args, Path("unused.npz"))[0]), 2000)
        Xtr, ytr, Xv = implementation.call_args.args[:3]
        self.assertEqual((len(Xtr), len(ytr), len(Xv)), (400, 400, 100))

    def test_small_datasets_do_not_duplicate_or_overlap_rows(self):
        for n in (500, 200, 10):
            Xtr, Xv, _, _ = algorithms._method_screen_split("tabicl_shap", self.prepared(n), self.args)
            self.assertEqual(len(Xtr) + len(Xv), n)
            self.assertEqual(len(Xv), int(np.ceil(n * 0.2)))
            self.assertFalse(set(Xtr[:, 0]) & set(Xv[:, 0]))

    def test_regression_and_determinism(self):
        self.args.current_task = "reg"
        block = self.prepared(2000, "reg")
        first = algorithms._method_screen_split("tabicl_shap", block, self.args)
        second = algorithms._method_screen_split("tabicl_shap", block, self.args)
        self.assertEqual((len(first[0]), len(first[1])), (400, 100))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_loco_methods_keep_existing_split_and_parameters(self):
        for method in ("loco_simple", "loco_cv"):
            Xtr, Xv, _, _ = algorithms._method_screen_split(method, self.prepared(20972), self.args)
            self.assertEqual((len(Xtr), len(Xv)), (1300, 700))
            parameters = score._run_parameters(self.args, method)
            self.assertEqual(parameters["max_screen_samples"], 2000)
            self.assertNotIn("sampling_protocol", parameters)
            self.assertEqual(common.score_file("outputs", "bin", "example", 42, method).name,
                             method + ".npz")

    def test_lower_global_cap_still_works_for_validation(self):
        self.args.max_screen_samples = 64
        Xtr, Xv, _, _ = algorithms._method_screen_split("tabicl_shap", self.prepared(900), self.args)
        self.assertEqual((len(Xtr), len(Xv)), (51, 13))
        self.assertEqual(score._run_parameters(self.args, "tabicl_shap")["max_screen_samples"], 64)

    def test_new_artifact_name_shared_with_prediction(self):
        args = ("outputs", "bin", "Amazon_employee_access", 42, "tabicl_shap")
        path = common.score_file(*args)
        self.assertEqual(path.name, "tabicl_shap_rows500_split80_20_v1.npz")
        self.assertEqual(predict.score_file(*args), path)
        self.assertNotEqual(algorithms._partial_path(path).name, "tabicl_shap.partial.npz")
        self.assertEqual(score._run_parameters(self.args, "tabicl_shap")["max_screen_samples"], 500)
        self.assertEqual(score._run_parameters(self.args, "tabicl_shap")["screen_validation_fraction"], 0.2)

    def test_old_or_missing_protocol_rejected_by_predictor_validation(self):
        prepared = {}
        meta = {"schema": common.SCHEMA, "strict_full_p": True, "noise_scores_computed": True,
                "n_features_scored": 2000, "task_type": "bin", "dataset": "example",
                "seed": 42, "method": "tabicl_shap", **prepared,
                "configuration": {"sampling_protocol": common.TABICL_SHAP_SAMPLING_ID}}
        scores = np.zeros(2000)
        common.validate_score(scores, meta, prepared, "bin", "example", 42, "tabicl_shap")
        for config in ({}, {"sampling_protocol": "old_1300_256"}):
            with self.assertRaisesRegex(ValueError, "sampling protocol mismatch"):
                common.validate_score(scores, {**meta, "configuration": config}, prepared,
                                      "bin", "example", 42, "tabicl_shap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
