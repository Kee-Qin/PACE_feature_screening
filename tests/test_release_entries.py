"""Release-entry regression tests; no GPU or model checkpoints required."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from src import simulation_protocol as protocol
from src import data_generators_classification as classification
from src import data_generators_regression as regression
from src.pace_real import common, prediction
import f2ves_real_eval_full as scoring

ROOT = Path(__file__).resolve().parent.parent


class SimulationTests(unittest.TestCase):
    def test_registry_contains_only_paper_settings(self):
        total = repetitions = 0
        for task, module in [('classification', classification), ('regression', regression)]:
            self.assertEqual(list(module.EXPERIMENTS), list(protocol.scenarios(task)))
            for exp, ps, reps in protocol.execution_plan(task):
                self.assertEqual(module.PARAMETER_SETS[exp], ps)
                self.assertTrue(module.EXPERIMENTS[exp].__name__.startswith(exp + '_'))
                total += len(ps)
                repetitions += len(ps) * reps
                for invalid in set([1, 2, 3]) - set(ps):
                    with self.assertRaises(KeyError):
                        module.EXPERIMENTS[exp](param_set=invalid, seed=0)
        self.assertEqual((total, repetitions), (25, 1100))

    def test_selection_order_validation_and_determinism(self):
        self.assertEqual(protocol.experiments('classification', 'exp6,exp1'), ['exp1', 'exp6'])
        self.assertEqual(protocol.settings('regression', 'exp3', [3, 2]), [2, 3])
        with self.assertRaises(ValueError):
            protocol.experiments('classification', 'exp10')
        with self.assertRaises(ValueError):
            protocol.settings('regression', 'exp3', [1])
        with self.assertRaises(ValueError):
            protocol.execution_plan('classification', n_reps=0)
        code = "from src import simulation_protocol as p; print(p.simulation_seed('regression','exp4',2,1))"
        outputs = [subprocess.check_output([sys.executable, '-B', '-c', code], cwd=ROOT) for _ in range(2)]
        self.assertEqual(outputs[0], outputs[1])

    def test_output_protocol_rejects_historical_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'new'
            protocol.ensure_output_protocol(root, 'classification', 'pace')
            protocol.ensure_output_protocol(root, 'classification', 'pace')
            with self.assertRaises(ValueError):
                protocol.ensure_output_protocol(root, 'regression', 'pace')
            old = Path(tmp) / 'old'
            old.mkdir()
            (old / 'checkpoint.json').write_text('{}')
            with self.assertRaises(ValueError):
                protocol.ensure_output_protocol(old, 'classification', 'pace')
            self.assertEqual((old / 'checkpoint.json').read_text(), '{}')

    def test_all_runner_plans_match(self):
        import final_experiments_pace as pace
        import final_experiments_nostats as nostats
        import f2ves_ablation as ablation
        for task in ('classification', 'regression'):
            expected = protocol.execution_plan(task)
            self.assertEqual(pace.resolve_exp_keys('all', task), [x[0] for x in expected])
            for exp, settings, _ in expected:
                self.assertEqual(pace.get_param_sets(exp, task), settings)
            args = argparse.Namespace(task=task, exp='all', param_sets=None, n_reps=None,
                                      ultra_reps=25, ultra_reps_reg=25)
            for track in 'ABCDE':
                self.assertEqual(nostats._resolve_track_track(track, args), expected)
            prefix = 'cls_' if task == 'classification' else 'reg_'
            self.assertEqual(ablation._resolve_plan(args), [(prefix+e, ps, n) for e, ps, n in expected])
        args = argparse.Namespace(task='all', exp='cls_exp6,reg_exp4', param_sets=None,
                                  n_reps=1, ultra_reps=25, ultra_reps_reg=25)
        self.assertEqual(ablation._resolve_plan(args), [('cls_exp6', [1, 2], 1), ('reg_exp4', [2, 3], 1)])

    def test_generated_data_is_reproducible(self):
        for module, exp, setting in [(classification, 'exp1', 1), (regression, 'exp1', 2)]:
            first = module.EXPERIMENTS[exp](param_set=setting, seed=42)
            second = module.EXPERIMENTS[exp](param_set=setting, seed=42)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(first[0].shape, (first[3], first[4]))


class StageIntegrationTests(unittest.TestCase):
    def test_loco_executes_full_width_cpu_fits(self):
        rng = np.random.RandomState(8)
        X = rng.normal(size=(60, 2000)).astype('float32')
        for task in ('bin', 'reg'):
            y = np.arange(60) % 2 if task == 'bin' else X[:, 0] + rng.normal(size=60)
            args = scoring.build_score_parser().parse_args(['--task_type', task, '--rf-trees', '1', '--rf-n-jobs', '1', '--rf-max-depth', '2'])
            for method in ('loco_simple', 'loco_cv'):
                result = scoring._validation_probe(method, X[:40], y[:40], X[40:], y[40:], task, args, Path('unused.npz'))
                self.assertEqual(result['P'], 2000)

    def test_saved_scores_feed_predictions_and_resume(self):
        """Exercise actual artifacts/selection/resume with only the model fits mocked."""
        X = np.random.RandomState(1).normal(size=(30, 2000)).astype('float32')
        block = {'n': 30, 'p_original': 2, 'n_noise': 1998,
                 'Xc_train': X[:20], 'Xc_val': X[20:25], 'Xc_test': X[25:],
                 'y_train': np.arange(20) % 2, 'y_val': np.arange(5) % 2,
                 'y_test': np.arange(5) % 2, 'cat_indices': []}
        scores = np.arange(2000, dtype=float)
        model_output = {'accuracy': .8, 'auc': .75}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            score_root, pred_root = Path(tmp)/'scores', Path(tmp)/'predictions'
            base = ['runner', '--task_type', 'bin', '--datasets', 'banknote_authentication', '--seeds', '42', '--methods', 'sage_cb']
            config = {'sampling_protocol': common.SAGE_SAMPLING_ID}
            with patch.object(sys, 'argv', base + ['--output-root', str(score_root)]), \
                 patch.object(scoring, 'prepare_full2000', return_value=block), \
                 patch.object(scoring, '_run_method', return_value=(scores, config)) as fit:
                scoring.score_main()
                self.assertEqual(fit.call_count, 1)
            saved = common.score_file(score_root, 'bin', 'banknote_authentication', 42, 'sage_cb')
            sc, meta = common.read_score(saved)
            np.testing.assert_array_equal(sc, scores)
            common.validate_score(sc, meta, block, 'bin', 'banknote_authentication', 42, 'sage_cb')
            self.assertNotIn('feature_fingerprint', meta)
            self.assertNotIn('target_fingerprint', meta)
            cmd = base + ['--score-roots', str(score_root), '--output-root', str(pred_root), '--model-group', 'cpu', '--models', 'lgb', '--resume']
            old_temp = tempfile.tempdir
            try:
                with patch.object(sys, 'argv', cmd), \
                     patch.object(prediction, 'prepare_full2000', return_value=block), \
                     patch.object(prediction, 'configure_model_paths', return_value=object()), \
                     patch.object(prediction, '_model_preflight'), \
                     patch.object(prediction, '_evaluate', return_value=model_output) as evaluate:
                    prediction.main()
                    self.assertEqual(evaluate.call_args.args[2].shape, (20, 6))
                    np.testing.assert_array_equal(evaluate.call_args.args[2], X[:20, 1999:1993:-1])
                    prediction.main()
                    self.assertEqual(evaluate.call_count, 1)
            finally:
                tempfile.tempdir = old_temp
            output = json.loads(common.prediction_file(pred_root, 'bin', 'banknote_authentication', 42).read_text())
            self.assertEqual(set(output['models']['lgb']), {'g3_sage_cb'})
            self.assertEqual(output['models']['lgb']['g3_sage_cb']['noise_selected'], 6)

    def test_parallel_children_use_canonical_stage(self):
        from unittest.mock import MagicMock
        for stage, module in [('score', scoring), ('predict', prediction)]:
            with tempfile.TemporaryDirectory() as tmp:
                parser = module.build_score_parser() if stage == 'score' else module.build_parser()
                args = parser.parse_args(['--task_type', 'bin', '--datasets', 'banknote_authentication', '--seeds', '42', '--gpus', '0', '--per_gpu', '2', '--output-root', tmp])
                process = MagicMock()
                process.poll.return_value = 0
                with patch('torch.cuda.device_count', return_value=1), \
                     patch('subprocess.Popen', return_value=process) as launch:
                    common.dispatch_gpu_workers(args, parser, ROOT/'f2ves_real_eval_full.py', stage=stage)
                cmd = launch.call_args.args[0]
                self.assertEqual(Path(cmd[2]).name, 'f2ves_real_eval_full.py')
                self.assertEqual(cmd[3:5], ['--stage', stage])
                self.assertIn('--worker', cmd)
                self.assertNotIn('--per_gpu', cmd)

    def test_correct_canonical_module_is_imported(self):
        import f2ves_real_eval_full as module
        self.assertTrue(callable(module.train_and_evaluate))
        self.assertTrue(callable(module.train_and_evaluate_dl))
        self.assertTrue(set(module.STAGED_METHODS) <= set(module.SCREEN_METHODS))
        self.assertFalse(set(module.STAGED_METHODS) & set(module.GPU12))


class GOTabPFNTests(unittest.TestCase):
    def test_both_heads_use_large_table_option(self):
        for name in ('eval_gotabpfn.py',):
            tree = ast.parse((ROOT/name).read_text(encoding='utf-8'))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('TabPFNClassifier', 'TabPFNRegressor')]
            self.assertEqual(len(calls), 2)
            for call in calls:
                kw = {x.arg: x.value for x in call.keywords}
                self.assertTrue(ast.literal_eval(kw['ignore_pretraining_limits']))
                self.assertEqual(ast.unparse(kw['random_state']), 'seed')

    def test_g2_flags_and_resume_are_canonical(self):
        source = (ROOT/'eval_gotabpfn.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        # Execute the actual completion function against fixture JSONs, without
        # importing incompatible model-library versions into the test environment.
        names = {'_has_primary_metric', '_dataset_json_path', '_unit_done'}
        subset = ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in names], type_ignores=[])
        ns = {'json': json, 'Path': Path, 'MODEL_NAME': 'gotabpfn', 'TASK_DIR_MAP': {'bin': 'binclass'}, 'TASK_DIR': {'bin': 'binclass'}}
        exec(compile(subset, str(ROOT/'eval_gotabpfn.py'), 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = ns['_dataset_json_path'](root, 'bin', 'example', 42)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'gotabpfn': {'g2_gaussian': {'auc': .8, 'accuracy': .75, 'n_features': 2000}}}))
            self.assertTrue(ns['_unit_done'](root, 'bin', 'example', [42], require_g2=True, g2_only=True))
            self.assertFalse(ns['_unit_done'](root, 'bin', 'example', [42], require_g2=True, g2_only=False))
        self.assertIn('cmd.append("--g2-only")', source)
        self.assertIn('cmd.append("--g2")', source)
        self.assertIn('"--g2-only"', source)
        self.assertIn('"--per_gpu"', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
