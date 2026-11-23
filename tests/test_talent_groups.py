"""TALENT fixed-budget evaluation and resume contracts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import f2ves_real_eval_full as full
import f2ves_real_gaussian_large_experiments as data


class TalentGroupTests(unittest.TestCase):
    def test_talent_sources_use_supported_groups(self):
        root = Path(__file__).resolve().parents[1]
        names = ('f2ves_real_eval_full.py', 'f2ves_real_gaussian_large_experiments.py',
                 'eval_gotabpfn.py', 'eval_tabpfnwide.py', 'src/pace_real/prediction.py')
        for name in names:
            self.assertNotIn('g' + str(4), (root/name).read_text(encoding='utf-8').lower(), name)

    def test_main_runner_fits_baselines_and_one_selected_table(self):
        frame = pd.DataFrame(np.arange(24).reshape(12, 2).astype(float))
        y = np.arange(12) % 2
        splits = (frame, y, frame.copy(), y.copy(), frame.copy(), y.copy())
        scores = {'sis': np.arange(12, dtype=float)}
        for task in ('bin', 'reg'):
            with patch.object(data, 'load_dataset', return_value=splits), \
                 patch.object(full, 'TARGET_TOTAL_FEATURES', 12), \
                 patch.object(full, 'load_screening_scores', return_value=(scores, {}, set())), \
                 patch.object(full, 'train_and_evaluate', return_value={'acc': .8}) as fit:
                result = full.run_single_dataset('fixture', task, 42, 'cpu',
                    models_filter=['lgb'], scr_filter=['sis'], predict_only=True)
                self.assertEqual(set(result['lgb']), {'g1_clean', 'g2_gaussian', 'g3_sis'})
                self.assertEqual([call.args[1].shape[1] for call in fit.call_args_list], [2, 12, 6])
                fit.reset_mock()
                full.run_single_dataset('fixture', task, 42, 'cpu', models_filter=['lgb'],
                    scr_filter=['sis'], predict_only=True, existing=result)
                fit.assert_not_called()

    def test_dataset_runner_fits_only_supported_groups(self):
        frame = pd.DataFrame(np.arange(24).reshape(12, 2).astype(float))
        y = np.arange(12) % 2
        splits = (frame, y, frame.copy(), y.copy(), frame.copy(), y.copy())
        with patch.object(data, 'load_dataset', return_value=splits), \
             patch.object(data, 'MODELS', ['lgb']), \
             patch.object(data, 'TARGET_TOTAL_FEATURES', 12), \
             patch.object(data, 'SCREENING_METHODS', {'bin':['sis_emb']}), \
             patch.object(data, 'compute_screening_scores', return_value={'sis_emb': np.arange(12)}), \
             patch.object(data, 'train_and_evaluate', return_value={'acc': .8}) as fit:
            result=data.run_single_dataset('fixture','bin',42,'cpu')
            self.assertEqual(set(result['lgb']), {'g1_clean','g2_gaussian','g3_sis_emb'})
            self.assertEqual(fit.call_count,3)
            fit.reset_mock()
            data.run_single_dataset('fixture','bin',42,'cpu',existing_result=result)
            fit.assert_not_called()

    def test_deep_and_nn_completion_checks(self):
        with patch.object(full, 'GPU12', ['sis']):
            self.assertTrue(full._model_complete_dl({'g1_clean':{},'g2_gaussian':{},'g3_sis':{}},'bin'))
        result={'lgb':{'g1_clean':{},'g2_gaussian':{},'g3_sand_score':{},'g_native_sand_score':{}}}
        self.assertTrue(full._E_models_complete(result,['sand_score'],['lgb'],'bin'))


if __name__ == '__main__':
    unittest.main()
