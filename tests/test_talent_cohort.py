"""Dataset defaults, filtering, and worker-sharding checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from src.pace_real import common, prediction
import f2ves_real_eval_full as scoring


class DatasetTests(unittest.TestCase):
    def test_full_catalog_is_default(self):
        for task in common.TASKS:
            self.assertEqual(common.available_datasets(task), list(common.data_impl.DATASET_MAP[task]))
        units = list(common.iter_units('all', None, [42, 123, 456]))
        self.assertEqual(len(units), 900)
        self.assertEqual(len(set(units)), 900)

    def test_optional_names(self):
        self.assertEqual(list(common.iter_units('bin', ['banknote_authentication'], [42])),
                         [('bin', 'banknote_authentication', 42)])
        with self.assertRaises(ValueError):
            list(common.iter_units('bin', ['unknown-dataset'], [42]))

    def test_entry_defaults(self):
        for parser in (scoring.build_score_parser(), prediction.build_parser()):
            args = parser.parse_args(['--task_type', 'all'])
            self.assertEqual(args.cohort, 'all')
            self.assertIsNone(args.datasets)
            self.assertNotIn('--cohort', parser.format_help())

    def test_shards_cover_all_units(self):
        units = list(common.iter_units('all', None, [42, 123, 456]))
        flat = [unit for shard in range(3) for unit in units[shard::3]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(units))
