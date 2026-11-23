"""Validate package imports and the experiment-to-src boundary."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHARED = {p.stem for p in (ROOT/'src').glob('*.py')} | {'pace_real', 'talent_benchmark'}


class SrcLayoutTests(unittest.TestCase):
    def test_no_unqualified_shared_imports(self):
        for path in ROOT.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    self.assertNotIn(node.module.split('.')[0], SHARED, str(path))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name.split('.')[0], SHARED, str(path))

    def test_ablation_encoders_are_shared(self):
        import f2ves_ablation as experiment
        from src import ablation_encoders
        for name in ('TabICLProjectionEmbedder', 'TabPFNColumnEmbedder', 'TabPFNV3ColumnEmbedder'):
            self.assertIs(getattr(experiment, name), getattr(ablation_encoders, name))

    def test_prediction_worker_entry_exists(self):
        from src import model_path
        self.assertTrue((model_path.CODE_ROOT/'f2ves_real_eval_full.py').is_file())
        self.assertFalse((ROOT/'summarize_gotabpfn_progress.py').exists())


if __name__ == '__main__':
    unittest.main()
