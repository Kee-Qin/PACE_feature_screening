"""Keep published code comments and CLI text English and free of change history."""
import ast
import io
from pathlib import Path
import re
import tokenize
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceHygieneTests(unittest.TestCase):
    def test_python_source_contains_no_chinese(self):
        paths = [*ROOT.glob('*.py'), *ROOT.glob('src/**/*.py'), *ROOT.glob('tests/*.py')]
        for path in paths:
            self.assertIsNone(re.search(r'[\u3400-\u9fff]', path.read_text(encoding='utf-8-sig')), str(path))

    def test_comments_describe_current_behavior(self):
        pattern = re.compile(r'legacy|historical|previously|supersed|retired|migration|'
                             r'formerly|CLEANUP_REVIEW|verbatim|unchanged|merged from|used to|\bold\b|'
                             r'original (?:numerical|call|implementation|framework|_unit)|stuck after', re.I)
        for path in [*ROOT.glob('*.py'), *ROOT.glob('src/**/*.py')]:
            source = path.read_text(encoding='utf-8-sig')
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    self.assertIsNone(pattern.search(token.string), f'{path}:{token.start[0]}')
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(pattern.search(ast.get_docstring(node) or ''), str(path))


if __name__ == '__main__':
    unittest.main()
