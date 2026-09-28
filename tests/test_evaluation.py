import json
import tempfile
import unittest
from pathlib import Path

from local_rag.evaluation import evaluate, load_cases


class EvaluationTest(unittest.TestCase):
    def test_loads_cases_and_calculates_recall_and_mrr(self):
        cases = [
            {"question": "one", "expected_source": "one.md"},
            {"question": "two", "expected_text": "needle"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            loaded = load_cases(path)

        def retrieve(profile, question):
            if question == "one":
                return [{"source": "/docs/one.md", "text": "anything"}]
            return [
                {"source": "wrong.md", "text": "no"},
                {"source": "right.md", "text": "the needle is here"},
            ]

        result = evaluate(loaded, ["test"], retrieve)[0]

        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["mrr"], 0.75)

    def test_rejects_cases_without_expected_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text('[{"question": "missing"}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
