import importlib.util
import json
import pathlib
import tempfile
import unittest

SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_quality_retention.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_quality_retention", SCRIPT_PATH
)
retention = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(retention)


class QualityRetentionScriptTest(unittest.TestCase):
    def test_paired_analysis_rejects_missing_nonfinite_and_boolean_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = pathlib.Path(temp_dir) / "base.jsonl"
            cand_path = pathlib.Path(temp_dir) / "candidate.jsonl"
            row = {
                "benchmark": "mmlu",
                "task": "x",
                "item_id": "1",
                "ok": True,
                "score": 1.0,
            }
            base_path.write_text(json.dumps(row))
            for value in (None, float("nan"), float("inf"), True, "1"):
                with self.subTest(value=value):
                    cand_path.write_text(json.dumps({**row, "score": value}))
                    with self.assertRaisesRegex(ValueError, "finite numeric scores"):
                        retention.paired_analysis(base_path, cand_path)

    def test_paired_analysis_rejects_truthy_request_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "rows.jsonl"
            row = {
                "benchmark": "mmlu",
                "task": "x",
                "item_id": "1",
                "ok": "false",
                "score": 1,
            }
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "successful requests"):
                retention.paired_analysis(path, path)

    def test_paired_analysis_rejects_empty_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "empty.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                retention.paired_analysis(path, path)

    def test_retention_ratio(self):
        self.assertAlmostEqual(retention.compute_retention(0.78, 0.8), 0.975)

    def test_missing_or_zero_baseline_returns_none(self):
        self.assertIsNone(retention.compute_retention(0.78, 0.0))
        self.assertIsNone(retention.compute_retention(None, 0.8))

    def test_paired_analysis_preserves_item_identity(self):
        baseline = [
            {
                "benchmark": "mmlu",
                "task": "x",
                "item_id": "1",
                "ok": True,
                "score": 1.0,
            },
            {
                "benchmark": "mmlu",
                "task": "x",
                "item_id": "2",
                "ok": True,
                "score": 0.0,
            },
        ]
        candidate = [
            {
                "benchmark": "mmlu",
                "task": "x",
                "item_id": "2",
                "ok": True,
                "score": 1.0,
            },
            {
                "benchmark": "mmlu",
                "task": "x",
                "item_id": "1",
                "ok": True,
                "score": 1.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = pathlib.Path(temp_dir) / "base.jsonl"
            cand_path = pathlib.Path(temp_dir) / "cand.jsonl"
            base_path.write_text("".join(json.dumps(row) + "\n" for row in baseline))
            cand_path.write_text("".join(json.dumps(row) + "\n" for row in candidate))
            result = retention.paired_analysis(base_path, cand_path)
        self.assertEqual(result["mmlu"]["candidate_only_wins"], 1)
        self.assertEqual(result["mmlu"]["paired_items"], 2)


if __name__ == "__main__":
    unittest.main()
