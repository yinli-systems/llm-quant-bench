import importlib.util
import pathlib
import sys
import unittest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "summarize_repeats_ci.py"
SPEC = importlib.util.spec_from_file_location("summarize_repeats_ci", SCRIPT_PATH)
ci = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["summarize_repeats_ci"] = ci
SPEC.loader.exec_module(ci)


class RepeatsCITest(unittest.TestCase):
    def test_describe_three_values(self):
        row = ci.describe([10.0, 12.0, 14.0])
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["mean"], 12.0)
        self.assertGreater(row["ci95_half_width"], 0.0)

    def test_summarize_repeats_groups_by_concurrency(self):
        summary = ci.summarize_repeats(
            [
                {
                    "concurrency": 1,
                    "requests": {"success_rate": 1.0},
                    "tokens": {"output_token_throughput": 10.0},
                    "goodput": {"request_goodput": 2.0},
                },
                {
                    "concurrency": 1,
                    "requests": {"success_rate": 1.0},
                    "tokens": {"output_token_throughput": 12.0},
                    "goodput": {"request_goodput": 4.0},
                },
            ]
        )
        metric = summary["groups"]["1"]["metrics"]["output_token_throughput"]
        self.assertAlmostEqual(metric["mean"], 11.0)
        goodput = summary["groups"]["1"]["metrics"]["request_goodput"]
        self.assertAlmostEqual(goodput["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
