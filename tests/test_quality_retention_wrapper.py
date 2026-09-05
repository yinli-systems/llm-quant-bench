import argparse
import importlib.util
import pathlib
import unittest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_quality_retention.py"
SPEC = importlib.util.spec_from_file_location("run_quality_retention", SCRIPT_PATH)
wrapper = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(wrapper)


class QualityRetentionWrapperTest(unittest.TestCase):
    def test_quality_eval_command_reuses_same_benchmarks(self):
        args = argparse.Namespace(
            benchmarks=["mmlu", "cmmlu"],
            concurrency=4,
            cmmlu_dir="/data/cmmlu/test",
            longbench_dir=None,
            longbench_tasks=None,
            mt_bench_file=None,
            limit=10,
            max_per_task=None,
        )
        cmd = wrapper.quality_eval_command(
            out=pathlib.Path("out"),
            base_url="http://baseline/v1",
            model="qwen-bf16",
            api_key=None,
            args=args,
        )
        self.assertIn("--benchmarks", cmd)
        self.assertIn("mmlu", cmd)
        self.assertIn("cmmlu", cmd)
        self.assertIn("--cmmlu-dir", cmd)
        self.assertIn("/data/cmmlu/test", cmd)


if __name__ == "__main__":
    unittest.main()
