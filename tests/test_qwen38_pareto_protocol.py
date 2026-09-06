import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "protocols" / "qwen38_27b_bf16_fp8_pareto_paracloud_v1.json"
PROTOCOL_V2_PATH = REPO_ROOT / "protocols" / "qwen38_27b_bf16_fp8_pareto_paracloud_v2.json"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_qwen38_pareto_protocol.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_qwen38_standard_eval.py"
STAGER_PATH = REPO_ROOT / "scripts" / "stage_qwen38_models.py"
SUMMARY_PATH = REPO_ROOT / "scripts" / "summarize_qwen38_quality.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


validator = load_module("validate_qwen38_pareto_protocol", VALIDATOR_PATH)
runner = load_module("run_qwen38_standard_eval", RUNNER_PATH)
stager = load_module("stage_qwen38_models", STAGER_PATH)
quality_summary = load_module("summarize_qwen38_quality", SUMMARY_PATH)


class Qwen38ParetoProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_frozen_protocol_passes_static_validation(self):
        checks = validator.validate_static(self.protocol)
        self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_local_gpqa_rejects_unverified_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv = pathlib.Path(temp_dir) / "gpqa.csv"
            csv.write_text("Question,Correct Answer\nexample,A\n")
            self.assertFalse(validator.validate_gpqa_csv(csv))
            with self.assertRaisesRegex(ValueError, "pinned upstream content"):
                runner.prepare_task_overlays(
                    self.protocol, lm_eval_root=pathlib.Path("/pinned/lm-eval"),
                    out=pathlib.Path(temp_dir), tasks=["gpqa_diamond_cot_zeroshot"],
                    gpqa_csv=csv,
                )

    def test_v3_reasoning_budget_is_enforced_and_passed_to_harness(self):
        protocol = json.loads((REPO_ROOT / "protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v3.json").read_text())
        self.assertTrue(all(c["ok"] for c in validator.validate_static(protocol)))
        command = runner.build_command(
            protocol, model_path=pathlib.Path("/models/q38"), out=pathlib.Path("/runs/test"),
            tasks=["gpqa_diamond_cot_zeroshot"], include_path=pathlib.Path("/overlays"),
            execution_tasks=["qwen38_gpqa_diamond_cot_zeroshot_pinned"], limit=1,
        )
        self.assertEqual(json.loads(command[command.index("--gen_kwargs") + 1]), {"max_gen_toks": 32768})
        protocol["quality"]["max_gen_toks"] = 256
        self.assertFalse(all(c["ok"] for c in validator.validate_static(protocol)))

    def test_storage_recovery_protocol_passes_static_validation(self):
        protocol = json.loads(PROTOCOL_V2_PATH.read_text(encoding="utf-8"))
        checks = validator.validate_static(protocol)
        self.assertTrue(all(check["ok"] for check in checks), checks)
        preflight = protocol["resource_preflight"]
        self.assertEqual(preflight["minimum_free_disk_gib_before_model_staging"], 128)
        self.assertEqual(preflight["minimum_free_disk_gib_before_runtime_staging"], 48)
        self.assertEqual(preflight["minimum_free_disk_gib_for_execution"], 32)

    def test_v4_preserves_text_backend_and_thinking_arguments(self):
        protocol = json.loads((REPO_ROOT / "protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v4.json").read_text())
        self.assertTrue(all(c["ok"] for c in validator.validate_static(protocol)))
        command = runner.build_command(
            protocol, model_path=pathlib.Path("/models/q38"), out=pathlib.Path("/runs/test"),
            tasks=["gpqa_diamond_cot_zeroshot"], include_path=pathlib.Path("/overlays"),
            execution_tasks=["qwen38_gpqa_diamond_cot_zeroshot_pinned"], limit=1,
        )
        self.assertEqual(command[command.index("--model") + 1], "vllm")
        arguments = json.loads(command[command.index("--model_args") + 1])
        self.assertTrue(arguments["enable_thinking"])
        self.assertEqual(arguments["think_end_token"], "</think>")
        self.assertEqual(arguments["chat_template_args"], {"reasoning_effort": "xhigh", "preserve_thinking": True})
        protocol["quality"]["backend"] = "vllm-vlm"
        self.assertFalse(all(c["ok"] for c in validator.validate_static(protocol)))

    def test_slurm_caches_are_job_local(self):
        script = (REPO_ROOT / "cluster/slurm/qwen38_27b_standard_quality_4x4090.sbatch").read_text()
        self.assertIn('source "$SRC/scripts/qwen38_job_env.sh"', script)
        environment = (REPO_ROOT / "scripts/qwen38_job_env.sh").read_text()
        for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "FLASHINFER_WORKSPACE_BASE"):
            self.assertRegex(environment, rf'export {name}="\$TMPDIR/[^"\n]+"')
        self.assertIn('[[ -x "$CUDA_HOME/bin/nvcc"', environment)
        self.assertIn('$(dirname "$PY"):$CUDA_HOME/bin:$PATH', environment)
        self.assertIn('command -v ninja', environment)

    def test_mutated_gate_is_rejected(self):
        self.protocol["gates"]["minimum_quality_retention"] = 0.9
        failures = [
            check["name"]
            for check in validator.validate_static(self.protocol)
            if not check["ok"]
        ]
        self.assertIn("pre-registered acceptance gates", failures)

    def test_task_version_parser_reads_pinned_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = pathlib.Path(temp_dir) / "task.yaml"
            config.write_text("task: example\nmetadata:\n  version: 3.1\n", encoding="utf-8")
            self.assertEqual(validator.metadata_version(config), "3.1")

    def test_runtime_uses_execution_disk_floor_after_receipts_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            for arm in self.protocol["arms"].values():
                model = root / arm["runtime_relative_path"]
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}\n", encoding="utf-8")
                stager.write_receipt(model, arm)
            self._write_dataset_receipt(root)
            original = validator.python_package_version
            validator.python_package_version = lambda _python, _package: "0.23.0"
            try:
                checks = validator.validate_runtime(self.protocol, root)
            finally:
                validator.python_package_version = original
            disk_check = next(check for check in checks if check["name"] == "runtime free disk")
            self.assertIn("minimum_free_disk_gib_for_execution", disk_check["detail"])

    def _write_dataset_receipt(self, root):
        preflight = self.protocol["resource_preflight"]
        datasets = []
        for task in self.protocol["quality"]["standard_lane"]["tasks"]:
            split_rows = {}
            for split, rows in task["expected_split_rows"].items():
                cache = root / "hf-datasets" / f"{task['name']}-{split}.arrow"
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(f"{task['name']}:{split}".encode())
                split_rows[split] = {
                    "rows": rows,
                    "fingerprint": f"fingerprint-{split}",
                    "cache_files": [
                        {
                            "path": str(cache.relative_to(root)),
                            "bytes": cache.stat().st_size,
                            "sha256": validator.sha256(cache),
                        }
                    ],
                }
            datasets.append(
                {
                    "name": task["name"],
                    "dataset_id": task["dataset_id"],
                    "revision": task["dataset_revision"],
                    "config": task.get("dataset_config"),
                    "splits": split_rows,
                }
            )
        receipt = {
            "protocol_version": self.protocol["protocol_version"],
            "offline_reload_verified": True,
            "datasets": datasets,
        }
        receipt_path = root / preflight["dataset_receipt_relative_path"]
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_bytes = (json.dumps(receipt, indent=2) + "\n").encode()
        receipt_path.write_bytes(receipt_bytes)
        marker = root / preflight["dataset_ready_marker_relative_path"]
        marker.write_text(validator.hashlib.sha256(receipt_bytes).hexdigest() + "\n", encoding="utf-8")

    def test_model_receipt_verifies_revision_and_file_content(self):
        arm = self.protocol["arms"]["baseline"]
        with tempfile.TemporaryDirectory() as temp_dir:
            model = pathlib.Path(temp_dir)
            config = model / "config.json"
            config.write_text('{"model_type":"qwen"}\n', encoding="utf-8")
            stager.write_receipt(model, arm)
            ok, _ = validator.validate_model_receipt(
                model,
                arm,
                marker_name=".READY",
                require_hashes=True,
                verify_content=True,
            )
            self.assertTrue(ok)
            original = config.read_bytes()
            config.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            ok, detail = validator.validate_model_receipt(
                model,
                arm,
                marker_name=".READY",
                require_hashes=True,
                verify_content=True,
            )
            self.assertFalse(ok)
            self.assertIn("content hash mismatch", detail)

    def test_runner_builds_matched_frozen_command(self):
        command = runner.build_command(
            self.protocol,
            model_path=pathlib.Path("/models/qwen38"),
            out=pathlib.Path("/runs/baseline"),
            tasks=["gpqa_diamond_cot_zeroshot", "mmlu_pro"],
            include_path=pathlib.Path("/runs/baseline/task_overrides"),
            execution_tasks=[
                "qwen38_gpqa_diamond_cot_zeroshot_pinned",
                "qwen38_mmlu_pro_pinned",
            ],
            limit=1,
        )
        model_args = json.loads(command[command.index("--model_args") + 1])
        self.assertEqual(command[command.index("--model") + 1], "vllm-vlm")
        self.assertEqual(model_args["tensor_parallel_size"], 4)
        self.assertTrue(model_args["enable_thinking"])
        self.assertEqual(model_args["chat_template_args"]["reasoning_effort"], "xhigh")
        self.assertNotIn("--check_integrity", command)
        self.assertIn("--log_samples", command)
        self.assertIn("--include_path", command)

    def test_runner_writes_dataset_pinned_overlays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = pathlib.Path(temp_dir)
            include_path, tasks = runner.prepare_task_overlays(
                self.protocol,
                lm_eval_root=pathlib.Path("/pinned/lm-eval"),
                out=out,
                tasks=["gpqa_diamond_cot_zeroshot", "mmlu_pro"],
            )
            self.assertEqual(
                tasks,
                ["qwen38_gpqa_diamond_cot_zeroshot_pinned", "qwen38_mmlu_pro_pinned"],
            )
            gpqa = (include_path / "gpqa_diamond_pinned.yaml").read_text(encoding="utf-8")
            self.assertIn("633f5ee89ab8ad4522a9f850766b73f62147ffdd", gpqa)
            mmlu = (include_path / "mmlu_pro_biology_pinned.yaml").read_text(encoding="utf-8")
            self.assertIn("b189ec765aa7ed75c8acfea42df31fdae71f97be", mmlu)
            group = (include_path / "mmlu_pro_group_pinned.yaml").read_text(encoding="utf-8")
            self.assertIn("qwen38_mmlu_pro_biology_pinned", group)

    def test_source_state_records_checkout(self):
        revision, dirty = runner.source_state(REPO_ROOT)
        self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertIsInstance(dirty, bool)

    def test_runner_rejects_unregistered_task(self):
        with self.assertRaises(ValueError):
            runner.build_command(
                self.protocol,
                model_path=pathlib.Path("/models/qwen38"),
                out=pathlib.Path("/runs/baseline"),
                tasks=["mmlu"],
                include_path=pathlib.Path("/runs/baseline/task_overrides"),
                execution_tasks=["mmlu"],
                limit=None,
            )

    def test_paired_quality_analysis_and_gates(self):
        def row(doc_id, score):
            return {
                "doc_id": doc_id,
                "doc_hash": f"doc-{doc_id}",
                "target_hash": f"target-{doc_id}",
                "prompt_hash": f"prompt-{doc_id}",
                "exact_match": score,
            }

        baseline = {("task", index): row(index, score) for index, score in enumerate([1, 1, 0, 1])}
        candidate = {("task", index): row(index, score) for index, score in enumerate([1, 1, 0, 1])}
        result = quality_summary.analyze_task(baseline, candidate, 4)
        self.assertEqual(result["quality_retention"], 1.0)
        self.assertEqual(result["candidate_minus_baseline_percentage_points"], 0.0)
        gates = quality_summary.apply_gates(
            self.protocol,
            {"gpqa_diamond_cot_zeroshot": result, "mmlu_pro": result},
        )
        self.assertTrue(all(gate["passed"] for gate in gates))

    def test_paired_quality_analysis_rejects_hash_drift(self):
        baseline = {
            ("task", 0): {
                "doc_hash": "a",
                "target_hash": "t",
                "prompt_hash": "p",
                "exact_match": 1.0,
            }
        }
        candidate = {
            ("task", 0): {
                "doc_hash": "b",
                "target_hash": "t",
                "prompt_hash": "p",
                "exact_match": 1.0,
            }
        }
        with self.assertRaises(ValueError):
            quality_summary.analyze_task(baseline, candidate, 1)

    def test_dataset_receipt_detects_content_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            self._write_dataset_receipt(root)
            ready, _ = validator.validate_dataset_receipt(
                root,
                self.protocol,
                verify_content=True,
            )
            self.assertTrue(ready)
            cache = next((root / "hf-datasets").glob("*.arrow"))
            content = cache.read_bytes()
            cache.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            ready, detail = validator.validate_dataset_receipt(
                root,
                self.protocol,
                verify_content=True,
            )
            self.assertFalse(ready)
            self.assertIn("dataset content hash mismatch", detail)


if __name__ == "__main__":
    unittest.main()
