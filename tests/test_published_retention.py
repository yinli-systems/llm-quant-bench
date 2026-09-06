"""Check compact receipts without pretending to replay remote licensed samples."""

import hashlib
import json
import unittest
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "results/qwen25-72b-retention-v1"


class PublishedRetentionTest(unittest.TestCase):
    def test_all_eleven_manifest_payloads_match(self):
        lines = (PACKAGE / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(lines), 11)
        for line in lines:
            digest, name = line.split(maxsplit=1)
            self.assertEqual(Path(name).name, name)
            self.assertEqual(
                hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest(), digest
            )

    def test_aggregate_quality_counts_and_paired_transitions_agree(self):
        def read(name):
            return json.loads((PACKAGE / name).read_text())

        base, candidate = read("bf16-summary.json"), read("awq-summary.json")
        result = read("quality-retention.json")
        self.assertEqual(base["items_sha256"], candidate["items_sha256"])
        self.assertEqual(result["items_sha256"], base["items_sha256"])
        self.assertEqual(base["total_items"], 26943)
        self.assertEqual(candidate["total_items"], 26943)
        self.assertEqual(base["failed_items"], 0)
        self.assertEqual(candidate["failed_items"], 0)
        self.assertEqual(
            {row["benchmark"] for row in result["rows"]}, {"mmlu", "cmmlu", "gsm8k"}
        )
        for row in result["rows"]:
            name = row["benchmark"]
            b, c = base["by_benchmark"][name], candidate["by_benchmark"][name]
            self.assertEqual(row["baseline_items"], b["items"])
            self.assertEqual(row["candidate_items"], c["items"])
            self.assertEqual(b["scored_items"], b["items"])
            self.assertEqual(c["scored_items"], c["items"])
            self.assertEqual(row["baseline_score"], b["score_mean"])
            self.assertEqual(row["candidate_score"], c["score_mean"])
            delta = c["score_mean"] - b["score_mean"]
            self.assertAlmostEqual(row["delta_percentage_points"], 100 * delta)
            self.assertAlmostEqual(
                row["quality_retention"], c["score_mean"] / b["score_mean"]
            )
            paired = result["paired_analysis"][name]
            n = paired["paired_items"]
            self.assertEqual(n, b["items"])
            self.assertEqual(
                paired["ties"]
                + paired["baseline_only_wins"]
                + paired["candidate_only_wins"],
                n,
            )
            self.assertAlmostEqual(
                (paired["candidate_only_wins"] - paired["baseline_only_wins"]) / n,
                delta,
            )
        # This topology mismatch is intentional and forbids same-hardware speed claims.
        self.assertEqual(result["execution"]["baseline_gpus"], 8)
        self.assertEqual(result["execution"]["candidate_gpus"], 2)
