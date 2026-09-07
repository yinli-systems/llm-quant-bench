import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_qwen38_pilot import digest, read_rows, strings


class PilotAuditHelpersTest(unittest.TestCase):
    def test_unicode_separators_are_not_jsonl_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            rows = [{"text": "synthetic\u2028one\u2029two"}, {"text": "second"}]
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
            self.assertEqual(read_rows(path), rows)

    def test_nested_response_strings(self):
        self.assertEqual(strings([["[invalid]"], ["answer"], None]), ["[invalid]", "answer"])
        self.assertEqual(strings([]), [])

    def test_hash_is_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"abc")
            self.assertEqual(digest(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
