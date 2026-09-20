import json
import tempfile
import unittest
from pathlib import Path

from finetuning.build_audited_subset import build_subset, transcript_from_training_text


class BuildAuditedSubsetTests(unittest.TestCase):
    def test_prefix_removal(self):
        self.assertEqual(transcript_from_training_text("language Hindi<asr_text>नमस्ते"), "नमस्ते")

    def test_exclusion_filtering_and_synthetic_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            rows = []
            for index in range(20):
                dataset = "synthetic" if index < 10 else "real"
                rows.append({
                    "audio": f"{index}.wav", "text": "language Test<asr_text>hello world",
                    "duration": 2, "language": "test", "source": dataset,
                })
            rows.append({"audio": "bad.wav", "text": "x" * 100, "duration": 1, "language": "test", "source": "real"})
            rows.append({"audio": "excluded.wav", "text": "hello", "duration": 2, "language": "other", "source": "quarantine"})
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = build_subset(source, root / "out.jsonl", 10, {"quarantine"}, "synthetic", 0.2, 0.8, 3)

        self.assertEqual(report["selected_total"], 10)
        self.assertEqual(report["selection"]["test"]["by_source"]["synthetic"], 2)
        self.assertEqual(report["selection"]["test"]["by_source"]["real"], 8)
        self.assertEqual(report["selection"]["other"]["status"], "no_eligible_source")
        self.assertEqual(report["rejected"]["chars_per_second_above_35"], 1)


if __name__ == "__main__":
    unittest.main()
