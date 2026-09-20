import json
import tempfile
import unittest
from pathlib import Path

from finetuning.audit_manifest_quality import acoustic_character_count, audit


class AuditManifestQualityTests(unittest.TestCase):
    def test_acoustic_character_count_is_unicode_aware(self):
        self.assertEqual(acoustic_character_count("नमस्ते, 12!"), 8)

    def test_audit_groups_and_flags_implausible_density(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "data.jsonl"
            rows = [
                {"audio": "a.wav", "text": "hello world", "duration": 2, "language": "English", "source": "one"},
                {"audio": "b.wav", "text": "x" * 50, "duration": 1, "language": "English", "source": "one"},
                {"audio": "c.wav", "text": "नमस्ते", "duration": 2, "language": "Hindi", "source": "two"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = audit(manifest, reservoir_size=10)

        self.assertEqual(report["scope"]["samples"], 3)
        self.assertEqual(report["scope"]["language_source_groups"], 2)
        english = next(group for group in report["groups"] if group["language"] == "english")
        self.assertEqual(english["flagged"]["chars_per_second_above_35"], 1)
        self.assertEqual(english["flagged"]["at_least_40_chars_in_at_most_1_second"], 1)


if __name__ == "__main__":
    unittest.main()
