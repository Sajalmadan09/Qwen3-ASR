import argparse
import json
import tempfile
import unittest
from pathlib import Path

from finetuning.prepare_multilingual_data import normalize_transcript, prepare


class PrepareMultilingualDataTests(unittest.TestCase):
    def test_normalize_transcript_collapses_whitespace(self):
        self.assertEqual(normalize_transcript("  hello\n  world  "), "hello world")

    def test_prepare_filters_and_removes_validation_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "processed_data" / "audio"
            audio_dir.mkdir(parents=True)
            (audio_dir / "a.wav").touch()
            (audio_dir / "b.wav").touch()
            (audio_dir / "c.wav").touch()

            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            languages = root / "languages.json"
            output = root / "out"
            languages.write_text(json.dumps({"tamil": "Tamil", "hindi": "Hindi"}), encoding="utf-8")
            train.write_text(
                "\n".join(
                    [
                        json.dumps({"audio": "processed_data/audio/a.wav", "text": "same text", "duration": 1.0, "language": "tamil", "source": "x", "dataset_id": 1}),
                        json.dumps({"audio": "processed_data/audio/b.wav", "text": "short", "duration": 0.1, "language": "hindi", "source": "x", "dataset_id": 2}),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                "\n".join(
                    [
                        json.dumps({"audio": "processed_data/audio/b.wav", "text": " same   text ", "duration": 1.0, "language": "tamil", "source": "x", "dataset_id": 1}),
                        json.dumps({"audio": "processed_data/audio/c.wav", "text": "नया", "duration": 2.0, "language": "hindi", "source": "x", "dataset_id": 2}),
                    ]
                ) + "\n",
                encoding="utf-8",
            )

            args = argparse.Namespace(
                workspace_root=root,
                train_manifest=train,
                validation_manifest=validation,
                output_dir=output,
                languages=languages,
                min_duration=0.5,
                max_duration=30.0,
                balanced_eval_per_language=1,
                train_smoke_per_language=1,
                seed=47803,
                check_audio_exists=True,
            )
            report = prepare(args)

            train_rows = [json.loads(line) for line in (output / "train.qwen.jsonl").read_text(encoding="utf-8").splitlines()]
            validation_rows = [json.loads(line) for line in (output / "validation.qwen.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(train_rows[0]["text"], "language Tamil<asr_text>same text")
            self.assertTrue(Path(train_rows[0]["audio"]).is_absolute())
            self.assertEqual(len(validation_rows), 1)
            self.assertEqual(validation_rows[0]["language"], "hindi")
            self.assertEqual(report["leakage"]["validation_rows_removed"], 1)
            self.assertEqual(report["train"]["rejected"]["too_short"], 1)
            self.assertEqual(report["outputs"]["train_smoke_balanced_rows"], 1)


if __name__ == "__main__":
    unittest.main()
