import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from finetuning.evaluate_multilingual_asr import (
    edit_counts,
    extract_reference,
    normalize_for_scoring,
    prepare_eval_audio,
    score_predictions,
)


class EvaluateMultilingualASRTests(unittest.TestCase):
    def test_reference_and_normalization(self):
        self.assertEqual(extract_reference("language Tamil<asr_text> வணக்கம்! "), " வணக்கம்! ")
        self.assertEqual(normalize_for_scoring("Hello—WORLD!"), "hello world")

    def test_edit_counts(self):
        self.assertEqual(edit_counts("a b c".split(), "a x c d".split()), (1, 1, 0))
        self.assertEqual(edit_counts("a b".split(), "a".split()), (0, 0, 1))

    def test_score_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            rows = [
                {"expected_language": "tamil", "predicted_language": "Tamil", "source": "a", "reference": "one two", "hypothesis": "one two", "error": ""},
                {"expected_language": "tamil", "predicted_language": "Hindi", "source": "b", "reference": "one two", "hypothesis": "one", "error": ""},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            metrics = score_predictions(path)
            self.assertEqual(metrics["overall"]["samples"], 2)
            self.assertEqual(metrics["overall"]["wer"], 0.25)
            self.assertEqual(metrics["overall"]["lid_accuracy"], 0.5)
            self.assertEqual(metrics["by_source"]["a"]["wer"], 0.0)
            self.assertEqual(metrics["by_source"]["b"]["wer"], 0.5)
            self.assertEqual(metrics["lid_confusion_matrix"]["tamil"], {"hindi": 1, "tamil": 1})
            self.assertEqual(
                metrics["top_lid_confusions"],
                [{"expected": "tamil", "predicted": "hindi", "samples": 1}],
            )

    def test_forced_language_is_not_counted_as_lid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            row = {
                "expected_language": "tamil",
                "predicted_language": "Tamil",
                "language_was_forced": True,
                "reference": "வணக்கம்",
                "hypothesis": "வணக்கம்",
                "error": "",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            metrics = score_predictions(path)
            self.assertIsNone(metrics["overall"]["lid_accuracy"])
            self.assertEqual(metrics["overall"]["lid_samples"], 0)

    def test_augmented_eval_audio_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.wav"
            sample_rate = 16000
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            sf.write(path, 0.2 * np.sin(2 * np.pi * 440 * time), sample_rate)
            row = {"audio": str(path)}
            first, first_rate = prepare_eval_audio(
                row,
                condition="synthetic",
                rpca_channels=[],
                snr_choices=(-5.0, 0.0, 5.0),
                seed=47803,
            )
            second, second_rate = prepare_eval_audio(
                row,
                condition="synthetic",
                rpca_channels=[],
                snr_choices=(-5.0, 0.0, 5.0),
                seed=47803,
            )
            self.assertEqual(first_rate, sample_rate)
            self.assertEqual(second_rate, sample_rate)
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
