import unittest

import numpy as np

from finetuning.telephony_augmentation import (
    mix_at_snr,
    robust_pca,
    select_low_rms_segment,
    synthetic_telephone_channel,
    telephony_augment,
)


class TelephonyAugmentationTests(unittest.TestCase):
    def test_robust_pca_reconstructs_matrix(self):
        rng = np.random.default_rng(7)
        left = rng.normal(size=(12, 2))
        right = rng.normal(size=(2, 10))
        matrix = left @ right
        matrix[1, 2] += 20
        matrix[8, 7] -= 15
        low_rank, sparse, report = robust_pca(matrix, lam=0.1, tolerance=1e-6, max_iterations=500)
        relative_error = np.linalg.norm(matrix - low_rank - sparse) / np.linalg.norm(matrix)
        self.assertLess(relative_error, 1e-5)
        self.assertTrue(report["converged"])

    def test_mix_at_snr_is_finite_and_length_preserving(self):
        rng = np.random.default_rng(1)
        speech = np.ones(1600, dtype=np.float32) * 0.1
        noise = rng.normal(size=300).astype(np.float32)
        mixed = mix_at_snr(speech, noise, 0.0, rng=rng)
        self.assertEqual(mixed.shape, speech.shape)
        self.assertTrue(np.all(np.isfinite(mixed)))
        self.assertLessEqual(np.max(np.abs(mixed)), 0.99001)

    def test_synthetic_and_rpca_paths_preserve_shape(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        speech = 0.2 * np.sin(2 * np.pi * 440 * time)
        channel = 0.01 * np.sin(2 * np.pi * 50 * np.arange(8000, dtype=np.float32) / 8000)
        synthetic = synthetic_telephone_channel(speech, sample_rate)
        augmented = telephony_augment(speech, sample_rate=sample_rate, channel=channel, seed=3)
        self.assertEqual(synthetic.shape, speech.shape)
        self.assertEqual(augmented.shape, speech.shape)
        self.assertTrue(np.all(np.isfinite(augmented)))

    def test_select_low_rms_segment(self):
        waveform = np.concatenate(
            [np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32) * 0.1]
        )
        segment, start = select_low_rms_segment(
            waveform, 10, segment_seconds=10, hop_seconds=10
        )
        self.assertEqual(start, 10.0)
        self.assertAlmostEqual(float(segment.mean()), 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
