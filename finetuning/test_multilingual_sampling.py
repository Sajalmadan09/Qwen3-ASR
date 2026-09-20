import unittest

from finetuning.multilingual_sampling import LanguageDurationSampler


class LanguageDurationSamplerTests(unittest.TestCase):
    def test_length_reproducibility_and_duration_buckets(self):
        languages = ["large"] * 80 + ["small"] * 20
        durations = [1.0] * 50 + [6.0] * 50
        sampler = LanguageDurationSampler(languages, durations, batch_size=5, language_alpha=0.0, seed=9)
        first = list(sampler)
        second = list(sampler)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        for start in range(0, len(first), 5):
            bucket_memberships = [durations[index] <= 2.0 for index in first[start : start + 5]]
            self.assertEqual(len(set(bucket_memberships)), 1)

    def test_distributed_ranks_have_expected_length(self):
        languages = ["a"] * 7 + ["b"] * 3
        durations = [3.0] * 10
        rank0 = LanguageDurationSampler(languages, durations, batch_size=2, num_replicas=2, rank=0)
        rank1 = LanguageDurationSampler(languages, durations, batch_size=2, num_replicas=2, rank=1)
        self.assertEqual(len(list(rank0)), 5)
        self.assertEqual(len(list(rank1)), 5)

    def test_alpha_zero_balances_languages(self):
        sampler = LanguageDurationSampler(
            ["large"] * 90 + ["small"] * 10,
            [1.0] * 100,
            batch_size=4,
            language_alpha=0.0,
        )
        self.assertEqual(sampler.expected_language_probabilities(), {"large": 0.5, "small": 0.5})

    def test_explicit_language_weights_adjust_probabilities(self):
        sampler = LanguageDurationSampler(
            ["english", "hindi", "tamil"],
            [3.0, 3.0, 3.0],
            batch_size=1,
            language_alpha=0.0,
            language_weights={"english": 2.0, "hindi": 2.0},
        )
        self.assertEqual(
            sampler.expected_language_probabilities(),
            {"english": 0.4, "hindi": 0.4, "tamil": 0.2},
        )


if __name__ == "__main__":
    unittest.main()
