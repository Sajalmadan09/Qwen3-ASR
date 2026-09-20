"""Language-balanced, duration-bucketed sampling for multilingual ASR."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class LanguageDurationSampler(Sampler[int]):
    """Sample languages by temperature while keeping local batches length-compatible.

    Every rank uses the same duration-bucket schedule and a rank-specific example
    stream. Epoch length matches a conventional distributed sampler. Sampling is
    with replacement, which is required to upsample low-resource languages.
    """

    def __init__(
        self,
        languages: Sequence[str],
        durations: Sequence[float],
        *,
        batch_size: int,
        language_alpha: float = 0.5,
        language_weights: dict[str, float] | None = None,
        duration_boundaries: Sequence[float] = (2.0, 4.0, 8.0, 12.0, 20.0, 30.0),
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if len(languages) != len(durations) or not languages:
            raise ValueError("languages and durations must be non-empty and have equal length")
        if batch_size <= 0 or num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid batch/distributed sampler configuration")
        if language_alpha < 0:
            raise ValueError("language_alpha must be non-negative")

        self.languages = [str(value) for value in languages]
        self.durations = [float(value) for value in durations]
        self.batch_size = int(batch_size)
        self.language_alpha = float(language_alpha)
        self.language_weights = {
            str(key).lower(): float(value) for key, value in (language_weights or {}).items()
        }
        if any(value <= 0 for value in self.language_weights.values()):
            raise ValueError("language weights must be positive")
        self.boundaries = tuple(float(value) for value in duration_boundaries)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.languages) / self.num_replicas))

        self.by_bucket_language: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        temporary: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, (language, duration) in enumerate(zip(self.languages, self.durations)):
            bucket = int(np.searchsorted(self.boundaries, duration, side="left"))
            temporary[(bucket, language)].append(index)
        for (bucket, language), indices in temporary.items():
            self.by_bucket_language[bucket][language] = np.asarray(indices, dtype=np.int64)

        bucket_counts = {
            bucket: sum(len(indices) for indices in groups.values())
            for bucket, groups in self.by_bucket_language.items()
        }
        self.buckets = np.asarray(sorted(bucket_counts), dtype=np.int64)
        counts = np.asarray([bucket_counts[int(bucket)] for bucket in self.buckets], dtype=np.float64)
        self.bucket_probabilities = counts / counts.sum()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        schedule_rng = np.random.default_rng(self.seed + self.epoch)
        sample_rng = np.random.default_rng(self.seed + self.epoch * 1009 + self.rank * 1_000_003)
        produced = 0
        while produced < self.num_samples:
            bucket = int(schedule_rng.choice(self.buckets, p=self.bucket_probabilities))
            groups = self.by_bucket_language[bucket]
            language_names = sorted(groups)
            counts = np.asarray([len(groups[name]) for name in language_names], dtype=np.float64)
            probabilities = np.power(counts, self.language_alpha)
            probabilities *= np.asarray(
                [self.language_weights.get(name.lower(), 1.0) for name in language_names],
                dtype=np.float64,
            )
            probabilities /= probabilities.sum()
            block_size = min(self.batch_size, self.num_samples - produced)
            chosen_languages = sample_rng.choice(language_names, size=block_size, p=probabilities)
            for language in chosen_languages:
                indices = groups[str(language)]
                yield int(indices[int(sample_rng.integers(0, len(indices)))])
            produced += block_size

    def expected_language_probabilities(self) -> dict[str, float]:
        counts = Counter(self.languages)
        weights = {
            key: count ** self.language_alpha * self.language_weights.get(key.lower(), 1.0)
            for key, count in counts.items()
        }
        total = sum(weights.values())
        return {key: value / total for key, value in sorted(weights.items())}
