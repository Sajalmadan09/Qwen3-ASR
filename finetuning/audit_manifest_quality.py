#!/usr/bin/env python3
"""Stream a JSONL ASR manifest and report transcript/duration anomalies.

This is a metadata audit, not an audio/text alignment verifier.  It is useful for
finding impossible transcript densities and for deciding which source/language
groups require listening or ASR spot checks before training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def acoustic_character_count(text: str) -> int:
    """Count Unicode letters, marks and numbers, excluding punctuation/spaces."""
    return sum(unicodedata.category(char)[0] in {"L", "M", "N"} for char in text)


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass
class GroupStats:
    reservoir_size: int
    rng: random.Random
    count: int = 0
    hours: float = 0.0
    flagged: Counter = field(default_factory=Counter)
    character_rates: list[float] = field(default_factory=list)
    word_rates: list[float] = field(default_factory=list)

    def add(self, duration: float, text: str) -> None:
        self.count += 1
        self.hours += duration / 3600.0
        chars = acoustic_character_count(text)
        words = len(text.split())
        char_rate = chars / duration
        word_rate = words / duration

        if char_rate < 0.5:
            self.flagged["chars_per_second_below_0.5"] += 1
        if char_rate > 35.0:
            self.flagged["chars_per_second_above_35"] += 1
        if word_rate > 8.0:
            self.flagged["words_per_second_above_8"] += 1
        if duration <= 1.0 and chars >= 40:
            self.flagged["at_least_40_chars_in_at_most_1_second"] += 1

        if len(self.character_rates) < self.reservoir_size:
            self.character_rates.append(char_rate)
            self.word_rates.append(word_rate)
            return
        replacement = self.rng.randrange(self.count)
        if replacement < self.reservoir_size:
            self.character_rates[replacement] = char_rate
            self.word_rates[replacement] = word_rate

    def report(self) -> dict:
        def distribution(values: list[float]) -> dict:
            return {
                "p01": quantile(values, 0.01),
                "p10": quantile(values, 0.10),
                "p50": quantile(values, 0.50),
                "p90": quantile(values, 0.90),
                "p99": quantile(values, 0.99),
            }

        return {
            "samples": self.count,
            "hours": self.hours,
            "flagged": dict(self.flagged),
            "chars_per_second": distribution(self.character_rates),
            "words_per_second": distribution(self.word_rates),
        }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def audit(path: Path, reservoir_size: int = 10_000, seed: int = 17) -> dict:
    groups: dict[tuple[str, str], GroupStats] = {}
    malformed = Counter()
    total_samples = 0
    total_hours = 0.0

    for line_number, row in iter_jsonl(path):
        try:
            language = str(row["language"]).strip().lower()
            source = str(row.get("source") or "unknown").strip()
            text = str(row["text"]).strip()
            duration = float(row["duration"])
            if not language or not text or not math.isfinite(duration) or duration <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            malformed["invalid_required_field"] += 1
            continue

        total_samples += 1
        total_hours += duration / 3600.0
        key = (language, source)
        if key not in groups:
            # Give each group an independent deterministic random stream.
            group_seed = seed + sum(ord(char) for char in f"{language}\0{source}")
            groups[key] = GroupStats(reservoir_size, random.Random(group_seed))
        groups[key].add(duration, text)

    rows = []
    for (language, source), stats in sorted(groups.items()):
        rows.append({"language": language, "source": source, **stats.report()})
    return {
        "manifest": str(path.resolve()),
        "scope": {
            "samples": total_samples,
            "hours": total_hours,
            "language_source_groups": len(rows),
            "malformed": dict(malformed),
        },
        "method": {
            "distribution_sample_per_group": reservoir_size,
            "character_definition": "Unicode letters, marks, and numbers",
            "warning": "Metadata-rate flags do not prove audio/transcript alignment.",
        },
        "groups": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.reservoir_size <= 0:
        parser.error("--reservoir-size must be positive")

    report = audit(args.manifest, args.reservoir_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["scope"], indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
