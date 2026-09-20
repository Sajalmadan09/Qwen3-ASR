#!/usr/bin/env python3
"""Build a deterministic, source-balanced ASR subset with conservative filters."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .audit_manifest_quality import acoustic_character_count
except ImportError:
    from audit_manifest_quality import acoustic_character_count


ASR_TAG = "<asr_text>"


def transcript_from_training_text(text: str) -> str:
    return text.split(ASR_TAG, 1)[1] if ASR_TAG in text else text


def stable_priority(seed: int, key: str) -> int:
    digest = hashlib.blake2b(f"{seed}\0{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def passes_metadata_filters(row: dict) -> tuple[bool, str | None]:
    try:
        duration = float(row["duration"])
        transcript = transcript_from_training_text(str(row["text"]).strip())
        if not transcript or not math.isfinite(duration) or duration <= 0:
            return False, "invalid_text_or_duration"
    except (KeyError, TypeError, ValueError):
        return False, "invalid_text_or_duration"
    chars = acoustic_character_count(transcript)
    rate = chars / duration
    if rate < 0.5:
        return False, "chars_per_second_below_0.5"
    if rate > 35.0:
        return False, "chars_per_second_above_35"
    if duration <= 1.0 and chars >= 40:
        return False, "at_least_40_chars_in_at_most_1_second"
    return True, None


def build_subset(
    input_path: Path,
    output_path: Path,
    target_per_language: int,
    excluded_sources: set[str],
    synthetic_source: str,
    max_synthetic_fraction: float,
    max_source_fraction: float,
    seed: int,
) -> dict:
    # Each source pool only needs target_per_language candidates. Keeping the
    # lowest stable hashes makes selection independent of manifest order.
    pools: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    rejected = Counter()
    input_counts = Counter()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                language = str(row["language"]).strip().lower()
                source = str(row.get("source") or "unknown").strip()
                audio = str(row["audio"])
                if not language:
                    raise ValueError
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                rejected["malformed"] += 1
                continue
            input_counts[language] += 1
            if source in excluded_sources:
                rejected[f"excluded_source:{source}"] += 1
                continue
            accepted, reason = passes_metadata_filters(row)
            if not accepted:
                rejected[reason or "metadata_filter"] += 1
                continue

            priority = stable_priority(seed, audio)
            item = (-priority, audio, line.rstrip("\n"))
            pool = pools[(language, source)]
            if len(pool) < target_per_language:
                heapq.heappush(pool, item)
            elif priority < -pool[0][0]:
                heapq.heapreplace(pool, item)

    by_language: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(dict)
    for (language, source), heap in pools.items():
        by_language[language][source] = sorted(
            [(-negative_priority, raw_line) for negative_priority, _audio, raw_line in heap]
        )

    selected: list[tuple[int, str]] = []
    selection_report = {}
    for language in sorted(input_counts):
        source_pools = by_language.get(language, {})
        sources = sorted(source_pools)
        if not sources:
            selection_report[language] = {"selected": 0, "by_source": {}, "status": "no_eligible_source"}
            continue
        # Raise the general cap when necessary so languages with only one or two
        # sources can still reach the requested size. Synthetic data keeps its
        # stricter independent cap.
        general_cap = max(
            math.ceil(target_per_language * max_source_fraction),
            math.ceil(target_per_language / len(sources)),
        )
        synthetic_cap = math.floor(target_per_language * max_synthetic_fraction)
        positions = Counter()
        chosen_counts = Counter()
        chosen: list[tuple[int, str]] = []

        while len(chosen) < target_per_language:
            progressed = False
            for source in sources:
                cap = synthetic_cap if source == synthetic_source else general_cap
                position = positions[source]
                pool = source_pools[source]
                if chosen_counts[source] >= cap or position >= len(pool):
                    continue
                chosen.append(pool[position])
                positions[source] += 1
                chosen_counts[source] += 1
                progressed = True
                if len(chosen) == target_per_language:
                    break
            if not progressed:
                break
        selected.extend(chosen)
        selection_report[language] = {
            "selected": len(chosen),
            "by_source": dict(sorted(chosen_counts.items())),
            "status": "complete" if len(chosen) == target_per_language else "insufficient_eligible_rows",
        }

    selected.sort(key=lambda item: item[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _priority, raw_line in selected:
            handle.write(raw_line + "\n")

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "policy": {
            "target_per_language": target_per_language,
            "excluded_sources": sorted(excluded_sources),
            "synthetic_source": synthetic_source,
            "max_synthetic_fraction": max_synthetic_fraction,
            "max_source_fraction": max_source_fraction,
            "metadata_filters": "0.5 <= Unicode letter/mark/number chars per second <= 35; reject >=40 chars at <=1s",
            "seed": seed,
        },
        "input_samples_by_language": dict(sorted(input_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "selection": selection_report,
        "selected_total": len(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-per-language", type=int, default=500)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--synthetic-source", default="adjaysagar/high-quality-tts")
    parser.add_argument("--max-synthetic-fraction", type=float, default=0.25)
    parser.add_argument("--max-source-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.target_per_language <= 0:
        parser.error("--target-per-language must be positive")
    for name, value in (("--max-synthetic-fraction", args.max_synthetic_fraction), ("--max-source-fraction", args.max_source_fraction)):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1")

    report = build_subset(
        args.manifest,
        args.output,
        args.target_per_language,
        set(args.exclude_source),
        args.synthetic_source,
        args.max_synthetic_fraction,
        args.max_source_fraction,
        args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report['selected_total']} rows to {args.output}")
    incomplete = {lang: info["status"] for lang, info in report["selection"].items() if info["status"] != "complete"}
    if incomplete:
        print("Incomplete languages: " + json.dumps(incomplete, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
