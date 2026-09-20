#!/usr/bin/env python3
"""Prepare VoxCPM manifests for multilingual Qwen3-ASR fine-tuning.

The converter is intentionally streaming: the source corpus contains millions
of rows. It validates the fields needed by Qwen3-ASR, resolves audio paths,
adds the expected language prefix, and removes validation utterances whose
normalized transcript appears in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


ASR_TAG = "<asr_text>"


def normalize_transcript(text: str) -> str:
    """Normalize only whitespace for conservative exact-overlap detection."""
    return " ".join(str(text).split())


def transcript_digest(text: str) -> bytes:
    return hashlib.blake2b(normalize_transcript(text).encode("utf-8"), digest_size=16).digest()


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def load_languages(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path}: expected a non-empty object mapping language keys to labels")
    return {str(k).strip().lower(): str(v).strip() for k, v in value.items()}


def resolve_audio_path(raw_path: str, workspace_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve(strict=False)


@dataclass
class SplitStats:
    input_rows: int = 0
    kept_rows: int = 0
    kept_hours: float = 0.0
    rejected: Counter[str] = field(default_factory=Counter)
    input_by_language: Counter[str] = field(default_factory=Counter)
    kept_by_language: Counter[str] = field(default_factory=Counter)
    kept_hours_by_language: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    source_counts: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "kept_rows": self.kept_rows,
            "kept_hours": round(self.kept_hours, 6),
            "rejected": dict(sorted(self.rejected.items())),
            "input_by_language": dict(sorted(self.input_by_language.items())),
            "kept_by_language": dict(sorted(self.kept_by_language.items())),
            "kept_hours_by_language": {
                key: round(value, 6) for key, value in sorted(self.kept_hours_by_language.items())
            },
            "source_counts": dict(sorted(self.source_counts.items())),
        }


def validate_and_convert(
    row: dict[str, Any],
    *,
    languages: dict[str, str],
    workspace_root: Path,
    min_duration: float,
    max_duration: float,
    check_audio_exists: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    language = str(row.get("language", "")).strip().lower()
    if language not in languages:
        return None, "unknown_language"

    transcript = normalize_transcript(row.get("text", ""))
    if not transcript:
        return None, "empty_transcript"

    try:
        duration = float(row.get("duration"))
    except (TypeError, ValueError):
        return None, "invalid_duration"
    if duration < min_duration:
        return None, "too_short"
    if duration > max_duration:
        return None, "too_long"

    raw_audio = str(row.get("audio", "")).strip()
    if not raw_audio:
        return None, "missing_audio_path"
    audio_path = resolve_audio_path(raw_audio, workspace_root)
    if check_audio_exists and not audio_path.is_file():
        return None, "audio_not_found"

    converted = {
        "audio": str(audio_path),
        "text": f"language {languages[language]}{ASR_TAG}{transcript}",
        "language": language,
        "duration": duration,
        "source": row.get("source", ""),
        "dataset_id": row.get("dataset_id"),
        "is_code_mixed": bool(row.get("is_code_mixed", False)),
    }
    return converted, None


def update_stats(stats: SplitStats, source_row: dict[str, Any], converted: dict[str, Any] | None, reason: str | None) -> None:
    stats.input_rows += 1
    language = str(source_row.get("language", "")).strip().lower() or "<missing>"
    stats.input_by_language[language] += 1
    if reason:
        stats.rejected[reason] += 1
        return
    assert converted is not None
    stats.kept_rows += 1
    duration_hours = float(converted["duration"]) / 3600.0
    stats.kept_hours += duration_hours
    stats.kept_by_language[language] += 1
    stats.kept_hours_by_language[language] += duration_hours
    stats.source_counts[str(converted.get("source", ""))] += 1


def write_jsonl_line(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def collect_validation_digests(path: Path) -> set[bytes]:
    digests: set[bytes] = set()
    for _, row in read_jsonl(path):
        text = normalize_transcript(row.get("text", ""))
        if text:
            digests.add(transcript_digest(text))
    return digests


def reservoir_add(
    reservoirs: dict[str, list[dict[str, Any]]],
    seen: Counter[str],
    row: dict[str, Any],
    limit: int,
    rng: random.Random,
) -> None:
    language = row["language"]
    seen[language] += 1
    bucket = reservoirs[language]
    if len(bucket) < limit:
        bucket.append(row)
        return
    index = rng.randrange(seen[language])
    if index < limit:
        bucket[index] = row


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = load_languages(args.languages.resolve())

    validation_digests = collect_validation_digests(validation_path)
    overlapping_validation_digests: set[bytes] = set()
    train_stats = SplitStats()
    validation_stats = SplitStats()

    train_output = output_dir / "train.qwen.jsonl"
    validation_output = output_dir / "validation.qwen.jsonl"
    balanced_output = output_dir / "validation_balanced.qwen.jsonl"
    train_smoke_output = output_dir / "train_smoke_balanced.qwen.jsonl"
    train_reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_reservoir_seen: Counter[str] = Counter()
    rng = random.Random(args.seed)

    with train_output.open("w", encoding="utf-8") as out:
        for _, row in read_jsonl(train_path):
            converted, reason = validate_and_convert(
                row,
                languages=languages,
                workspace_root=workspace_root,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                check_audio_exists=args.check_audio_exists,
            )
            update_stats(train_stats, row, converted, reason)
            text = normalize_transcript(row.get("text", ""))
            if text:
                digest = transcript_digest(text)
                if digest in validation_digests:
                    overlapping_validation_digests.add(digest)
            if converted is not None:
                write_jsonl_line(out, converted)
                reservoir_add(
                    train_reservoirs,
                    train_reservoir_seen,
                    converted,
                    args.train_smoke_per_language,
                    rng,
                )

    train_smoke_rows = [
        row for language in sorted(train_reservoirs) for row in train_reservoirs[language]
    ]
    rng.shuffle(train_smoke_rows)
    with train_smoke_output.open("w", encoding="utf-8") as out:
        for row in train_smoke_rows:
            write_jsonl_line(out, row)

    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reservoir_seen: Counter[str] = Counter()
    overlap_rows = 0
    with validation_output.open("w", encoding="utf-8") as out:
        for _, row in read_jsonl(validation_path):
            text = normalize_transcript(row.get("text", ""))
            if text and transcript_digest(text) in overlapping_validation_digests:
                validation_stats.input_rows += 1
                language = str(row.get("language", "")).strip().lower() or "<missing>"
                validation_stats.input_by_language[language] += 1
                validation_stats.rejected["transcript_overlap_with_train"] += 1
                overlap_rows += 1
                continue
            converted, reason = validate_and_convert(
                row,
                languages=languages,
                workspace_root=workspace_root,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                check_audio_exists=args.check_audio_exists,
            )
            update_stats(validation_stats, row, converted, reason)
            if converted is not None:
                write_jsonl_line(out, converted)
                reservoir_add(
                    reservoirs,
                    reservoir_seen,
                    converted,
                    args.balanced_eval_per_language,
                    rng,
                )

    balanced_rows = [row for language in sorted(reservoirs) for row in reservoirs[language]]
    rng.shuffle(balanced_rows)
    with balanced_output.open("w", encoding="utf-8") as out:
        for row in balanced_rows:
            write_jsonl_line(out, row)

    report = {
        "configuration": {
            "workspace_root": str(workspace_root),
            "train_manifest": str(train_path),
            "validation_manifest": str(validation_path),
            "languages": languages,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "check_audio_exists": args.check_audio_exists,
            "balanced_eval_per_language": args.balanced_eval_per_language,
            "train_smoke_per_language": args.train_smoke_per_language,
            "seed": args.seed,
        },
        "train": train_stats.to_dict(),
        "validation": validation_stats.to_dict(),
        "leakage": {
            "unique_validation_transcripts_seen_in_train": len(overlapping_validation_digests),
            "validation_rows_removed": overlap_rows,
        },
        "outputs": {
            "train": str(train_output),
            "validation": str(validation_output),
            "validation_balanced": str(balanced_output),
            "validation_balanced_rows": len(balanced_rows),
            "train_smoke_balanced": str(train_smoke_output),
            "train_smoke_balanced_rows": len(train_smoke_rows),
        },
    }
    report_path = output_dir / "preparation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", type=Path, default=here / "languages_indic25.json")
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--balanced-eval-per-language", type=positive_int, default=200)
    parser.add_argument("--train-smoke-per-language", type=positive_int, default=200)
    parser.add_argument("--seed", type=int, default=47803)
    parser.add_argument("--check-audio-exists", action="store_true")
    args = parser.parse_args(argv)
    if args.min_duration < 0 or args.max_duration <= args.min_duration:
        parser.error("duration limits must satisfy 0 <= min-duration < max-duration")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = prepare(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
