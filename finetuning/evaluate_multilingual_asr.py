#!/usr/bin/env python3
"""Run Qwen3-ASR inference and report per-language WER, CER, and LID accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ASR_TAG = "<asr_text>"


def extract_reference(target: str) -> str:
    value = str(target)
    return value.split(ASR_TAG, 1)[1] if ASR_TAG in value else value


def display_language(language: str) -> str:
    value = str(language).strip()
    return value[:1].upper() + value[1:].lower() if value else ""


def normalize_for_scoring(text: str) -> str:
    """Apply script-neutral normalization suitable for reproducible scoring."""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    chars: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    """Return substitutions, insertions, and deletions with O(len(hypothesis)) memory."""
    # Each cell contains (total edits, substitutions, insertions, deletions).
    previous = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, ref_item in enumerate(reference, 1):
        current = [(i, 0, 0, i)]
        for j, hyp_item in enumerate(hypothesis, 1):
            if ref_item == hyp_item:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            ins = current[j - 1]
            delete = previous[j]
            candidates = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2], delete[3] + 1),
                (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),
            ]
            current.append(min(candidates))
        previous = current
    _, substitutions, insertions, deletions = previous[-1]
    return substitutions, insertions, deletions


@dataclass
class MetricAccumulator:
    samples: int = 0
    failed_samples: int = 0
    lid_correct: int = 0
    lid_samples: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    reference_words: int = 0
    char_substitutions: int = 0
    char_insertions: int = 0
    char_deletions: int = 0
    reference_characters: int = 0

    def add(self, row: dict[str, Any]) -> None:
        reference = normalize_for_scoring(row.get("reference", ""))
        hypothesis = normalize_for_scoring(row.get("hypothesis", ""))
        expected_language = display_language(row.get("expected_language", ""))
        predicted_language = display_language(row.get("predicted_language", ""))
        word_counts = edit_counts(reference.split(), hypothesis.split())
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        char_counts = edit_counts(ref_chars, hyp_chars)

        self.samples += 1
        self.failed_samples += bool(row.get("error"))
        if not row.get("language_was_forced", False):
            self.lid_samples += 1
            self.lid_correct += bool(expected_language and predicted_language == expected_language)
        self.substitutions += word_counts[0]
        self.insertions += word_counts[1]
        self.deletions += word_counts[2]
        self.reference_words += len(reference.split())
        self.char_substitutions += char_counts[0]
        self.char_insertions += char_counts[1]
        self.char_deletions += char_counts[2]
        self.reference_characters += len(ref_chars)

    def to_dict(self) -> dict[str, Any]:
        word_errors = self.substitutions + self.insertions + self.deletions
        char_errors = self.char_substitutions + self.char_insertions + self.char_deletions
        return {
            "samples": self.samples,
            "failed_samples": self.failed_samples,
            "wer": word_errors / self.reference_words if self.reference_words else None,
            "cer": char_errors / self.reference_characters if self.reference_characters else None,
            "lid_accuracy": self.lid_correct / self.lid_samples if self.lid_samples else None,
            "lid_samples": self.lid_samples,
            "word_counts": {
                "substitutions": self.substitutions,
                "insertions": self.insertions,
                "deletions": self.deletions,
                "reference": self.reference_words,
            },
            "character_counts": {
                "substitutions": self.char_substitutions,
                "insertions": self.char_insertions,
                "deletions": self.char_deletions,
                "reference": self.reference_characters,
            },
        }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def score_predictions(path: Path) -> dict[str, Any]:
    overall = MetricAccumulator()
    by_language: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_source: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_language_source: dict[tuple[str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
    lid_confusions: Counter[tuple[str, str]] = Counter()
    for row in read_jsonl(path):
        language = str(row.get("expected_language", "")).strip().lower() or "<unknown>"
        source = str(row.get("source", "")).strip() or "<unknown>"
        overall.add(row)
        by_language[language].add(row)
        by_source[source].add(row)
        by_language_source[(language, source)].add(row)
        if not row.get("language_was_forced", False):
            predicted = str(row.get("predicted_language", "")).strip().lower() or "<unknown>"
            lid_confusions[(language, predicted)] += 1

    per_language = {key: value.to_dict() for key, value in sorted(by_language.items())}
    per_source = {key: value.to_dict() for key, value in sorted(by_source.items())}
    per_language_source = {
        f"{language}\t{source}": value.to_dict()
        for (language, source), value in sorted(by_language_source.items())
    }
    language_metrics = list(per_language.values())
    macro = {}
    for metric in ("wer", "cer", "lid_accuracy"):
        values = [value[metric] for value in language_metrics if value[metric] is not None]
        macro[metric] = sum(values) / len(values) if values else None
    confusion_matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for (expected, predicted), count in sorted(lid_confusions.items()):
        confusion_matrix[expected][predicted] = count
    top_confusions = [
        {"expected": expected, "predicted": predicted, "samples": count}
        for (expected, predicted), count in sorted(
            lid_confusions.items(), key=lambda item: (-item[1], item[0])
        )
        if expected != predicted
    ]
    return {
        "overall": overall.to_dict(),
        "macro_average": macro,
        "by_language": per_language,
        "by_source": per_source,
        "by_language_source": per_language_source,
        "lid_confusion_matrix": dict(confusion_matrix),
        "top_lid_confusions": top_confusions,
    }


def batched(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _row_seed(audio: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{audio}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def prepare_eval_audio(
    row: dict[str, Any],
    *,
    condition: str,
    rpca_channels: list[tuple[Any, int]],
    snr_choices: tuple[float, ...],
    seed: int,
) -> Any:
    """Load and deterministically degrade one evaluation sample when requested."""
    if condition == "clean":
        return row["audio"]

    import numpy as np
    import soundfile as sf

    try:
        from .telephony_augmentation import resample_waveform, telephony_augment
    except ImportError:
        from telephony_augmentation import resample_waveform, telephony_augment

    waveform, source_rate = sf.read(str(row["audio"]), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    sample_rate = 16000
    waveform = resample_waveform(waveform, int(source_rate), sample_rate)

    rng = np.random.default_rng(_row_seed(str(row["audio"]), seed))
    use_rpca = condition == "rpca" or (condition == "mixed" and bool(rng.integers(0, 2)))
    channel = None
    channel_rate = 8000
    if use_rpca:
        channel, channel_rate = rpca_channels[int(rng.integers(0, len(rpca_channels)))]
    augmented = telephony_augment(
        waveform,
        sample_rate=sample_rate,
        channel=channel,
        channel_sample_rate=channel_rate,
        snr_db=float(rng.choice(snr_choices)),
        seed=int(rng.integers(0, 2**32 - 1)),
    )
    return augmented, sample_rate


def infer(args: argparse.Namespace) -> None:
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if args.adapter_path:
        from peft import PeftModel

        try:
            from .qwen3_asr_sft import patch_outer_forward
        except ImportError:
            from qwen3_asr_sft import patch_outer_forward

        patch_outer_forward(model.model)
        model.model = PeftModel.from_pretrained(model.model, args.adapter_path)
        model.model.eval()

    rpca_channels: list[tuple[Any, int]] = []
    if args.rpca_channel:
        import soundfile as sf

        for path in args.rpca_channel:
            channel, channel_rate = sf.read(path, dtype="float32", always_2d=True)
            rpca_channels.append((channel.mean(axis=1), int(channel_rate)))

    completed: set[str] = set()
    if args.resume and args.predictions.exists():
        completed = {str(row.get("audio", "")) for row in read_jsonl(args.predictions)}
    rows = [row for row in read_jsonl(args.manifest) if str(row.get("audio", "")) not in completed]
    mode = "a" if args.resume else "w"
    with args.predictions.open(mode, encoding="utf-8") as handle:
        for batch in batched(rows, args.batch_size):
            audios = [
                prepare_eval_audio(
                    row,
                    condition=args.audio_condition,
                    rpca_channels=rpca_channels,
                    snr_choices=args.snr_choices,
                    seed=args.augmentation_seed,
                )
                for row in batch
            ]
            forced_languages = None
            if args.language_mode == "expected":
                forced_languages = [display_language(row["language"]) for row in batch]
            try:
                results = model.transcribe(audio=audios, language=forced_languages)
                errors = [""] * len(results)
            except Exception as batch_exc:
                results = []
                errors = []
                for audio, language in zip(audios, forced_languages or [None] * len(audios)):
                    try:
                        results.append(model.transcribe(audio=audio, language=language)[0])
                        errors.append("")
                    except Exception as exc:
                        results.append(None)
                        errors.append(f"{type(exc).__name__}: {exc}")
                print(f"batch fallback after {type(batch_exc).__name__}: {batch_exc}", file=sys.stderr)

            for source, result, error in zip(batch, results, errors):
                prediction = {
                    "audio": source["audio"],
                    "source": source.get("source", ""),
                    "dataset_id": source.get("dataset_id"),
                    "is_code_mixed": bool(source.get("is_code_mixed", False)),
                    "audio_condition": args.audio_condition,
                    "expected_language": source["language"],
                    "predicted_language": result.language if result is not None else "",
                    "language_was_forced": args.language_mode == "expected",
                    "reference": extract_reference(source["text"]),
                    "hypothesis": result.text if result is not None else "",
                    "error": error,
                }
                handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--language-mode", choices=("auto", "expected"), default="auto")
    parser.add_argument(
        "--audio-condition",
        choices=("clean", "synthetic", "rpca", "mixed"),
        default="clean",
        help="deterministic evaluation-time telephone degradation",
    )
    parser.add_argument("--rpca-channel", type=Path, action="append", default=[])
    parser.add_argument("--snr-choices", default="-5,0,5")
    parser.add_argument("--augmentation-seed", type=int, default=47803)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args(argv)
    if not args.score_only and args.manifest is None:
        parser.error("--manifest is required unless --score-only is used")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    try:
        args.snr_choices = tuple(
            float(value.strip()) for value in args.snr_choices.split(",") if value.strip()
        )
    except ValueError as exc:
        parser.error(f"--snr-choices must contain comma-separated numbers: {exc}")
    if not args.snr_choices:
        parser.error("--snr-choices must contain at least one value")
    if not args.score_only and args.audio_condition in {"rpca", "mixed"} and not args.rpca_channel:
        parser.error(f"--audio-condition {args.audio_condition} requires --rpca-channel")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.predictions = args.predictions.resolve()
    args.metrics_output = args.metrics_output.resolve()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        args.manifest = args.manifest.resolve()
        infer(args)
    metrics = score_predictions(args.predictions)
    with args.metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
