#!/usr/bin/env python3
"""RPCA channel extraction and deterministic telephony augmentation utilities.

RPCA follows Mo and Lam (SSCI 2020): an 8 kHz magnitude spectrogram is
decomposed into low-rank and sparse matrices, then an ideal binary mask keeps
bins where the low-rank component dominates. The paper's literal lambda of
``0.5 / max(shape)`` degenerates to an all-sparse solution on the supplied
calls, so the operational default is ``0.5 / sqrt(max(shape))``. The exact
paper value remains reproducible through ``--lam``. Extracted channels should
be audited for residual intelligible speech before use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy import signal


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def singular_value_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    left, singular_values, right = np.linalg.svd(values, full_matrices=False)
    keep = singular_values > threshold
    if not np.any(keep):
        return np.zeros_like(values)
    shrunk = singular_values[keep] - threshold
    return (left[:, keep] * shrunk) @ right[keep]


def robust_pca(
    matrix: np.ndarray,
    *,
    lam: float | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Decompose ``matrix = low_rank + sparse`` with inexact ALM."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    lam = float(lam if lam is not None else 0.5 / math.sqrt(max(matrix.shape)))
    if lam <= 0:
        raise ValueError("lambda must be positive")

    spectral_norm = float(np.linalg.norm(matrix, ord=2))
    infinity_norm = float(np.max(np.abs(matrix)) / lam)
    dual_norm = max(spectral_norm, infinity_norm, np.finfo(np.float64).eps)
    multiplier = matrix / dual_norm
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    matrix_norm = max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(np.float64).eps)
    mu = 1.25 / max(spectral_norm, np.finfo(np.float64).eps)
    mu_limit = mu * 1e7
    rho = 1.5
    residual = math.inf

    for iteration in range(1, max_iterations + 1):
        low_rank = singular_value_threshold(matrix - sparse + multiplier / mu, 1.0 / mu)
        sparse = soft_threshold(matrix - low_rank + multiplier / mu, lam / mu)
        difference = matrix - low_rank - sparse
        multiplier += mu * difference
        mu = min(mu * rho, mu_limit)
        residual = float(np.linalg.norm(difference, ord="fro") / matrix_norm)
        if residual < tolerance:
            break

    report: dict[str, float | int | bool] = {
        "iterations": iteration,
        "relative_residual": residual,
        "converged": residual < tolerance,
        "lambda": lam,
        "rank": int(np.linalg.matrix_rank(low_rank)),
        "sparse_fraction": float(np.count_nonzero(sparse) / sparse.size),
    }
    return low_rank, sparse, report


def extract_rpca_channel(
    waveform: np.ndarray,
    *,
    sample_rate: int = 8000,
    n_fft: int = 1024,
    hop_length: int = 256,
    lam: float | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("waveform is empty")
    _, _, spectrum = signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    low_rank, sparse, report = robust_pca(
        magnitude,
        lam=lam,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    mask = np.abs(low_rank) > np.abs(sparse)
    channel_spectrum = spectrum * mask
    _, channel = signal.istft(
        channel_spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    channel = channel[: waveform.size]
    if channel.size < waveform.size:
        channel = np.pad(channel, (0, waveform.size - channel.size))
    channel = np.asarray(channel, dtype=np.float32)
    report.update(
        {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "mask_fraction": float(mask.mean()),
            "input_rms": rms(waveform),
            "channel_rms": rms(channel),
        }
    )
    return channel, report


def rms(waveform: np.ndarray) -> float:
    value = np.asarray(waveform, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value))) if value.size else 0.0


def fit_channel_length(channel: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32).reshape(-1)
    if not channel.size:
        raise ValueError("channel characteristic is empty")
    if channel.size < length:
        channel = np.tile(channel, int(math.ceil(length / channel.size)))
    if channel.size == length:
        return channel.copy()
    start = int(rng.integers(0, channel.size - length + 1))
    return channel[start : start + length].copy()


def mix_at_snr(
    speech: np.ndarray,
    channel: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator,
    peak: float = 0.99,
) -> np.ndarray:
    speech = np.asarray(speech, dtype=np.float32).reshape(-1)
    noise = fit_channel_length(channel, speech.size, rng)
    speech_rms = rms(speech)
    noise_rms = rms(noise)
    if noise_rms <= 1e-12 or speech_rms <= 1e-12:
        return speech.copy()
    noise *= speech_rms / (noise_rms * (10.0 ** (float(snr_db) / 20.0)))
    mixed = speech + noise
    maximum = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if maximum > peak:
        mixed *= peak / maximum
    return mixed.astype(np.float32, copy=False)


def mu_law_roundtrip(waveform: np.ndarray, quantization_channels: int = 256) -> np.ndarray:
    waveform = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    mu = float(quantization_channels - 1)
    encoded = np.sign(waveform) * np.log1p(mu * np.abs(waveform)) / np.log1p(mu)
    quantized = np.round((encoded + 1.0) * 0.5 * mu)
    encoded_quantized = 2.0 * quantized / mu - 1.0
    decoded = np.sign(encoded_quantized) * np.expm1(np.abs(encoded_quantized) * np.log1p(mu)) / mu
    return decoded.astype(np.float32)


def resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return signal.resample_poly(
        np.asarray(waveform, dtype=np.float32),
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32)


def select_low_rms_segment(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    segment_seconds: float,
    hop_seconds: float = 1.0,
    minimum_rms: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Select a quiet, non-digital-silent segment and return it plus start time."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    segment_length = max(1, int(round(segment_seconds * sample_rate)))
    if waveform.size <= segment_length:
        return waveform, 0.0
    hop = max(1, int(round(hop_seconds * sample_rate)))
    candidates: list[tuple[float, int]] = []
    for start in range(0, waveform.size - segment_length + 1, hop):
        value = rms(waveform[start : start + segment_length])
        if value >= minimum_rms:
            candidates.append((value, start))
    if not candidates:
        start = 0
    else:
        _, start = min(candidates)
    return waveform[start : start + segment_length], start / sample_rate


def synthetic_telephone_channel(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply narrowband telephone coloration plus an 8-bit mu-law round trip."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    nyquist = sample_rate / 2.0
    high_hz = min(3400.0, nyquist * 0.95)
    sos = signal.butter(6, [300.0 / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    if waveform.size > 3 * (2 * len(sos) + 1):
        filtered = signal.sosfiltfilt(sos, waveform)
    else:
        filtered = signal.sosfilt(sos, waveform)
    narrowband = signal.resample_poly(filtered, 8000, sample_rate)
    restored = signal.resample_poly(narrowband, sample_rate, 8000)
    restored = restored[: waveform.size]
    if restored.size < waveform.size:
        restored = np.pad(restored, (0, waveform.size - restored.size))
    return mu_law_roundtrip(restored)


def telephony_augment(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    channel: np.ndarray | None = None,
    channel_sample_rate: int = 8000,
    snr_db: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Create narrowband speech and optionally mix an RPCA characteristic."""
    output = synthetic_telephone_channel(waveform, sample_rate=sample_rate)
    if channel is None:
        return output
    channel = np.asarray(channel, dtype=np.float32)
    if channel_sample_rate != sample_rate:
        channel = resample_waveform(channel, channel_sample_rate, sample_rate)
    return mix_at_snr(output, channel, snr_db, rng=np.random.default_rng(seed))


def extract_files(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for source in args.inputs:
        waveform, source_rate = sf.read(source, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        waveform = resample_waveform(waveform, int(source_rate), args.sample_rate)
        original_duration = waveform.size / args.sample_rate
        segment_start = 0.0
        if args.segment_seconds > 0:
            waveform, segment_start = select_low_rms_segment(
                waveform,
                args.sample_rate,
                segment_seconds=args.segment_seconds,
                hop_seconds=args.segment_hop_seconds,
                minimum_rms=args.minimum_segment_rms,
            )
        channel, report = extract_rpca_channel(
            waveform,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            lam=args.lam,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        destination = args.output_dir / f"{source.stem}.rpca_channel.wav"
        sf.write(destination, channel, args.sample_rate, subtype="PCM_16")
        reports.append(
            {
                "input": str(source.resolve()),
                "output": str(destination.resolve()),
                "original_duration_seconds": original_duration,
                "selected_segment_start_seconds": segment_start,
                "selected_segment_duration_seconds": waveform.size / args.sample_rate,
                **report,
            }
        )
    result = {
        "warning": "Audit extracted files for residual intelligible speech before training.",
        "files": reports,
    }
    report_path = args.output_dir / "rpca_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract RPCA characteristics from call recordings")
    extract.add_argument("inputs", type=Path, nargs="+")
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--sample-rate", type=int, default=8000)
    extract.add_argument("--n-fft", type=int, default=1024)
    extract.add_argument("--hop-length", type=int, default=256)
    extract.add_argument("--lam", type=float)
    extract.add_argument("--tolerance", type=float, default=1e-7)
    extract.add_argument("--max-iterations", type=int, default=1000)
    extract.add_argument("--segment-seconds", type=float, default=10.0)
    extract.add_argument("--segment-hop-seconds", type=float, default=1.0)
    extract.add_argument("--minimum-segment-rms", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "extract":
        result = extract_files(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
