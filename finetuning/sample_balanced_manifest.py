#!/usr/bin/env python3
"""Create a deterministic per-language reservoir sample from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=47803)
    args = parser.parse_args()
    if args.per_language <= 0:
        parser.error("--per-language must be positive")

    rng = random.Random(args.seed)
    seen: Counter[str] = Counter()
    samples: dict[str, list[dict]] = defaultdict(list)
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{args.input}:{line_number}: invalid JSON: {exc}") from exc
            language = str(row.get("language", "")).strip().lower()
            if not language:
                continue
            seen[language] += 1
            bucket = samples[language]
            if len(bucket) < args.per_language:
                bucket.append(row)
            else:
                index = rng.randrange(seen[language])
                if index < args.per_language:
                    bucket[index] = row

    rows = [row for language in sorted(samples) for row in samples[language]]
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows), "languages": len(samples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
