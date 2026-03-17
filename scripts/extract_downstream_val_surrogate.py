#!/usr/bin/env python3
"""Extract a compact downstream validation surrogate summary from a TIGER train run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_METRICS = [
    "val/recall@5",
    "val/recall@10",
    "val/ndcg@5",
    "val/ndcg@10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the best validation row from a downstream train metrics.csv."
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--primary-metric", default="val/recall@5")
    parser.add_argument("--metric", action="append", default=[])
    return parser.parse_args()


def coerce_scalar(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def find_latest_metrics_file(train_dir: Path) -> Optional[Path]:
    csv_dir = train_dir / "csv"
    if not csv_dir.exists():
        return None

    version_dirs = [
        path for path in csv_dir.iterdir() if path.is_dir() and path.name.startswith("version_")
    ]
    if not version_dirs:
        return None

    def version_key(path: Path) -> int:
        suffix = path.name.removeprefix("version_")
        try:
            return int(suffix)
        except ValueError:
            return -1

    latest_version_dir = max(version_dirs, key=version_key)
    metrics_file = latest_version_dir / "metrics.csv"
    return metrics_file if metrics_file.exists() else None


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    metrics_file = find_latest_metrics_file(train_dir)
    if metrics_file is None:
        raise FileNotFoundError(f"No metrics.csv found under {train_dir}")

    tracked_metrics = list(dict.fromkeys(DEFAULT_METRICS + args.metric))
    best_row = None
    best_score = None
    with metrics_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            primary_value = coerce_scalar(row.get(args.primary_metric))
            if not isinstance(primary_value, (int, float)):
                continue
            if best_score is None or float(primary_value) > best_score:
                best_score = float(primary_value)
                best_row = row

    best_metrics: Dict[str, Any] = {}
    if best_row is not None:
        for key in tracked_metrics + ["step", "epoch"]:
            value = coerce_scalar(best_row.get(key))
            if value is not None:
                best_metrics[key] = value

    payload = {
        "train_dir": str(train_dir),
        "metrics_file": str(metrics_file),
        "primary_metric": args.primary_metric,
        "best_metrics": best_metrics,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
