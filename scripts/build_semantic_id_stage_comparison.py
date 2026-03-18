#!/usr/bin/env python3
"""Merge training summaries and proxy metrics into a compact stage comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TRAIN_METRIC_CANDIDATES = {
    "final_training_loss": [
        "train/loss_epoch",
        "train/loss",
        "train/loss_step",
    ],
    "final_quantization_loss": [
        "train/quantization_loss_epoch",
        "train/quantization_loss",
        "train/quantization_loss_step",
    ],
    "final_reconstruction_loss": [
        "train/reconstruction_loss_epoch",
        "train/reconstruction_loss",
        "train/reconstruction_loss_step",
    ],
    "final_hierarchy_loss": [
        "train/hierarchy_loss_epoch",
        "train/hierarchy_loss",
        "train/hierarchy_loss_step",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact comparison table for the semantic-ID stage."
    )
    parser.add_argument(
        "--train-run",
        action="append",
        required=True,
        help="Training run in the form method_name=/path/to/train/output",
    )
    parser.add_argument(
        "--inference-run",
        action="append",
        default=[],
        help="Inference run in the form method_name=/path/to/inference/output",
    )
    parser.add_argument(
        "--proxy-dir",
        required=True,
        help="Directory that contains per-method proxy metric JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the consolidated CSV table.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the merged rows as JSON.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path to write a short human-readable markdown summary.",
    )
    return parser.parse_args()


def parse_run_specs(run_specs: Iterable[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for run_spec in run_specs:
        if "=" not in run_spec:
            raise ValueError(
                f"Invalid run specification '{run_spec}'. Expected method_name=path."
            )
        method_name, path = run_spec.split("=", 1)
        method_name = method_name.strip()
        path = path.strip()
        if not method_name or not path:
            raise ValueError(
                f"Invalid run specification '{run_spec}'. Expected non-empty method_name=path."
            )
        parsed[method_name] = Path(path)
    return parsed


def coerce_scalar(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        if text.lower() in {"nan", "inf", "-inf"}:
            return text
    except AttributeError:
        return value
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def load_last_metrics_row(metrics_file: Path) -> Dict[str, str]:
    if not metrics_file.exists():
        return {}

    last_seen: Dict[str, str] = {}
    with metrics_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value not in ("", None):
                    last_seen[key] = value
    return last_seen


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


def pick_metric_value(last_metrics: Dict[str, str], candidates: List[str]) -> Optional[Any]:
    for candidate in candidates:
        value = coerce_scalar(last_metrics.get(candidate))
        if value is not None:
            return value
    return None


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def read_optional_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text().strip()
    return text or None


def flatten_per_layer(
    row: Dict[str, Any],
    prefix: str,
    values: List[Any],
    max_layers: int,
) -> None:
    for layer_idx in range(max_layers):
        key = f"{prefix}_{layer_idx + 1}"
        row[key] = values[layer_idx] if layer_idx < len(values) else None


def build_row(
    method_name: str,
    train_dir: Path,
    inference_dir: Optional[Path],
    proxy_result: Dict[str, Any],
) -> Dict[str, Any]:
    metrics_file = find_latest_metrics_file(train_dir)
    last_metrics = load_last_metrics_row(metrics_file) if metrics_file is not None else {}
    checkpoint_path = find_latest_checkpoint(train_dir / "checkpoints")

    row: Dict[str, Any] = {
        "method": method_name,
        "training_output_dir": str(train_dir),
        "inference_output_dir": str(inference_dir) if inference_dir is not None else None,
        "metrics_file": str(metrics_file) if metrics_file is not None else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "selected_weight": read_optional_text(train_dir / "selected_weight.txt"),
        "selection_mode": read_optional_text(train_dir / "selection_mode.txt"),
        "final_step": coerce_scalar(last_metrics.get("step")),
        "final_epoch": coerce_scalar(last_metrics.get("epoch")),
        "final_training_loss": pick_metric_value(
            last_metrics, TRAIN_METRIC_CANDIDATES["final_training_loss"]
        ),
        "final_quantization_loss": pick_metric_value(
            last_metrics, TRAIN_METRIC_CANDIDATES["final_quantization_loss"]
        ),
        "final_reconstruction_loss": pick_metric_value(
            last_metrics, TRAIN_METRIC_CANDIDATES["final_reconstruction_loss"]
        ),
        "final_hierarchy_loss": pick_metric_value(
            last_metrics, TRAIN_METRIC_CANDIDATES["final_hierarchy_loss"]
        ),
        "avg_utilization": proxy_result.get("avg_utilization"),
        "overall_utilization": proxy_result.get("overall_utilization"),
        "collision_rate": proxy_result.get("collision_rate"),
        "frac_unique_ids": proxy_result.get("frac_unique_ids"),
        "avg_sibling_separation": proxy_result.get("avg_sibling_separation"),
        "near_collision_separation": proxy_result.get("near_collision_separation"),
        "same_parent_leaf_uniqueness": proxy_result.get("same_parent_leaf_uniqueness"),
        "same_parent_collision_rate": proxy_result.get("same_parent_collision_rate"),
        "same_parent_assignment_margin": proxy_result.get("same_parent_assignment_margin"),
        "same_parent_positive_margin_fraction": proxy_result.get(
            "same_parent_positive_margin_fraction"
        ),
        "full_collision_distance_mean": proxy_result.get("full_collision_distance_mean"),
        "sibling_distance_mean": proxy_result.get("sibling_distance_mean"),
        "mid_prefix_distance_mean": proxy_result.get("mid_prefix_distance_mean"),
        "unrelated_distance_mean": proxy_result.get("unrelated_distance_mean"),
        "distance_gap_full_minus_sibling": proxy_result.get(
            "distance_gap_full_minus_sibling"
        ),
        "distance_gap_unrelated_minus_sibling": proxy_result.get(
            "distance_gap_unrelated_minus_sibling"
        ),
        "num_items": proxy_result.get("num_items"),
        "num_unique_ids": proxy_result.get("num_unique_ids"),
        "num_colliding_items": proxy_result.get("num_colliding_items"),
        "prefix_metric_type": proxy_result.get("prefix_metric_type"),
        "semantic_id_source": proxy_result.get("prediction_source"),
        "proxy_result_path": proxy_result.get("_proxy_result_path"),
        "notes": json.dumps(proxy_result.get("notes", []), ensure_ascii=True),
    }

    num_hierarchies = int(proxy_result.get("num_hierarchies", 0) or 0)
    entropy_values = proxy_result.get("entropy_per_layer", [])
    max_token_ratio_values = proxy_result.get("max_token_ratio_per_layer", [])
    topk_coverage_values = proxy_result.get("topk_token_coverage_per_layer", [])
    utilization_values = proxy_result.get("utilization_per_layer", [])
    sibling_separation_values = proxy_result.get("sibling_separation_per_layer", [])

    prefix_metric_type = proxy_result.get("prefix_metric_type")
    if prefix_metric_type == "purity":
        prefix_values = proxy_result.get("prefix_purity_per_layer", [])
    elif prefix_metric_type == "compactness":
        prefix_values = proxy_result.get("prefix_compactness_per_layer", [])
    else:
        prefix_values = []

    flatten_per_layer(row, "utilization_layer", utilization_values, num_hierarchies)
    flatten_per_layer(row, "entropy_layer", entropy_values, num_hierarchies)
    flatten_per_layer(
        row,
        "max_token_ratio_layer",
        max_token_ratio_values,
        num_hierarchies,
    )
    flatten_per_layer(
        row,
        "topk_token_coverage_layer",
        topk_coverage_values,
        num_hierarchies,
    )
    flatten_per_layer(row, "prefix_metric_layer", prefix_values, num_hierarchies)
    flatten_per_layer(
        row,
        "sibling_separation_layer",
        sibling_separation_values,
        max(num_hierarchies - 1, 0),
    )
    return row


def ordered_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    preferred_prefixes = [
        "method",
        "training_output_dir",
        "inference_output_dir",
        "metrics_file",
        "checkpoint_path",
        "selected_weight",
        "selection_mode",
        "semantic_id_source",
        "proxy_result_path",
        "final_step",
        "final_epoch",
        "final_training_loss",
        "final_quantization_loss",
        "final_reconstruction_loss",
        "final_hierarchy_loss",
        "avg_utilization",
        "overall_utilization",
        "collision_rate",
        "frac_unique_ids",
        "avg_sibling_separation",
        "near_collision_separation",
        "same_parent_leaf_uniqueness",
        "same_parent_collision_rate",
        "same_parent_assignment_margin",
        "same_parent_positive_margin_fraction",
        "full_collision_distance_mean",
        "sibling_distance_mean",
        "mid_prefix_distance_mean",
        "unrelated_distance_mean",
        "distance_gap_full_minus_sibling",
        "distance_gap_unrelated_minus_sibling",
        "num_items",
        "num_unique_ids",
        "num_colliding_items",
        "prefix_metric_type",
    ]

    all_keys = {key for row in rows for key in row.keys()}
    per_layer_keys = []
    for prefix in (
        "utilization_layer_",
        "entropy_layer_",
        "max_token_ratio_layer_",
        "topk_token_coverage_layer_",
        "prefix_metric_layer_",
        "sibling_separation_layer_",
    ):
        per_layer_keys.extend(sorted(key for key in all_keys if key.startswith(prefix)))

    ordered = preferred_prefixes + per_layer_keys
    ordered.extend(
        sorted(key for key in all_keys if key not in set(ordered) and key != "notes")
    )
    if "notes" in all_keys:
        ordered.append("notes")
    return [key for key in ordered if key in all_keys]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=True) + "\n")


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Semantic ID Stage Comparison",
        "",
        "| method | selected_weight | selection_mode | final_quantization_loss | final_hierarchy_loss | collision_rate | frac_unique_ids | avg_sibling_separation | near_collision_separation | same_parent_leaf_uniqueness | same_parent_assignment_margin | prefix_metric_type |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_metric(row.get("method")),
                    format_metric(row.get("selected_weight")),
                    format_metric(row.get("selection_mode")),
                    format_metric(row.get("final_quantization_loss")),
                    format_metric(row.get("final_hierarchy_loss")),
                    format_metric(row.get("collision_rate")),
                    format_metric(row.get("frac_unique_ids")),
                    format_metric(row.get("avg_sibling_separation")),
                    format_metric(row.get("near_collision_separation")),
                    format_metric(row.get("same_parent_leaf_uniqueness")),
                    format_metric(row.get("same_parent_assignment_margin")),
                    format_metric(row.get("prefix_metric_type")),
                ]
            )
            + " |"
        )

    lines.append("")
    for row in rows:
        entropy_values = [
            value
            for key, value in sorted(row.items())
            if key.startswith("entropy_layer_") and value is not None
        ]
        prefix_values = [
            value
            for key, value in sorted(row.items())
            if key.startswith("prefix_metric_layer_") and value is not None
        ]
        sibling_values = [
            value
            for key, value in sorted(row.items())
            if key.startswith("sibling_separation_layer_") and value is not None
        ]
        lines.append(f"## {row['method']}")
        lines.append("")
        lines.append(f"- entropy_per_layer: `{json.dumps(entropy_values, ensure_ascii=True)}`")
        lines.append(
            f"- prefix_metric_type: `{format_metric(row.get('prefix_metric_type'))}`"
        )
        lines.append(
            f"- prefix_metric_per_layer: `{json.dumps(prefix_values, ensure_ascii=True)}`"
        )
        lines.append(
            f"- sibling_separation_per_layer: `{json.dumps(sibling_values, ensure_ascii=True)}`"
        )
        lines.append(
            f"- same_parent_leaf_uniqueness: `{format_metric(row.get('same_parent_leaf_uniqueness'))}`"
        )
        lines.append(
            f"- same_parent_assignment_margin: `{format_metric(row.get('same_parent_assignment_margin'))}`"
        )
        lines.append(
            f"- pair_type_distance_means: `{json.dumps({'full': row.get('full_collision_distance_mean'), 'sibling': row.get('sibling_distance_mean'), 'mid': row.get('mid_prefix_distance_mean'), 'unrelated': row.get('unrelated_distance_mean')}, ensure_ascii=True)}`"
        )
        lines.append("")

    markdown = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown + "\n")
    return markdown


def main() -> None:
    args = parse_args()
    train_runs = parse_run_specs(args.train_run)
    inference_runs = parse_run_specs(args.inference_run)
    proxy_dir = Path(args.proxy_dir)

    rows = []
    for method_name, train_dir in train_runs.items():
        proxy_result_path = proxy_dir / f"{method_name}.json"
        if not proxy_result_path.exists():
            raise FileNotFoundError(
                f"Missing proxy result for method '{method_name}': {proxy_result_path}"
            )
        proxy_result = json.loads(proxy_result_path.read_text())
        proxy_result["_proxy_result_path"] = str(proxy_result_path)
        row = build_row(
            method_name=method_name,
            train_dir=train_dir,
            inference_dir=inference_runs.get(method_name),
            proxy_result=proxy_result,
        )
        rows.append(row)

    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print(f"Wrote stage comparison CSV to: {output_csv}")

    if args.output_json is not None:
        output_json = Path(args.output_json)
        write_json(output_json, rows)
        print(f"Wrote stage comparison JSON to: {output_json}")

    if args.output_md is not None:
        output_md = Path(args.output_md)
        markdown = write_markdown(output_md, rows)
        print(f"Wrote stage comparison markdown to: {output_md}")
        print()
        print(markdown)


if __name__ == "__main__":
    main()
