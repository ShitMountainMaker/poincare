#!/usr/bin/env python3
"""Standalone proxy analysis focused on deep-level Poincare diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.components.semantic_id_metrics import (
    compute_proxy_metrics,
    load_embedding_tensor,
    load_item_categories,
    load_semantic_id_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute standalone deep-level proxy metrics for semantic IDs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form method_name=path_to_pickle_dir_or_file.",
    )
    parser.add_argument(
        "--teacher-forced-run",
        action="append",
        default=[],
        help=(
            "Optional method_name=path_to_teacher_forced_json_or_train_dir. "
            "If a directory is provided, teacher_forced_val.json will be loaded from it."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--category-field", default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--num-hierarchies", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def parse_run_specs(run_specs: List[str]) -> List[Tuple[str, str]]:
    parsed_runs = []
    for run_spec in run_specs:
        if "=" not in run_spec:
            raise ValueError(f"Invalid --run value '{run_spec}'. Expected method_name=path.")
        method_name, path = run_spec.split("=", 1)
        method_name = method_name.strip()
        path = path.strip()
        if not method_name or not path:
            raise ValueError(f"Invalid --run value '{run_spec}'. Expected non-empty method_name=path.")
        parsed_runs.append((method_name, path))
    return parsed_runs


def parse_optional_mapping_specs(run_specs: List[str]) -> Dict[str, str]:
    parsed_runs: Dict[str, str] = {}
    for run_spec in run_specs:
        if "=" not in run_spec:
            raise ValueError(f"Invalid mapping value '{run_spec}'. Expected method_name=path.")
        method_name, path = run_spec.split("=", 1)
        method_name = method_name.strip()
        path = path.strip()
        if not method_name or not path:
            raise ValueError(f"Invalid mapping value '{run_spec}'. Expected non-empty method_name=path.")
        parsed_runs[method_name] = path
    return parsed_runs


def select_embeddings_for_items(
    full_embedding_tensor: torch.Tensor,
    item_ids: List[int],
) -> torch.Tensor:
    max_item_id = max(item_ids) if item_ids else -1
    if max_item_id >= full_embedding_tensor.size(0):
        raise IndexError(
            f"Embedding tensor has {full_embedding_tensor.size(0)} rows, but semantic ids reference item_id {max_item_id}."
        )
    return full_embedding_tensor[item_ids]


def avg_or_none(values: List[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def resolve_teacher_forced_file(path_or_dir: str) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        candidate = path / "teacher_forced_val.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No teacher_forced_val.json found in {path}")
    return path


def load_teacher_forced_summary(path_or_dir: str) -> Dict[str, Any]:
    resolved_path = resolve_teacher_forced_file(path_or_dir)
    payload = json.loads(resolved_path.read_text())
    payload["_teacher_forced_json"] = str(resolved_path)
    return payload


def summarize_proxy_result(
    result: Dict[str, Any],
    teacher_forced_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prefix_metric_type = result.get("prefix_metric_type")
    if prefix_metric_type == "compactness":
        prefix_values = result.get("prefix_compactness_per_layer", []) or []
    elif prefix_metric_type == "purity":
        prefix_values = result.get("prefix_purity_per_layer", []) or []
    else:
        prefix_values = []

    entropy_values = result.get("entropy_per_layer", []) or []
    summary = {
        "method": result.get("method"),
        "prediction_source": result.get("prediction_source"),
        "collision_rate": result.get("collision_rate"),
        "frac_unique_ids": result.get("frac_unique_ids"),
        "avg_utilization": result.get("avg_utilization"),
        "avg_entropy": avg_or_none([float(value) for value in entropy_values]),
        "avg_prefix_metric": avg_or_none([float(value) for value in prefix_values]),
        "prefix_metric_type": prefix_metric_type,
        "avg_sibling_separation": result.get("avg_sibling_separation"),
        "near_collision_separation": result.get("near_collision_separation"),
        "same_parent_leaf_uniqueness": result.get("same_parent_leaf_uniqueness"),
        "same_parent_collision_rate": result.get("same_parent_collision_rate"),
        "same_parent_assignment_margin": result.get("same_parent_assignment_margin"),
        "same_parent_positive_margin_fraction": result.get(
            "same_parent_positive_margin_fraction"
        ),
        "full_collision_distance_mean": result.get("full_collision_distance_mean"),
        "sibling_distance_mean": result.get("sibling_distance_mean"),
        "mid_prefix_distance_mean": result.get("mid_prefix_distance_mean"),
        "unrelated_distance_mean": result.get("unrelated_distance_mean"),
        "distance_gap_full_minus_sibling": result.get("distance_gap_full_minus_sibling"),
        "distance_gap_unrelated_minus_sibling": result.get(
            "distance_gap_unrelated_minus_sibling"
        ),
        "teacher_forced_last_ce": None,
        "teacher_forced_last_acc": None,
        "teacher_forced_last2_mean_ce": None,
        "teacher_forced_last2_mean_acc": None,
        "teacher_forced_leaf_ce": None,
        "teacher_forced_leaf_acc": None,
        "same_parent_hardest_sibling_margin": None,
        "same_parent_within_parent_accuracy": None,
        "rollout1_last_ce": None,
        "rollout1_last_acc": None,
        "rollout2_last2_mean_ce": None,
        "rollout2_last2_mean_acc": None,
        "rollout2_last2_sequence_acc": None,
        "teacher_forced_depth_metrics_path": None,
        "notes": result.get("notes", []),
    }
    if teacher_forced_result is not None:
        summary.update(
            {
                "teacher_forced_last_ce": teacher_forced_result.get(
                    "teacher_forced_last_ce"
                ),
                "teacher_forced_last_acc": teacher_forced_result.get(
                    "teacher_forced_last_acc"
                ),
                "teacher_forced_last2_mean_ce": teacher_forced_result.get(
                    "teacher_forced_last2_mean_ce"
                ),
                "teacher_forced_last2_mean_acc": teacher_forced_result.get(
                    "teacher_forced_last2_mean_acc"
                ),
                "teacher_forced_leaf_ce": teacher_forced_result.get(
                    "teacher_forced_leaf_ce"
                ),
                "teacher_forced_leaf_acc": teacher_forced_result.get(
                    "teacher_forced_leaf_acc"
                ),
                "same_parent_hardest_sibling_margin": teacher_forced_result.get(
                    "same_parent_hardest_sibling_margin"
                ),
                "same_parent_within_parent_accuracy": teacher_forced_result.get(
                    "same_parent_within_parent_accuracy"
                ),
                "rollout1_last_ce": teacher_forced_result.get("rollout1_last_ce"),
                "rollout1_last_acc": teacher_forced_result.get("rollout1_last_acc"),
                "rollout2_last2_mean_ce": teacher_forced_result.get(
                    "rollout2_last2_mean_ce"
                ),
                "rollout2_last2_mean_acc": teacher_forced_result.get(
                    "rollout2_last2_mean_acc"
                ),
                "rollout2_last2_sequence_acc": teacher_forced_result.get(
                    "rollout2_last2_sequence_acc"
                ),
                "teacher_forced_depth_metrics_path": teacher_forced_result.get(
                    "_teacher_forced_json"
                ),
            }
        )
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fields = [
        "method",
        "prediction_source",
        "collision_rate",
        "frac_unique_ids",
        "avg_utilization",
        "avg_entropy",
        "avg_prefix_metric",
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
        "teacher_forced_last_ce",
        "teacher_forced_last_acc",
        "teacher_forced_last2_mean_ce",
        "teacher_forced_last2_mean_acc",
        "teacher_forced_leaf_ce",
        "teacher_forced_leaf_acc",
        "same_parent_hardest_sibling_margin",
        "same_parent_within_parent_accuracy",
        "rollout1_last_ce",
        "rollout1_last_acc",
        "rollout2_last2_mean_ce",
        "rollout2_last2_mean_acc",
        "rollout2_last2_sequence_acc",
        "teacher_forced_depth_metrics_path",
        "prefix_metric_type",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            normalized["notes"] = json.dumps(row.get("notes", []), ensure_ascii=True)
            writer.writerow(normalized)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Deep-Level Poincare Proxy Summary",
        "",
        "| method | avg_prefix | avg_entropy | avg_sibling_sep | near_collision_sep | parent_unique | parent_margin | sib_margin | roll1_ce | roll1_acc | roll2_ce | roll2_acc | roll2_seq_acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method")),
                    f"{row.get('avg_prefix_metric', 0.0):.6f}" if row.get("avg_prefix_metric") is not None else "N/A",
                    f"{row.get('avg_entropy', 0.0):.6f}" if row.get("avg_entropy") is not None else "N/A",
                    f"{row.get('avg_sibling_separation', 0.0):.6f}" if row.get("avg_sibling_separation") is not None else "N/A",
                    f"{row.get('near_collision_separation', 0.0):.6f}" if row.get("near_collision_separation") is not None else "N/A",
                    f"{row.get('same_parent_leaf_uniqueness', 0.0):.6f}" if row.get("same_parent_leaf_uniqueness") is not None else "N/A",
                    f"{row.get('same_parent_assignment_margin', 0.0):.6f}" if row.get("same_parent_assignment_margin") is not None else "N/A",
                    f"{row.get('same_parent_hardest_sibling_margin', 0.0):.6f}" if row.get("same_parent_hardest_sibling_margin") is not None else "N/A",
                    f"{row.get('rollout1_last_ce', 0.0):.6f}" if row.get("rollout1_last_ce") is not None else "N/A",
                    f"{row.get('rollout1_last_acc', 0.0):.6f}" if row.get("rollout1_last_acc") is not None else "N/A",
                    f"{row.get('rollout2_last2_mean_ce', 0.0):.6f}" if row.get("rollout2_last2_mean_ce") is not None else "N/A",
                    f"{row.get('rollout2_last2_mean_acc', 0.0):.6f}" if row.get("rollout2_last2_mean_acc") is not None else "N/A",
                    f"{row.get('rollout2_last2_sequence_acc', 0.0):.6f}" if row.get("rollout2_last2_sequence_acc") is not None else "N/A",
                ]
            )
            + " |"
        )

    lines.append("")
    for row in rows:
        lines.append(f"## {row['method']}")
        lines.append("")
        lines.append(f"- prediction_source: `{row.get('prediction_source')}`")
        lines.append(f"- collision_rate: `{row.get('collision_rate')}`")
        lines.append(f"- frac_unique_ids: `{row.get('frac_unique_ids')}`")
        lines.append(f"- same_parent_leaf_uniqueness: `{row.get('same_parent_leaf_uniqueness')}`")
        lines.append(f"- same_parent_collision_rate: `{row.get('same_parent_collision_rate')}`")
        lines.append(f"- same_parent_assignment_margin: `{row.get('same_parent_assignment_margin')}`")
        lines.append(f"- same_parent_positive_margin_fraction: `{row.get('same_parent_positive_margin_fraction')}`")
        lines.append(
            "- pair_type_distance_means: "
            f"`{json.dumps({'full': row.get('full_collision_distance_mean'), 'sibling': row.get('sibling_distance_mean'), 'mid': row.get('mid_prefix_distance_mean'), 'unrelated': row.get('unrelated_distance_mean')}, ensure_ascii=True)}`"
        )
        lines.append(
            "- pair_type_distance_gaps: "
            f"`{json.dumps({'full_minus_sibling': row.get('distance_gap_full_minus_sibling'), 'unrelated_minus_sibling': row.get('distance_gap_unrelated_minus_sibling')}, ensure_ascii=True)}`"
        )
        lines.append(
            "- teacher_forced_depth_metrics: "
            f"`{json.dumps({'last_ce': row.get('teacher_forced_last_ce'), 'last_acc': row.get('teacher_forced_last_acc'), 'last2_mean_ce': row.get('teacher_forced_last2_mean_ce'), 'last2_mean_acc': row.get('teacher_forced_last2_mean_acc'), 'path': row.get('teacher_forced_depth_metrics_path')}, ensure_ascii=True)}`"
        )
        lines.append(f"- notes: `{json.dumps(row.get('notes', []), ensure_ascii=True)}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_forced_runs = parse_optional_mapping_specs(args.teacher_forced_run)

    embedding_tensor = (
        load_embedding_tensor(args.embedding_path) if args.embedding_path is not None else None
    )

    results = []
    summaries = []
    for method_name, prediction_path in parse_run_specs(args.run):
        item_ids, cluster_ids, prediction_metadata = load_semantic_id_predictions(
            prediction_path=prediction_path,
            num_hierarchies=args.num_hierarchies,
        )
        category_labels, category_metadata = load_item_categories(
            item_ids=item_ids,
            metadata_csv=args.metadata_csv,
            metadata_json=args.metadata_json,
            category_field=args.category_field,
        )
        method_embeddings = None
        if embedding_tensor is not None:
            method_embeddings = select_embeddings_for_items(embedding_tensor, item_ids)

        codebook_size = (
            args.codebook_size
            if args.codebook_size is not None
            else int(cluster_ids.max().item()) + 1
            if cluster_ids.numel() > 0
            else 0
        )
        metadata = {
            **prediction_metadata,
            **category_metadata,
            "codebook_size_source": "cli" if args.codebook_size is not None else "inferred",
        }

        result = compute_proxy_metrics(
            method_name=method_name,
            item_ids=item_ids,
            cluster_ids=cluster_ids,
            codebook_size=codebook_size,
            top_k=args.top_k,
            embeddings=method_embeddings,
            category_labels=category_labels,
            extra_metadata=metadata,
        )
        teacher_forced_result = (
            load_teacher_forced_summary(teacher_forced_runs[method_name])
            if method_name in teacher_forced_runs
            else None
        )
        if teacher_forced_result is not None:
            result["teacher_forced"] = teacher_forced_result
            result["teacher_forced_last_ce"] = teacher_forced_result.get(
                "teacher_forced_last_ce"
            )
            result["teacher_forced_last_acc"] = teacher_forced_result.get(
                "teacher_forced_last_acc"
            )
            result["teacher_forced_last2_mean_ce"] = teacher_forced_result.get(
                "teacher_forced_last2_mean_ce"
            )
            result["teacher_forced_last2_mean_acc"] = teacher_forced_result.get(
                "teacher_forced_last2_mean_acc"
            )
            result["teacher_forced_depth_metrics_path"] = teacher_forced_result.get(
                "_teacher_forced_json"
            )
        summary = summarize_proxy_result(
            result, teacher_forced_result=teacher_forced_result
        )
        results.append(result)
        summaries.append(summary)
        write_json(output_dir / f"{method_name}.json", result)

    write_json(output_dir / "comparison.json", {"results": summaries})
    write_csv(output_dir / "comparison.csv", summaries)
    write_markdown(output_dir / "SUMMARY.md", summaries)
    print(json.dumps({"output_dir": str(output_dir), "methods": [row["method"] for row in summaries]}, indent=2))


if __name__ == "__main__":
    main()
