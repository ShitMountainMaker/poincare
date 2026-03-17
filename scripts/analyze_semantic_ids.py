#!/usr/bin/env python3
"""Analyze semantic-id inference outputs and compute proxy metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.components.semantic_id_metrics import (
    compute_proxy_metrics,
    format_summary,
    load_embedding_tensor,
    load_item_categories,
    load_semantic_id_predictions,
    result_to_flat_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute proxy metrics for semantic ID inference outputs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form method_name=path_to_pickle_dir_or_file. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store comparison JSON/CSV and summary outputs. Defaults to outputs/semantic_id_proxy_metrics.",
    )
    parser.add_argument(
        "--embedding-path",
        default=None,
        help="Optional embedding tensor used for unsupervised prefix compactness fallback.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional CSV with item_id and category-like field for prefix purity.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON with item_id and category-like field for prefix purity.",
    )
    parser.add_argument(
        "--category-field",
        default=None,
        help="Optional explicit category field name when metadata contains multiple candidates.",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=None,
        help="Optional total codebook size per hierarchy. If omitted, inferred from the maximum observed token id + 1.",
    )
    parser.add_argument(
        "--num-hierarchies",
        type=int,
        default=None,
        help="Optional number of hierarchy digits to analyze. Useful when a dedup suffix exists in tensor files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k coverage to report for per-layer token distributions.",
    )
    return parser.parse_args()


def parse_run_specs(run_specs: List[str]) -> List[Tuple[str, str]]:
    parsed_runs = []
    for run_spec in run_specs:
        if "=" not in run_spec:
            raise ValueError(
                f"Invalid --run value '{run_spec}'. Expected method_name=path."
            )
        method_name, path = run_spec.split("=", 1)
        method_name = method_name.strip()
        path = path.strip()
        if not method_name or not path:
            raise ValueError(
                f"Invalid --run value '{run_spec}'. Expected non-empty method_name=path."
            )
        parsed_runs.append((method_name, path))
    return parsed_runs


def default_output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "outputs" / "semantic_id_proxy_metrics"


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


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, results: List[Dict[str, object]]) -> str:
    lines = ["# Semantic ID Proxy Metrics Summary", ""]
    for result in results:
        lines.append(f"## {result['method']}")
        lines.append("")
        lines.append("```text")
        lines.append(format_summary(result))
        lines.append("```")
        lines.append("")
    summary = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary + "\n")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = parse_run_specs(args.run)
    embedding_tensor = (
        load_embedding_tensor(args.embedding_path)
        if args.embedding_path is not None
        else None
    )

    results = []
    for method_name, prediction_path in run_specs:
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
            "codebook_size_source": "cli"
            if args.codebook_size is not None
            else "inferred_max_plus_one",
        }
        notes = metadata.get("notes", [])
        if args.codebook_size is None:
            notes.append(
                "Codebook size was inferred from observed token ids because --codebook-size was not provided."
            )
        metadata["notes"] = notes

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
        results.append(result)
        write_json(output_dir / f"{method_name}.json", result)

    comparison_payload = {"results": results}
    write_json(output_dir / "comparison.json", comparison_payload)
    write_csv(output_dir / "comparison.csv", [result_to_flat_row(result) for result in results])
    summary_text = write_summary(output_dir / "SUMMARY.md", results)
    print(summary_text)
    print(f"\nMachine-readable outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
