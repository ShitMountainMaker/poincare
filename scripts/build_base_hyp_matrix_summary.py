#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SEMANTIC_FIELDS = [
    "entropy_layer_2",
    "prefix_metric_layer_1",
    "prefix_metric_layer_2",
    "prefix_metric_layer_3",
    "avg_sibling_separation",
    "near_collision_separation",
    "collision_rate",
    "frac_unique_ids",
    "selected_weight",
]

DOWNSTREAM_FIELDS = [
    "test/recall@5",
    "test/recall@10",
    "test/ndcg@5",
    "test/ndcg@10",
    "val/recall@5",
    "val/recall@10",
    "val/ndcg@5",
    "val/ndcg@10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[])
    return parser.parse_args()


def format_float(value: str | float | None, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_delta(base: str | float | None, hyp: str | float | None, digits: int = 4) -> str:
    if base in (None, "") or hyp in (None, ""):
        return "-"
    try:
        return f"{float(hyp) - float(base):+.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def latest_metrics(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    last_seen: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            if value not in ("", None):
                last_seen[key] = value
    return last_seen


def latest_metrics_from_run_dir(run_dir: Path) -> dict[str, str]:
    csv_root = run_dir / "csv"
    if not csv_root.exists():
        return {}
    candidates = sorted(csv_root.glob("**/metrics.csv"))
    if not candidates:
        return {}
    return latest_metrics(candidates[-1])


def avg_prefix_metric(row: dict[str, str]) -> str:
    values = []
    for key in ("prefix_metric_layer_1", "prefix_metric_layer_2", "prefix_metric_layer_3"):
        value = row.get(key)
        if value not in (None, ""):
            values.append(float(value))
    if not values:
        return "-"
    return f"{sum(values) / len(values):.4f}"


def dataset_seed_from_run_name(run_name: str) -> tuple[str, str]:
    if "_seed" in run_name:
        dataset, seed = run_name.rsplit("_seed", 1)
        return dataset, seed
    return run_name, "-"


def collect_run_summary(run_root: Path) -> dict:
    run_name = run_root.name
    dataset, seed = dataset_seed_from_run_name(run_name)

    semantic_rows = read_csv_rows(run_root / "semantic_id_stage" / "semantic_id_stage_comparison.csv")
    semantic_by_method = {row["method"]: row for row in semantic_rows if row.get("method")}

    base_sem = semantic_by_method.get("base", {})
    hyp_sem = semantic_by_method.get("hyp_prefix", {})

    base_tiger = latest_metrics_from_run_dir(run_root / "recommendation_stage" / "tiger_train_base")
    hyp_tiger = latest_metrics_from_run_dir(run_root / "recommendation_stage" / "tiger_train_hyp_prefix")

    return {
        "run_name": run_name,
        "dataset": dataset,
        "seed": seed,
        "base_sem": base_sem,
        "hyp_sem": hyp_sem,
        "base_tiger": base_tiger,
        "hyp_tiger": hyp_tiger,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    run_names = args.run or [path.name for path in sorted(args.root.iterdir()) if path.is_dir() and (path / "semantic_id_stage" / "semantic_id_stage_comparison.csv").exists()]

    summaries = [collect_run_summary(args.root / run_name) for run_name in run_names]

    semantic_rows = []
    downstream_rows = []
    for item in summaries:
        base_sem = item["base_sem"]
        hyp_sem = item["hyp_sem"]
        base_tiger = item["base_tiger"]
        hyp_tiger = item["hyp_tiger"]

        semantic_rows.append([
            item["run_name"],
            item["dataset"],
            item["seed"],
            avg_prefix_metric(base_sem),
            avg_prefix_metric(hyp_sem),
            format_float(base_sem.get("collision_rate")),
            format_float(hyp_sem.get("collision_rate")),
            format_delta(base_sem.get("collision_rate"), hyp_sem.get("collision_rate")),
            format_float(base_sem.get("frac_unique_ids")),
            format_float(hyp_sem.get("frac_unique_ids")),
            format_delta(base_sem.get("frac_unique_ids"), hyp_sem.get("frac_unique_ids")),
            format_float(base_sem.get("entropy_layer_2")),
            format_float(hyp_sem.get("entropy_layer_2")),
            format_float(base_sem.get("avg_sibling_separation")),
            format_float(hyp_sem.get("avg_sibling_separation")),
            format_float(base_sem.get("near_collision_separation")),
            format_float(hyp_sem.get("near_collision_separation")),
            hyp_sem.get("selected_weight", "-") or "-",
        ])

        downstream_rows.append([
            item["run_name"],
            item["dataset"],
            item["seed"],
            format_float(base_tiger.get("test/recall@5")),
            format_float(hyp_tiger.get("test/recall@5")),
            format_delta(base_tiger.get("test/recall@5"), hyp_tiger.get("test/recall@5")),
            format_float(base_tiger.get("test/recall@10")),
            format_float(hyp_tiger.get("test/recall@10")),
            format_delta(base_tiger.get("test/recall@10"), hyp_tiger.get("test/recall@10")),
            format_float(base_tiger.get("test/ndcg@5")),
            format_float(hyp_tiger.get("test/ndcg@5")),
            format_delta(base_tiger.get("test/ndcg@5"), hyp_tiger.get("test/ndcg@5")),
            format_float(base_tiger.get("test/ndcg@10")),
            format_float(hyp_tiger.get("test/ndcg@10")),
            format_delta(base_tiger.get("test/ndcg@10"), hyp_tiger.get("test/ndcg@10")),
        ])

    lines = [
        "# Base + Hyp Matrix Summary",
        "",
        f"- root: `{args.root}`",
        f"- runs: `{', '.join(run_names)}`",
        "",
        "## Semantic Stage",
        "",
        markdown_table(
            [
                "run",
                "dataset",
                "seed",
                "base avg_prefix",
                "hyp avg_prefix",
                "base collision",
                "hyp collision",
                "delta",
                "base unique",
                "hyp unique",
                "delta",
                "base ent_l2",
                "hyp ent_l2",
                "base avg_sibling",
                "hyp avg_sibling",
                "base near_collision",
                "hyp near_collision",
                "selected hyp",
            ],
            semantic_rows,
        ),
        "",
        "## Downstream Test",
        "",
        markdown_table(
            [
                "run",
                "dataset",
                "seed",
                "base R@5",
                "hyp R@5",
                "delta",
                "base R@10",
                "hyp R@10",
                "delta",
                "base N@5",
                "hyp N@5",
                "delta",
                "base N@10",
                "hyp N@10",
                "delta",
            ],
            downstream_rows,
        ),
        "",
    ]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
