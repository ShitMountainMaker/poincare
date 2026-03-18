#!/usr/bin/env python3
"""Select a retained Poincare weight using deep-level proxy diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
        description="Select a retained Poincare hierarchy-loss weight from a sweep."
    )
    parser.add_argument("--method-family", required=True, help="Expected value: hyp_prefix")
    parser.add_argument("--baseline-train-dir", required=True)
    parser.add_argument("--baseline-proxy-json", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "Candidate spec in the form "
            "weight|method_name|train_dir|inference_dir|proxy_json"
            "[|surrogate_json]"
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--collision-worse-threshold", type=float, default=0.03)
    parser.add_argument("--frac-unique-min-improvement", type=float, default=-0.01)
    parser.add_argument("--entropy-avg-drop-threshold", type=float, default=0.15)
    parser.add_argument("--entropy-layer-drop-threshold", type=float, default=0.2)
    parser.add_argument("--quantization-loss-relative-threshold", type=float, default=0.05)
    parser.add_argument("--quantization-loss-absolute-threshold", type=float, default=1.0)
    parser.add_argument("--loss-explosion-threshold", type=float, default=1e5)
    parser.add_argument("--prefix-worse-threshold", type=float, default=0.0)
    parser.add_argument("--same-parent-uniqueness-drop-threshold", type=float, default=0.02)
    parser.add_argument("--same-parent-margin-drop-threshold", type=float, default=0.02)
    parser.add_argument(
        "--same-parent-positive-margin-drop-threshold", type=float, default=0.05
    )
    parser.add_argument("--full-sibling-gap-min", type=float, default=-0.01)
    parser.add_argument("--unrelated-sibling-gap-min", type=float, default=0.0)
    parser.add_argument("--surrogate-primary-metric", default="val/recall@5")
    parser.add_argument("--surrogate-secondary-metric", default="val/ndcg@10")
    return parser.parse_args()


def coerce_scalar(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"nan", "inf", "-inf"}:
        return lowered
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def avg_or_none(values: List[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def is_non_finite(value: Optional[float]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return True
    return not math.isfinite(float(value))


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


def pick_metric_value(last_metrics: Dict[str, str], candidates: Iterable[str]) -> Optional[float]:
    for candidate in candidates:
        value = coerce_scalar(last_metrics.get(candidate))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_training_summary(train_dir: Path) -> Dict[str, Any]:
    metrics_file = find_latest_metrics_file(train_dir)
    last_metrics = load_last_metrics_row(metrics_file) if metrics_file is not None else {}
    return {
        "metrics_file": str(metrics_file) if metrics_file is not None else None,
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
    }


def parse_candidate_specs(candidate_specs: List[str]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for candidate_spec in candidate_specs:
        parts = candidate_spec.split("|")
        if len(parts) not in {5, 6}:
            raise ValueError(
                f"Invalid --candidate value '{candidate_spec}'. "
                "Expected weight|method_name|train_dir|inference_dir|proxy_json"
                "[|surrogate_json]."
            )
        weight_text, method_name, train_dir, inference_dir, proxy_json = [
            part.strip() for part in parts[:5]
        ]
        surrogate_json = parts[5].strip() if len(parts) == 6 else None
        parsed.append(
            {
                "weight": float(weight_text),
                "weight_text": weight_text,
                "method": method_name,
                "train_dir": train_dir,
                "inference_dir": inference_dir,
                "proxy_json": proxy_json,
                "surrogate_json": surrogate_json or None,
            }
        )
    return parsed


def load_proxy_result(proxy_json: Path) -> Dict[str, Any]:
    with proxy_json.open() as handle:
        payload = json.load(handle)
    payload["_proxy_json"] = str(proxy_json)
    return payload


def load_surrogate_result(surrogate_json: Optional[Path]) -> Optional[Dict[str, Any]]:
    if surrogate_json is None or not surrogate_json.exists():
        return None
    with surrogate_json.open() as handle:
        payload = json.load(handle)
    payload["_surrogate_json"] = str(surrogate_json)
    return payload


def summarize_proxy_metrics(proxy_result: Dict[str, Any]) -> Dict[str, Any]:
    prefix_metric_type = proxy_result.get("prefix_metric_type")
    if prefix_metric_type == "purity":
        prefix_values = proxy_result.get("prefix_purity_per_layer", []) or []
        prefix_direction = "higher_better"
    elif prefix_metric_type == "compactness":
        prefix_values = proxy_result.get("prefix_compactness_per_layer", []) or []
        prefix_direction = "lower_better"
    else:
        prefix_values = []
        prefix_direction = "unknown"

    entropy_values = proxy_result.get("entropy_per_layer", []) or []
    sibling_values = proxy_result.get("sibling_separation_per_layer", []) or []
    return {
        "collision_rate": proxy_result.get("collision_rate"),
        "frac_unique_ids": proxy_result.get("frac_unique_ids"),
        "entropy_per_layer": entropy_values,
        "avg_entropy": avg_or_none([float(value) for value in entropy_values]),
        "prefix_metric_type": prefix_metric_type,
        "prefix_values": prefix_values,
        "avg_prefix_metric": avg_or_none([float(value) for value in prefix_values]),
        "prefix_direction": prefix_direction,
        "sibling_separation_per_layer": sibling_values,
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
        "distance_gap_full_minus_sibling": proxy_result.get("distance_gap_full_minus_sibling"),
        "distance_gap_unrelated_minus_sibling": proxy_result.get(
            "distance_gap_unrelated_minus_sibling"
        ),
        "teacher_forced_last_ce": proxy_result.get("teacher_forced_last_ce"),
        "teacher_forced_last_acc": proxy_result.get("teacher_forced_last_acc"),
        "teacher_forced_last2_mean_ce": proxy_result.get("teacher_forced_last2_mean_ce"),
        "teacher_forced_last2_mean_acc": proxy_result.get("teacher_forced_last2_mean_acc"),
        "teacher_forced_leaf_ce": proxy_result.get("teacher_forced_leaf_ce"),
        "teacher_forced_leaf_acc": proxy_result.get("teacher_forced_leaf_acc"),
        "same_parent_hardest_sibling_margin": proxy_result.get(
            "same_parent_hardest_sibling_margin"
        ),
        "same_parent_within_parent_accuracy": proxy_result.get(
            "same_parent_within_parent_accuracy"
        ),
        "rollout1_last_ce": proxy_result.get("rollout1_last_ce"),
        "rollout1_last_acc": proxy_result.get("rollout1_last_acc"),
        "rollout2_last2_mean_ce": proxy_result.get("rollout2_last2_mean_ce"),
        "rollout2_last2_mean_acc": proxy_result.get("rollout2_last2_mean_acc"),
        "rollout2_last2_sequence_acc": proxy_result.get("rollout2_last2_sequence_acc"),
        "num_items": proxy_result.get("num_items"),
    }


def candidate_feature_vector(candidate: Dict[str, Any]) -> Dict[str, float]:
    features = {
        "same_parent_assignment_margin": float(candidate["same_parent_assignment_margin"]),
        "same_parent_positive_margin_fraction": float(
            candidate["same_parent_positive_margin_fraction"]
        ),
        "same_parent_leaf_uniqueness": float(candidate["same_parent_leaf_uniqueness"]),
        "distance_gap_unrelated_minus_sibling": float(
            candidate["distance_gap_unrelated_minus_sibling"]
        ),
        "distance_gap_full_minus_sibling": float(candidate["distance_gap_full_minus_sibling"]),
        "avg_sibling_separation": float(candidate["avg_sibling_separation"]),
        "near_collision_separation": float(candidate["near_collision_separation"]),
        "avg_entropy": float(candidate["avg_entropy"]),
        "avg_prefix_metric": float(candidate["avg_prefix_metric"]),
    }
    if candidate.get("teacher_forced_last_ce") is not None:
        features["teacher_forced_last_ce"] = float(candidate["teacher_forced_last_ce"])
    if candidate.get("teacher_forced_last_acc") is not None:
        features["teacher_forced_last_acc"] = float(candidate["teacher_forced_last_acc"])
    if candidate.get("teacher_forced_last2_mean_ce") is not None:
        features["teacher_forced_last2_mean_ce"] = float(
            candidate["teacher_forced_last2_mean_ce"]
        )
    if candidate.get("teacher_forced_last2_mean_acc") is not None:
        features["teacher_forced_last2_mean_acc"] = float(
            candidate["teacher_forced_last2_mean_acc"]
        )
    if candidate.get("teacher_forced_leaf_ce") is not None:
        features["teacher_forced_leaf_ce"] = float(candidate["teacher_forced_leaf_ce"])
    if candidate.get("teacher_forced_leaf_acc") is not None:
        features["teacher_forced_leaf_acc"] = float(candidate["teacher_forced_leaf_acc"])
    if candidate.get("same_parent_hardest_sibling_margin") is not None:
        features["same_parent_hardest_sibling_margin"] = float(
            candidate["same_parent_hardest_sibling_margin"]
        )
    if candidate.get("same_parent_within_parent_accuracy") is not None:
        features["same_parent_within_parent_accuracy"] = float(
            candidate["same_parent_within_parent_accuracy"]
        )
    if candidate.get("rollout1_last_ce") is not None:
        features["rollout1_last_ce"] = float(candidate["rollout1_last_ce"])
    if candidate.get("rollout1_last_acc") is not None:
        features["rollout1_last_acc"] = float(candidate["rollout1_last_acc"])
    if candidate.get("rollout2_last2_mean_ce") is not None:
        features["rollout2_last2_mean_ce"] = float(candidate["rollout2_last2_mean_ce"])
    if candidate.get("rollout2_last2_mean_acc") is not None:
        features["rollout2_last2_mean_acc"] = float(candidate["rollout2_last2_mean_acc"])
    if candidate.get("rollout2_last2_sequence_acc") is not None:
        features["rollout2_last2_sequence_acc"] = float(
            candidate["rollout2_last2_sequence_acc"]
        )
    return features


def dominates(a: Dict[str, Any], b: Dict[str, Any], prefix_direction: str) -> bool:
    epsilon = 1e-12
    a_features = candidate_feature_vector(a)
    b_features = candidate_feature_vector(b)

    comparisons = [
        a_features["same_parent_assignment_margin"]
        >= b_features["same_parent_assignment_margin"] - epsilon,
        a_features["same_parent_positive_margin_fraction"]
        >= b_features["same_parent_positive_margin_fraction"] - epsilon,
        a_features["same_parent_leaf_uniqueness"]
        >= b_features["same_parent_leaf_uniqueness"] - epsilon,
        a_features["distance_gap_unrelated_minus_sibling"]
        >= b_features["distance_gap_unrelated_minus_sibling"] - epsilon,
        a_features["distance_gap_full_minus_sibling"]
        >= b_features["distance_gap_full_minus_sibling"] - epsilon,
        a_features["avg_sibling_separation"] >= b_features["avg_sibling_separation"] - epsilon,
        a_features["near_collision_separation"] >= b_features["near_collision_separation"] - epsilon,
        a_features["avg_entropy"] >= b_features["avg_entropy"] - epsilon,
    ]
    strict = [
        a_features["same_parent_assignment_margin"]
        > b_features["same_parent_assignment_margin"] + epsilon,
        a_features["same_parent_positive_margin_fraction"]
        > b_features["same_parent_positive_margin_fraction"] + epsilon,
        a_features["same_parent_leaf_uniqueness"]
        > b_features["same_parent_leaf_uniqueness"] + epsilon,
        a_features["distance_gap_unrelated_minus_sibling"]
        > b_features["distance_gap_unrelated_minus_sibling"] + epsilon,
        a_features["distance_gap_full_minus_sibling"]
        > b_features["distance_gap_full_minus_sibling"] + epsilon,
        a_features["avg_sibling_separation"] > b_features["avg_sibling_separation"] + epsilon,
        a_features["near_collision_separation"] > b_features["near_collision_separation"] + epsilon,
        a_features["avg_entropy"] > b_features["avg_entropy"] + epsilon,
    ]

    if prefix_direction == "lower_better":
        comparisons.append(
            a_features["avg_prefix_metric"] <= b_features["avg_prefix_metric"] + epsilon
        )
        strict.append(a_features["avg_prefix_metric"] < b_features["avg_prefix_metric"] - epsilon)
    elif prefix_direction == "higher_better":
        comparisons.append(
            a_features["avg_prefix_metric"] >= b_features["avg_prefix_metric"] - epsilon
        )
        strict.append(a_features["avg_prefix_metric"] > b_features["avg_prefix_metric"] + epsilon)

    return all(comparisons) and any(strict)


def selection_sort_key(candidate: Dict[str, Any], prefix_direction: str) -> Tuple[Any, ...]:
    avg_prefix_metric = float(candidate["avg_prefix_metric"])
    prefix_key = avg_prefix_metric if prefix_direction == "lower_better" else -avg_prefix_metric
    teacher_last_ce = (
        float(candidate["teacher_forced_last_ce"])
        if candidate.get("teacher_forced_last_ce") is not None
        else float("inf")
    )
    teacher_last_acc = (
        -float(candidate["teacher_forced_last_acc"])
        if candidate.get("teacher_forced_last_acc") is not None
        else float("inf")
    )
    teacher_last2_ce = (
        float(candidate["teacher_forced_last2_mean_ce"])
        if candidate.get("teacher_forced_last2_mean_ce") is not None
        else float("inf")
    )
    teacher_last2_acc = (
        -float(candidate["teacher_forced_last2_mean_acc"])
        if candidate.get("teacher_forced_last2_mean_acc") is not None
        else float("inf")
    )
    rollout1_last_ce = (
        float(candidate["rollout1_last_ce"])
        if candidate.get("rollout1_last_ce") is not None
        else float("inf")
    )
    rollout1_last_acc = (
        -float(candidate["rollout1_last_acc"])
        if candidate.get("rollout1_last_acc") is not None
        else float("inf")
    )
    rollout2_last2_ce = (
        float(candidate["rollout2_last2_mean_ce"])
        if candidate.get("rollout2_last2_mean_ce") is not None
        else float("inf")
    )
    rollout2_last2_acc = (
        -float(candidate["rollout2_last2_mean_acc"])
        if candidate.get("rollout2_last2_mean_acc") is not None
        else float("inf")
    )
    rollout2_seq_acc = (
        -float(candidate["rollout2_last2_sequence_acc"])
        if candidate.get("rollout2_last2_sequence_acc") is not None
        else float("inf")
    )
    hardest_sibling_margin = (
        -float(candidate["same_parent_hardest_sibling_margin"])
        if candidate.get("same_parent_hardest_sibling_margin") is not None
        else float("inf")
    )
    within_parent_acc = (
        -float(candidate["same_parent_within_parent_accuracy"])
        if candidate.get("same_parent_within_parent_accuracy") is not None
        else float("inf")
    )
    return (
        rollout1_last_ce,
        rollout1_last_acc,
        rollout2_last2_ce,
        rollout2_last2_acc,
        rollout2_seq_acc,
        hardest_sibling_margin,
        within_parent_acc,
        teacher_last_ce,
        teacher_last_acc,
        teacher_last2_ce,
        teacher_last2_acc,
        -float(candidate["same_parent_assignment_margin"]),
        -float(candidate["same_parent_positive_margin_fraction"]),
        -float(candidate["same_parent_leaf_uniqueness"]),
        -float(candidate["distance_gap_unrelated_minus_sibling"]),
        -float(candidate["distance_gap_full_minus_sibling"]),
        -float(candidate["avg_sibling_separation"]),
        -float(candidate["near_collision_separation"]),
        -float(candidate["avg_entropy"]),
        prefix_key,
        float(candidate["weight"]),
    )


def surrogate_sort_key(candidate: Dict[str, Any], prefix_direction: str) -> Tuple[Any, ...]:
    primary = candidate.get("surrogate_primary_score")
    secondary = candidate.get("surrogate_secondary_score")
    primary_key = -float(primary) if primary is not None else float("inf")
    secondary_key = -float(secondary) if secondary is not None else float("inf")
    return (primary_key, secondary_key, *selection_sort_key(candidate, prefix_direction))


def load_reference_metrics(
    baseline_train_dir: Path,
    baseline_proxy_json: Path,
) -> Dict[str, Any]:
    baseline_training = load_training_summary(baseline_train_dir)
    baseline_proxy = load_proxy_result(baseline_proxy_json)
    baseline_summary = summarize_proxy_metrics(baseline_proxy)
    return {
        **baseline_training,
        **baseline_summary,
        "train_dir": str(baseline_train_dir),
        "proxy_json": str(baseline_proxy_json),
    }


def extract_surrogate_scores(
    surrogate_payload: Optional[Dict[str, Any]],
    primary_metric: str,
    secondary_metric: str,
) -> Tuple[Optional[float], Optional[float]]:
    if surrogate_payload is None:
        return None, None
    best_metrics = surrogate_payload.get("best_metrics", {})
    primary = best_metrics.get(primary_metric)
    secondary = best_metrics.get(secondary_metric)
    primary_score = float(primary) if isinstance(primary, (int, float)) else None
    secondary_score = float(secondary) if isinstance(secondary, (int, float)) else None
    return primary_score, secondary_score


def evaluate_candidate(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    reasons: List[str] = []
    training_metrics = candidate["training"]
    proxy_metrics = candidate["proxy_summary"]

    required_scalars = [
        proxy_metrics["avg_entropy"],
        proxy_metrics["avg_prefix_metric"],
        proxy_metrics["avg_sibling_separation"],
        proxy_metrics["near_collision_separation"],
        proxy_metrics["same_parent_leaf_uniqueness"],
        proxy_metrics["same_parent_assignment_margin"],
        proxy_metrics["same_parent_positive_margin_fraction"],
        proxy_metrics["distance_gap_full_minus_sibling"],
        proxy_metrics["distance_gap_unrelated_minus_sibling"],
    ]
    if any(is_non_finite(value) for value in required_scalars):
        reasons.append("unstable_or_missing_metrics")

    for field_name in (
        "final_training_loss",
        "final_quantization_loss",
        "final_reconstruction_loss",
        "final_hierarchy_loss",
    ):
        field_value = training_metrics.get(field_name)
        if field_value is None:
            continue
        if is_non_finite(field_value):
            reasons.append(f"{field_name}_is_non_finite")
        elif abs(float(field_value)) > args.loss_explosion_threshold:
            reasons.append(
                f"{field_name}_exceeds_explosion_threshold({args.loss_explosion_threshold})"
            )

    collision_delta = float(proxy_metrics["collision_rate"]) - float(baseline["collision_rate"])
    frac_unique_delta = float(proxy_metrics["frac_unique_ids"]) - float(
        baseline["frac_unique_ids"]
    )
    entropy_deltas = [
        float(candidate_entropy) - float(baseline_entropy)
        for candidate_entropy, baseline_entropy in zip(
            proxy_metrics["entropy_per_layer"],
            baseline["entropy_per_layer"],
        )
    ]
    avg_entropy_delta = float(proxy_metrics["avg_entropy"]) - float(baseline["avg_entropy"])
    avg_prefix_delta = float(proxy_metrics["avg_prefix_metric"]) - float(
        baseline["avg_prefix_metric"]
    )
    avg_sibling_delta = float(proxy_metrics["avg_sibling_separation"]) - float(
        baseline["avg_sibling_separation"]
    )
    near_collision_delta = float(proxy_metrics["near_collision_separation"]) - float(
        baseline["near_collision_separation"]
    )
    same_parent_uniqueness_delta = float(proxy_metrics["same_parent_leaf_uniqueness"]) - float(
        baseline["same_parent_leaf_uniqueness"]
    )
    same_parent_margin_delta = float(proxy_metrics["same_parent_assignment_margin"]) - float(
        baseline["same_parent_assignment_margin"]
    )
    same_parent_positive_margin_delta = float(
        proxy_metrics["same_parent_positive_margin_fraction"]
    ) - float(baseline["same_parent_positive_margin_fraction"])
    full_gap_delta = float(proxy_metrics["distance_gap_full_minus_sibling"]) - float(
        baseline["distance_gap_full_minus_sibling"]
    )
    unrelated_gap_delta = float(
        proxy_metrics["distance_gap_unrelated_minus_sibling"]
    ) - float(baseline["distance_gap_unrelated_minus_sibling"])
    quantization_loss_delta = float(training_metrics["final_quantization_loss"]) - float(
        baseline["final_quantization_loss"]
    )

    if collision_delta > args.collision_worse_threshold:
        reasons.append(
            f"collision_rate_worse_than_baseline(+{collision_delta:.6f} > {args.collision_worse_threshold:.6f})"
        )

    if frac_unique_delta < args.frac_unique_min_improvement - 1e-12:
        reasons.append(
            "frac_unique_ids_regressed_too_much("
            f"{float(proxy_metrics['frac_unique_ids']):.6f} <= "
            f"{float(baseline['frac_unique_ids']) + args.frac_unique_min_improvement:.6f})"
        )

    if avg_entropy_delta < -args.entropy_avg_drop_threshold:
        reasons.append(
            f"avg_entropy_dropped_too_much({avg_entropy_delta:.6f} < "
            f"-{args.entropy_avg_drop_threshold:.6f})"
        )
    if entropy_deltas and min(entropy_deltas) < -args.entropy_layer_drop_threshold:
        reasons.append(
            f"layer_entropy_dropped_too_much(min_delta={min(entropy_deltas):.6f})"
        )
    if same_parent_uniqueness_delta < -args.same_parent_uniqueness_drop_threshold:
        reasons.append(
            f"same_parent_leaf_uniqueness_dropped_too_much({same_parent_uniqueness_delta:.6f})"
        )
    if same_parent_margin_delta < -args.same_parent_margin_drop_threshold:
        reasons.append(
            f"same_parent_assignment_margin_dropped_too_much({same_parent_margin_delta:.6f})"
        )
    if (
        same_parent_positive_margin_delta
        < -args.same_parent_positive_margin_drop_threshold
    ):
        reasons.append(
            "same_parent_positive_margin_fraction_dropped_too_much("
            f"{same_parent_positive_margin_delta:.6f})"
        )
    candidate["collision_rate"] = float(proxy_metrics["collision_rate"])
    candidate["frac_unique_ids"] = float(proxy_metrics["frac_unique_ids"])
    candidate["avg_entropy"] = float(proxy_metrics["avg_entropy"])
    candidate["entropy_per_layer"] = [float(value) for value in proxy_metrics["entropy_per_layer"]]
    candidate["avg_prefix_metric"] = float(proxy_metrics["avg_prefix_metric"])
    candidate["prefix_metric_type"] = proxy_metrics["prefix_metric_type"]
    candidate["prefix_values"] = [float(value) for value in proxy_metrics["prefix_values"]]
    candidate["avg_sibling_separation"] = float(proxy_metrics["avg_sibling_separation"])
    candidate["near_collision_separation"] = float(proxy_metrics["near_collision_separation"])
    candidate["same_parent_leaf_uniqueness"] = float(proxy_metrics["same_parent_leaf_uniqueness"])
    candidate["same_parent_collision_rate"] = float(proxy_metrics["same_parent_collision_rate"])
    candidate["same_parent_assignment_margin"] = float(
        proxy_metrics["same_parent_assignment_margin"]
    )
    candidate["same_parent_positive_margin_fraction"] = float(
        proxy_metrics["same_parent_positive_margin_fraction"]
    )
    candidate["full_collision_distance_mean"] = float(proxy_metrics["full_collision_distance_mean"])
    candidate["sibling_distance_mean"] = float(proxy_metrics["sibling_distance_mean"])
    candidate["mid_prefix_distance_mean"] = float(proxy_metrics["mid_prefix_distance_mean"])
    candidate["unrelated_distance_mean"] = float(proxy_metrics["unrelated_distance_mean"])
    candidate["distance_gap_full_minus_sibling"] = float(
        proxy_metrics["distance_gap_full_minus_sibling"]
    )
    candidate["distance_gap_unrelated_minus_sibling"] = float(
        proxy_metrics["distance_gap_unrelated_minus_sibling"]
    )
    candidate["teacher_forced_last_ce"] = (
        float(proxy_metrics["teacher_forced_last_ce"])
        if proxy_metrics.get("teacher_forced_last_ce") is not None
        else None
    )
    candidate["teacher_forced_last_acc"] = (
        float(proxy_metrics["teacher_forced_last_acc"])
        if proxy_metrics.get("teacher_forced_last_acc") is not None
        else None
    )
    candidate["teacher_forced_last2_mean_ce"] = (
        float(proxy_metrics["teacher_forced_last2_mean_ce"])
        if proxy_metrics.get("teacher_forced_last2_mean_ce") is not None
        else None
    )
    candidate["teacher_forced_last2_mean_acc"] = (
        float(proxy_metrics["teacher_forced_last2_mean_acc"])
        if proxy_metrics.get("teacher_forced_last2_mean_acc") is not None
        else None
    )
    candidate["teacher_forced_leaf_ce"] = (
        float(proxy_metrics["teacher_forced_leaf_ce"])
        if proxy_metrics.get("teacher_forced_leaf_ce") is not None
        else None
    )
    candidate["teacher_forced_leaf_acc"] = (
        float(proxy_metrics["teacher_forced_leaf_acc"])
        if proxy_metrics.get("teacher_forced_leaf_acc") is not None
        else None
    )
    candidate["same_parent_hardest_sibling_margin"] = (
        float(proxy_metrics["same_parent_hardest_sibling_margin"])
        if proxy_metrics.get("same_parent_hardest_sibling_margin") is not None
        else None
    )
    candidate["same_parent_within_parent_accuracy"] = (
        float(proxy_metrics["same_parent_within_parent_accuracy"])
        if proxy_metrics.get("same_parent_within_parent_accuracy") is not None
        else None
    )
    candidate["rollout1_last_ce"] = (
        float(proxy_metrics["rollout1_last_ce"])
        if proxy_metrics.get("rollout1_last_ce") is not None
        else None
    )
    candidate["rollout1_last_acc"] = (
        float(proxy_metrics["rollout1_last_acc"])
        if proxy_metrics.get("rollout1_last_acc") is not None
        else None
    )
    candidate["rollout2_last2_mean_ce"] = (
        float(proxy_metrics["rollout2_last2_mean_ce"])
        if proxy_metrics.get("rollout2_last2_mean_ce") is not None
        else None
    )
    candidate["rollout2_last2_mean_acc"] = (
        float(proxy_metrics["rollout2_last2_mean_acc"])
        if proxy_metrics.get("rollout2_last2_mean_acc") is not None
        else None
    )
    candidate["rollout2_last2_sequence_acc"] = (
        float(proxy_metrics["rollout2_last2_sequence_acc"])
        if proxy_metrics.get("rollout2_last2_sequence_acc") is not None
        else None
    )
    candidate["final_quantization_loss"] = float(training_metrics["final_quantization_loss"])
    candidate["final_hierarchy_loss"] = training_metrics["final_hierarchy_loss"]
    candidate["collision_delta_vs_baseline"] = collision_delta
    candidate["frac_unique_delta_vs_baseline"] = frac_unique_delta
    candidate["avg_entropy_delta_vs_baseline"] = avg_entropy_delta
    candidate["avg_prefix_delta_vs_baseline"] = avg_prefix_delta
    candidate["avg_sibling_separation_delta_vs_baseline"] = avg_sibling_delta
    candidate["near_collision_separation_delta_vs_baseline"] = near_collision_delta
    candidate["same_parent_leaf_uniqueness_delta_vs_baseline"] = same_parent_uniqueness_delta
    candidate["same_parent_assignment_margin_delta_vs_baseline"] = same_parent_margin_delta
    candidate["same_parent_positive_margin_fraction_delta_vs_baseline"] = (
        same_parent_positive_margin_delta
    )
    candidate["distance_gap_full_minus_sibling_delta_vs_baseline"] = full_gap_delta
    candidate["distance_gap_unrelated_minus_sibling_delta_vs_baseline"] = unrelated_gap_delta
    candidate["quantization_loss_delta_vs_baseline"] = quantization_loss_delta
    candidate["screening_reasons"] = reasons
    candidate["screening_status"] = "kept" if not reasons else "rejected"
    return candidate


def build_markdown_report(selection_payload: Dict[str, Any]) -> str:
    lines = [
        f"## {selection_payload['method_family']} Weight Sweep (Poincare v3.5)",
        "",
        f"- sweep_weights: `{selection_payload['weights']}`",
        f"- selected_weight: `{selection_payload['selected_weight']}`",
        f"- selection_mode: `{selection_payload['selection_mode']}`",
        "",
        "### Baseline Reference",
        "",
        f"- avg_prefix_metric: `{selection_payload['baseline']['avg_prefix_metric']:.6f}`",
        f"- avg_entropy: `{selection_payload['baseline']['avg_entropy']:.6f}`",
        f"- avg_sibling_separation: `{selection_payload['baseline']['avg_sibling_separation']:.6f}`",
        f"- near_collision_separation: `{selection_payload['baseline']['near_collision_separation']:.6f}`",
        f"- same_parent_leaf_uniqueness: `{selection_payload['baseline']['same_parent_leaf_uniqueness']:.6f}`",
        f"- same_parent_assignment_margin: `{selection_payload['baseline']['same_parent_assignment_margin']:.6f}`",
        f"- same_parent_positive_margin_fraction: `{selection_payload['baseline']['same_parent_positive_margin_fraction']:.6f}`",
        f"- distance_gap_full_minus_sibling: `{selection_payload['baseline']['distance_gap_full_minus_sibling']:.6f}`",
        f"- distance_gap_unrelated_minus_sibling: `{selection_payload['baseline']['distance_gap_unrelated_minus_sibling']:.6f}`",
        "",
        "### Candidate Results",
        "",
        "| weight | status | parent_margin | parent_unique | pos_margin_frac | full-sib_gap | unrelated-sib_gap | avg_prefix | avg_entropy | teacher(last) | surrogate | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]

    for candidate in selection_payload["candidates"]:
        notes = "selected" if candidate["selected"] else ""
        if candidate["screening_reasons"]:
            notes = ", ".join(candidate["screening_reasons"])
        elif candidate["pareto_status"] == "frontier":
            notes = "pareto_frontier"
        surrogate_text = (
            f"{candidate['surrogate_primary_score']:.6f}"
            if candidate["surrogate_primary_score"] is not None
            else "n/a"
        )
        teacher_text = (
            f"roll1_ce={candidate['rollout1_last_ce']:.4f},roll2_ce={candidate['rollout2_last2_mean_ce']:.4f}"
            if candidate.get("rollout1_last_ce") is not None
            else "n/a"
        )
        lines.append(
            "| "
            f"{candidate['weight_text']} | "
            f"{candidate['screening_status']} | "
            f"{candidate['same_parent_assignment_margin']:.6f} | "
            f"{candidate['same_parent_leaf_uniqueness']:.6f} | "
            f"{candidate['same_parent_positive_margin_fraction']:.6f} | "
            f"{candidate['distance_gap_full_minus_sibling']:.6f} | "
            f"{candidate['distance_gap_unrelated_minus_sibling']:.6f} | "
            f"{candidate['avg_prefix_metric']:.6f} | "
            f"{candidate['avg_entropy']:.6f} | "
            f"{teacher_text} | "
            f"{surrogate_text} | "
            f"{notes} |"
        )

    lines.extend(
        [
            "",
            "### Selection Logic",
            "",
            "- deep_level_priority: `rollout1_last_ce -> rollout1_last_acc -> rollout2_last2_mean_ce -> rollout2_last2_mean_acc -> rollout2_last2_sequence_acc -> hardest_sibling_margin -> within_parent_acc -> teacher_forced_last2_mean_ce -> teacher_forced_last2_mean_acc`",
            "- geometry_guardrails: `full/sibling and unrelated/sibling gaps are now soft-scored, not hard rejection rules.`",
            "- support_metrics: `avg_sibling_separation -> near_collision_separation -> avg_entropy -> avg_prefix_metric`",
            "- collision_and_unique_policy: `kept only as regression guardrails, not as primary ranking metrics.`",
            "",
            f"- selected_method: `{selection_payload['selected_method']}`",
            f"- selected_weight: `{selection_payload['selected_weight']}`",
            f"- selection_mode: `{selection_payload['selection_mode']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = load_reference_metrics(
        Path(args.baseline_train_dir),
        Path(args.baseline_proxy_json),
    )
    candidate_specs = parse_candidate_specs(args.candidate)
    prefix_direction = baseline["prefix_direction"]

    candidates: List[Dict[str, Any]] = []
    for candidate_spec in candidate_specs:
        candidate = {
            **candidate_spec,
            "training": load_training_summary(Path(candidate_spec["train_dir"])),
            "proxy": load_proxy_result(Path(candidate_spec["proxy_json"])),
            "surrogate": load_surrogate_result(
                Path(candidate_spec["surrogate_json"])
                if candidate_spec["surrogate_json"] is not None
                else None
            ),
        }
        candidate["proxy_summary"] = summarize_proxy_metrics(candidate["proxy"])
        (
            candidate["surrogate_primary_score"],
            candidate["surrogate_secondary_score"],
        ) = extract_surrogate_scores(
            candidate["surrogate"],
            args.surrogate_primary_metric,
            args.surrogate_secondary_metric,
        )
        if candidate["proxy_summary"]["prefix_direction"] != prefix_direction:
            raise ValueError("Prefix metric type changed across candidates; cannot compare reliably.")
        evaluate_candidate(baseline, candidate, args)
        candidates.append(candidate)

    survivors = [candidate for candidate in candidates if candidate["screening_status"] == "kept"]
    selection_mode = "deep_level_soft_score"
    selection_pool = survivors
    for candidate in candidates:
        candidate["pareto_status"] = "not_used"
        candidate["dominators"] = []
    if not selection_pool:
        selection_mode = "fallback_no_candidate_passed_filters"
        selection_pool = [
            candidate
            for candidate in candidates
            if "unstable_or_missing_metrics" not in candidate["screening_reasons"]
        ]
        if not selection_pool:
            selection_pool = candidates

    selection_pool_with_surrogate = [
        candidate
        for candidate in selection_pool
        if candidate.get("surrogate_primary_score") is not None
    ]
    if selection_pool_with_surrogate:
        selection_mode = "downstream_val_surrogate_then_deep_level_soft_score"
        selected_candidate = sorted(
            selection_pool_with_surrogate,
            key=lambda candidate: surrogate_sort_key(candidate, prefix_direction),
        )[0]
    else:
        selected_candidate = sorted(
            selection_pool,
            key=lambda candidate: selection_sort_key(candidate, prefix_direction),
        )[0]

    for candidate in candidates:
        candidate["selected"] = candidate["method"] == selected_candidate["method"]

    selection_payload = {
        "method_family": args.method_family,
        "weights": [candidate["weight_text"] for candidate in candidates],
        "selected_weight": selected_candidate["weight"],
        "selected_weight_text": selected_candidate["weight_text"],
        "selected_method": selected_candidate["method"],
        "selected_train_dir": selected_candidate["train_dir"],
        "selected_inference_dir": selected_candidate["inference_dir"],
        "selection_mode": selection_mode,
        "baseline": {
            "collision_rate": baseline["collision_rate"],
            "frac_unique_ids": baseline["frac_unique_ids"],
            "avg_entropy": baseline["avg_entropy"],
            "avg_prefix_metric": baseline["avg_prefix_metric"],
            "avg_sibling_separation": baseline["avg_sibling_separation"],
            "near_collision_separation": baseline["near_collision_separation"],
            "same_parent_leaf_uniqueness": baseline["same_parent_leaf_uniqueness"],
            "same_parent_collision_rate": baseline["same_parent_collision_rate"],
            "same_parent_assignment_margin": baseline["same_parent_assignment_margin"],
            "same_parent_positive_margin_fraction": baseline["same_parent_positive_margin_fraction"],
            "distance_gap_full_minus_sibling": baseline["distance_gap_full_minus_sibling"],
            "distance_gap_unrelated_minus_sibling": baseline["distance_gap_unrelated_minus_sibling"],
            "teacher_forced_last_ce": baseline.get("teacher_forced_last_ce"),
            "teacher_forced_last_acc": baseline.get("teacher_forced_last_acc"),
            "teacher_forced_last2_mean_ce": baseline.get("teacher_forced_last2_mean_ce"),
            "teacher_forced_last2_mean_acc": baseline.get("teacher_forced_last2_mean_acc"),
            "teacher_forced_leaf_ce": baseline.get("teacher_forced_leaf_ce"),
            "teacher_forced_leaf_acc": baseline.get("teacher_forced_leaf_acc"),
            "same_parent_hardest_sibling_margin": baseline.get(
                "same_parent_hardest_sibling_margin"
            ),
            "same_parent_within_parent_accuracy": baseline.get(
                "same_parent_within_parent_accuracy"
            ),
            "rollout1_last_ce": baseline.get("rollout1_last_ce"),
            "rollout1_last_acc": baseline.get("rollout1_last_acc"),
            "rollout2_last2_mean_ce": baseline.get("rollout2_last2_mean_ce"),
            "rollout2_last2_mean_acc": baseline.get("rollout2_last2_mean_acc"),
            "rollout2_last2_sequence_acc": baseline.get("rollout2_last2_sequence_acc"),
            "prefix_metric_type": baseline["prefix_metric_type"],
            "final_quantization_loss": baseline["final_quantization_loss"],
            "train_dir": baseline["train_dir"],
            "proxy_json": baseline["proxy_json"],
        },
        "thresholds": {
            "collision_worse_threshold": args.collision_worse_threshold,
            "frac_unique_min_improvement": args.frac_unique_min_improvement,
            "entropy_avg_drop_threshold": args.entropy_avg_drop_threshold,
            "entropy_layer_drop_threshold": args.entropy_layer_drop_threshold,
            "same_parent_uniqueness_drop_threshold": args.same_parent_uniqueness_drop_threshold,
            "same_parent_margin_drop_threshold": args.same_parent_margin_drop_threshold,
            "same_parent_positive_margin_drop_threshold": args.same_parent_positive_margin_drop_threshold,
            "full_sibling_gap_min": args.full_sibling_gap_min,
            "unrelated_sibling_gap_min": args.unrelated_sibling_gap_min,
            "quantization_loss_relative_threshold": args.quantization_loss_relative_threshold,
            "quantization_loss_absolute_threshold": args.quantization_loss_absolute_threshold,
            "loss_explosion_threshold": args.loss_explosion_threshold,
            "prefix_worse_threshold": args.prefix_worse_threshold,
        },
        "surrogate": {
            "primary_metric": args.surrogate_primary_metric,
            "secondary_metric": args.surrogate_secondary_metric,
        },
        "candidates": [
            {
                "weight": candidate["weight"],
                "weight_text": candidate["weight_text"],
                "method": candidate["method"],
                "train_dir": candidate["train_dir"],
                "inference_dir": candidate["inference_dir"],
                "proxy_json": candidate["proxy_json"],
                "surrogate_json": candidate["surrogate_json"],
                "screening_status": candidate["screening_status"],
                "screening_reasons": candidate["screening_reasons"],
                "pareto_status": candidate["pareto_status"],
                "dominators": candidate["dominators"],
                "selected": candidate["selected"],
                "collision_rate": candidate["collision_rate"],
                "frac_unique_ids": candidate["frac_unique_ids"],
                "avg_entropy": candidate["avg_entropy"],
                "entropy_per_layer": candidate["entropy_per_layer"],
                "avg_prefix_metric": candidate["avg_prefix_metric"],
                "prefix_metric_type": candidate["prefix_metric_type"],
                "prefix_values": candidate["prefix_values"],
                "avg_sibling_separation": candidate["avg_sibling_separation"],
                "near_collision_separation": candidate["near_collision_separation"],
                "same_parent_leaf_uniqueness": candidate["same_parent_leaf_uniqueness"],
                "same_parent_collision_rate": candidate["same_parent_collision_rate"],
                "same_parent_assignment_margin": candidate["same_parent_assignment_margin"],
                "same_parent_positive_margin_fraction": candidate["same_parent_positive_margin_fraction"],
                "full_collision_distance_mean": candidate["full_collision_distance_mean"],
                "sibling_distance_mean": candidate["sibling_distance_mean"],
                "mid_prefix_distance_mean": candidate["mid_prefix_distance_mean"],
                "unrelated_distance_mean": candidate["unrelated_distance_mean"],
                "distance_gap_full_minus_sibling": candidate["distance_gap_full_minus_sibling"],
                "distance_gap_unrelated_minus_sibling": candidate["distance_gap_unrelated_minus_sibling"],
                "teacher_forced_last_ce": candidate["teacher_forced_last_ce"],
                "teacher_forced_last_acc": candidate["teacher_forced_last_acc"],
                "teacher_forced_last2_mean_ce": candidate["teacher_forced_last2_mean_ce"],
                "teacher_forced_last2_mean_acc": candidate["teacher_forced_last2_mean_acc"],
                "teacher_forced_leaf_ce": candidate["teacher_forced_leaf_ce"],
                "teacher_forced_leaf_acc": candidate["teacher_forced_leaf_acc"],
                "same_parent_hardest_sibling_margin": candidate[
                    "same_parent_hardest_sibling_margin"
                ],
                "same_parent_within_parent_accuracy": candidate[
                    "same_parent_within_parent_accuracy"
                ],
                "rollout1_last_ce": candidate["rollout1_last_ce"],
                "rollout1_last_acc": candidate["rollout1_last_acc"],
                "rollout2_last2_mean_ce": candidate["rollout2_last2_mean_ce"],
                "rollout2_last2_mean_acc": candidate["rollout2_last2_mean_acc"],
                "rollout2_last2_sequence_acc": candidate[
                    "rollout2_last2_sequence_acc"
                ],
                "surrogate_primary_score": candidate["surrogate_primary_score"],
                "surrogate_secondary_score": candidate["surrogate_secondary_score"],
                "final_quantization_loss": candidate["final_quantization_loss"],
                "final_hierarchy_loss": candidate["final_hierarchy_loss"],
                "collision_delta_vs_baseline": candidate["collision_delta_vs_baseline"],
                "frac_unique_delta_vs_baseline": candidate["frac_unique_delta_vs_baseline"],
                "avg_entropy_delta_vs_baseline": candidate["avg_entropy_delta_vs_baseline"],
                "avg_prefix_delta_vs_baseline": candidate["avg_prefix_delta_vs_baseline"],
                "avg_sibling_separation_delta_vs_baseline": candidate["avg_sibling_separation_delta_vs_baseline"],
                "near_collision_separation_delta_vs_baseline": candidate["near_collision_separation_delta_vs_baseline"],
                "same_parent_leaf_uniqueness_delta_vs_baseline": candidate["same_parent_leaf_uniqueness_delta_vs_baseline"],
                "same_parent_assignment_margin_delta_vs_baseline": candidate["same_parent_assignment_margin_delta_vs_baseline"],
                "same_parent_positive_margin_fraction_delta_vs_baseline": candidate["same_parent_positive_margin_fraction_delta_vs_baseline"],
                "distance_gap_full_minus_sibling_delta_vs_baseline": candidate["distance_gap_full_minus_sibling_delta_vs_baseline"],
                "distance_gap_unrelated_minus_sibling_delta_vs_baseline": candidate["distance_gap_unrelated_minus_sibling_delta_vs_baseline"],
                "quantization_loss_delta_vs_baseline": candidate["quantization_loss_delta_vs_baseline"],
            }
            for candidate in candidates
        ],
    }

    markdown_report = build_markdown_report(selection_payload)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(selection_payload, indent=2, ensure_ascii=True) + "\n")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report + "\n")
    print(markdown_report)


if __name__ == "__main__":
    main()
