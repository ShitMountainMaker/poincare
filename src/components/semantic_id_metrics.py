import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


DEFAULT_CATEGORY_FIELDS = (
    "category",
    "taxonomy",
    "path",
    "category_path",
    "taxonomy_path",
)


def pairwise_lcp_matrix(cluster_ids: torch.Tensor) -> torch.Tensor:
    if cluster_ids.dim() != 2:
        raise ValueError("cluster_ids must have shape [batch_size, num_hierarchies].")

    equality_matrix = cluster_ids.unsqueeze(1).eq(cluster_ids.unsqueeze(0))
    prefix_matches = equality_matrix.to(dtype=torch.int64).cumprod(dim=-1)
    return prefix_matches.sum(dim=-1)


def pairwise_squared_euclidean_distance(embeddings: torch.Tensor) -> torch.Tensor:
    sq_norm = embeddings.pow(2).sum(dim=-1, keepdim=True)
    return (
        sq_norm + sq_norm.transpose(0, 1) - 2.0 * embeddings @ embeddings.transpose(0, 1)
    ).clamp_min(0.0)


def upper_triangle_pair_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def _normalize_item_id(item_id: Any) -> int:
    if isinstance(item_id, torch.Tensor):
        return int(item_id.item())
    return int(item_id)


def _normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return str(value.item())
        return json.dumps(value.tolist(), ensure_ascii=True)
    if isinstance(value, (list, tuple)):
        return " > ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    text = str(value).strip()
    return text if text else None


def _resolve_prediction_file(prediction_path: str) -> Path:
    path = Path(prediction_path)
    if path.is_dir():
        pkl_path = path / "merged_predictions.pkl"
        pt_path = path / "merged_predictions_tensor.pt"
        if pkl_path.exists():
            return pkl_path
        if pt_path.exists():
            return pt_path
        raise FileNotFoundError(
            f"No merged_predictions.pkl or merged_predictions_tensor.pt found in {path}"
        )
    return path


def load_semantic_id_predictions(
    prediction_path: str,
    num_hierarchies: Optional[int] = None,
) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
    resolved_path = _resolve_prediction_file(prediction_path)
    metadata = {
        "prediction_source": str(resolved_path),
        "prediction_format": resolved_path.suffix.lstrip("."),
        "notes": [],
    }

    if resolved_path.suffix == ".pkl":
        with open(resolved_path, "rb") as handle:
            rows = pickle.load(handle)

        normalized_rows = []
        for row in rows:
            item_id = _normalize_item_id(row["item_id"])
            cluster_ids = torch.as_tensor(row["cluster_ids"]).view(-1).long()
            normalized_rows.append((item_id, cluster_ids))

        normalized_rows.sort(key=lambda pair: pair[0])
        item_ids = [row[0] for row in normalized_rows]
        cluster_ids = torch.stack([row[1] for row in normalized_rows], dim=0)
    elif resolved_path.suffix == ".pt":
        tensor = torch.load(resolved_path, map_location="cpu")
        if tensor.dim() != 2:
            raise ValueError("Semantic id tensor file must be 2D.")

        if tensor.size(0) <= tensor.size(1):
            cluster_ids = tensor.t().contiguous().long()
            metadata["notes"].append(
                "Transposed semantic id tensor from [D, N] to [N, D]."
            )
        else:
            cluster_ids = tensor.contiguous().long()
        item_ids = list(range(cluster_ids.size(0)))
        metadata["notes"].append(
            "Using row index as item_id because tensor outputs do not store explicit ids."
        )
    else:
        raise ValueError(
            f"Unsupported semantic id file format: {resolved_path.suffix or '<none>'}"
        )

    if num_hierarchies is not None and cluster_ids.size(1) > num_hierarchies:
        metadata["notes"].append(
            f"Trimmed semantic ids from {cluster_ids.size(1)} to {num_hierarchies} hierarchies."
        )
        cluster_ids = cluster_ids[:, :num_hierarchies].contiguous()

    return item_ids, cluster_ids, metadata


def load_embedding_tensor(embedding_path: str) -> torch.Tensor:
    tensor = torch.load(embedding_path, map_location="cpu")
    if tensor.dim() != 2:
        raise ValueError("Embedding tensor must be 2D.")
    return tensor.float().contiguous()


def load_item_categories(
    item_ids: Iterable[int],
    metadata_csv: Optional[str] = None,
    metadata_json: Optional[str] = None,
    category_field: Optional[str] = None,
) -> Tuple[Optional[List[Optional[str]]], Dict[str, Any]]:
    if metadata_csv and metadata_json:
        raise ValueError("Only one of metadata_csv or metadata_json may be provided.")

    if metadata_csv is None and metadata_json is None:
        return None, {"category_source": None, "category_field": None, "notes": []}

    if metadata_csv is not None:
        raw_mapping = _load_category_mapping_from_csv(metadata_csv, category_field)
        category_source = metadata_csv
    else:
        raw_mapping = _load_category_mapping_from_json(metadata_json, category_field)
        category_source = metadata_json

    labels = [_normalize_label(raw_mapping.get(item_id)) for item_id in item_ids]
    available_count = sum(label is not None for label in labels)
    if available_count == 0:
        return None, {
            "category_source": category_source,
            "category_field": category_field,
            "notes": ["Category metadata file was provided but no matching labels were found."],
        }
    return labels, {
        "category_source": category_source,
        "category_field": category_field,
        "notes": [f"Loaded category labels for {available_count} items."],
    }


def _load_category_mapping_from_csv(
    metadata_csv: str,
    category_field: Optional[str] = None,
) -> Dict[int, Any]:
    with open(metadata_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV metadata file has no header: {metadata_csv}")

        item_id_field = "item_id" if "item_id" in reader.fieldnames else None
        if item_id_field is None:
            raise ValueError("CSV metadata must contain an item_id column.")

        resolved_category_field = category_field or _resolve_category_field(reader.fieldnames)
        if resolved_category_field is None:
            raise ValueError(
                "CSV metadata must contain one of the supported category fields or specify --category-field."
            )

        mapping = {}
        for row in reader:
            mapping[int(row[item_id_field])] = row.get(resolved_category_field)
        return mapping


def _load_category_mapping_from_json(
    metadata_json: str,
    category_field: Optional[str] = None,
) -> Dict[int, Any]:
    with open(metadata_json) as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        sample_value = next(iter(data.values()), None)
        if isinstance(sample_value, dict):
            resolved_category_field = category_field or _resolve_category_field(sample_value.keys())
            if resolved_category_field is None:
                raise ValueError(
                    "JSON metadata dict values must contain a category-like field or specify --category-field."
                )
            return {
                int(item_id): item_data.get(resolved_category_field)
                for item_id, item_data in data.items()
            }
        return {int(item_id): value for item_id, value in data.items()}

    if isinstance(data, list):
        if len(data) == 0:
            return {}
        if not isinstance(data[0], dict):
            raise ValueError("JSON metadata list must contain objects with item_id and category fields.")
        resolved_category_field = category_field or _resolve_category_field(data[0].keys())
        if resolved_category_field is None:
            raise ValueError(
                "JSON metadata objects must contain a category-like field or specify --category-field."
            )
        return {
            int(row["item_id"]): row.get(resolved_category_field)
            for row in data
            if "item_id" in row
        }

    raise ValueError("Unsupported JSON metadata structure.")


def _resolve_category_field(field_names: Iterable[str]) -> Optional[str]:
    field_set = set(field_names)
    for field in DEFAULT_CATEGORY_FIELDS:
        if field in field_set:
            return field
    return None


def infer_codebook_size(cluster_ids: torch.Tensor) -> int:
    return int(cluster_ids.max().item()) + 1 if cluster_ids.numel() > 0 else 0


def compute_codebook_utilization(
    cluster_ids: torch.Tensor,
    codebook_size: int,
) -> Dict[str, Any]:
    used_codes_per_layer = [
        int(torch.unique(cluster_ids[:, layer_idx]).numel())
        for layer_idx in range(cluster_ids.size(1))
    ]
    utilization_per_layer = [
        used_codes / codebook_size if codebook_size > 0 else 0.0
        for used_codes in used_codes_per_layer
    ]
    total_capacity = codebook_size * cluster_ids.size(1)
    overall_utilization = (
        sum(used_codes_per_layer) / total_capacity if total_capacity > 0 else 0.0
    )
    avg_utilization = (
        sum(utilization_per_layer) / len(utilization_per_layer)
        if utilization_per_layer
        else 0.0
    )
    return {
        "used_codes_per_layer": used_codes_per_layer,
        "utilization_per_layer": utilization_per_layer,
        "overall_utilization": overall_utilization,
        "avg_utilization": avg_utilization,
    }


def compute_collision_metrics(cluster_ids: torch.Tensor) -> Dict[str, Any]:
    ids_as_tuples = [tuple(row.tolist()) for row in cluster_ids]
    counts = Counter(ids_as_tuples)
    num_items = len(ids_as_tuples)
    num_unique_ids = len(counts)
    num_colliding_items = sum(count for count in counts.values() if count > 1)
    collision_rate = num_colliding_items / num_items if num_items > 0 else 0.0
    frac_unique_ids = num_unique_ids / num_items if num_items > 0 else 0.0
    num_collision_groups = sum(1 for count in counts.values() if count > 1)
    return {
        "num_items": num_items,
        "num_unique_ids": num_unique_ids,
        "num_colliding_items": num_colliding_items,
        "num_collision_groups": num_collision_groups,
        "collision_rate": collision_rate,
        "frac_unique_ids": frac_unique_ids,
    }


def compute_distribution_health(
    cluster_ids: torch.Tensor,
    top_k: int = 5,
) -> Dict[str, Any]:
    entropy_per_layer = []
    max_token_ratio_per_layer = []
    topk_token_coverage_per_layer = []

    for layer_idx in range(cluster_ids.size(1)):
        _, counts = torch.unique(cluster_ids[:, layer_idx], return_counts=True)
        probs = counts.float() / counts.sum().float().clamp_min(1.0)
        entropy = float((-probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        max_ratio = float(probs.max().item())
        sorted_probs = torch.sort(probs, descending=True).values
        coverage = float(sorted_probs[:top_k].sum().item())

        entropy_per_layer.append(entropy)
        max_token_ratio_per_layer.append(max_ratio)
        topk_token_coverage_per_layer.append(coverage)

    return {
        "entropy_per_layer": entropy_per_layer,
        "max_token_ratio_per_layer": max_token_ratio_per_layer,
        "topk_token_coverage_per_layer": topk_token_coverage_per_layer,
        "top_k": top_k,
    }


def compute_prefix_purity(
    cluster_ids: torch.Tensor,
    labels: List[Optional[str]],
) -> Dict[str, Any]:
    purity_per_layer = []
    labeled_fraction_per_layer = []

    for depth in range(1, cluster_ids.size(1) + 1):
        groups = defaultdict(list)
        for idx, row in enumerate(cluster_ids[:, :depth].tolist()):
            label = labels[idx]
            if label is not None:
                groups[tuple(row)].append(label)

        labeled_items = sum(len(group_labels) for group_labels in groups.values())
        majority_sum = sum(
            Counter(group_labels).most_common(1)[0][1]
            for group_labels in groups.values()
            if len(group_labels) > 0
        )
        purity = majority_sum / labeled_items if labeled_items > 0 else 0.0
        purity_per_layer.append(purity)
        labeled_fraction_per_layer.append(
            labeled_items / len(labels) if len(labels) > 0 else 0.0
        )

    return {
        "prefix_metric_type": "purity",
        "prefix_purity_per_layer": purity_per_layer,
        "labeled_fraction_per_layer": labeled_fraction_per_layer,
    }


def compute_prefix_compactness(
    cluster_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> Dict[str, Any]:
    compactness_per_layer = []
    non_singleton_fraction_per_layer = []

    for depth in range(1, cluster_ids.size(1) + 1):
        groups = defaultdict(list)
        for idx, row in enumerate(cluster_ids[:, :depth].tolist()):
            groups[tuple(row)].append(idx)

        weighted_compactness = 0.0
        total_items = 0
        non_singleton_items = 0
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue
            group_embeddings = embeddings[group_indices]
            centroid = group_embeddings.mean(dim=0, keepdim=True)
            group_spread = (group_embeddings - centroid).pow(2).sum(dim=-1).mean().item()
            weighted_compactness += group_spread * len(group_indices)
            total_items += len(group_indices)
            non_singleton_items += len(group_indices)

        compactness = weighted_compactness / total_items if total_items > 0 else 0.0
        compactness_per_layer.append(compactness)
        non_singleton_fraction_per_layer.append(
            non_singleton_items / cluster_ids.size(0) if cluster_ids.size(0) > 0 else 0.0
        )

    return {
        "prefix_metric_type": "compactness",
        "prefix_compactness_per_layer": compactness_per_layer,
        "non_singleton_fraction_per_layer": non_singleton_fraction_per_layer,
    }


def _weighted_centroid_pair_distance(
    child_groups: List[List[int]],
    embeddings: torch.Tensor,
) -> Tuple[float, int]:
    if len(child_groups) <= 1:
        return 0.0, 0

    centroids: List[torch.Tensor] = []
    sizes: List[int] = []
    for group_indices in child_groups:
        if not group_indices:
            continue
        group_embeddings = embeddings[group_indices]
        centroids.append(group_embeddings.mean(dim=0))
        sizes.append(len(group_indices))

    if len(centroids) <= 1:
        return 0.0, 0

    weighted_distance = 0.0
    total_pairs = 0
    for left_idx in range(len(centroids)):
        for right_idx in range(left_idx + 1, len(centroids)):
            pair_weight = sizes[left_idx] * sizes[right_idx]
            pair_distance = (
                centroids[left_idx] - centroids[right_idx]
            ).pow(2).sum().item()
            weighted_distance += pair_distance * pair_weight
            total_pairs += pair_weight
    return weighted_distance, total_pairs


def compute_sibling_separation(
    cluster_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> Dict[str, Any]:
    sibling_separation_per_layer = []
    multi_child_parent_fraction_per_layer = []

    num_items = cluster_ids.size(0)
    num_hierarchies = cluster_ids.size(1)
    for shared_prefix_depth in range(1, num_hierarchies):
        parents: Dict[Tuple[int, ...], Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for item_idx, row in enumerate(cluster_ids.tolist()):
            parent_prefix = tuple(row[:shared_prefix_depth])
            child_token = int(row[shared_prefix_depth])
            parents[parent_prefix][child_token].append(item_idx)

        weighted_distance = 0.0
        total_pairs = 0
        multi_child_items = 0
        for child_map in parents.values():
            child_groups = list(child_map.values())
            if len(child_groups) <= 1:
                continue
            weighted_part, pair_count = _weighted_centroid_pair_distance(
                child_groups,
                embeddings,
            )
            weighted_distance += weighted_part
            total_pairs += pair_count
            multi_child_items += sum(len(group_indices) for group_indices in child_groups)

        sibling_separation = weighted_distance / total_pairs if total_pairs > 0 else 0.0
        sibling_separation_per_layer.append(sibling_separation)
        multi_child_parent_fraction_per_layer.append(
            multi_child_items / num_items if num_items > 0 else 0.0
        )

    return {
        "sibling_separation_per_layer": sibling_separation_per_layer,
        "avg_sibling_separation": (
            sum(sibling_separation_per_layer) / len(sibling_separation_per_layer)
            if sibling_separation_per_layer
            else 0.0
        ),
        "near_collision_separation": (
            sibling_separation_per_layer[-1] if sibling_separation_per_layer else 0.0
        ),
        "multi_child_parent_fraction_per_layer": multi_child_parent_fraction_per_layer,
    }


def _masked_distance_mean(
    pairwise_dist: torch.Tensor,
    mask: torch.Tensor,
) -> Optional[float]:
    values = pairwise_dist[mask]
    if values.numel() == 0:
        return None
    return float(values.mean().item())


def compute_pair_type_distance_ordering(
    cluster_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> Dict[str, Any]:
    if cluster_ids.size(0) < 2:
        return {
            "full_collision_distance_mean": None,
            "sibling_distance_mean": None,
            "mid_prefix_distance_mean": None,
            "unrelated_distance_mean": None,
            "distance_gap_full_minus_sibling": None,
            "distance_gap_unrelated_minus_sibling": None,
        }

    pairwise_dist = pairwise_squared_euclidean_distance(embeddings.float()).sqrt()
    lcp = pairwise_lcp_matrix(cluster_ids.long())
    pair_mask = upper_triangle_pair_mask(cluster_ids.size(0), device=cluster_ids.device)
    num_hierarchies = cluster_ids.size(1)

    full_mask = pair_mask & lcp.eq(num_hierarchies)
    if num_hierarchies > 1:
        sibling_mask = pair_mask & lcp.eq(num_hierarchies - 1)
    else:
        sibling_mask = torch.zeros_like(pair_mask)
    if num_hierarchies > 2:
        mid_prefix_mask = pair_mask & lcp.eq(num_hierarchies - 2)
    else:
        mid_prefix_mask = torch.zeros_like(pair_mask)
    unrelated_mask = pair_mask & lcp.eq(0)

    full_mean = _masked_distance_mean(pairwise_dist, full_mask)
    sibling_mean = _masked_distance_mean(pairwise_dist, sibling_mask)
    mid_prefix_mean = _masked_distance_mean(pairwise_dist, mid_prefix_mask)
    unrelated_mean = _masked_distance_mean(pairwise_dist, unrelated_mask)

    return {
        "full_collision_distance_mean": full_mean,
        "sibling_distance_mean": sibling_mean,
        "mid_prefix_distance_mean": mid_prefix_mean,
        "unrelated_distance_mean": unrelated_mean,
        "distance_gap_full_minus_sibling": (
            full_mean - sibling_mean
            if full_mean is not None and sibling_mean is not None
            else None
        ),
        "distance_gap_unrelated_minus_sibling": (
            unrelated_mean - sibling_mean
            if unrelated_mean is not None and sibling_mean is not None
            else None
        ),
    }


def compute_same_parent_leaf_metrics(
    cluster_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> Dict[str, Any]:
    num_hierarchies = cluster_ids.size(1)
    if num_hierarchies < 2:
        return {
            "same_parent_leaf_uniqueness": None,
            "same_parent_collision_rate": None,
            "same_parent_assignment_margin": None,
            "same_parent_positive_margin_fraction": None,
        }

    parents: Dict[Tuple[int, ...], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for item_idx, row in enumerate(cluster_ids.tolist()):
        parent_prefix = tuple(row[:-1])
        leaf_token = int(row[-1])
        parents[parent_prefix][leaf_token].append(item_idx)

    weighted_leaf_uniqueness = 0.0
    weighted_collision_rate = 0.0
    uniqueness_items = 0
    assignment_margin_sum = 0.0
    assignment_margin_positive = 0
    assignment_margin_items = 0

    for child_map in parents.values():
        all_item_count = sum(len(group_indices) for group_indices in child_map.values())
        if all_item_count <= 1:
            continue

        unique_leaf_count = len(child_map)
        colliding_item_count = sum(
            len(group_indices)
            for group_indices in child_map.values()
            if len(group_indices) > 1
        )
        weighted_leaf_uniqueness += (unique_leaf_count / all_item_count) * all_item_count
        weighted_collision_rate += (colliding_item_count / all_item_count) * all_item_count
        uniqueness_items += all_item_count

        if len(child_map) <= 1:
            continue

        child_tokens = list(child_map.keys())
        child_centroids = {
            child_token: embeddings[group_indices].float().mean(dim=0)
            for child_token, group_indices in child_map.items()
        }
        for child_token in child_tokens:
            own_centroid = child_centroids[child_token]
            other_centroids = [
                child_centroids[other_child_token]
                for other_child_token in child_tokens
                if other_child_token != child_token
            ]
            if not other_centroids:
                continue
            other_centroids_tensor = torch.stack(other_centroids, dim=0)
            for item_idx in child_map[child_token]:
                item_embedding = embeddings[item_idx].float()
                own_distance = (item_embedding - own_centroid).pow(2).sum().item()
                other_distances = (
                    other_centroids_tensor - item_embedding.unsqueeze(0)
                ).pow(2).sum(dim=-1)
                nearest_sibling_distance = float(other_distances.min().item())
                margin = nearest_sibling_distance - own_distance
                assignment_margin_sum += margin
                assignment_margin_positive += int(margin > 0.0)
                assignment_margin_items += 1

    return {
        "same_parent_leaf_uniqueness": (
            weighted_leaf_uniqueness / uniqueness_items if uniqueness_items > 0 else None
        ),
        "same_parent_collision_rate": (
            weighted_collision_rate / uniqueness_items if uniqueness_items > 0 else None
        ),
        "same_parent_assignment_margin": (
            assignment_margin_sum / assignment_margin_items
            if assignment_margin_items > 0
            else None
        ),
        "same_parent_positive_margin_fraction": (
            assignment_margin_positive / assignment_margin_items
            if assignment_margin_items > 0
            else None
        ),
    }


def compute_proxy_metrics(
    method_name: str,
    item_ids: List[int],
    cluster_ids: torch.Tensor,
    codebook_size: int,
    top_k: int = 5,
    embeddings: Optional[torch.Tensor] = None,
    category_labels: Optional[List[Optional[str]]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "method": method_name,
        "num_hierarchies": int(cluster_ids.size(1)),
        "codebook_size": codebook_size,
    }
    if extra_metadata:
        result.update(extra_metadata)

    result.update(compute_codebook_utilization(cluster_ids, codebook_size))
    result.update(compute_collision_metrics(cluster_ids))
    result.update(compute_distribution_health(cluster_ids, top_k=top_k))

    notes = list(result.get("notes", []))
    if category_labels is not None:
        result.update(compute_prefix_purity(cluster_ids, category_labels))
    elif embeddings is not None:
        result.update(compute_prefix_compactness(cluster_ids, embeddings))
        result.update(compute_sibling_separation(cluster_ids, embeddings))
        result.update(compute_pair_type_distance_ordering(cluster_ids, embeddings))
        result.update(compute_same_parent_leaf_metrics(cluster_ids, embeddings))
        notes.append("Category metadata unavailable; used embedding compactness fallback.")
    else:
        result["prefix_metric_type"] = "unavailable"
        result["prefix_compactness_per_layer"] = []
        result["sibling_separation_per_layer"] = []
        result["avg_sibling_separation"] = None
        result["near_collision_separation"] = None
        result["full_collision_distance_mean"] = None
        result["sibling_distance_mean"] = None
        result["mid_prefix_distance_mean"] = None
        result["unrelated_distance_mean"] = None
        result["distance_gap_full_minus_sibling"] = None
        result["distance_gap_unrelated_minus_sibling"] = None
        result["same_parent_leaf_uniqueness"] = None
        result["same_parent_collision_rate"] = None
        result["same_parent_assignment_margin"] = None
        result["same_parent_positive_margin_fraction"] = None
        notes.append(
            "Category metadata and embedding tensor were both unavailable; prefix purity/compactness not computed."
        )

    result["notes"] = notes
    result["item_id_min"] = min(item_ids) if item_ids else None
    result["item_id_max"] = max(item_ids) if item_ids else None
    return result


def format_summary(result: Dict[str, Any]) -> str:
    lines = [
        f"Method: {result['method']}",
        f"  num_items: {result['num_items']}",
        f"  avg_utilization: {result['avg_utilization']:.6f}",
        f"  collision_rate: {result['collision_rate']:.6f}",
        f"  frac_unique_ids: {result['frac_unique_ids']:.6f}",
        f"  utilization_per_layer: {json.dumps(result['utilization_per_layer'])}",
        f"  entropy_per_layer: {json.dumps(result['entropy_per_layer'])}",
        f"  max_token_ratio_per_layer: {json.dumps(result['max_token_ratio_per_layer'])}",
        f"  top{result['top_k']}_token_coverage_per_layer: {json.dumps(result['topk_token_coverage_per_layer'])}",
    ]

    prefix_metric_type = result.get("prefix_metric_type", "unavailable")
    if prefix_metric_type == "purity":
        lines.append(
            f"  prefix_purity_per_layer: {json.dumps(result['prefix_purity_per_layer'])}"
        )
    elif prefix_metric_type == "compactness":
        lines.append(
            f"  prefix_compactness_per_layer: {json.dumps(result['prefix_compactness_per_layer'])}"
        )
        lines.append(
            f"  sibling_separation_per_layer: {json.dumps(result.get('sibling_separation_per_layer', []))}"
        )
        lines.append(
            f"  near_collision_separation: {result.get('near_collision_separation', 0.0):.6f}"
        )
        lines.append(
            f"  same_parent_leaf_uniqueness: {result.get('same_parent_leaf_uniqueness', 0.0):.6f}"
        )
        lines.append(
            f"  same_parent_assignment_margin: {result.get('same_parent_assignment_margin', 0.0):.6f}"
        )
        lines.append(
            "  pair_type_distance_means: "
            f"{json.dumps({'full': result.get('full_collision_distance_mean'), 'sibling': result.get('sibling_distance_mean'), 'mid': result.get('mid_prefix_distance_mean'), 'unrelated': result.get('unrelated_distance_mean')}, ensure_ascii=True)}"
        )
    else:
        lines.append("  prefix_metric: unavailable")

    if result.get("notes"):
        lines.append(f"  notes: {json.dumps(result['notes'], ensure_ascii=True)}")
    return "\n".join(lines)


def result_to_flat_row(result: Dict[str, Any]) -> Dict[str, Any]:
    flat_row = {}
    for key, value in result.items():
        if isinstance(value, (list, dict)):
            flat_row[key] = json.dumps(value, ensure_ascii=True)
        else:
            flat_row[key] = value
    return flat_row
