#!/usr/bin/env python3
"""Compute teacher-forced next-token CE and accuracy by depth for a trained TIGER run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Tuple

import rootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.custom_hydra_resolvers import *  # noqa: F401,F403,E402
from src.utils.launcher_utils import initialize_pipeline_modules  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute teacher-forced next-token CE/accuracy by depth."
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def resolve_checkpoint(train_dir: Path) -> Path:
    for candidate in (
        train_dir / "checkpoints" / "best.ckpt",
        train_dir / "checkpoints" / "last.ckpt",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {train_dir / 'checkpoints'}")


def load_semantic_id_tensor(cfg: DictConfig) -> torch.Tensor:
    semantic_id_path = Path(str(cfg.semantic_id_path))
    if not semantic_id_path.exists():
        raise FileNotFoundError(f"Missing semantic id tensor: {semantic_id_path}")
    semantic_id_tensor = torch.load(
        semantic_id_path, map_location="cpu", weights_only=False
    )
    if semantic_id_tensor.ndim != 2:
        raise ValueError(
            f"Expected semantic id tensor to have shape [N, D] or [D, N], got {tuple(semantic_id_tensor.shape)}"
        )

    # Downstream semantic-ID tensors are stored as [num_hierarchies, num_items].
    # Teacher-forced analysis needs item-major rows to recover same-parent sibling sets.
    if semantic_id_tensor.size(0) <= 16 and semantic_id_tensor.size(0) < semantic_id_tensor.size(1):
        semantic_id_tensor = semantic_id_tensor.t().contiguous()
    return semantic_id_tensor


def build_same_parent_leaf_map(
    semantic_id_tensor: torch.Tensor,
    leaf_idx: int,
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    parent_to_leafs: Dict[Tuple[int, ...], set[int]] = defaultdict(set)
    for row in semantic_id_tensor.tolist():
        if len(row) <= leaf_idx:
            continue
        parent_prefix = tuple(int(token) for token in row[:leaf_idx])
        leaf_token = int(row[leaf_idx])
        parent_to_leafs[parent_prefix].add(leaf_token)
    return {
        parent_prefix: tuple(sorted(leaf_tokens))
        for parent_prefix, leaf_tokens in parent_to_leafs.items()
    }


def prepare_cfg(cfg: DictConfig, device_name: str) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg):
        cfg.train = False
        cfg.test = False
        cfg.ckpt_path = None
        cfg.logger = None
        cfg.callbacks = None
        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.strategy = "auto"
        cfg.trainer.sync_batchnorm = False
        cfg.trainer.logger = False
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.enable_model_summary = False
        cfg.trainer.accelerator = "gpu" if device_name == "cuda" else "cpu"
        cfg.data_loading.val_dataloader_config.dataloader.num_workers = 0
        cfg.data_loading.test_dataloader_config.dataloader.num_workers = 0
        cfg.data_loading.val_dataloader_config.dataloader.persistent_workers = False
        cfg.data_loading.test_dataloader_config.dataloader.persistent_workers = False
        cfg.data_loading.val_dataloader_config.dataloader.timeout = 0
        cfg.data_loading.test_dataloader_config.dataloader.timeout = 0
    return cfg


def move_batch_to_device(
    batch: Tuple[Any, Any],
    device: torch.device,
) -> Tuple[Any, Any]:
    model_input = batch[0]
    label_data = batch[1]
    model_input_cls = type(model_input)
    label_data_cls = type(label_data)
    moved_model_input = model_input_cls(
        user_id_list=(
            model_input.user_id_list.to(device)
            if isinstance(model_input.user_id_list, torch.Tensor)
            else model_input.user_id_list
        ),
        transformed_sequences={
            key: value.to(device) for key, value in model_input.transformed_sequences.items()
        },
        mask=model_input.mask.to(device) if model_input.mask is not None else None,
    )
    moved_label_data = label_data_cls(
        labels={
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in label_data.labels.items()
        },
        label_location={
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in label_data.label_location.items()
        },
        attention_mask={
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in label_data.attention_mask.items()
        },
    )
    return moved_model_input, moved_label_data


def extract_future_ids(
    model_input: Any,
    label_data: Any,
) -> torch.Tensor:
    fut_ids = None
    for _, labels in label_data.labels.items():
        fut_ids = labels.reshape(model_input.mask.size(0), -1)
    if fut_ids is None:
        raise ValueError("Expected one label tensor for teacher-forced analysis.")
    return fut_ids


def compute_logits_by_depth(model: Any, model_input: Any, future_ids: torch.Tensor) -> list[torch.Tensor]:
    model_output = model.forward(
        attention_mask_encoder=model_input.mask,
        future_ids=future_ids,
        **{
            model.feature_to_model_input_map.get(key, key): value
            for key, value in model_input.transformed_sequences.items()
        },
    )
    model_output = model_output[:, :-1]
    return [
        model.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
        for hierarchy in range(int(model.num_hierarchies))
    ]


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    hydra_cfg_path = train_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {hydra_cfg_path}")

    raw_cfg = OmegaConf.load(hydra_cfg_path)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    cfg = prepare_cfg(raw_cfg, device_name)
    pipeline_modules = initialize_pipeline_modules(cfg)
    datamodule = pipeline_modules.datamodule
    model = pipeline_modules.model

    datamodule.trainer = SimpleNamespace(world_size=1, global_rank=0)
    datamodule.setup()

    checkpoint_path = resolve_checkpoint(train_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(device_name)
    model.to(device)
    model.eval()

    dataloader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()
    if isinstance(dataloader, (tuple, list)):
        dataloader = dataloader[0]
    num_hierarchies = int(model.num_hierarchies)
    last_idx = num_hierarchies - 1
    leaf_idx = num_hierarchies - 2 if num_hierarchies >= 2 else last_idx
    semantic_id_tensor = load_semantic_id_tensor(cfg)
    same_parent_leaf_map = build_same_parent_leaf_map(semantic_id_tensor, leaf_idx)
    ce_sums = torch.zeros(num_hierarchies, dtype=torch.float64)
    correct_sums = torch.zeros(num_hierarchies, dtype=torch.float64)
    counts = torch.zeros(num_hierarchies, dtype=torch.float64)
    rollout1_last_ce_sum = 0.0
    rollout1_last_correct_sum = 0.0
    rollout1_last_count = 0.0
    rollout2_last2_ce_sum = 0.0
    rollout2_last2_correct_sum = 0.0
    rollout2_last2_token_count = 0.0
    rollout2_last2_sequence_correct_sum = 0.0
    rollout2_last2_sequence_count = 0.0
    same_parent_hardest_sibling_margin_sum = 0.0
    same_parent_hardest_sibling_margin_count = 0.0
    same_parent_within_parent_correct_sum = 0.0
    same_parent_within_parent_count = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            model_input, label_data = move_batch_to_device(batch, device)
            fut_ids = extract_future_ids(model_input, label_data)
            logits_by_depth = compute_logits_by_depth(model, model_input, fut_ids)
            for hierarchy in range(num_hierarchies):
                logits = logits_by_depth[hierarchy]
                target = fut_ids[:, hierarchy].long()
                ce = F.cross_entropy(logits, target, reduction="none")
                pred = logits.argmax(dim=-1)
                ce_sums[hierarchy] += ce.sum().double().cpu()
                correct_sums[hierarchy] += (pred == target).sum().double().cpu()
                counts[hierarchy] += float(target.numel())

            if num_hierarchies >= 2:
                leaf_logits = logits_by_depth[leaf_idx]
                leaf_target = fut_ids[:, leaf_idx].long()
                leaf_pred = leaf_logits.argmax(dim=-1)
                leaf_ce = F.cross_entropy(leaf_logits, leaf_target, reduction="none")

                parent_prefixes = fut_ids[:, :leaf_idx].detach().cpu().tolist()
                for row_idx, parent_prefix in enumerate(parent_prefixes):
                    sibling_tokens = same_parent_leaf_map.get(
                        tuple(int(token) for token in parent_prefix),
                        (),
                    )
                    target_token = int(leaf_target[row_idx].item())
                    sibling_tokens = [token for token in sibling_tokens if token != target_token]
                    if not sibling_tokens:
                        continue
                    gt_logit = float(leaf_logits[row_idx, target_token].item())
                    hardest_sibling_logit = float(leaf_logits[row_idx, sibling_tokens].max().item())
                    same_parent_hardest_sibling_margin_sum += gt_logit - hardest_sibling_logit
                    same_parent_hardest_sibling_margin_count += 1.0
                    same_parent_within_parent_correct_sum += 1.0 if gt_logit > hardest_sibling_logit else 0.0
                    same_parent_within_parent_count += 1.0

                rollout_future_ids = fut_ids.clone()
                rollout_future_ids[:, leaf_idx] = leaf_pred
                rollout_logits_by_depth = compute_logits_by_depth(model, model_input, rollout_future_ids)
                last_logits_rollout = rollout_logits_by_depth[last_idx]
                last_target = fut_ids[:, last_idx].long()
                last_ce_rollout = F.cross_entropy(last_logits_rollout, last_target, reduction="none")
                last_pred_rollout = last_logits_rollout.argmax(dim=-1)
                rollout1_last_ce_sum += float(last_ce_rollout.sum().item())
                rollout1_last_correct_sum += float((last_pred_rollout == last_target).sum().item())
                rollout1_last_count += float(last_target.numel())

                rollout2_last2_ce_sum += float(leaf_ce.sum().item() + last_ce_rollout.sum().item())
                rollout2_last2_correct_sum += float(
                    (leaf_pred == leaf_target).sum().item()
                    + (last_pred_rollout == last_target).sum().item()
                )
                rollout2_last2_token_count += float(2 * last_target.numel())
                rollout2_last2_sequence_correct_sum += float(
                    ((leaf_pred == leaf_target) & (last_pred_rollout == last_target))
                    .sum()
                    .item()
                )
                rollout2_last2_sequence_count += float(last_target.numel())

    by_depth = []
    for hierarchy in range(num_hierarchies):
        count = float(counts[hierarchy].item())
        ce_value = float(ce_sums[hierarchy].item() / count) if count > 0 else None
        acc_value = float(correct_sums[hierarchy].item() / count) if count > 0 else None
        by_depth.append(
            {
                "depth": hierarchy + 1,
                "teacher_forced_ce": ce_value,
                "teacher_forced_accuracy": acc_value,
                "count": int(count),
            }
        )

    last = by_depth[-1]
    last_two = by_depth[-2:] if len(by_depth) >= 2 else by_depth
    leaf_depth_metrics = by_depth[leaf_idx]
    payload = {
        "train_dir": str(train_dir),
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "device": device_name,
        "num_hierarchies": num_hierarchies,
        "leaf_depth_index": leaf_idx + 1,
        "last_depth_index": last_idx + 1,
        "by_depth": by_depth,
        "teacher_forced_leaf_ce": leaf_depth_metrics["teacher_forced_ce"],
        "teacher_forced_leaf_acc": leaf_depth_metrics["teacher_forced_accuracy"],
        "teacher_forced_last_ce": last["teacher_forced_ce"],
        "teacher_forced_last_acc": last["teacher_forced_accuracy"],
        "teacher_forced_last2_mean_ce": sum(
            metric["teacher_forced_ce"] for metric in last_two if metric["teacher_forced_ce"] is not None
        )
        / len(last_two),
        "teacher_forced_last2_mean_acc": sum(
            metric["teacher_forced_accuracy"]
            for metric in last_two
            if metric["teacher_forced_accuracy"] is not None
        )
        / len(last_two),
        "same_parent_hardest_sibling_margin": (
            same_parent_hardest_sibling_margin_sum / same_parent_hardest_sibling_margin_count
            if same_parent_hardest_sibling_margin_count > 0
            else None
        ),
        "same_parent_within_parent_accuracy": (
            same_parent_within_parent_correct_sum / same_parent_within_parent_count
            if same_parent_within_parent_count > 0
            else None
        ),
        "rollout1_last_ce": (
            rollout1_last_ce_sum / rollout1_last_count if rollout1_last_count > 0 else None
        ),
        "rollout1_last_acc": (
            rollout1_last_correct_sum / rollout1_last_count
            if rollout1_last_count > 0
            else None
        ),
        "rollout2_last2_mean_ce": (
            rollout2_last2_ce_sum / rollout2_last2_token_count
            if rollout2_last2_token_count > 0
            else None
        ),
        "rollout2_last2_mean_acc": (
            rollout2_last2_correct_sum / rollout2_last2_token_count
            if rollout2_last2_token_count > 0
            else None
        ),
        "rollout2_last2_sequence_acc": (
            rollout2_last2_sequence_correct_sum / rollout2_last2_sequence_count
            if rollout2_last2_sequence_count > 0
            else None
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
