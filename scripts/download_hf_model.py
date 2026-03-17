#!/usr/bin/env python3
"""Download a Hugging Face model through a China mirror for GRID."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "google/flan-t5-xl"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "models" / "google" / "flan-t5-xl"
    default_cache_dir = repo_root / ".cache" / "huggingface"

    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model for GRID via a mirror."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model repo id or local path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help=f"Local directory to store the downloaded model. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(default_cache_dir),
        help=f"Hugging Face cache directory. Default: {default_cache_dir}",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Hugging Face mirror endpoint. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Falls back to HF_TOKEN if set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent download workers. Lower this if the mirror is unstable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "hub")

    print(f"Downloading model: {args.model}")
    print(f"Mirror endpoint: {args.endpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")

    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_dir),
        cache_dir=str(cache_dir),
        endpoint=args.endpoint,
        token=args.token,
        resume_download=True,
        max_workers=args.max_workers,
    )

    print("\nDownload finished.")
    print("Use this local model path with GRID:")
    print(
        "python -m src.inference "
        "experiment=sem_embeds_inference_flat "
        "data_dir=/data/user/cwu319/RC/hyper/data/amazon_data/beauty "
        f"embedding_model={output_dir}"
    )


if __name__ == "__main__":
    main()
