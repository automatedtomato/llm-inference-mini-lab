from __future__ import annotations

import argparse
import csv
import gc
import os
import sys

import torch

from pathlib import Path
from typing import Any

from utils import get_logger, load_config
from utils.tools.layer_by_layer_compare import clear_hooks, register_original_hooks, register_target_hooks
from utils.utils import load_model_and_tokenizer, prepare_dataset

logger = get_logger("layer_by_layer")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Layer-wise Sensitivity Analysis")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "-o",
        "--out-dump-dir",
        type=str,
        default="src/results/layer_by_layer_results.csv",
        help="Specify output CSV directory",
    )
    return parser.parse_args()

def _cleanup_gpu() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def save_results_to_csv(results: dict[str, dict[str, float]], out_path: Path) -> None:
    """Save layer-by-layer analysis to csv."""
    rows: list[dict[str, Any]] = []
    for layer_name, metrics in results.items():
        row = {
            "layer": layer_name,
            "neg_mse": metrics["neg_mse"],
            "cosine_similarity": metrics["cosine_similarity"],
        }
        rows.append(row)

    try:
        # sort by layer index
        rows.sort(key=lambda x: int(str(x["layer"])))
    except ValueError:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["layer", "neg_mse", "cosine_similarity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    """Run tow-pass layer-by-layer analysis."""
    args = parse_arguments()

    config = load_config(args.config_path)
    model_config = config["model"]
    model_name = model_config["model_name"]

    test_text = prepare_dataset(length=1000)

    org_activations: dict[str, torch.Tensor] = {}
    results: dict[str, dict[str, float]] = {}

    logger.info("Loading original model (fp16).")
    try:
        model_org, tokenizer = load_model_and_tokenizer(
            model_name, torch.float16, quantization_config=None,
        )
    except Exception as e:
        logger.error(f"Failed to load original model: {e}")
        sys.exit(1)
    example_inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

    logger.info("Registering original hooks.")
    hooks = register_original_hooks(model_org, org_activations)

    with torch.no_grad():
        model_org(**example_inputs)
    logger.info(f"Captured activations from {len(org_activations)} layers.")

    clear_hooks(hooks)
    del model_org
    del tokenizer
    _cleanup_gpu()

    q_bit = "int4" if model_config["quantization_config"]["load_in_4bit"] else "int8"
    logger.info(f"Loading quantized model ({q_bit}).")

    try:
        model_quant, _ = load_model_and_tokenizer(**model_config)
    except Exception as e:
        logger.error(f"Failed to load quaintized model: {e}")
        sys.exit(1)

    logger.info("Registering quantized hooks")
    hooks = register_target_hooks(model_quant, org_activations, results)

    with torch.no_grad():
        model_quant(**example_inputs)

    clear_hooks(hooks)

    logger.info(f"Analysis completed. Saving results to {args.out_dump_dir}")
    save_results_to_csv(results, args.out_dump_dir)

if __name__ == "__main__":
    main()
