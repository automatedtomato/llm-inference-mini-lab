from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

from utils import get_logger, load_config
from utils.eval_utils import NUM_TRIALS, run_evaluation, save_results_to_csv
from utils.utils import cleanup_gpu, load_model_and_tokenizer, prepare_dataset

save_results = os.getenv("SAVE_EVAL_RESULTS", None) is not None

logger = get_logger("llm_lab")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM mini lab")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Specify model architecture. Default to Llama-3.2-3B.",
    )
    parser.add_argument(
        "--qconfig",
        type=str,
        help="Path to quantization config file.",
    )
    parser.add_argument(
        "--float-eval",
        action="store_true",
        help="If specified, float model will be evaluated.",
    )
    parser.add_argument(
        "--quant-eval",
        action="store_true",
        help="If specified, quantized model will be evaluated.",
    )
    parser.add_argument(
        "--lbl-dump-dir",
        type=str,
        help=(
            "If specified, layer-by-layer analysis tool is enabled. "
            "Analysis report will be saved to the specified dir."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run laboratory."""
    args = parse_arguments()

    qconfig = load_config(args.qconfig)

    prompt = prepare_dataset(length=1000)
    results: list[tuple[dict, Any]] = []  # (qconfig, metrics)

    if args.float_eval:
        logger.info("Start float model evaluation.")
        try:
            float_model, tokenizer = load_model_and_tokenizer(
                args.arch, dtype=torch.float16, quantization_config=None
            )
            ret_float = run_evaluation(float_model, tokenizer, prompt)
            print(f"--- Float Model Eval. Result ({NUM_TRIALS} avg.): {args.arch} ---")
            print(f"Avg. Prefill Speed: {ret_float[0]:.4f} tokens/sec")
            print(f"Avg. Decode Speed : {ret_float[1]:.4f} tokens/sec")
            print(f"Max VRAM          : {ret_float[2]:.4f} GB")
            print(f"Perplexity.       : {ret_float[3]:.4f}\n")
            results.append(({"method": "FP16"}, ret_float))

            del float_model
            del tokenizer
            cleanup_gpu()

        except Exception as e:
            logger.error(f"Float eval failed: {e}")

    if args.quant_eval:
        logger.info("Start quantized model evaluation.")
        if not qconfig:
            logger.error("Quantization config is missing. Use --qconfig.")
            sys.exit(1)

        try:
            quant_model, tokenizer = load_model_and_tokenizer(
                args.arch, dtype=torch.float16, **qconfig
            )
            ret_quant = run_evaluation(quant_model, tokenizer, prompt)
            print(
                f"--- Quantized Model Eval. Result ({NUM_TRIALS} avg.): {args.arch} ---"
            )
            print(f"Avg. Prefill Speed: {ret_quant[0]:.4f} tokens/sec")
            print(f"Avg. Decode Speed : {ret_quant[1]:.4f} tokens/sec")
            print(f"Max VRAM          : {ret_quant[2]:.4f} GB")
            print(f"Perplexity.       : {ret_quant[3]:.4f}\n")
            results.append((qconfig, ret_quant))

            del quant_model
            del tokenizer
            cleanup_gpu()

        except Exception as e:
            logger.error(f"Quantized eval failed: {e}")

    if save_results and results:
        save_path = Path("src/results/") / "metrics.csv"
        for qc, met in results:
            save_results_to_csv(args.arch, qc, met, save_path)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
