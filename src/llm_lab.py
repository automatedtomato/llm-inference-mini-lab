from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

from tools.layer_by_layer_analysis import create_lbl_analysis_report
from utils import get_logger, load_config
from utils.eval_utils import run_evaluation, save_results_to_csv
from utils.utils import cleanup_gpu, load_model_and_tokenizer, prepare_dataset

save_results = os.getenv("SAVE_EVAL_RESULTS", None) is not None

logger = get_logger("llm_lab")

LBL_INPUTS_LENGTH = 500  # For memory limitation


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
        "-l",
        "--input-length",
        type=int,
        default=2000,
        help=(
            "Specify input length for the model. "
            "OOM error occurs if inputs is too long. Default to 2000. "
        ),
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
        type=Path,
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
    prompt = prepare_dataset(length=args.input_length)
    results: list[tuple[dict, Any]] = []  # (qconfig, metrics)

    if args.float_eval:
        logger.info(f"Start float model evaluation: {args.arch}")
        try:
            float_model, tokenizer = load_model_and_tokenizer(
                args.arch, dtype=torch.float16, quantization_config=None
            )
            ret_float = run_evaluation(
                float_model, tokenizer, prompt, "float", args.input_length
            )
            results.append(({"method": "FP16"}, ret_float))

        except Exception as e:
            logger.error(f"Float eval failed: {e}")
            sys.exit(1)
        finally:
            del float_model
            del tokenizer
            cleanup_gpu()

    if args.quant_eval:
        logger.info(f"Start quantized model evaluation: {args.arch}")
        if not qconfig:
            logger.error("Quantization config is missing. Use --qconfig.")
            sys.exit(1)

        try:
            quant_model, tokenizer = load_model_and_tokenizer(
                args.arch, dtype=torch.float16, quantization_config=qconfig
            )
            ret_quant = run_evaluation(
                quant_model, tokenizer, prompt, "quantized", args.input_length
            )
            results.append((qconfig, ret_quant))

        except Exception as e:
            logger.error(f"Quantized eval failed: {e}")
            sys.exit(1)
        finally:
            del quant_model
            del tokenizer
            cleanup_gpu()

    if save_results and results:
        save_path = Path("src/results/") / "metrics.csv"
        for qc, met in results:
            save_results_to_csv(args.arch, qc, met, save_path)
        print(f"Results saved to {save_path}")

    if args.lbl_dump_dir:
        test_prompt = prepare_dataset(length=LBL_INPUTS_LENGTH)
        create_lbl_analysis_report(args.arch, qconfig, test_prompt, args.lbl_dump_dir)
        print(
            "Layer by layer analysis report saved to "
            f"{args.lbl_dump_dir}/lbl_analysis.md"
        )


if __name__ == "__main__":
    main()
