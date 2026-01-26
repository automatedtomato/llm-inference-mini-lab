from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal

from src.quantizer.passes import quantize_linear
from src.tools.layer_by_layer_analysis import create_lbl_analysis_report
from src.utils import get_logger, load_config
from src.utils.eval_utils import run_evaluation, save_results_to_csv
from src.utils.lab_patch import apply_patch
from src.utils.utils import cleanup_gpu, load_model_and_tokenizer, prepare_dataset

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
        "--bnb-quant-eval",
        action="store_true",
        help="If specified, model quantized by BitsAndBytes will be evaluated.",
    )
    parser.add_argument(
        "--custom-quant-eval",
        action="store_true",
        help="If specified, model quantized by custom naive int8 will be evaluated.",
    )
    parser.add_argument(
        "--lbl-dump-dir",
        type=Path,
        help=(
            "If specified, layer-by-layer analysis tool is enabled. "
            "Analysis report will be saved to the specified dir."
        ),
    )
    parser.add_argument(
        "--mixed-prec",
        action="store_true",
        help="If specified, replace layers identified in confif with float16",
    )
    return parser.parse_args()


def run_benchmark(
    model_arch: str,
    q_config: dict[str, Any] | None,
    prompt: str,
    mode: Literal["float", "bnb_quant", "custom_quant"],
    input_legth: int,
    mixed_prec: bool,  # noqa: FBT001
) -> tuple[dict[str, Any], Any] | None:
    """Run benchmark task.

    Args:
        model_arch (str): model architecture
        q_config (dict): loaded quantization config
        prompt (str): evaluation prompt
        mode (Literal): evaluation mode ("float", "bnb_quant", "custom_quant")
        input_legth (int): input length for evaluation
        mixed_prec (bool): if True, apply mixed precision patch for BNB quant


    Returns:
        tuple[dict, Any] | None: (result_confg, metrics) or None if failed.

    """
    cur_config = q_config.copy() if q_config else {}
    load_config = None

    if mode == "bnb_quant":
        if not cur_config:
            logger.error(
                "Quantization config for BNB not found. Use `--qconfig` option."
            )
            return None
        load_config = cur_config

    model = None
    tokenizer = None

    try:
        model, tokenizer = load_model_and_tokenizer(
            model_arch, quantization_config=load_config
        )

        if mode == "bnb_quant" and mixed_prec:
            fp16_layers = cur_config.get("fp16_layers", [])
            if fp16_layers:
                model = apply_patch(model, model_arch, fp16_layers)
                cur_config["method"] = f"MixedPrecision(n={len(fp16_layers)})"
            else:
                logger.warning("Mixed precision requested but no layers declared.")

        elif mode == "custom_quant":
            model = quantize_linear(model)
            cur_config = {"method": "CustomInt8"}

        elif mode == "float":
            cur_config = {"method": "FP16"}

        metrics = run_evaluation(model, tokenizer, prompt, mode, input_legth)
        return (cur_config, metrics)  # noqa: TRY300

    except Exception as e:
        logger.error(f"{mode} eval failed: {e}")
        return None

    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        cleanup_gpu()


def main() -> None:
    """Run laboratory."""
    args = parse_arguments()

    qconfig = load_config(args.qconfig)
    prompt = prepare_dataset(length=args.input_length)
    results: list[tuple[dict, Any]] = []  # (qconfig, metrics)

    if args.float_eval:
        res = run_benchmark(
            args.arch,
            qconfig,
            prompt,
            mode="float",
            input_legth=args.input_length,
            mixed_prec=False,
        )
        if res:
            results.append(res)

    if args.bnb_quant_eval:
        res = run_benchmark(
            args.arch,
            qconfig,
            prompt,
            mode="bnb_quant",
            input_legth=args.input_length,
            mixed_prec=args.mixed_prec,
        )
        if res:
            results.append(res)

    if args.custom_quant_eval:
        res = run_benchmark(
            args.arch,
            qconfig,
            prompt,
            mode="custom_quant",
            input_legth=args.input_length,
            mixed_prec=False,
        )
        if res:
            results.append(res)

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
