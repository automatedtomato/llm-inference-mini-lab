from __future__ import annotations

import argparse
import os

from utils import get_logger

save_results = os.getenv("SAVE_EVAL_RESULTS", None) is not None

logger = get_logger("llm_lab")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM mini lab")
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        required=True,
        help="ath to config file.",
    )
    parser.add_argument(
        "--only-float",
        action="store_true",
        help="If specified, only float model will be evaluated.",
    )
    parser.add_argument(
        "--only-quant",
        action="store_true",
        help="If specified, only quantized model will be evaluated.",
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
