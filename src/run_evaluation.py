from __future__ import annotations

import argparse

from utils import get_logger, load_config
from utils.utils import load_model_and_tokenizer, measure_perplexity, prepare_dataset

logger = get_logger("evaluation")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="benchmark")

    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        required=True,
        help="Specify the path to config file.",
    )
    return parser.parse_args()


def main() -> None:
    """RUn main logic of evaluation."""
    args = parse_arguments()
    config = load_config(args.config_path)

    model_config = config["model"]
    model, tokenizer = load_model_and_tokenizer(**model_config)

    logger.info("Preparing dataset.")
    text = prepare_dataset(split="test")

    logger.info("Calculating perplexity.")
    ppl = measure_perplexity(model, tokenizer, text, stride=512)

    print(f"\n--- Evaluation Result: {model_config['model_name']} ---")
    print(
        f"Quantization: "
        f"{model_config.get('quantization_config', {}).get('method', 'FP16')}"
    )
    print(f"Perplexity  : {ppl:.4f}")


if __name__ == "__main__":
    main()
