from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any

import torch

from utils import get_logger, load_config
from utils.utils import load_model_and_tokenizer, save_results_to_csv

logger = get_logger("benchmark")

save_results = os.getenv("SAVE_BENCHMARK_RESULTS", None) is not None


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


def measure_metrics(
    prompt: str,
    tokenizer: Any,  # noqa: ANN401
    model: Any,  # noqa: ANN401
    *,
    device: str = "cuda",
    max_new_tokens: int = 50,
) -> dict[str, float]:
    """Measure metrics (prefill, decode) for the given model. Return in dictionary.

    Args:
        prompt (str): prompt text.
        tokenizer (Any): loaded tokenizer.
        model (Any): loaded model.
        device (str): device. Defualt to 'cuda'
        max_new_tokens (int): maximum num of new tokens.

    Returns:
        Dictionary of metrics (dict[str, Any])

    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    num_imput_tokens = input_ids.shape[1]  # Get squence length

    # Synchronize GPU
    logger.info("Synching for calculating prefill latency.")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time_p = time.perf_counter()

    # Get prefil + first token
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1)

    torch.cuda.synchronize()
    end_time_p = time.perf_counter()

    prefill_latency = end_time_p - start_time_p
    prefill_tps = num_imput_tokens / prefill_latency

    logger.info("Synching for calculating decode latency.")
    torch.cuda.synchronize()
    start_time_d = time.perf_counter()

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    torch.cuda.synchronize()
    end_time_d = time.perf_counter()

    total_latency = end_time_d - start_time_d

    decode_latency = total_latency - prefill_latency

    total_tokens = output.shape[1]
    generated_tokens = total_tokens - num_imput_tokens
    decode_tps = generated_tokens / decode_latency if decode_latency > 0 else 0.0

    max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "prefill_latency": prefill_latency,
        "prefill_tps": prefill_tps,
        "decode_latency": decode_latency,
        "decode_tps": decode_tps,
        "max_vram_gb": max_memory_gb,
    }


def main() -> None:
    """Run the main logic of benchmarking."""
    args = parse_arguments()
    config = load_config(args.config_path)

    model_config = config["model"]
    prompt = config["prompt"]["content"]

    benchmark_config = config["benchmark"]
    max_new_tokens = benchmark_config.get("max_new_tokens", 50)

    model, tokenizer = load_model_and_tokenizer(**model_config)

    warmup_steps = benchmark_config.get("warmup_steps", 2)
    num_trials = benchmark_config.get("num_trials", 3)

    logger.info(f"Starting warmup ({warmup_steps} steps).")
    for _ in range(warmup_steps):
        _ = measure_metrics(prompt, tokenizer, model, max_new_tokens=max_new_tokens)

    logger.info(f"Starting benchmark measurement ({num_trials} trials).")
    results = []
    for i in range(num_trials):
        metrics = measure_metrics(
            prompt, tokenizer, model, max_new_tokens=max_new_tokens
        )
        results.append(metrics)
        print(
            f"Trial {i + 1}: Prefill={metrics['prefill_tps']:.1f},"
            f"Decode={metrics['decode_tps']:.1f}"
        )

    # Calculate avg. metrics
    avg_prefill = statistics.mean([m["prefill_tps"] for m in results])
    avg_decode = statistics.mean([m["decode_tps"] for m in results])
    max_vram = max([m["max_vram_gb"] for m in results])

    if save_results:
        out_path = config["output"]["csv_path"]
        save_results_to_csv(model_config, metrics, out_path)

    print(f"--- Result ({num_trials} avg.): {model_config['model_name']} ---")
    print(f"Prefill Speed: {avg_prefill:.2f} tokens/sec")
    print(f"Decode Speed : {avg_decode:.2f} tokens/sec")
    print(f"Max VRAM     : {max_vram:.2f} GB")


if __name__ == "__main__":
    main()
