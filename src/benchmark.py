from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_logger, load_config

if TYPE_CHECKING:
    from pathlib import Path

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


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    **kwargs,  # noqa: ANN003, ARG001
) -> tuple[Any, Any]:
    """Load model and tokenizer from model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loading model: {model_name}.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def measure_metrics(
    prompt: str,
    tokenizer: Any,  # noqa: ANN401
    model: Any,  # noqa: ANN401
    *,
    device: str = "cuda",
    max_new_tokens: int = 50,
) -> dict[str, float]:
    """Measure metrics (prefill, decode) for the given model. Return in dictionary."""
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


def save_results_to_csv(
    model_config: dict[str, Any], metrics: dict[str, Any], out_path: Path
) -> None:
    """Save benchmark results to a CSV file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.exists(out_path)

    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_config["model_name"],
        "dtype": str(model_config.get("dtype", "unknown")),
        "prefill_tps": f"{metrics['prefill_tps']:f}",
        "decode_tps": f"{metrics['decode_tps']:f}",
        "max_vram_gb": f"{metrics['max_vram_gb']:f}",
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
    }

    fieldnames = list(row.keys())

    with open(out_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # If newly created, write header
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Results saved to {out_path}.")


def main() -> None:
    """Run the main logic."""
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

    logger.info(f"Starting benchmark measurement ({num_trials} trials)")
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

    print(f"--- Result ({num_trials} avg.): {model_config['model_name']} ---")
    print(f"Prefill Speed: {avg_prefill:.2f} tokens/sec")
    print(f"Decode Speed : {avg_decode:.2f} tokens/sec")
    print(f"Max VRAM     : {max_vram:.2f} GB")

    if save_results:
        out_path = config["output"]["csv_path"]
        save_results_to_csv(model_config, metrics, out_path)


if __name__ == "__main__":
    main()
