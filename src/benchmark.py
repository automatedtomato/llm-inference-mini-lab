from __future__ import annotations

import argparse
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import get_logger

logger = get_logger("benchmark")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help=("Specify hte model name. Default to 'meta-llama/Llama-3.2-3B-Instruct'."),
    )

    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Set the number of max_new_tokens."
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_name: str, *, dtype: torch.dtype = torch.float16
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
        "max_memory_gb": max_memory_gb,
    }


def main() -> None:
    """Run the main logic."""
    args = parse_arguments()
    model, tokenizer = load_model_and_tokenizer(args.arch)
    prompt = "How do you say 'how do you say in Dutch?' in Dutch?"
    metrics = measure_metrics(prompt, tokenizer, model, max_new_tokens=args.max_tokens)
    print(f"--- Result: {args.arch} ---")
    print(f"Prefill Speed: {metrics['prefill_tps']:.2f} tokens/sec")
    print(f"Decode Speed : {metrics['decode_tps']:.2f} tokens/sec")
    print(f"Max VRAM     : {metrics['max_memory_gb']:.2f} GB")

if __name__ == "__main__":
    main()
