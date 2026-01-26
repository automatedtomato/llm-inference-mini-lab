from __future__ import annotations

import csv
import os
import statistics
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from . import get_logger

if TYPE_CHECKING:
    from pathlib import Path

GPU_LIMIT = 4096
WARMUP_STEPS = 2
NUM_TRIALS = 3
MAX_NEW_TOKENS = 50


def _compute_metrics(
    model: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    prompt: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 50,
) -> dict[str, Any]:
    """Measure metrics (prefill, decode) for the given model. Return in dictionary.

    Args:
        prompt (str): prompt text.
        tokenizer (Any): loaded tokenizer.
        model (Any): loaded model.
        device (str): device. Defualt to 'cuda'
        max_new_tokens (int): maximum num of new tokens.

    Returns:
        dict of metrics (tuple[float, ...])

    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids_all = inputs.input_ids
    num_imput_tokens = input_ids_all.shape[1]  # Get squence length

    # Synchronize GPU
    print("Synching for calculating prefill latency.")
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

    print("Synching for calculating decode latency.")
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

    max_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "max_vram_gb": max_vram_gb,
    }


def _compute_perplexity(
    model: Any,  # noqa: ANN401
    input_ids_all: torch.Tensor,
    max_length: int,
    stride: int,
    *,
    device: str = "cuda",
) -> float:
    """Inner logic of calculating perplexity."""
    nlls = []  # Negative Log Likelihoods
    seq_len = input_ids_all.size(1)
    prev_end_loc = 0

    pbar = tqdm(range(0, seq_len, stride), desc="Perplexity Evaluation")

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = input_ids_all[:, begin_loc:end_loc].to(device)

        target_ids = input_ids.clone()

        if trg_len < input_ids.size(1):
            target_ids[:, :-trg_len] = -100

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / prev_end_loc)

    return float(ppl)


def run_evaluation(
    model: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    prompt: str,
    mode: str,
    prompt_length: int,
    stride: int = 512,
    device: str = "cuda",
) -> tuple[float, ...]:
    """Calculate metrics."""
    logger = get_logger("metrics")

    logger.info(f"Starting warmup ({WARMUP_STEPS} steps).")
    for _ in range(WARMUP_STEPS):
        _ = _compute_metrics(
            model, tokenizer, prompt, device=device, max_new_tokens=MAX_NEW_TOKENS
        )

    logger.info(f"Starting metrics measurement ({NUM_TRIALS} trials).")
    results = []
    for _ in range(NUM_TRIALS):
        metrics = _compute_metrics(
            model, tokenizer, prompt, device=device, max_new_tokens=MAX_NEW_TOKENS
        )
        results.append(metrics)

    logger.info("Start perplexity measurement.")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_all = inputs.input_ids

    model_lim = getattr(model.config, "max_position_embeddings", GPU_LIMIT)
    gpu_lim = GPU_LIMIT
    max_length = min(model_lim, gpu_lim)
    ppl = _compute_perplexity(model, input_ids_all, max_length, stride, device=device)

    # Calculate avg. metrics
    avg_prefill_tps = statistics.mean(m["prefill_tps"] for m in results)
    avg_decode_tps = statistics.mean([m["decode_tps"] for m in results])
    max_vram = max([m["max_vram_gb"] for m in results])
    print("\n")
    print(f"--- Eval. Result ({mode} model - {NUM_TRIALS} avg. - {prompt_length=}) ---")
    print(f"Avg. Prefill Speed: {avg_prefill_tps:.4f} tokens/sec")
    print(f"Avg. Decode Speed : {avg_decode_tps:.4f} tokens/sec")
    print(f"Max VRAM          : {max_vram:.4f} GB")
    print(f"Perplexity.       : {ppl:.4f}\n")

    return avg_prefill_tps, avg_decode_tps, max_vram, ppl, prompt_length


def save_results_to_csv(
    model_name: str,
    qconfig: dict[str, Any],
    metrics: tuple[float, ...],
    save_path: Path,
) -> None:
    """Save benchmark results to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.exists(save_path)

    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "quant_method": qconfig.get("method", "FP16"),
        "load_8bit": qconfig.get("load_in_8bit", False),
        "load_4bit": qconfig.get("load_in_4bit", False),
        "quant_type": qconfig.get("bnb_4bit_quant_type", "n/a"),
        "prompt_length": metrics[4],
        "avg_prefill_tps": f"{metrics[0]:.6f}",
        "avg_decode_tps": f"{metrics[1]:.6f}",
        "max_vram_gb": f"{metrics[2]:.6f}",
        "perplexity": f"{metrics[3]:.6f}",
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
    }

    fieldnames = list(row.keys())

    with open(save_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # If newly created, write header
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
