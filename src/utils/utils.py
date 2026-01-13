from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from pathlib import Path


GPU_LIMIT = 4096


def load_model_and_tokenizer(  # noqa: D417
    model_name: str,
    dtype: torch.dtype = torch.float16,
    quantization_config: dict[str, Any] | None = None,
    **kwargs,  # noqa: ANN003, ARG001
) -> tuple[Any, Any]:
    """Load model and tokenizer from model name.

    Args:
        model_name (str): model name to be loaded.
        dtype (torch.dtype): dtype in which model is loaded.
        quantization_config (dict[str, Any]): configuration for quantization.

    Returns:
        Tuple of model and tokenizer.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = None
    if quantization_config:
        if quantization_config.get("method") == "bitsandbytes":
            quantization_config["bnb_4bit_compute_type"] = getattr(
                torch, str(quantization_config.get("bnb_4bit_compute_dtype"))
            )
            print(f"Configuring bitsandbytes quantization. {quantization_config}")
            bnb_config = BitsAndBytesConfig(
                **quantization_config,
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise NotImplementedError

    print(f"Model: {model_name}. Quantized={bnb_config is not None}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, quantization_config=bnb_config, device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def save_results_to_csv(
    model_config: dict[str, Any], metrics: dict[str, Any], out_path: Path
) -> None:
    """Save benchmark results to a CSV file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.exists(out_path)

    quant_config = model_config.get("quantization_config", {})

    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_config["model_name"],
        "dtype": str(model_config.get("dtype", "unknown")),
        "quant_method": quant_config.get("method", "None"),
        "load_8bit": quant_config.get("load_in_8bit", False),
        "load_4bit": quant_config.get("load_in_4bit", False),
        "quant_type": quant_config.get("bnb_4bit_quant_type", "n/a"),
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


def prepare_dataset(split: str = "test", length: int = -1) -> str:
    """Load WikiText-2 dataset and concat into unified string."""
    print("Loading WikiText-2 dataset.")
    dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split=split)
    # Concat each text with two line breaks so to evaluate while persist log context
    text = "\n\n".join(dataset["text"])

    if length > 0:
        text = text[:length]

    print(f"{len(text)=}")

    return text


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


def measure_perplexity(
    model: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    text: str,
    stride: int = 512,
    *,
    device: str = "cuda",
) -> float:
    """Caluculate perplexity.

    Args:
        model: Any
        tokenizer: Any
        text (str): text to be used to calculate perplexity
        stride: window stride
        device: loading device
    Returns:
        perplexity score (float)

    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_all = encodings.input_ids

    model_lim = getattr(model.config, "max_position_embeddings", GPU_LIMIT)
    gpu_lim = GPU_LIMIT
    max_length = min(model_lim, gpu_lim)

    ppl = _compute_perplexity(model, input_ids_all, max_length, stride, device=device)
    print(f"Perplexity: {ppl:.2f}")
    return ppl
