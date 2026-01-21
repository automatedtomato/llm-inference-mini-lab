from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import get_logger
from utils.utils import cleanup_gpu, load_model_and_tokenizer

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger("lbl-tool")


def _compute_mse_and_cos_sim(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Compute MSE and cosine similarity between two tensors.

    Args:
        a (torch.Tensor): first tensor (e.g. quantized output)
        b (torch.Tensor): second tensor (e.g. original output)

    Returns:
        dictionary: MSE and cosine similarity

    """
    if a.device != b.device:
        b = b.to(a.device)
    if a.dtype != b.dtype:
        b = b.to(a.dtype)

    mse = torch.nn.functional.mse_loss(a, b).item()

    if a.dim() > 1:
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    else:
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0), dim=-1
        ).item()

    return {"neg_mse": -mse, "cosine_similarity": cos_sim}


def _get_leaf_module(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Recursively find leaf modules."""
    target_modules = []
    ignored_containers = (torch.nn.ModuleList, torch.nn.Sequential)

    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) == 0:
            target_modules.append((name, module))
        elif isinstance(module, ignored_containers):
            continue

    return target_modules


def _register_original_hooks(
    model: torch.nn.Module,
    org_activations: dict[str, torch.Tensor],
) -> list[Any]:
    """Regster hooks for FP16 output.

    Args:
        model: pytorch model (fp16/original)
        org_activations: dict to store captured activations

    Returns:
        list of hook handles

    """
    hooks: list[Any] = []
    modules = _get_leaf_module(model)
    print(f"Total {len(modules)} modules detected.")

    for name, module in modules:

        def _hook_fn(
            _model: Any,  # noqa: ANN401
            _in: Any,  # noqa: ANN401
            out: torch.Tensor | tuple[torch.Tensor, ...],
            name: str = name,
        ) -> None:
            if isinstance(out, tuple):
                out = out[0]
                # Offload to CPU immediately to save VRAM
            org_activations[name] = out.detach().to("cpu").float()

        handle = module.register_forward_hook(_hook_fn)
        hooks.append(handle)
    return hooks


def _register_target_hooks_and_compare(
    model: torch.nn.Module,
    ref_activations: dict[str, torch.Tensor],
    metrics_results: dict[str, dict[str, float]],
    compare_results: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> list[tuple[str, torch.nn.Module]]:
    """Regiser hooks to compare quantized activation with original.

    Args:
        model: pytorch model (quantized)
        ref_activations: dict containing original fp16 activation
        metrics_results: dict to store the comparison metrics
        compare_results: data for scatter plot

    Returns:
        list of hook handles

    """
    hooks: list[Any] = []
    modules = _get_leaf_module(model)
    for name, module in modules:

        def _hook_fn(
            _model: Any,  # noqa: ANN401
            _in: Any,  # noqa: ANN401
            out: torch.Tensor | tuple[torch.Tensor, ...],
            name: str = name,
        ) -> None:
            if isinstance(out, tuple):
                out = out[0]
            current_act = out.detach()

            if name not in ref_activations:
                return

            ref_act = ref_activations[name].to(current_act.device)
            metrics = _compute_mse_and_cos_sim(current_act, ref_act)
            metrics_results[name] = metrics
            if compare_results is not None:
                # Downsample to avoid massive RAM usage
                ref_flat = ref_act.cpu().float().numpy().flatten()
                cur_flat = current_act.cpu().float().numpy().flatten()
                lim = 2000
                if len(ref_flat) > lim:
                    indices = np.random.choice(len(ref_flat), lim, replace=False)
                    ref_flat = ref_flat[indices]
                    cur_flat = cur_flat[indices]
                compare_results[name] = (ref_flat, cur_flat)

        handle = module.register_forward_hook(_hook_fn)
        hooks.append(handle)
    return hooks


def _clear_hooks(hooks: list[Any]) -> None:
    """Remove all registered hooks."""
    for handle in hooks:
        handle.remove()
    hooks.clear()


def _plot_comparison_results(
    metrics_results: dict[str, dict[str, float]],
    compare_results: dict[str, tuple[np.ndarray, np.ndarray]],
    fig_dir: Path,
) -> None:
    name_list: list[str] = []
    # plot scatter for each node
    for i, (name, metrics) in enumerate(compare_results.items()):
        name_list.append("_".join([str(i), name]))
        org_data, quant_data = metrics
        plt.figure()
        plt.title(name_list[i])
        plt.xlabel("float")
        plt.ylabel("quantized")
        plt.axline((0, 0), slope=1.0)
        plt.plot(org_data, quant_data, ".")
        plt.savefig(fig_dir / (name_list[i] + ".png"), bbox_inches="tight")
        plt.close()
        plt.clf()

    # plot metrics shift
    mses = [r["neg_mse"] for r in metrics_results.values()]
    sims = [r["cosine_similarity"] for r in metrics_results.values()]
    x = range(len(name_list))

    max_name_len = max(len(name) for name in name_list)

    w_per_data = 0.5
    fig_width = len(mses) * w_per_data

    h_per_char = 0.6
    fig_hight = 6.0 + (max_name_len * h_per_char)

    _, ax1 = plt.subplots(figsize=(fig_width, fig_hight))
    ax1.set_xlabel("op index")
    ax1.set_ylabel("neg_mse", color="red")
    ax1.plot(x, mses, color="red", alpha=0.5, label="negative MSE")
    ax1.tick_params(axis="y", labelcolor="red")

    ax1.set_xticks(x)
    ax1.set_xticklabels(name_list, rotation=90, fontsize=8)
    ax1.set_xlim(-0.5, len(name_list) - 0.5)
    ax1.grid(axis="x", linestyle=":", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("cosine_sim", color="blue")
    ax2.plot(x, sims, color="blue", alpha=0.5, label="Cosine Simimilarity")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title("Neg. MSE and Cosine Similarity Changes")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "summary.png")
    plt.close()
    plt.clf()


def create_lbl_analysis_report(  # noqa: PLR0915
    model_arch: str,
    qconfig: dict[str, Any],
    test_prompt: str,
    out_dir: Path,
) -> None:
    """Create layer-by-layer analysis report."""
    activations: dict[str, torch.Tensor] = {}
    metrics_results: dict[str, dict[str, float]] = {}
    compare_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    logger.info("Loading float model for layer-by-layer analysis")
    try:
        model_float, tokenizer = load_model_and_tokenizer(
            model_arch,
            dtype=torch.float16,
            quantization_config=None,
        )
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        hooks = _register_original_hooks(model_float, activations)

        model_float.eval()
        with torch.no_grad():
            model_float(**inputs)

        _clear_hooks(hooks)

    except Exception as e:
        logger.error(f"Failed to get hooks from float model: {e}")
        sys.exit(1)
    finally:
        del model_float
        del tokenizer
        cleanup_gpu()

    logger.info("Loading quantized model for layer-by-layer analysis")
    try:
        model_quant, tokenizer = load_model_and_tokenizer(
            model_arch,
            dtype=torch.float16,
            quantization_config=qconfig,
        )
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        hooks = _register_target_hooks_and_compare(
            model_quant,
            activations,
            metrics_results,
            compare_results,
        )

        model_quant.eval()
        with torch.no_grad():
            model_quant(**inputs)

        _clear_hooks(hooks)

    except Exception as e:
        logger.error(f"Failed to get hooks from quant model: {e}")
    finally:
        del model_quant
        del tokenizer
        cleanup_gpu()

    if not metrics_results:
        logger.error("No metrics collected. Skipping report generation.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "images"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_comparison_results(metrics_results, compare_results, fig_dir)

    md_path = out_dir / "lbl_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Layer-By-Layer MSE and Cosine Similarity Analysis\n")
        f.write("## Global Changes\n")
        f.write("![Summary](images/summary.png)\n\n")
        f.write("## Layer-By-Layer Sensitivity\n")
        for i, (name, met) in enumerate(metrics_results.items()):
            mse, sim = met["neg_mse"], met["cosine_similarity"]
            f.write(f"### {i} {name}\n")
            f.write(f"![{i}_{name}](images/{i}_{name}.png)\n")
            f.write(f"- `Neg. MSE: {mse}`\n")
            f.write(f"- `Cosine Similarity: {sim}`\n")
