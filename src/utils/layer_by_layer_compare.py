from __future__ import annotations

from typing import Any

import torch


class TransformerBlockNotFoundError(Exception):
    """Custom exception fo Transformer block not found in the given model."""

    def __init__(self, model: torch.nn.Module) -> None:
        msg = (
            "Unsupported architecture cannot find 'layers: "
            f"arch = {[n for n, _ in model.named_children()]}"
        )
        super().__init__(msg)

def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
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
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim = -1).mean().item()

    return {"neg_mse": -mse, "cosine_similarity": cos_sim}

def _get_transformer_layers(
    model: Any,  # noqa: ANN401
) -> list[tuple[str, torch.nn.Module]]:
    # General composition (Llama, Mistral, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers.named_children())
    # Falcon, GPT-NeoX, etc.
    if hasattr(model, "layers"):
        return list(model.layers.named_children())
    # GPT-2, etc.
    if hasattr(model, "h"):
        return list(model.h.named_children())

    raise TransformerBlockNotFoundError(model)


def register_original_hooks(
    model: Any,  # noqa: ANN401
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
    layers = _get_transformer_layers(model)
    print(f"Original model layers: {layers}...")
    print(f"{len(layers)} layers detected.")

    for name, module in layers:

        def _hook_fn(
            _model: Any,  # noqa: ANN401
            _in: Any,  # noqa: ANN401
            out: torch.Tensor | tuple[torch.Tensor, ...],
            name: str = name,
        ) -> None:
            if isinstance(out, tuple):
                out = out[0]
                # Offload to CPU immediately to save VRAM
            org_activations[name] = out.detach().to("cpu")

        handle = module.register_forward_hook(_hook_fn)
        hooks.append(handle)
    return hooks


def register_target_hooks(
    model: torch.nn.Module,
    ref_activations: dict[str, torch.Tensor],
    results: dict[str, dict[str, float]],
) -> list[tuple[str, torch.nn.Module]]:
    """Regiser hooks to compare quantized activation with original.

    Args:
        model: pytorch model (quantized)
        ref_activations: dict containing original fp16 activation
        results: dict to store the comparison metrics

    Returns:
        list of hook handles

    """
    hooks: list[Any] = []
    layers = _get_transformer_layers(model)
    for name, module in layers:

        def _hook_fn(
            _model: Any,  # noqa: ANN401
            _in: Any,  # noqa: ANN401
            out: torch.Tensor | tuple[torch.Tensor, ...],
            name: str = name,
        ) -> None:
            if isinstance(out, tuple):
                out = out[0]
            current_act = out.data
            if name not in ref_activations:
                return
            ref_act = ref_activations[name].to(current_act.device)
            metrics = compute_metrics(current_act, ref_act)
            results[name] = metrics

        handle = module.register_forward_hook(_hook_fn)
        hooks.append(handle)
    return hooks


def clear_hooks(hooks: list[Any]) -> None:
    """Remove all registered hooks."""
    for handle in hooks:
        handle.remove()
    hooks.clear()
