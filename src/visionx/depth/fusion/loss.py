"""Pérdidas y métrica de Fusion (T12), sobre mapas en d/W_ref.

Todas reciben `mask` (bool, N x 1 x H x W): GT válida Y base válida. Fuera de la máscara el GT
puede traer cualquier valor (el dataset pone 0, pero acá se toleran hasta NaN o Inf): se lo
reemplaza con `torch.where` antes de operar, así no entra NaN ni gradiente desde afuera.

Las cuentas se hacen en **píxeles de referencia** (multiplicando por W_ref) para que los
valores sean comparables entre frames de distinto ancho y legibles (EPE en px).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from visionx.shared.errors import InvalidInput


def _width(ref_width: float | Tensor, like: Tensor) -> Tensor:
    """W_ref como tensor broadcasteable contra N x 1 x H x W (escalar o uno por muestra)."""
    w = torch.as_tensor(ref_width, dtype=like.dtype, device=like.device)
    if w.ndim == 0:
        return w
    if w.ndim == 1 and w.shape[0] == like.shape[0]:
        return w.reshape(-1, 1, 1, 1)
    raise InvalidInput(f"ref_width debe ser escalar o de largo N={like.shape[0]}, vino {w.shape}")


def _check(disparity: Tensor, gt: Tensor, mask: Tensor) -> None:
    if disparity.shape != gt.shape or disparity.shape != mask.shape:
        raise InvalidInput(
            f"disparity, gt y mask deben tener la misma forma: "
            f"{tuple(disparity.shape)}, {tuple(gt.shape)}, {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise InvalidInput(f"mask debe ser bool, vino {mask.dtype}")


def _masked_mean(values: Tensor, mask: Tensor, graph_anchor: Tensor) -> Tensor:
    """Media sobre la máscara; vacía -> 0 con grafo válido (no NaN)."""
    count = mask.sum()
    if int(count) == 0:
        return graph_anchor.sum() * 0.0
    return torch.where(mask, values, torch.zeros_like(values)).sum() / count


def laplace_nll(
    disparity: Tensor, b: Tensor, gt: Tensor, mask: Tensor, ref_width: float | Tensor
) -> Tensor:
    """NLL de Laplace en px de referencia: |d-gt|/b + log(2b), media sobre la máscara."""
    _check(disparity, gt, mask)
    if b.shape != disparity.shape:
        raise InvalidInput("b debe tener la forma de disparity")
    w = _width(ref_width, disparity)
    # Valores seguros fuera de la máscara: ni NaN en la cuenta ni gradiente hacia afuera.
    d = torch.where(mask, disparity, torch.zeros_like(disparity))
    g = torch.where(mask, gt, torch.zeros_like(gt))
    b_safe = torch.where(mask, b, torch.ones_like(b))
    err_px = (d - g).abs() * w
    b_px = b_safe * w
    nll = err_px / b_px + torch.log(2.0 * b_px)
    return _masked_mean(nll, mask, d + b_safe)


def l1_loss(disparity: Tensor, gt: Tensor, mask: Tensor, ref_width: float | Tensor) -> Tensor:
    """Error absoluto medio en px de referencia (ablación sin incertidumbre)."""
    _check(disparity, gt, mask)
    w = _width(ref_width, disparity)
    d = torch.where(mask, disparity, torch.zeros_like(disparity))
    g = torch.where(mask, gt, torch.zeros_like(gt))
    return _masked_mean((d - g).abs() * w, mask, d)


def combined_loss(
    out: dict[str, Tensor],
    gt: Tensor,
    mask: Tensor,
    ref_width: float | Tensor,
    kind: Literal["nll", "l1"] = "nll",
    l1_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pérdida de entrenamiento y sus partes (detached, para registrar)."""
    if kind not in ("nll", "l1"):
        raise InvalidInput(f"kind debe ser 'nll' o 'l1', vino {kind!r}")
    if l1_weight < 0:
        raise InvalidInput(f"l1_weight debe ser >= 0, vino {l1_weight}")
    disparity = out["disparity"]
    parts: dict[str, Tensor] = {}
    loss = disparity.sum() * 0.0
    if kind == "nll":
        nll = laplace_nll(disparity, out["b"], gt, mask, ref_width)
        parts["nll"] = nll.detach()
        loss = loss + nll
    if kind == "l1" or l1_weight > 0:
        l1 = l1_loss(disparity, gt, mask, ref_width)
        parts["l1"] = l1.detach()
        loss = loss + (1.0 if kind == "l1" else l1_weight) * l1
    parts["total"] = loss.detach()
    return loss, parts


@torch.no_grad()
def epe_px(disparity: Tensor, gt: Tensor, mask: Tensor, ref_width: float | Tensor) -> Tensor:
    """Error de punto final medio (|d - gt| en px de referencia) sobre la máscara."""
    return l1_loss(disparity, gt, mask, ref_width)
