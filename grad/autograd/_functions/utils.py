from __future__ import annotations

import math
from typing import Any

from grad.autograd.function import Function
from grad.tensor import Tensor


def _reduce_to_shape(grad: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    if tuple(grad.shape) == tuple(target_shape):
        return grad

    out_shape = tuple(grad.shape)
    ndiff = len(out_shape) - len(target_shape)
    padded_target = (1,) * ndiff + tuple(target_shape)

    # Axes to reduce: dimensions introduced by broadcasting or target dim == 1
    reduce_axes = [
        i for i, (gdim, tdim) in enumerate(zip(out_shape, padded_target)) if tdim == 1 and gdim != 1
    ]

    reduced = grad
    for axis in sorted(reduce_axes, reverse=True):
        reduced = reduced.sum(dim=axis, keepdims=True)

    if ndiff > 0:
        for _ in range(ndiff):
            reduced = reduced.sum(dim=0, keepdims=False)

    if tuple(reduced.shape) != tuple(target_shape):
        reduced = reduced.reshape(target_shape)
    return reduced


def _target_shape(
    ctx: Function, saved: Any, shape_attr: str, fallback: tuple[int, ...]
) -> tuple[int, ...]:
    if hasattr(ctx, shape_attr):
        return tuple(getattr(ctx, shape_attr))
    if isinstance(saved, Tensor):
        return tuple(saved.shape)
    return fallback


def _safe_divide(numerator: Any, denominator: Any) -> Any:
    if isinstance(denominator, (int, float)) and denominator == 0:
        return math.copysign(math.inf, float(numerator))
    return numerator / denominator
