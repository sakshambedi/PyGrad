from __future__ import annotations

from itertools import product as iter_product


def tensor_stride(shape) -> tuple[int, ...]:
    """
    O(len(shape)) time, O(1) extra memory.
    pytorch stride : https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html
    """
    if not shape:
        return ()

    acc = 1
    stride = []
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    stride.reverse()
    return tuple(stride)


def _nd_indices(shape):
    """
    Yields all possible n-dimensional indices for a given shape.
    Example: shape = (2, 3) yields (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    """

    return iter_product(*(range(s) for s in shape))


def broadcast_shape(shp1: tuple[int, ...], shp2: tuple[int, ...]) -> list[int]:
    """
    Computes the broadcasted shape of two input shapes according to broadcasting rules.
    Raises ValueError if the shapes are not broadcast-compatible.
    """
    result = []
    for a_dim, b_dim in zip(reversed(shp1), reversed(shp2)):
        if a_dim == b_dim or a_dim == 1 or b_dim == 1:
            result.append(max(a_dim, b_dim))
        else:
            raise ValueError(f"Shapes {shp1} and {shp2} are not broadcast-compatible.")
    # Leading dimensions of the longer shape carry over unchanged. Clamp at 0:
    # a negative slice end would otherwise pull dimensions from the shorter shape.
    result.extend(reversed(shp1[: max(len(shp1) - len(shp2), 0)]))
    result.extend(reversed(shp2[: max(len(shp2) - len(shp1), 0)]))
    return list(reversed(result))


def can_broadcast(shp1: tuple[int, ...], shp2: tuple[int, ...]):
    for a_dim, b_dim in zip(reversed(shp1), reversed(shp2)):
        if a_dim != b_dim and a_dim != 1 and b_dim != 1:
            return False
    return True
