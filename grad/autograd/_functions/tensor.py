from __future__ import annotations

from typing import Any

from grad.autograd import operations
from grad.buffer import Buffer
from grad.dtype import dtypes
from grad.tensor import Tensor
from grad.utils.misc import broadcast_shape, tensor_stride


def _elementwise_operation(ctx, a: Tensor, b: Tensor, op_type: operations.BinaryOpType):
    if a.storage is None or b.storage is None:
        raise ValueError(f"Cannot perform {op_type} on tensors with no storage")

    rdtype = dtypes._upcast(a.dtype, b.dtype)
    out_shape = tuple(broadcast_shape(a.shape, b.shape))

    # Materialize scalar/broadcasted inputs so backend binary op receives same-sized buffers.
    if tuple(a.shape) == ():
        a_in = Tensor.full(out_shape, a.item(), dtype=rdtype)
    else:
        a_in = a if tuple(a.shape) == out_shape else Tensor._contiguous_tensor(a.expand(*out_shape))

    if tuple(b.shape) == ():
        b_in = Tensor.full(out_shape, b.item(), dtype=rdtype)
    else:
        b_in = b if tuple(b.shape) == out_shape else Tensor._contiguous_tensor(b.expand(*out_shape))

    ctx.save_for_backward(a, b)
    ctx.a_shape = tuple(a.shape)
    ctx.b_shape = tuple(b.shape)
    ctx.out_shape = out_shape

    cpp_result_buffer = operations.binary_op(
        a_in.storage._storage, b_in.storage._storage, op_type, rdtype.name
    )
    result = Tensor.__new__(Tensor)
    result.shape = out_shape
    result._stride = tensor_stride(result.shape)
    result.device = (a.device or b.device) or "cpu"
    result._contiguous = True
    result.base_offset = 0
    result.storage = Buffer._from_cpp_buffer(cpp_result_buffer, rdtype)
    result.grad, result.grad_fn, result.requires_grad = None, None, None

    return result


def _unary_operation(ctx, a: Tensor, op_type: operations.UnaryOpType):
    if a.storage is None:
        raise ValueError(f"Cannot perform {op_type} on tensors with no storage")

    ctx.save_for_backward(a)

    cpp_result_buffer = operations.unary_op(a.storage._storage, op_type, a.dtype.name)
    result = Tensor.__new__(Tensor)
    result.shape = tuple(a.shape)
    result._stride = tensor_stride(result.shape)
    result.device = a.device or "cpu"
    result._contiguous = a._contiguous
    result.base_offset = 0
    result.storage = Buffer._from_cpp_buffer(cpp_result_buffer, a.dtype)
    result.grad, result.grad_fn, result.requires_grad = None, None, None

    return result


def _materialize_operand(operand: Any, out_shape: tuple[int, ...], ref: Tensor) -> Tensor:
    if isinstance(operand, Tensor):
        if tuple(operand.shape) == out_shape:
            return operand
        if tuple(operand.shape) == ():
            return Tensor.full(
                out_shape, operand.item(), dtype=operand.dtype, device=operand.device or "cpu"
            )
        return Tensor._contiguous_tensor(operand.expand(*out_shape))

    return Tensor.full(out_shape, operand, dtype=ref.dtype, device=ref.device or "cpu")
