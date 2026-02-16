from __future__ import annotations

from typing import Any

from grad.autograd import operations  # type:ignore
from grad.autograd.function import Function
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


def _materialize_to_shape(t: Tensor, out_shape: tuple[int, ...]) -> Tensor:
    if tuple(t.shape) == out_shape:
        return t
    if tuple(t.shape) == ():
        return Tensor.full(out_shape, t.item(), dtype=t.dtype, device=t.device or "cpu")
    return Tensor._contiguous_tensor(t.expand(*out_shape))


def _tensor_log(t: Tensor) -> Tensor:
    import numpy as np

    return Tensor(np.log(t.to_numpy()).tolist())


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.ADD)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # For addition, L = a + b ; dL/da = grad_output, dL/db = grad_output
        # If inputs were broadcast in forward, reduce gradients back to input shapes.
        grad_output = grad_outputs[0]
        a_shape = tuple(getattr(ctx, "a_shape", ()))
        b_shape = tuple(getattr(ctx, "b_shape", ()))

        grad_a = _reduce_to_shape(grad_output, a_shape)
        grad_b = _reduce_to_shape(grad_output, b_shape)
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.SUB)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # For subtraction, L = a - b; dL/da = grad_output, dL/db = -grad_output
        grad_output = grad_outputs[0]
        a_shape = tuple(getattr(ctx, "a_shape", ()))
        b_shape = tuple(getattr(ctx, "b_shape", ()))
        grad_a = _reduce_to_shape(grad_output, a_shape)
        grad_b = _reduce_to_shape(-grad_output, b_shape)
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.MUL)
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor
        out_shape = tuple(getattr(ctx, "out_shape", tuple(grad_output.shape)))

        a_in = _materialize_to_shape(a, out_shape)
        b_in = _materialize_to_shape(b, out_shape)
        grad_a_full = grad_output * b_in
        grad_b_full = grad_output * a_in

        grad_a = _reduce_to_shape(grad_a_full, tuple(getattr(ctx, "a_shape", a.shape)))
        grad_b = _reduce_to_shape(grad_b_full, tuple(getattr(ctx, "b_shape", b.shape)))
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise division of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.DIV)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor
        out_shape = tuple(getattr(ctx, "out_shape", tuple(grad_output.shape)))

        a_in = _materialize_to_shape(a, out_shape)
        b_in = _materialize_to_shape(b, out_shape)
        grad_a_full = grad_output / b_in
        grad_b_full = -(grad_output * a_in / (b_in * b_in))

        grad_a = _reduce_to_shape(grad_a_full, tuple(getattr(ctx, "a_shape", a.shape)))
        grad_b = _reduce_to_shape(grad_b_full, tuple(getattr(ctx, "b_shape", b.shape)))
        return grad_a, grad_b


class Pow(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise power operation."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.POW)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # Power rule: if y = x^n, then dy/dx = n*x^(n-1)
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor
        out_shape = tuple(getattr(ctx, "out_shape", tuple(grad_output.shape)))

        a_in = _materialize_to_shape(a, out_shape)
        b_in = _materialize_to_shape(b, out_shape)

        grad_a_full = grad_output * b_in * (a_in ** (b_in - 1))
        grad_b_full = grad_output * (a_in**b_in) * _tensor_log(a_in)

        grad_a = _reduce_to_shape(grad_a_full, tuple(getattr(ctx, "a_shape", a.shape)))
        grad_b = _reduce_to_shape(grad_b_full, tuple(getattr(ctx, "b_shape", b.shape)))
        return grad_a, grad_b


class Neg(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor) -> Tensor:
        """Element-wise negation."""
        return _unary_operation(ctx, a, operations.UnaryOpType.NEG)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # For negation: L = -a, dL/da = -grad_output
        grad_output = grad_outputs[0]
        return (-grad_output,)
