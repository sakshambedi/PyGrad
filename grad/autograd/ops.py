from __future__ import annotations

import math
from typing import Any

from grad.autograd import operations
from grad.autograd._functions import (
    _elementwise_operation,
    _materialize_operand,
    _reduce_to_shape,
    _safe_divide,
    _target_shape,
    _unary_operation,
)
from grad.autograd.function import Function
from grad.tensor import Tensor


def _apply_elementwise(t: Tensor, fn) -> list:
    """Apply a scalar function element-wise, returning a nested list matching t.shape."""
    from grad.utils.misc import _nd_indices

    if t.shape == ():
        return fn(t.item())
    flat = [fn(t[idx]) for idx in _nd_indices(t.shape)]
    return Tensor._nest(flat, list(t.shape))


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
        if not isinstance(grad_output, Tensor):
            return grad_output, grad_output

        a, b = ctx.saved_tensor
        a_shape = tuple(getattr(ctx, "a_shape", ()))
        b_shape = tuple(getattr(ctx, "b_shape", ()))
        if not a_shape:
            a_shape = _target_shape(ctx, a, "a_shape", tuple(grad_output.shape))
        if not b_shape:
            b_shape = _target_shape(ctx, b, "b_shape", tuple(grad_output.shape))

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
        if not isinstance(grad_output, Tensor):
            return grad_output, -grad_output

        a, b = ctx.saved_tensor
        a_shape = _target_shape(ctx, a, "a_shape", tuple(grad_output.shape))
        b_shape = _target_shape(ctx, b, "b_shape", tuple(grad_output.shape))

        grad_a = _reduce_to_shape(grad_output, a_shape)
        grad_b = _reduce_to_shape(-grad_output, b_shape)
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.MUL)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor
        if not isinstance(grad_output, Tensor):
            return b * grad_output, a * grad_output

        out_shape = tuple(getattr(ctx, "out_shape", tuple(grad_output.shape)))
        a_in = _materialize_operand(a, out_shape, grad_output)
        b_in = _materialize_operand(b, out_shape, grad_output)

        grad_a_full = b_in * grad_output
        grad_b_full = a_in * grad_output

        a_shape = _target_shape(ctx, a, "a_shape", tuple(grad_output.shape))
        b_shape = _target_shape(ctx, b, "b_shape", tuple(grad_output.shape))
        grad_a = _reduce_to_shape(grad_a_full, a_shape)
        grad_b = _reduce_to_shape(grad_b_full, b_shape)
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
        if not isinstance(grad_output, Tensor):
            grad_a = _safe_divide(grad_output, b)
            grad_b = -_safe_divide(a * grad_output, b**2)
            return grad_a, grad_b

        out_shape = tuple(getattr(ctx, "out_shape", tuple(grad_output.shape)))
        a_in = _materialize_operand(a, out_shape, grad_output)
        b_in = _materialize_operand(b, out_shape, grad_output)

        grad_a_full = grad_output / b_in
        grad_b_full = -(a_in * grad_output / (b_in**2))

        a_shape = _target_shape(ctx, a, "a_shape", tuple(grad_output.shape))
        b_shape = _target_shape(ctx, b, "b_shape", tuple(grad_output.shape))
        grad_a = _reduce_to_shape(grad_a_full, a_shape)
        grad_b = _reduce_to_shape(grad_b_full, b_shape)
        return grad_a, grad_b


class Pow(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise power operation."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.POW)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor

        if not isinstance(grad_output, Tensor):
            grad_a = b * (a ** (b - 1)) * grad_output
            ln_a = math.log(a) if not isinstance(a, Tensor) else math.log(float(a.item()))
            grad_b = (a**b) * ln_a * grad_output
            return grad_a, grad_b

        out_shape = tuple(getattr(ctx, "out_shape", tuple(grad_output.shape)))
        a_in = _materialize_operand(a, out_shape, grad_output)
        b_in = _materialize_operand(b, out_shape, grad_output)

        grad_a_full = b_in * (a_in ** (b_in - 1)) * grad_output
        ln_a = Tensor(
            _apply_elementwise(a_in, math.log),
            dtype=a_in.dtype,
            device=a_in.device,
        )
        grad_b_full = (a_in**b_in) * ln_a * grad_output

        a_shape = _target_shape(ctx, a, "a_shape", tuple(grad_output.shape))
        b_shape = _target_shape(ctx, b, "b_shape", tuple(grad_output.shape))
        grad_a = _reduce_to_shape(grad_a_full, a_shape)
        grad_b = _reduce_to_shape(grad_b_full, b_shape)
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


class Sum(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, *, dim: int | None = None, keepdims: bool = False) -> Tensor:
        """Sum reduction along a dimension."""
        from grad.utils.misc import _nd_indices

        ctx.save_for_backward(a)
        ctx.a_shape = tuple(a.shape)
        ctx.dim = dim
        ctx.keepdims = keepdims

        if dim is None:
            total = sum(a[idx] for idx in _nd_indices(a.shape))
            return Tensor(total, dtype=a.dtype, device=a.device)

        ndim = len(a.shape)
        axis = dim + ndim if dim < 0 else dim

        if keepdims:
            out_shape = a.shape[:axis] + (1,) + a.shape[axis + 1 :]
        else:
            out_shape = a.shape[:axis] + a.shape[axis + 1 :]

        out = Tensor.zeros(out_shape, dtype=a.dtype, device=a.device)
        for idx in _nd_indices(a.shape):
            out_idx = idx[:axis] + ((0,) if keepdims else ()) + idx[axis + 1 :]
            out[out_idx] = out[out_idx] + a[idx]
        return out

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        from grad.utils.misc import _nd_indices

        grad_output = grad_outputs[0]
        a_shape = ctx.a_shape

        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)

        if ctx.dim is None:
            grad_val = grad_output.item() if grad_output.shape == () else grad_output[(0,)]
            return (Tensor.full(a_shape, grad_val, dtype=grad_output.dtype),)

        # Broadcast gradient back along the reduced dimension
        out = Tensor.zeros(a_shape, dtype=grad_output.dtype)
        ndim = len(a_shape)
        axis = ctx.dim + ndim if ctx.dim < 0 else ctx.dim
        for idx in _nd_indices(a_shape):
            grad_idx = idx[:axis] + ((0,) if ctx.keepdims else ()) + idx[axis + 1 :]
            out[idx] = grad_output[grad_idx]
        return (out,)


class Exp(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor) -> Tensor:
        """Element-wise exponential: e^a."""
        result = Tensor(
            _apply_elementwise(a, math.exp),
            dtype=a.dtype,
            device=a.device,
        )
        ctx.save_for_backward(a)
        ctx.exp_result = result
        return result

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # d/da e^a = e^a * grad_output
        grad_output = grad_outputs[0]
        exp_result = ctx.exp_result
        if not isinstance(grad_output, Tensor):
            return (exp_result * grad_output,)
        return (exp_result * grad_output,)


class Log(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor) -> Tensor:
        """Element-wise natural logarithm: ln(a)."""
        result = Tensor(
            _apply_elementwise(a, math.log),
            dtype=a.dtype,
            device=a.device,
        )
        ctx.save_for_backward(a)
        return result

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # d/da ln(a) = 1/a * grad_output
        grad_output = grad_outputs[0]
        (a,) = ctx.saved_tensor
        if not isinstance(grad_output, Tensor):
            return (grad_output / a,)
        return (grad_output / a,)
