from __future__ import annotations

import math
from typing import Any

import numpy as np

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
        ln_a = Tensor(np.log(a_in.to_numpy()).tolist(), dtype=a_in.dtype, device=a_in.device)
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
        a = ctx.saved_tensor
        return (-grad_output,)


class Exp(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, *, out=None) -> Tensor:
        """Element-wise negation."""
        ...

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # implement backward for
        ...
