from __future__ import annotations

import math
from typing import Any

import numpy as np
from grad.autograd import operations  # type:ignore
from grad.autograd.function import Function
from grad.buffer import Buffer
from grad.dtype import dtypes
from grad.tensor import Tensor
from grad.utils.misc import broadcast_shape, tensor_stride


def _elementwise_operation(ctx, a: Tensor, b: Tensor, op_type: operations.BinaryOpType):
    if a.storage is None or b.storage is None:
        raise ValueError(f"Cannot perform {op_type} on tensors with no storage")

    ctx.save_for_backward(a, b)
    rdtype = dtypes._upcast(a.dtype, b.dtype)
    out_shape = broadcast_shape(a.shape, b.shape)
    # cpp_result_buffer = operations.buffer_add(a.storage._storage, b.storage._storage, rdtype.name)
    cpp_result_buffer = operations.binary_op(
        a.storage._storage, b.storage._storage, op_type, rdtype.name
    )
    result = Tensor.__new__(Tensor)
    result.shape = tuple(out_shape)
    result._stride = tensor_stride(result.shape)
    result.device = (a.device or b.device) or "cpu"
    result._contiguous = a._contiguous and b._contiguous
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


def _safe_divide(numerator: Any, denominator: Any) -> Any:
    if isinstance(denominator, (int, float)) and denominator == 0:
        return math.copysign(math.inf, float(numerator))
    return numerator / denominator


class Add(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.ADD)

    @staticmethod
    def backward(ctx: Function, *grad_output: Any) -> Any:
        # For addition, L = a + b ;  dL/da = grad_output, dL/db = grad_output
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.SUB)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        # For subtraction, L = a - b; dL/da = grad_output, dL/db = -grad_output
        grad_output = grad_outputs[0]
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.MUL)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor
        return b * grad_output, a * grad_output


class Div(Function):
    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise division of two tensors."""
        return _elementwise_operation(ctx, a, b, operations.BinaryOpType.DIV)

    @staticmethod
    def backward(ctx: Function, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        a, b = ctx.saved_tensor
        grad_a = _safe_divide(grad_output, b)
        grad_b = -_safe_divide(a * grad_output, b**2)
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
        grad_a = b * (a ** (b - 1)) * grad_output
        if isinstance(a, Tensor):
            ln_a = Tensor(np.log(a.to_numpy()).tolist(), dtype=a.dtype, device=a.device)
        else:
            ln_a = math.log(a)
        grad_b = (a**b) * ln_a * grad_output
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
