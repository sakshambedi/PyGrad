from __future__ import annotations

import random
from collections.abc import Generator, Iterable, Sequence
from math import prod as _prod
from typing import Any

from grad.autograd.function import Function
from grad.buffer import Buffer
from grad.dtype import DType, DTypeLike, dtypes
from grad.utils.misc import _nd_indices, tensor_stride

__all__ = ["Tensor"]


class Tensor:
    """Tiny, PyTorch‑like dense tensor backed by a contiguous buffer."""

    __slots__ = (
        "shape",
        "device",
        "requires_grad",
        "grad",
        "storage",
        "grad_fn",
        "_stride",
        "_contiguous",
        "base_offset",
    )

    def __init__(
        self,
        data: Iterable | int | float | None = None,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: bool | None = None,
    ) -> None:
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self.storage: Buffer | None = None
        self.grad_fn: Function | None = None
        self._contiguous: bool = True
        self.base_offset: int = 0

        if data is None:
            self.shape: tuple[int, ...] = ()
            self.storage = Buffer([0], dtype)
        elif isinstance(data, (list, tuple)):
            self.shape: tuple[int, ...] = self._infer_shape(data)
            self.storage = Buffer(data, dtype)
        else:  # Handles single int or float
            self.shape: tuple[int, ...] = ()
            self.storage = Buffer([data], dtype)

        self._stride: tuple[int, ...] = tensor_stride(self.shape)

    @classmethod
    def zeros(cls, shape: Sequence[int], **kw) -> Tensor:
        """Create a tensor filled with zeros."""
        return cls._filled(shape, 0, **kw)

    @classmethod
    def ones(cls, shape: Sequence[int], **kw) -> Tensor:
        """Create a tensor filled with ones."""
        return cls._filled(shape, 1, **kw)

    def is_contigous(self) -> bool:
        return self._contiguous

    @classmethod
    def arange(
        cls,
        end: int,
        start: int = 0,
        step: int = 1,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: bool = False,
    ) -> Tensor:
        """Return a 1D tensor with values from start to end (exclusive) with a given step.
        # For some reason using a list comprehension for making arange is faster than using array.array
        """
        if step == 0:
            raise ValueError("step must not be zero")

        size = max(0, (end - start + (step - (1 if step > 0 else -1))) // step)

        inst: Tensor = cls.__new__(cls)
        inst.storage = Buffer([start + i * step for i in range(size)], dtype)
        inst.shape = (size,)
        inst._stride = tensor_stride(inst.shape)
        inst.device, inst.requires_grad = device, requires_grad
        inst.grad, inst.grad_fn, inst._contiguous, inst.base_offset = (
            None,
            None,
            True,
            0,
        )
        return inst

    @classmethod
    def rand(cls, *shape: int, **kw) -> Tensor:
        """Create a tensor with random numbers from a uniform distribution [0, 1).
        Read More: https://docs.pytorch.org/docs/stable/generated/torch.rand.html
        """
        size = _prod(shape) if shape else 1
        random_data = [random.random() for _ in range(size)]

        inst: Tensor = cls.__new__(cls)
        inst.storage = Buffer(random_data, kw.get("dtype", dtypes.float32))
        inst.shape = tuple(shape)
        inst._stride = tensor_stride(inst.shape)
        inst.device = kw.get("device", "cpu")
        inst.requires_grad = kw.get("requires_grad", False)
        inst.grad, inst.grad_fn, inst._contiguous, inst.base_offset = (
            None,
            None,
            True,
            0,
        )
        return inst

    @classmethod
    def eye(cls, n: int, m: int | None = None, **kw) -> Tensor:
        """Create a 2D identity tensor (ones on diagonal, zeros elsewhere).
        Read More: https://docs.pytorch.org/docs/stable/generated/torch.eye.html
        """
        if m is None:
            m = n
        data = [1 if i == j else 0 for i in range(n) for j in range(m)]

        inst: Tensor = cls.__new__(cls)
        inst.storage = Buffer(data, kw.get("dtype", dtypes.float32))
        inst.shape = (n, m)
        inst._stride = tensor_stride(inst.shape)
        inst.device = kw.get("device", "cpu")
        inst.requires_grad = kw.get("requires_grad", False)
        inst.grad, inst.grad_fn, inst._contiguous, inst.base_offset = (
            None,
            None,
            True,
            0,
        )
        return inst

    @classmethod
    def randn(cls, *shape: int, **kw) -> Tensor:
        """Create a tensor with random numbers from a standard normal distribution.
        Read More : https://docs.pytorch.org/docs/stable/generated/torch.randn.html
        """
        size = _prod(shape) if shape else 1
        random_data = [random.gauss(0.0, 1.0) for _ in range(size)]

        inst: Tensor = cls.__new__(cls)

        inst.storage = Buffer(random_data, kw.get("dtype", dtypes.float32))
        inst.shape = tuple(shape)
        inst._stride = tensor_stride(inst.shape)
        inst.device = kw.get("device", "cpu")
        inst.requires_grad = kw.get("requires_grad", False)
        inst.grad, inst.grad_fn, inst._contiguous, inst.base_offset = (
            None,
            None,
            True,
            0,
        )
        return inst

    @classmethod
    def full(cls, shape: Sequence[int], fill_value: Any, **kw) -> Tensor:
        """Create a tensor filled with the specified value."""
        return cls._filled(shape, fill_value, **kw)

    # ---- Shape Manipulation Methods ----

    def expand(self, *shape: int) -> Tensor:
        """Expand the tensor to a larger shape without allocating new memory."""
        if len(shape) < len(self.shape):
            raise ValueError(
                "The expanded shape must have at least as many dimensions as the original shape."
            )

        new_shape = list(shape)
        new_stride = [0] * len(new_shape)

        # Align shapes to the right
        shape_offset = len(new_shape) - len(self.shape)

        for i in range(len(self.shape) - 1, -1, -1):
            if self.shape[i] == new_shape[i + shape_offset]:
                new_stride[i + shape_offset] = self._stride[i]
            elif self.shape[i] == 1:
                new_stride[i + shape_offset] = 0
            else:
                raise ValueError(
                    f"Cannot expand dimension {i} of size {self.shape[i]} to {new_shape[i + shape_offset]}"
                )

        return self._create_view(tuple(new_shape), stride=tuple(new_stride))

    def item(self) -> Any:
        if (tshape := self.shape) == ():
            storage = self.storage
            if storage is None:
                raise AttributeError("Tensor with data is not initialized yet!")
            return storage[0]
        raise IndexError(
            f"Unable to return item from a Tensor of shape {tshape}. Supports tensors with shape ()"
        )

    def view(self, *shape: int) -> Tensor:
        """Return a tensor with the same data but a different shape."""
        shape_tuple = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        new_size, old_size = _prod(shape_tuple), _prod(self.shape)

        if new_size != old_size:
            raise ValueError(
                f"Cannot view tensor of shape {self.shape} with {old_size} elements as shape {shape_tuple} with {new_size} elements"
            )

        return self._create_view(shape_tuple)

    def reshape(self, *shape: int) -> Tensor:
        """Alias for view method."""
        return self.view(*shape)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Swap dimensions dim0 and dim1 of the tensor."""
        if dim0 == dim1:
            return self

        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

        new_stride = list(self._stride)
        new_stride[dim0], new_stride[dim1] = new_stride[dim1], new_stride[dim0]

        return self._create_view(tuple(new_shape), stride=tuple(new_stride))

    @staticmethod
    def T(ten: Tensor) -> Tensor:
        """Transpose the tensor"""
        if len(ten.shape) <= 1:
            return ten
        elif len(ten.shape) == 2:
            return ten.transpose(0, 1)
        raise BufferError(
            f"Input tensor with shape({ten.shape}) has len: ({len(ten.shape)})>= 2 for transpose not supported"
        )

    @staticmethod
    def permute(ten: Tensor, *idx: int) -> Tensor:
        """Permute the tensor. Read more: https://docs.pytorch.org/docs/stable/generated/torch.permute.html"""
        if len(idx) == 1 and isinstance(idx[0], tuple):
            idx_tup = tuple(int(i) for i in idx[0])
        else:
            idx_tup = idx
        if len(idx_tup) != len(ten.shape):
            raise ValueError(
                f"Number of permutation indices ({len(idx)}) must match tensor dimensions ({len(ten.shape)})"
            )
        if len(set(idx_tup)) != len(idx_tup):
            raise ValueError(f"Permutation indices contain duplicates: {idx}")
        if sorted(idx_tup) != list(range(len(ten.shape))):
            raise ValueError(
                f"Invalid permutation indices: {idx}. Must be a permutation of {list(range(len(ten.shape)))}"
            )

        shape_n = [ten.shape[d] for d in idx_tup]
        stride_n = [ten._stride[d] for d in idx_tup]
        return ten._create_view(tuple(shape_n), stride=tuple(stride_n))

    @property
    def dtype(self) -> DType:
        """Return the data type of the tensor."""
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage._dtype

    @property
    def vector_dtype(self) -> str:
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage.dtype

    @staticmethod
    def iterbuffer(t: Tensor, rdtype: DType) -> Iterable[Any]:
        if t.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")

        for i in t.buffer if t.is_contigous() else (t[idx] for idx in _nd_indices(t.shape)):
            yield i

    @property
    def buffer(self) -> memoryview:
        """Return a memoryview of the underlying storage."""
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return self.storage._storage

    def buffer_id(self) -> int:
        """Returns the memory address of the underlying storage."""
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")
        return id(self.buffer)

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        """Return the stride of the tensor. If dim is specified, return the stride for that dimension."""
        return self._stride if dim is None else self._stride[dim % len(self.shape)]

    @staticmethod
    def matmul(t1: Tensor, t2: Tensor, /, dtype: DTypeLike | None = None) -> Tensor:
        """Matrix multiplication of two tensors (1D/2D).
        Read More: https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
        """
        if len(t1.shape) < 1 or len(t2.shape) < 1:
            raise ValueError("matmul requires tensors with at least 1 dimension")

        out_dtype = dtype or dtypes._upcast(t1.dtype, t2.dtype)
        dev = t1.device or "cpu"
        rg = t1.requires_grad or t2.requires_grad

        # 1D x 1D: dot product → scalar
        if len(t1.shape) == 1 and len(t2.shape) == 1:
            n = t1.shape[0]
            if n != t2.shape[0]:
                raise ValueError(
                    f"Dot product requires same length, got {n} and {t2.shape[0]}"
                )
            dot = sum(t1[(i,)] * t2[(i,)] for i in range(n))
            return Tensor(dot, dtype=out_dtype, device=dev)

        # 2D x 2D
        if len(t1.shape) == 2 and len(t2.shape) == 2:
            M, K = t1.shape
            K2, N = t2.shape
            if K != K2:
                raise ValueError(
                    f"matmul shape mismatch: ({M}x{K}) @ ({K2}x{N})"
                )
            result = [
                [sum(t1[(i, k)] * t2[(k, j)] for k in range(K)) for j in range(N)]
                for i in range(M)
            ]
            return Tensor(result, dtype=out_dtype, device=dev, requires_grad=rg)

        # 2D x 1D: matrix-vector → 1D
        if len(t1.shape) == 2 and len(t2.shape) == 1:
            M, K = t1.shape
            if K != t2.shape[0]:
                raise ValueError(
                    f"matmul shape mismatch: ({M}x{K}) @ ({t2.shape[0]},)"
                )
            result = [sum(t1[(i, k)] * t2[(k,)] for k in range(K)) for i in range(M)]
            return Tensor(result, dtype=out_dtype, device=dev, requires_grad=rg)

        # 1D x 2D: vector-matrix → 1D
        if len(t1.shape) == 1 and len(t2.shape) == 2:
            K = t1.shape[0]
            K2, N = t2.shape
            if K != K2:
                raise ValueError(
                    f"matmul shape mismatch: ({K},) @ ({K2}x{N})"
                )
            result = [sum(t1[(k,)] * t2[(k, j)] for k in range(K)) for j in range(N)]
            return Tensor(result, dtype=out_dtype, device=dev, requires_grad=rg)

        raise ValueError(
            f"matmul not supported for tensors with shapes {t1.shape} and {t2.shape}"
        )

    def mean(
        self,
        dim: int | None = None,
        keepdims: bool = False,
        *,
        dtype: DType | None = None,
    ) -> Tensor:
        """Compute the mean along the given dimension.
        Read More: https://docs.pytorch.org/docs/stable/generated/torch.mean.html
        """
        out_dtype = dtype if dtype is not None else self.dtype

        if dim is None:
            s = self.sum(dtype=out_dtype)
            count = _prod(self.shape) if self.shape else 1
            return s / Tensor(count, dtype=out_dtype)

        ndim = len(self.shape)
        axis = dim + ndim if dim < 0 else dim
        if axis < 0 or axis >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )

        s = self.sum(dim=dim, keepdims=keepdims, dtype=out_dtype)
        count = self.shape[axis]
        return s / Tensor(count, dtype=out_dtype)

    def sum(
        self,
        dim: int | None = None,
        keepdims: bool = False,
        *,
        dtype: DType | None = None,
    ) -> Tensor:
        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")

        # Use autograd Sum when gradient tracking is needed
        if self.requires_grad:
            from grad.autograd.ops import Sum

            result = Sum.apply(self, dim=dim, keepdims=keepdims)
            if dtype is not None and dtype != self.dtype:
                pass  # dtype cast not yet supported in autograd path
            return result

        out_dtype = dtype if dtype is not None else self.dtype

        if dim is None:
            return Tensor(
                sum(Tensor.iterbuffer(self, out_dtype)),
                dtype=out_dtype,
                requires_grad=self.requires_grad,
            )

        if not isinstance(dim, int):
            raise TypeError(f"dim must be int or None, got {type(dim).__name__}")

        ndim = len(self.shape)
        axis = dim + ndim if dim < 0 else dim
        if axis < 0 or axis >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )

        if keepdims:
            out_shape = list(self.shape)
            out_shape[axis] = 1
            out_shape = tuple(out_shape)
        else:
            out_shape = self.shape[:axis] + self.shape[axis + 1 :]

        out = Tensor.zeros(
            out_shape,
            dtype=out_dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        for idx in _nd_indices(self.shape):
            out_idx = idx[:axis] + ((0,) if keepdims else ()) + idx[axis + 1 :]
            out[out_idx] = out[out_idx] + self[idx]

        return out

    # ---- Default override fuctions ----
    def __add__(self, other):
        """Element-wise addition that integrates with autograd."""
        from grad.autograd.ops import Add

        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return Add.apply(self, other)

    def __sub__(self, other):
        from grad.autograd.ops import Sub

        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return Sub.apply(self, other)

    def __mul__(self, other):
        from grad.autograd.ops import Mul

        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return Mul.apply(self, other)

    def __truediv__(self, other):
        from grad.autograd.ops import Div

        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return Div.apply(self, other)

    def __pow__(self, other):
        from grad.autograd.ops import Pow

        return Pow.apply(self, other)

    def __neg__(self):
        from grad.autograd.ops import Neg

        return Neg.apply(self)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return Tensor.matmul(self, other)

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other + self

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other - self

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other * self

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other / self

    def __rpow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        return other ** self

    def exp(self) -> Tensor:
        from grad.autograd.ops import Exp

        return Exp.apply(self)

    def log(self) -> Tensor:
        from grad.autograd.ops import Log

        return Log.apply(self)

    def backward(self, gradient: Tensor | None = None) -> None:
        """Compute gradients via reverse-mode autodiff (backpropagation)."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that doesn't require grad")

        if gradient is None:
            if self.shape == () or _prod(self.shape) == 1:
                gradient = Tensor.ones(self.shape if self.shape else (1,), dtype=self.dtype)
            else:
                raise RuntimeError(
                    "gradient must be specified for non-scalar outputs"
                )

        # Topological sort
        topo_order: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(t: Tensor) -> None:
            tid = id(t)
            if tid in visited:
                return
            visited.add(tid)
            if t.grad_fn is not None:
                for inp in t.grad_fn.saved_tensor:
                    if isinstance(inp, Tensor) and inp.requires_grad:
                        _build_topo(inp)
            topo_order.append(t)

        _build_topo(self)

        self.grad = gradient

        for node in reversed(topo_order):
            if node.grad_fn is None:
                continue
            grads = node.grad_fn.backward(node.grad_fn, node.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)

            saved = node.grad_fn.saved_tensor
            if not isinstance(saved, tuple):
                saved = (saved,)

            for inp, g in zip(saved, grads):
                if not isinstance(inp, Tensor) or not inp.requires_grad:
                    continue
                if g is None:
                    continue
                if not isinstance(g, Tensor):
                    g = Tensor(g, dtype=inp.dtype)
                inp.grad = g if inp.grad is None else inp.grad + g

    def _offset(self, index):
        return self.base_offset + sum(i * s for i, s in zip(index, self._stride))

    def __getitem__(self, index):
        """Access tensor data by index."""
        storage = self.storage
        if storage is None:
            raise AttributeError("Tensor with a storage has not been initialized yet!")
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape):
            raise IndexError(
                f"Incorrect number of indices for tensor of shape {self.shape!r}. "
                f"Expected {len(self.shape)} index{'es' if len(self.shape) != 1 else ''}, "
                f"but got {len(index)}: {index!r}."
            )
        norm = []
        for axis, (idx, dim) in enumerate(zip(index, self.shape)):
            if not isinstance(idx, int):
                raise TypeError(
                    f"indices must be integers, got {type(idx).__name__} at axis {axis}"
                )

            if idx < 0:
                idx += dim
            if idx >= dim or idx < 0:
                raise IndexError(
                    f"index {index[axis]} is out of bounds for axis {axis} with size {dim}"
                )
            norm.append(idx)

        offsetval = self._offset(tuple(norm))
        return storage[offsetval]

    def tolist(self) -> list | int | float:
        """Convert tensor to a nested Python list (or scalar)."""
        return self._to_nested()

    def to_numpy(self):
        """Convert tensor to numpy array. Requires numpy to be installed."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy is required for to_numpy(). Install it with: pip install numpy"
            ) from None

        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")

        if self._contiguous:
            arr = np.array(self.storage.to_list(), dtype=self.dtype.fmt)
            return arr.reshape(self.shape)

        flat = [self[idx] for idx in _nd_indices(self.shape)]
        return np.array(flat, dtype=self.dtype.fmt).reshape(self.shape)

    def __setitem__(self, idx, value):
        """Standard function for setting values by indexing"""
        if not isinstance(idx, tuple):
            idx = (idx,)  # in case of 1d tensor
        if len(idx) != len(self.shape):
            raise IndexError(
                f"Incorrect number of indices for tensor of shape {self.shape!r}. "
                f"Expected {len(self.shape)} index{'es' if len(self.shape) != 1 else ''}, "
                f"but got {len(idx)}: {idx!r}."
            )
        if self.storage is None:
            raise AttributeError("Tensor with a storage has not been initialized yet!")

        norm = []
        for axis, (i, dim) in enumerate(zip(idx, self.shape)):
            if not isinstance(i, int):
                raise TypeError(f"indices must be integers, got {type(i).__name__} at axis {axis}")
            if i < 0:
                i += dim
            if i >= dim or i < 0:
                raise IndexError(
                    f"index {idx[axis]} is out of bounds for axis {axis} with size {dim}"
                )
            norm.append(i)

        offsetval = self._offset(index=tuple(norm))
        self.storage[offsetval] = value

    def __repr__(self) -> str:
        """Return a string representation of the tensor"""
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype.name}, "
            f"device={self.device}, contiguous={self.is_contigous()}, requires_grad={self.requires_grad}, "
            f"data={self._to_nested()})"
        )

    def __str__(self) -> str:
        """Return a string representation of the tensor data."""
        return str(self._to_nested())

    # ---- Internal Helper Methods ----
    @classmethod
    def _filled(
        cls: type[Tensor],
        shape: Sequence[int],
        value: Any,
        *,
        dtype: DTypeLike = dtypes.float32,
        device: str = "cpu",
        requires_grad: bool | None = None,
    ) -> Tensor:
        """Internal method for creating tensors filled with a value."""
        inst: Tensor = cls.__new__(cls)
        inst.storage = Buffer._filled(dtype, _prod(shape), value)
        inst.shape = tuple(shape)
        inst._stride = tensor_stride(inst.shape)
        inst.device, inst.requires_grad = device, requires_grad
        inst.grad, inst.grad_fn, inst._contiguous, inst.base_offset = (
            None,
            None,
            True,
            0,
        )
        return inst

    def _create_view(
        self,
        shape: tuple[int, ...],
        *,
        stride: tuple[int, ...] | None = None,
        base_offset: int | None = None,
    ) -> Tensor:
        """Create a new tensor that shares storage with self but has a different shape."""
        result = Tensor.__new__(Tensor)
        result.shape = shape
        result._stride = tensor_stride(shape) if stride is None else stride
        result.device = self.device
        result.requires_grad = self.requires_grad
        result.grad, result.grad_fn, result._contiguous = None, None, False
        result.storage = self.storage.share() if self.storage is not None else None
        result.base_offset = 0 if base_offset is None else base_offset
        return result

    @staticmethod
    def _infer_shape(seq: Sequence) -> tuple[int, ...]:
        """Infer the shape of a nested sequence."""
        if not isinstance(seq, (list, tuple)):
            return ()
        if not seq:
            return (0,)
        inner = Tensor._infer_shape(seq[0])
        if any(Tensor._infer_shape(s) != inner for s in seq[1:]):
            raise IndexError("Inconsistent tensor shape")
        return (len(seq),) + inner

    @staticmethod
    def _flatten_gen(x: Any) -> Generator:
        """Flatten a nested sequence into a generator."""
        stack = [x]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                for item in reversed(current):
                    stack.append(item)
            else:
                yield current

    @staticmethod
    def _contiguous_tensor(t: Tensor) -> Tensor:
        """Make the cheap view/permuate to a buffer on device"""
        if t._contiguous:
            return t

        out = Tensor.zeros(t.shape, dtype=t.dtype, device=t.device, requires_grad=t.requires_grad)
        for idx in _nd_indices(t.shape):
            out[idx] = t[idx]

        return out

    def _to_nested(self) -> Any:
        """Convert the flat buffer to a nested list structure matching the tensor's shape."""

        if not self.shape or _prod(self.shape) == 0:
            if not self.shape:
                return None if self.storage is None else self.storage[0]

            return [self._nest([], list(self.shape[1:])) for _ in range(self.shape[0])]

        if self.storage is None:
            raise AttributeError("Tensor with data is not initialized yet!")

        # Contiguous tensor case
        if self._contiguous:
            flat = self.storage.to_list()
            return self._nest(flat, list(self.shape))

        # Non-contiguous tensor - need to go through indices
        flat_ordered_data = []
        for idx in _nd_indices(self.shape):
            flat_ordered_data.append(self.__getitem__(idx))
        return self._nest(flat_ordered_data, list(self.shape))

    @staticmethod
    def _nest(flat: list[Any], dims: list[int]) -> Any:
        """Recursively nest a flat list according to the provided dimensions."""
        if not dims:
            return flat.pop(0) if flat else None
        return [Tensor._nest(flat, dims[1:]) for _ in range(dims[0])]
