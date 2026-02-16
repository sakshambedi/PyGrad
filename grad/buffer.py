import weakref
from collections.abc import Iterable
from typing import Any

from grad.device import Device
from grad.dtype import DType, DTypeLike, to_dtype
from grad.kernels import cpu_kernel


class Buffer:
    """A buffer class for storing data with a specific data type.

    This class wraps around a C++ buffer implementation to store data efficiently.
    It provides methods for creating, accessing, and modifying data buffers.
    """

    __slots__ = ("_dtype", "_storage", "_parent", "__weakref__")

    def __init__(self, iterable: Iterable[Any], dtype: DTypeLike, *, copy: bool = True):
        self._dtype: DType = to_dtype(dtype)
        self._storage = cpu_kernel.Buffer(iterable, self._dtype.name)
        self._parent = None

    def to(self, device: Device): ...  # noqa: E704

    @property
    def dtype(self) -> str:
        return str(self._storage.get_dtype())

    def __len__(self):
        return self._storage.size()

    def __repr__(self) -> str:
        return str(self.to_list())

    def share(self) -> "Buffer":
        """Return a new Buffer instance that shares underlying storage."""
        new_buff = Buffer.__new__(Buffer)
        new_buff._dtype = self._dtype
        new_buff._storage = self._storage
        # Create a weakref to the original buffer for updates
        new_buff._parent = weakref.ref(self)
        return new_buff

    def shares_storage_with(self, other: "Buffer") -> bool:
        """Check if this buffer shares its storage with other."""
        return self._storage is getattr(other, "_storage", None)

    def iterstorage(self) -> Iterable[Any]:
        for i in range(self.__len__()):
            yield self._storage[i]

    def to_list(self) -> list[Any]:
        return [val for val in self.iterstorage()]

    def __del__(self): ...  # noqa : E704

    @staticmethod
    def _make_buffer(type_: DTypeLike, iterable: Iterable[Any]) -> memoryview: ...  # noqa : E704

    def clone(self) -> "Buffer":
        """Create a copy of this buffer"""
        return Buffer(self.to_list(), self._dtype)

    def resize(self, new_size: int) -> "Buffer":
        """Create a new buffer with different size, preserving existing data where possible"""
        current_data = self.to_list()
        if new_size > len(current_data):
            current_data.extend([0] * (new_size - len(current_data)))
        else:
            current_data = current_data[:new_size]
        return Buffer(current_data, self._dtype)

    @classmethod
    def _filled(cls, dtype: DTypeLike, num_elem: int, val: int | float) -> "Buffer":
        buff = cls.__new__(cls)
        buff._dtype = out_dtype = to_dtype(dtype)
        # buff._storage = cpu_kernel.Buffer(num_elem, out_dtype.name, val)
        buff._storage = cpu_kernel.Buffer._filled(val, out_dtype.name, num_elem)
        buff._parent = None
        return buff

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Buffer index {idx} out of range")
        return self._storage[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._storage[idx] = value

    def size_bytes(self) -> int:
        """Return the size of this buffer in bytes"""
        return len(self) * self._dtype.itemsize

    @classmethod
    def _from_cpp_buffer(cls, cpp_buffer: Any, dtype: DType):
        """Create a Python Buffer from a C++ Buffer"""
        buff = cls.__new__(cls)
        buff._dtype = to_dtype(dtype)
        buff._storage = cpp_buffer
        buff._parent = None
        return buff
