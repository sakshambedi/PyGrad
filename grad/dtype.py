from __future__ import annotations

import array
from dataclasses import dataclass
from typing import Final, Literal, Union

ARRAY_E_SUPPORTED = "e" in array.typecodes

FmtStr = Literal["?", "b", "B", "h", "H", "i", "I", "q", "Q", "e", "f", "d"]


class DTypeMetaClass(type):
    dcache: dict[tuple, DType] = {}

    def __call__(self, *args, **kwds):
        if ret := DTypeMetaClass.dcache.get(args, None):
            return ret

        DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
        return ret


@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
    priority: int
    itemsize: int
    name: str
    fmt: FmtStr | None
    count: int
    _scalar: DType | None

    @staticmethod
    def new(priority: int, itemsize: int, name: str, fmt: FmtStr | None):
        return DType(priority, itemsize, name, fmt, 1, None)


class dtypes:
    bool: Final[DType] = DType.new(0, 1, "bool", "?")
    int8: Final[DType] = DType.new(1, 1, "int8", "b")
    uint8: Final[DType] = DType.new(2, 1, "uint8", "B")
    int16: Final[DType] = DType.new(3, 2, "uint16", "h")
    uint16: Final[DType] = DType.new(4, 2, "uint16", "H")
    int32: Final[DType] = DType.new(5, 4, "int32", "i")
    uint32: Final[DType] = DType.new(6, 4, "uint32", "I")
    int64: Final[DType] = DType.new(7, 8, "int64", "q")
    uint64: Final[DType] = DType.new(8, 8, "uint64", "Q")

    # Floating point
    float16: Final[DType] = DType.new(9, 2, "float16", "e")
    float32: Final[DType] = DType.new(10, 4, "float32", "f")
    float64: Final[DType] = DType.new(11, 8, "float64", "d")

    # alias
    fp16: Final[DType] = float16
    fp32: Final[DType] = float32
    fp64: Final[DType] = float64
    double: Final[DType] = float64

    @staticmethod
    def _upcast(t1: DType, t2: DType) -> DType:
        if t1 and t2:
            return t1 if t1.priority >= t2.priority else t2
        raise TypeError(f"Cannot upcast dtypes {t1.name} and {t2.name}")


DTypeLike = Union[str, DType]


def to_dtype(dtype: DTypeLike) -> DType:
    return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())
