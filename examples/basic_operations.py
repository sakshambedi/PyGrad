#!/usr/bin/env python3
"""
Basic operations example for pygrad tensor library.
"""

from examples.print_utils import print_banner, print_kv, print_section
from grad.dtype import dtypes
from grad.tensor import Tensor


def demo_tensor_creation() -> None:
    """Demonstrate various ways to create tensors."""
    print_section("Tensor Creation")

    # Create tensors with different methods
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([[1, 2, 3], [4, 5, 6]])
    t3 = Tensor(5)  # Scalar tensor
    t4 = Tensor.zeros((2, 3))
    t5 = Tensor.ones((2, 2))
    t6 = Tensor.arange(10)
    t7 = Tensor.randn(2, 3)
    t8 = Tensor.full((2, 2), 7)

    print_kv("1D tensor", t1)
    print_kv("2D tensor", t2)
    print_kv("Scalar tensor", t3)
    print_kv("Zeros tensor", t4)
    print_kv("Ones tensor", t5)
    print_kv("Range tensor", t6)
    print_kv("Random tensor", t7)
    print_kv("Full tensor", t8)

    # Create tensors with different dtypes
    t9 = Tensor([1, 2, 3], dtype=dtypes.int32)
    t10 = Tensor([1.5, 2.5, 3.5], dtype=dtypes.float64)

    print_kv("Int32 tensor", t9)
    print_kv("Float64 tensor", t10)


def demo_math_operations() -> None:
    """Demonstrate basic mathematical operations."""
    print_section("Mathematical Operations")

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    c = a + b
    d = a - b
    e = a * b
    f = b / a

    print_kv("Addition", f"{a} + {b} = {c}")
    print_kv("Subtraction", f"{a} - {b} = {d}")
    print_kv("Multiplication", f"{a} * {b} = {e}")
    print_kv("Division", f"{b} / {a} = {f}")


def demo_shape_operations() -> None:
    """Demonstrate shape manipulation operations."""
    print_section("Shape Operations")

    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.reshape(3, 2)
    c = a.view(6, 1)
    d = Tensor.T(a)
    e = Tensor.arange(24).reshape(2, 3, 4)
    f = Tensor.permute(e, 2, 0, 1)

    print_kv("Original tensor", f"{a} | shape={a.shape}")
    print_kv("Reshaped tensor", f"{b} | shape={b.shape}")
    print_kv("Viewed tensor", f"{c} | shape={c.shape}")
    print_kv("Transposed tensor", f"{d} | shape={d.shape}")
    print_kv("Original 3D shape", e.shape)
    print_kv("Permuted 3D shape", f.shape)


def demo_numpy_interop() -> None:
    """Demonstrate NumPy interoperability."""
    print_section("NumPy Interoperability")

    t = Tensor([[1, 2, 3], [4, 5, 6]])
    np_array = t.to_numpy()
    result_np = np_array * 2
    t_result = Tensor(result_np.tolist())

    print_kv("Tensor", t)
    print_kv("NumPy array", np_array)
    print_kv("NumPy array type", type(np_array))
    print_kv("NumPy result * 2", result_np)
    print_kv("Back to tensor", t_result)


def demo_autograd_basics() -> None:
    """Demonstrate basic autograd functionality."""
    print_section("Autograd Basics")

    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a + b

    print_kv("Tensor a", f"{a}, requires_grad={a.requires_grad}")
    print_kv("Tensor b", f"{b}, requires_grad={b.requires_grad}")
    print_kv("Result c = a + b", f"{c}, requires_grad={c.requires_grad}")
    print_kv("Note", "Backward pass implementation is in progress")


def main() -> None:
    """Run all examples."""
    print_banner("pygrad Tensor Library Examples")

    demo_tensor_creation()
    demo_math_operations()
    demo_shape_operations()
    demo_numpy_interop()
    demo_autograd_basics()


if __name__ == "__main__":
    main()
