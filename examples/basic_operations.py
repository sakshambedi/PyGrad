#!/usr/bin/env python3
"""
Basic operations example for pygrad tensor library
"""

from grad.dtype import dtypes
from grad.tensor import Tensor


def test_tensor_creation():
    """Demonstrate various ways to create tensors"""
    print("\n=== Tensor Creation ===")

    # Create tensors with different methods
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([[1, 2, 3], [4, 5, 6]])
    t3 = Tensor(5)  # Scalar tensor
    t4 = Tensor.zeros((2, 3))
    t5 = Tensor.ones((2, 2))
    t6 = Tensor.arange(10)
    t7 = Tensor.randn(2, 3)
    t8 = Tensor.full((2, 2), 7)

    print(f"1D tensor: {t1}")
    print(f"2D tensor: {t2}")
    print(f"Scalar tensor: {t3}")
    print(f"Zeros tensor: {t4}")
    print(f"Ones tensor: {t5}")
    print(f"Range tensor: {t6}")
    print(f"Random tensor: {t7}")
    print(f"Full tensor: {t8}")

    # Create tensors with different dtypes
    t9 = Tensor([1, 2, 3], dtype=dtypes.int32)
    t10 = Tensor([1.5, 2.5, 3.5], dtype=dtypes.float64)

    print(f"Int32 tensor: {t9}")
    print(f"Float64 tensor: {t10}")


def test_math_operations():
    """Demonstrate basic mathematical operations"""
    print("\n=== Mathematical Operations ===")

    # Create input tensors
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    # Addition
    c = a + b
    print(f"Addition: {a} + {b} = {c}")

    # Subtraction
    d = a - b
    print(f"Subtraction: {a} - {b} = {d}")

    # Multiplication
    e = a * b
    print(f"Multiplication: {a} * {b} = {e}")

    # Division
    f = b / a
    print(f"Division: {b} / {a} = {f}")

    # Scalar operations
    # g = a + 5
    # print(f"Add scalar: {a} + 5 = {g}")

    # h = a * 2
    # print(f"Multiply by scalar: {a} * 2 = {h}")


def test_shape_operations():
    """Demonstrate shape manipulation operations"""
    print("\n=== Shape Operations ===")

    # Create a 2D tensor
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Original tensor: {a} with shape {a.shape}")

    # Reshape
    b = a.reshape(3, 2)
    print(f"Reshaped tensor: {b} with shape {b.shape}")

    # View (alias for reshape)
    c = a.view(6, 1)
    print(f"Viewed tensor: {c} with shape {c.shape}")

    # Transpose 2D
    d = Tensor.T(a)
    print(f"Transposed tensor: {d} with shape {d.shape}")

    # Permute dimensions (for 3D or higher)
    e = Tensor.arange(24).reshape(2, 3, 4)
    f = Tensor.permute(e, 2, 0, 1)  # Permute dimensions
    print(f"Original 3D tensor shape: {e.shape}")
    print(f"Permuted tensor shape: {f.shape}")


def test_numpy_interop():
    """Demonstrate NumPy interoperability"""
    print("\n=== NumPy Interoperability ===")

    # Create a tensor
    t = Tensor([[1, 2, 3], [4, 5, 6]])

    # Convert to NumPy array
    np_array = t.to_numpy()
    print(f"Tensor: {t}")
    print(f"NumPy array: {np_array}")
    print(f"NumPy array type: {type(np_array)}")

    # Operations with NumPy arrays
    result_np = np_array * 2

    # Convert back to tensor
    t_result = Tensor(result_np.tolist())
    print(f"NumPy result * 2: {result_np}")
    print(f"Back to tensor: {t_result}")


def test_autograd_basics():
    """Demonstrate basic autograd functionality"""
    print("\n=== Autograd Basics ===")

    # Create tensors with requires_grad=True
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    print(f"Tensor a: {a}, requires_grad={a.requires_grad}")
    print(f"Tensor b: {b}, requires_grad={b.requires_grad}")

    # Forward pass
    c = a + b
    print(f"Result c = a + b: {c}, requires_grad={c.requires_grad}")

    # The backward computation will be implemented in future versions
    print("Note: Backward pass implementation is in progress")


def main():
    """Run all examples"""
    print("pygrad Tensor Library Examples")
    print("================================")

    test_tensor_creation()
    test_math_operations()
    test_shape_operations()
    test_numpy_interop()
    test_autograd_basics()


if __name__ == "__main__":
    main()
