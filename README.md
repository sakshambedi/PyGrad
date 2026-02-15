# pygrad

<p align="center">
  <img src="./imgs/micrograd-banner-m.jpeg" alt="pygrad" width="600"/>
</p>

<p align="center">
  <strong>An Educational Deep Learning Framework with High-Performance C++ Backend</strong>
</p>

<p align="center">
  <a href="https://github.com/sakshambedi/micrograd/releases"><img src="https://img.shields.io/badge/version-0.12.61-blue.svg" alt="Version 0.12.61"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-beta-orange.svg" alt="Beta Status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python 3.7+"></a>
  <a href="https://github.com/sakshambedi/micrograd"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platform"></a>
</p>

---

> **Warning**
> This project is in **beta** (v0.12.61). APIs may change between releases. Not recommended for production use.
>
> **Missing features:** matrix multiplication, broadcasting, neural network layers, GPU support.

---

## Overview

**pygrad** is an educational tensor library that implements automatic differentiation from scratch. Inspired by [tinygrad](https://github.com/geohot/tinygrad) and [PyTorch](https://pytorch.org/), this project provides a hands-on way to understand how modern deep learning frameworks work internally.

The library features a Python frontend with a high-performance C++ backend using [pybind11](https://github.com/pybind/pybind11), [Eigen](https://eigen.tuxfamily.org/), and [xsimd](https://github.com/xtensor-stack/xsimd) for SIMD-optimized operations. It's designed for learners, educators, and anyone curious about the internals of tensor computation and autograd systems.

## Quick Start

### Installation

```bash
git clone https://github.com/sakshambedi/micrograd.git
cd micrograd

make setup-env  # Install prerequisites (including xsimd headers)
make build      # Build C++ extensions
make install    # Install Python package
```

### Hello World

```python
from grad.tensor import Tensor
from grad.dtype import dtypes

# Create tensors
a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, requires_grad=True)
b = Tensor.randn(2, 3)
c = Tensor.zeros((2, 3))

# Operations (with autograd support)
result = a + b
product = a * b
power = a ** 2

# Shape manipulation
reshaped = a.view(3, 2)
transposed = Tensor.T(a)

# NumPy interoperability (zero-copy when contiguous)
numpy_array = result.to_numpy()

print(f"Shape: {result.shape}, dtype: {result.dtype.name}")
```

## Features

### Tensor Operations

| Feature             | Status   | Description                                                                           |
| ------------------- | -------- | ------------------------------------------------------------------------------------- |
| Multi-dtype support | Complete | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `float16`, `float32`, `float64` |
| Factory methods     | Complete | `zeros`, `ones`, `full`, `arange`, `randn`                                            |
| Shape operations    | Complete | `view`, `reshape`, `transpose`, `permute`, `expand`                                   |
| Indexing            | Complete | Multi-dimensional indexing with negative indices, assignment                          |
| NumPy integration   | Complete | Zero-copy buffer protocol via `to_numpy()`                                            |

### Automatic Differentiation

| Feature              | Status   | Description                                   |
| -------------------- | -------- | --------------------------------------------- |
| Forward pass         | Complete | All operations support forward computation    |
| Gradient tracking    | Complete | `requires_grad`, `grad`, `grad_fn` attributes |
| Add/Sub backward     | Complete | Full gradient computation                     |
| Mul/Div/Pow backward | Partial  | Forward pass complete, backward in progress   |

### C++ Backend

| Feature          | Status   | Description                        |
| ---------------- | -------- | ---------------------------------- |
| SIMD operations  | Complete | AVX2/FMA (x86), NEON (ARM)         |
| Loop unrolling   | Complete | 4x unrolled SIMD kernels           |
| Type casting     | Complete | Automatic dtype upcasting          |
| Memory alignment | Complete | Aligned loads/stores when possible |

## Installation

### Prerequisites

| Dependency   | Version     | Purpose                |
| ------------ | ----------- | ---------------------- |
| Python       | >= 3.7      | Runtime                |
| NumPy        | >= 1.21.6   | Array interoperability |
| CMake        | >= 3.14     | Build system           |
| C++ Compiler | C++17       | Native extensions      |
| pybind11     | >= 2.11     | Python/C++ bindings    |
| xsimd        | header-only | SIMD acceleration      |

Eigen is fetched automatically during build.
xsimd must be available as headers on your system (recommended: run `make setup-env`, which installs it via vcpkg).

### Platform-Specific Setup

#### macOS (Intel & Apple Silicon)

```bash
# Install Xcode command line tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Clone and build
git clone https://github.com/sakshambedi/micrograd.git
cd micrograd
make setup-env
make build
```

#### Linux (Ubuntu/Debian)

```bash
# Install build dependencies
sudo apt update
sudo apt install cmake build-essential python3-dev

# Clone and build
git clone https://github.com/sakshambedi/micrograd.git
cd micrograd
make setup-env
make build
```

#### Windows

```bash
# Requires Visual Studio Build Tools with C++ support
# Install CMake from https://cmake.org/download/

git clone https://github.com/sakshambedi/micrograd.git
cd micrograd
python build.py
```

### Build Commands

| Command          | Description                 |
| ---------------- | --------------------------- |
| `make setup-env` | Install all prerequisites   |
| `make build`     | Release build (no tests)    |
| `make debug`     | Debug build with symbols    |
| `make test`      | Build and run C++ tests     |
| `make install`   | Install Python package      |
| `make clean`     | Remove build artifacts      |
| `make help`      | Show all available commands |

## API Reference

### Tensor Creation

```python
from grad.tensor import Tensor
from grad.dtype import dtypes

# From data
Tensor([1, 2, 3])                              # 1D tensor
Tensor([[1, 2], [3, 4]])                       # 2D tensor
Tensor(5)                                       # Scalar tensor

# Factory methods
Tensor.zeros((2, 3))                           # All zeros
Tensor.ones((2, 3))                            # All ones
Tensor.full((2, 3), 7)                         # Fill with value
Tensor.arange(10)                              # Range [0, 10)
Tensor.arange(10, start=5, step=2)             # Range [5, 10) step 2
Tensor.randn(2, 3)                             # Random normal distribution

# With options
Tensor([1, 2, 3], dtype=dtypes.float64, requires_grad=True)
```

### Data Types

```python
from grad.dtype import dtypes

# Integer types
dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64
dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64

# Floating-point types
dtypes.float16, dtypes.float32, dtypes.float64

# Aliases
dtypes.fp16, dtypes.fp32, dtypes.fp64, dtypes.double
```

### Operations

```python
# Arithmetic (with autograd support)
c = a + b          # Addition
c = a - b          # Subtraction
c = a * b          # Multiplication
c = a / b          # Division
c = a ** 2         # Power
c = -a             # Negation

# Reduction
s = Tensor.sum(t)  # Sum all elements
```

### Shape Manipulation

```python
t.view(2, 3)                      # Reshape (shares storage)
t.reshape(2, 3)                   # Alias for view
t.transpose(0, 1)                 # Swap dimensions
Tensor.T(t)                       # 2D transpose
Tensor.permute(t, 2, 0, 1)        # Arbitrary dimension reorder
t.expand(4, 3, 2)                 # Broadcast to larger shape
```

### Tensor Properties

```python
t.shape          # Tuple of dimensions
t.dtype          # DType object
t.device         # "cpu" (GPU planned)
t.requires_grad  # Gradient tracking enabled
t.grad           # Gradient tensor (after backward)
t.grad_fn        # Computation graph node
t.stride()       # Memory strides
t.is_contigous() # Contiguity check
```

### NumPy Interoperability

```python
# Tensor to NumPy (zero-copy when contiguous)
arr = t.to_numpy()

# NumPy to Tensor
t = Tensor(np_array.tolist())
```

## Architecture

```
pygrad/
├── grad/                         # Python package
│   ├── tensor.py                # Core Tensor class
│   ├── dtype.py                 # Data type definitions
│   ├── buffer.py                # Python buffer wrapper
│   ├── device.py                # Device abstraction
│   ├── autograd/
│   │   ├── function.py          # Base Function class
│   │   └── ops.py               # Add, Sub, Mul, Div, Pow, Neg
│   └── kernels/                 # Compiled C++ modules
├── kernels/                      # C++ source
│   ├── cpu_kernel.cpp/.h        # Buffer implementation
│   ├── vecbuffer.cpp/.h         # SIMD-optimized storage
│   └── operations.cpp/.h        # Binary/unary operations
├── tests/                        # Test suite
│   ├── tensor_test.py           # Tensor functionality
│   ├── ops_test.py              # Operation tests
│   └── test_buffer.py           # Buffer tests
├── examples/                     # Usage examples
├── CMakeLists.txt               # CMake configuration
└── setup.py                     # Python packaging
```

### Design Principles

1. **Python-First API**: Familiar PyTorch-like interface
2. **C++ Performance**: Hot paths in optimized native code
3. **SIMD Everywhere**: xsimd for portable vectorization
4. **Zero-Copy Design**: Buffer sharing between Python and C++
5. **Educational Clarity**: Code prioritizes readability over micro-optimizations

## Performance

### SIMD Optimizations

The C++ backend uses xsimd for portable SIMD operations:

- **x86_64**: AVX2 + FMA instructions
- **ARM64**: NEON instructions (Apple Silicon optimized)
- **Fallback**: Scalar operations for unsupported platforms

### Optimization Techniques

- 4x loop unrolling for SIMD operations
- Aligned memory access when possible
- Compile-time dispatch based on alignment
- Template-based type specialization

### Comparison with Established Libraries

| Capability       | pygrad  | NumPy | PyTorch  |
| ---------------- | ------- | ----- | -------- |
| Element-wise ops | SIMD    | SIMD  | SIMD/GPU |
| Broadcasting     | Planned | Full  | Full     |
| Matrix multiply  | Planned | BLAS  | cuBLAS   |
| Autograd         | Partial | No    | Full     |
| GPU support      | Planned | No    | Full     |

> **Note**: pygrad prioritizes educational value over raw performance. For production workloads, use PyTorch, JAX, or TensorFlow.

## Testing

### Running Tests

```bash
# Build and run C++ tests
make test

# Run Python tests
pytest -v

# Run both C++ and Python tests
make test && pytest -v

# Specific test file
pytest tests/tensor_test.py -v

# C++ tests only
cd build && ctest --output-on-failure
```

### Test Coverage

The project includes **500+ unit tests** covering:

- Tensor creation and manipulation
- All arithmetic operations
- Shape operations (view, transpose, permute)
- Data type handling and casting
- Buffer management and sharing
- NumPy equivalence validation

## Roadmap

### v0.13 (Next Release)

- [ ] Broadcasting support
- [ ] Complete backward pass for Mul, Div, Pow
- [ ] `backward()` method on Tensor

### v0.14

- [ ] Matrix multiplication (`@` operator, `matmul`)
- [ ] Reduction operations (`mean`, `max`, `min`)
- [ ] Concatenation and stacking

### v1.0 (Future)

- [ ] Neural network layers (Linear, Conv2d)
- [ ] Activation functions (ReLU, Sigmoid, Tanh, GELU)
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Optimizers (SGD, Adam)
- [ ] Model serialization
- [ ] GPU support (CUDA/Metal)

### Non-Goals

- Production-ready performance benchmarks
- Full PyTorch API compatibility
- Distributed training support

## Contributing

This is primarily an educational project. Contributions are welcome!

### Ways to Contribute

1. **Report Bugs**: Open an issue with reproduction steps
2. **Suggest Features**: Discuss in GitHub Issues first
3. **Add Tests**: Edge cases and NumPy equivalence tests
4. **Improve Docs**: Clarifications and examples
5. **Educational Content**: Tutorials explaining internals

### Development Setup

```bash
git clone https://github.com/sakshambedi/micrograd.git
cd micrograd
make setup-env
make debug    # Build with debug symbols
pytest -v     # Verify tests pass
```

### Code Style

- **Python**: PEP 8 (enforced via flake8)
- **C++**: Google C++ Style Guide (enforced via cpplint)
- **Commits**: Conventional Commits format

## Acknowledgments

pygrad is inspired by:

- [tinygrad](https://github.com/geohot/tinygrad) by George Hotz
- [PyTorch](https://pytorch.org/) by Meta AI
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy

### Dependencies

- [Eigen](https://eigen.tuxfamily.org/) - Linear algebra
- [pybind11](https://github.com/pybind/pybind11) - Python/C++ bindings
- [xsimd](https://github.com/xtensor-stack/xsimd) - SIMD abstraction
- [GoogleTest](https://github.com/google/googletest) - C++ testing

## License

MIT License - See [LICENSE](LICENSE) for details.

Copyright (c) 2025 Saksham Bedi

---

<p align="center">
  <em>Built for learning</em> | <a href="https://github.com/sakshambedi/micrograd">GitHub</a>
</p>
