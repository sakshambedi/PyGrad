# pygrad

<p align="center">
  <strong>An educational tensor/autograd library with a high-performance C++ backend</strong>
</p>

<p align="center">
  <a href="https://github.com/sakshambedi/micrograd/releases"><img src="https://img.shields.io/badge/version-0.12.61-blue.svg" alt="Version 0.12.61"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-beta-orange.svg" alt="Beta Status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
</p>

---

> **Warning**
> This project is in **beta** (`v0.12.61`). APIs may change between releases and some functionality is intentionally incomplete.

---

## Overview

**pygrad** is a minimal, educational tensor library focused on learning how tensor storage, shape/stride views, and automatic differentiation systems work.

It provides:

- A Python-first `Tensor` API (`grad.tensor`)
- A C++ backend for core operations and buffer handling
- Python/C++ bindings via `pybind11`
- NumPy interoperability via `memoryview`/buffer conversion
- SIMD-oriented native kernels in the backend

This project is inspired by frameworks like tinygrad and PyTorch, but prioritizes **clarity and internals exploration** over full production feature parity.

---

## Current Status (What works today)

### Implemented

- Tensor creation from scalars/lists
- Factory methods:
    - `Tensor.zeros`
    - `Tensor.ones`
    - `Tensor.full`
    - `Tensor.arange`
    - `Tensor.randn`
- Shape/view operations:
    - `view`, `reshape`
    - `transpose`, `Tensor.T`
    - `Tensor.permute`
    - `expand`
- Elementwise forward ops:
    - `+`, `-`, `*`, `/`, unary negation
    - power op is wired in autograd ops module
- DType system with integer and floating dtypes
- DType upcasting for binary ops
- C++-accelerated operation dispatch for unary/binary ops
- Test suite (Python + C++ integration flow via build/test targets)

### In progress / partial

- Autograd backward coverage beyond basic ops:
    - `Add.backward`, `Sub.backward`, `Neg.backward` are implemented
    - `Mul.backward`, `Div.backward`, `Pow.backward` are currently placeholders
- Full training-oriented API surface (e.g. end-to-end `.backward()` UX)

### Not implemented yet

- Matrix multiplication (`matmul`, `@`) as public-ready feature
- Neural network layers / optimizers
- GPU backend

---

## Installation

## Prerequisites

- Python `>=3.8`
- C++17 compiler
- CMake (recommended `>=3.14`)
- `pybind11`
- NumPy (`>=1.21.6`)

Project metadata lives in `pyproject.toml` with package name `pygrad`.

---

### Quick setup (Makefile)

```bash
git clone https://github.com/sakshambedi/micrograd.git
cd micrograd

make setup-env      # install/check system prerequisites
make build          # build native extension(s)
make install        # pip install -e .
```

### Useful development commands

```bash
make debug          # debug build
make test           # build + run tests
make test-debug     # debug build + tests
make clean          # remove build artifacts
make help           # show all make targets
```

---

## Quick Start

```python
from grad.tensor import Tensor
from grad.dtype import dtypes

# Create tensors
a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32, requires_grad=True)
b = Tensor.randn(2, 3, dtype=dtypes.float32)

# Elementwise operations
c = a + b
d = a - b
e = a * b
f = a / b
g = -a

# Shape/view operations
v = a.view(3, 2)
t = a.transpose(0, 1)
p = Tensor.permute(a, 1, 0)

print("shape:", c.shape)
print("dtype:", c.dtype.name)
print("stride:", c.stride())
print("contiguous:", c.is_contigous())  # note: API spelling is currently is_contigous
```

---

## API Snapshot

### Tensor creation

- `Tensor(data, dtype=..., device="cpu", requires_grad=...)`
- `Tensor.zeros(shape, **kwargs)`
- `Tensor.ones(shape, **kwargs)`
- `Tensor.full(shape, fill_value, **kwargs)`
- `Tensor.arange(end, start=0, step=1, dtype=..., device="cpu", requires_grad=False)`
- `Tensor.randn(*shape, **kwargs)`

### Shape/view ops

- `view(*shape)`
- `reshape(*shape)` (alias of `view`)
- `transpose(dim0, dim1)`
- `Tensor.T(tensor)` (2D convenience transpose)
- `Tensor.permute(tensor, *dims)`
- `expand(*shape)`

### Reductions

- `Tensor.sum(tensor, dtype=...)` (current scalar reduction behavior)

### Arithmetic operators

- `+`, `-`, `*`, `/`, unary `-`, power (`**`) path exists through autograd op wiring

---

## Data Types

`grad.dtype.dtypes` includes integer and floating families, including aliases like `fp16/fp32/fp64`.

Examples:

- Integers: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`
- Floats: `float16`, `float32`, `float64`

---

## Project Structure

```text
micrograd/
├── grad/
│   ├── tensor.py              # Core Tensor API
│   ├── dtype.py               # DType definitions
│   ├── buffer.py              # Buffer abstraction
│   ├── device.py              # Device abstraction (CPU-oriented currently)
│   ├── autograd/
│   │   ├── function.py        # Function base class
│   │   ├── ops.py             # Python autograd ops wrappers
│   │   └── operations*.so     # Native extension bindings
│   └── kernels/
├── kernels/
│   ├── cpu_kernel.cpp/.h
│   ├── vecbuffer.cpp/.h
│   └── operations.cpp/.h
├── tests/
│   ├── tensor_test.py
│   ├── ops_test.py
│   └── test_buffer.py
├── pyproject.toml
├── setup.py
└── makefile
```

---

## Testing

```bash
# Run full test flow
make test

# Run Python tests directly
pytest -v

# Run a single test file
pytest tests/tensor_test.py -v
```

---

## Roadmap (short-term)

- Complete backward implementations for `Mul`, `Div`, and `Pow`
- Improve gradient propagation UX for practical training loops
- Add matrix multiplication support
- Expand reductions and tensor composition ops

---

## Contributing

Contributions are welcome, especially for:

- Autograd correctness and gradient tests
- Edge-case tensor shape/stride tests
- Documentation improvements and educational walkthroughs
- Backend operation coverage and performance-safe refactors

---

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2025 Saksham Bedi
