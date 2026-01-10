# Micrograd - Current State

*Last updated: January 2025*

This document provides a comprehensive overview of the micrograd ML library's current implementation status, architecture, and roadmap considerations.

## Architecture Overview

```
Python Layer (grad/)          C++ Layer (kernels/)
├── tensor.py (484 lines)     ├── operations.cpp (SIMD kernels)
├── buffer.py (wraps C++)     ├── cpu_kernel.cpp (buffer mgmt)
├── dtype.py (11 types)       ├── vecbuffer.cpp (Eigen-aligned)
└── autograd/                 └── pybind11 bindings
    ├── function.py (base)
    └── ops.py (operations)
```

### Design Principles

- **Clean separation**: Python handles metadata (shapes, strides, autograd graph), C++ handles computation
- **Type-safe storage**: Variant-based buffer supporting 11 data types
- **SIMD-first**: All operations use xsimd for portable vectorization
- **Zero-copy views**: Tensor views share underlying buffer storage
- **Cross-platform**: Supports Linux, macOS (Intel + Apple Silicon), Windows

---

## Implementation Status

### Core Components

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Tensor** | Complete | `grad/tensor.py` | Full n-dimensional array with strides |
| **Buffer** | Complete | `grad/buffer.py` | C++ backend wrapper |
| **DType** | Complete | `grad/dtype.py` | 11 types with upcasting |
| **Device** | Stub | `grad/device.py` | Placeholder for GPU support |
| **Function** | Complete | `grad/autograd/function.py` | Base class for autograd |
| **Operations** | Partial | `grad/autograd/ops.py` | See operation table below |

### Tensor Operations

#### Creation Methods

| Method | Status | Description |
|--------|--------|-------------|
| `Tensor(data)` | Complete | From Python list or scalar |
| `Tensor.zeros()` | Complete | Zero-filled tensor |
| `Tensor.ones()` | Complete | One-filled tensor |
| `Tensor.arange()` | Complete | Range tensor |
| `Tensor.randn()` | Complete | Random normal distribution |
| `Tensor.full()` | Complete | Fill with specific value |

#### Shape Operations

| Method | Status | Description |
|--------|--------|-------------|
| `view()` | Complete | Reshape without copy |
| `reshape()` | Complete | Reshape with optional copy |
| `transpose()` | Complete | Swap two dimensions |
| `permute()` | Complete | Arbitrary dimension reordering |
| `expand()` | Complete | Broadcast to larger shape |

#### Indexing

| Operation | Status | Description |
|-----------|--------|-------------|
| `__getitem__` | Complete | Multi-dimensional indexing |
| `__setitem__` | Complete | Element/slice assignment |

### Arithmetic Operations

| Operation | Forward | Backward | SIMD | Notes |
|-----------|---------|----------|------|-------|
| Add | Complete | Complete | Yes | Full autograd support |
| Sub | Complete | Complete | Yes | Full autograd support |
| Neg | Complete | Complete | Yes | Full autograd support |
| Mul | Complete | **Missing** | Yes | Forward only |
| Div | Complete | **Missing** | Yes | Forward only |
| Pow | Complete | **Missing** | Yes | Forward only |

### Reduction Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `sum()` | Partial | Basic sum, missing `axis`/`keepdim` parameters |

---

## C++ Kernel Architecture

### SIMD Implementation (`kernels/operations.cpp`)

The kernel uses a hybrid SIMD approach:

```cpp
// Unrolled SIMD loop (4x batching for better vectorization)
for (size_t i = 0; i + 4 * batch_size <= n; i += 4 * batch_size) {
    // Process 4 SIMD batches per iteration
    auto batch1 = xsimd::load_aligned(&a[i]);
    auto batch2 = xsimd::load_aligned(&a[i + batch_size]);
    // ... apply operation and store
}

// Scalar fallback for remainder
for (size_t i = simd_end; i < n; ++i) {
    result[i] = op(a[i], b[i]);
}
```

**Performance characteristics:**
- SIMD threshold: operations < 16 elements use scalar path
- Alignment-aware: separate aligned/unaligned code paths
- Loop unrolling: 4x batches per iteration
- Platform-specific: AVX2/FMA on x86, NEON on ARM

### Buffer Management (`kernels/cpu_kernel.cpp`)

- **VecBuffer<T>**: Eigen-aligned vector container
- **Variant storage**: Runtime type erasure for multiple dtypes
- **NumPy protocol**: `__array_interface__` for zero-copy interop

---

## Data Types

| Type | Size | Priority | Format |
|------|------|----------|--------|
| int8 | 1 | 1 | 'b' |
| uint8 | 1 | 2 | 'B' |
| int16 | 2 | 3 | 'h' |
| uint16 | 2 | 4 | 'H' |
| int32 | 4 | 5 | 'i' |
| uint32 | 4 | 6 | 'I' |
| int64 | 8 | 7 | 'q' |
| uint64 | 8 | 8 | 'Q' |
| float16 | 2 | 9 | 'e' |
| **float32** | 4 | 10 | 'f' |
| float64 | 8 | 11 | 'd' |

*float32 is the default dtype*

**Upcasting rules**: Operations between different types promote to the higher priority type.

---

## Test Coverage

### C++ Tests (GoogleTest)

| Test File | Coverage |
|-----------|----------|
| `test_cpu_kernel.cpp` | Buffer initialization, indexing, dtype conversion |
| `test_operations.cpp` | Binary/unary SIMD operations |
| `test_vecbuffer.cpp` | Container behavior, alignment |
| `test_dtype_enum.cpp` | Type system |

### Python Tests (pytest)

| Test File | Coverage | Test Count |
|-----------|----------|------------|
| `tensor_test.py` | Creation, shapes, indexing, utilities | ~405 cases |
| `ops_test.py` | All arithmetic operations, gradients | ~100+ cases |
| `test_buffer.py` | Buffer creation, access, types | ~80+ cases |

**Skipped tests**: Sum operations, full transpose support

---

## Build System

### Dependencies (via CMake FetchContent)

| Dependency | Version | Purpose |
|------------|---------|---------|
| Eigen | 3.4.0 | Linear algebra, alignment utilities |
| pybind11 | 2.11.1 | Python bindings |
| GoogleTest | 1.14.0 | C++ testing |
| xsimd | latest | SIMD operations |

### Platform-Specific Flags

| Platform | Flags |
|----------|-------|
| x86_64 | `-mavx2 -mfma -msse4.2 -O3` |
| Apple Silicon | `-mcpu=apple-m1 -O3` |
| Windows (MSVC) | `/arch:AVX2 /O2` |

### Make Targets

```bash
make setup-env    # Install prerequisites
make build        # Release build
make debug        # Debug build
make test         # Build + run all tests
make clean        # Clean build directory
```

---

## What's Missing for Complete ML Library

### Critical (Required for Training)

- [ ] **Matrix multiplication** (`matmul`) - Essential for neural networks
- [ ] **Backward passes** for Mul, Div, Pow
- [ ] **Sum with axis/keepdim** - Needed for loss computation
- [ ] **Broadcasting rules** - Framework exists, not complete

### Activation Functions

- [ ] ReLU
- [ ] Sigmoid
- [ ] Tanh
- [ ] GELU
- [ ] Softmax

### Loss Functions

- [ ] MSE (Mean Squared Error)
- [ ] CrossEntropyLoss
- [ ] BCELoss

### Training Infrastructure

- [ ] Optimizers (SGD, Adam, AdamW)
- [ ] Learning rate schedulers
- [ ] Layer abstractions (Linear, Conv2d, BatchNorm)
- [ ] Model serialization (save/load)

---

## GPU Integration Roadmap

### Current Architecture (CPU Only)

```
Tensor → Buffer → VecBuffer<T> → CPU SIMD (xsimd)
```

### Proposed Multi-Backend Architecture

```
Tensor → Buffer → dispatch() → CPU (xsimd)
                            → MLX (Metal)
                            → CUDA
```

### Integration Points

| File | Changes Needed |
|------|----------------|
| `grad/device.py` | Implement actual device dispatch |
| `grad/buffer.py:17-30` | Backend-specific buffer creation |
| `CMakeLists.txt` | Conditional GPU library compilation |

### Proposed Directory Structure

```
kernels/
├── cpu_kernel.cpp      # Existing
├── operations.cpp      # Existing - template for GPU kernels
├── mlx_kernel.mm       # New - Objective-C++ for Metal
├── cuda_kernel.cu      # New - CUDA
└── dispatch.cpp        # New - Runtime backend selection
```

### MLX-Specific Considerations

- Requires Objective-C++ (`.mm` files)
- Metal shaders for compute kernels
- Apple Silicon only
- Can leverage MLX's existing tensor operations

### CUDA-Specific Considerations

- NVCC compiler integration
- Shared memory optimization for reductions
- cuBLAS for matmul operations
- Memory management (cudaMalloc/cudaFree)

---

## Priority Recommendations

### Phase 1: Complete Autograd

1. Implement backward passes for Mul, Div, Pow
2. Add `sum()` with axis parameter
3. Test gradient flow through computation graph

### Phase 2: Basic Neural Network Support

1. Implement `matmul` operation
2. Add ReLU and Sigmoid activations
3. Implement MSE loss
4. Create basic SGD optimizer

### Phase 3: GPU Acceleration

1. Abstract device dispatch in buffer layer
2. Implement MLX backend (Apple Silicon)
3. Implement CUDA backend
4. Add memory transfer utilities (to/from device)

---

## Code Quality Notes

### Strengths

- Clean Python/C++ separation
- Type-safe variant storage
- Comprehensive test coverage for implemented features
- SIMD-aware with alignment optimization
- Cross-platform support

### Design Patterns Used

- **Variant pattern**: Type erasure for buffer storage
- **Template metaprogramming**: SIMD kernel dispatch
- **Function-based autograd**: PyTorch-style with context saving
- **Metaclass caching**: DType singleton pattern
- **Slot-based classes**: Memory-efficient tensor attributes

---

## Recent Development

| Commit | Description |
|--------|-------------|
| c8d2dc7 | Add Buffer `__setitem__` support |
| 8821860 | Merge Python-C integration phase 2 |
| 970d72d | Optimize binary kernels with unrolled SIMD |
| 3bb7b82 | Refactor elementwise ops, add `expand()` |
| e517dca | Implement power and negation ops |

**Active development areas**: Python-C integration, SIMD optimization, operation implementations
