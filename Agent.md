# PyGrad C++ Kernel Development Guide

This document provides comprehensive guidance for developing and compiling the C++ kernel components of PyGrad.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Build System](#build-system)
4. [Compilation Methods](#compilation-methods)
5. [Development Workflow](#development-workflow)
6. [Adding New Features](#adding-new-features)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

## Project Structure

The C++ kernel consists of three main components:

```markdown
kernels/
├── cpu_kernel.h/cpp # Python bindings and Buffer class
├── vecbuffer.h/cpp # SIMD-optimized vector buffer
├── operations.h/cpp # Mathematical operations (add, sub, mul, div)
└── cpu_kernel.h # Header with DType enum and Buffer interface
```

### Key Components

- **Buffer Class**: Python-compatible buffer with multiple data type support
- **VecBuffer**: SIMD-optimized vector operations using xsimd
- **Operations**: Mathematical operations with SIMD acceleration
- **Python Bindings**: pybind11-based Python interface

## Dependencies

### Required Dependencies

1. **CMake** (3.14+)
2. **C++17 compatible compiler** (GCC 7+, Clang 5+, MSVC 2017+)
3. **Python 3.6+** with development headers
4. **vcpkg** (for dependency management)

### External Libraries (managed by vcpkg)

- **Eigen 3.4.0**: Linear algebra operations
- **xsimd**: SIMD operations
- **pybind11**: Python bindings
- **GoogleTest**: Unit testing

### Installing Dependencies

#### Using vcpkg (Recommended)

```bash
# Install vcpkg first (if not already installed)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # On Windows: bootstrap-vcpkg.bat

# Install required packages
vcpkg install eigen3 xsimd pybind11
```

#### Manual Installation

If you prefer manual installation, ensure these libraries are available:

- Eigen headers in your include path
- xsimd headers in your include path
- pybind11 headers in your include path

## Build System

The project uses CMake as the primary build system with multiple build scripts for convenience.

### Build Scripts

1. **`build.sh`** (Unix/Linux/macOS)
2. **`build.bat`** (Windows)
3. **`Makefile`** (Cross-platform with make)

### CMake Configuration

The main `CMakeLists.txt` handles:

- Dependency detection and configuration
- SIMD optimization flags
- Python module generation
- Test suite setup

## Compilation Methods

### Method 1: Using Build Scripts (Recommended)

#### Unix/Linux/macOS

```bash
# Standard release build
./build.sh

# Debug build with tests
./build.sh --debug --tests

# Clean build with verbose output
./build.sh --clean --verbose

# Install dependencies and build
./build.sh --install-deps --tests
```

#### Windows

```cmd
# Standard release build
build.bat

# Debug build with tests
build.bat --debug --tests

# Clean build with verbose output
build.bat --clean --verbose
```

### Method 2: Using Makefile

```bash
# Show all available targets
make help

# Build in release mode
make build

# Build in debug mode
make debug

# Build and run tests
make test

# Development workflow (debug + tests)
make dev

# Check dependencies
make check

# Format code (requires clang-format)
make format

# Lint code (requires cpplint)
make lint

# Quick test without rebuilding
make quick-test

# Show build artifacts
make artifacts

# Install the Python module
make install

# Uninstall the Python module
make uninstall

# Test Python module integration
make python-test
```

### Method 3: Manual CMake

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) ..

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Development Workflow

### 1. Setting Up Development Environment

```bash
# Clone and setup
git clone <repository>
cd pygrad

# Install dependencies
./build.sh --install-deps

# Initial build
./build.sh --debug --tests
```

### 2. Daily Development Cycle

```bash
# Quick development build and test
make dev

# Or using build script
./build.sh --debug --tests
```

### 3. Adding New Features

1. **Modify C++ code** in `kernels/` directory
2. **Update tests** in `tests/kernels/` directory
3. **Build and test**:

    ```bash
    make dev
    ```

4. **Update Python bindings** if needed in `cpu_kernel.cpp`

## Adding New Features

### Adding New Operations

1. **Define operation in `operations.h`**:

    ```cpp
    template <typename T>
    void new_operation(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept;
    ```

2. **Implement in `operations.cpp`**:

    ```cpp
    template <typename T>
    void new_operation(const T *lhs, const T *rhs, T *out, std::size_t n) noexcept {
        binary_kernel<T, NewOp>(lhs, rhs, out, n);
    }
    ```

3. **Add template instantiations**:

    ```cpp
    template void new_operation<float>(const float *, const float *, float *, std::size_t) noexcept;
    // ... for other types
    ```

4. **Add Python binding** in `cpu_kernel.cpp`:

    ```cpp
    .def("new_operation", &buffer_new_operation)
    ```

### Adding New Data Types

1. **Update `DType` enum** in `cpu_kernel.h`
2. **Add to `str_to_dtype` map** in `cpu_kernel.cpp`
3. **Update `BufferVariant`** to include new VecBuffer type
4. **Add to `init()` method** in `cpu_kernel.cpp`
5. **Update template instantiations** in `operations.cpp`

### Adding New SIMD Operations

1. **Define operation struct** in `operations.cpp`:

    ```cpp
    struct NewOp {
        template <typename T> static constexpr T apply_scalar(T a, T b) noexcept {
            return /* operation */;
        }

        template <typename Batch>
        static constexpr Batch apply_simd(const Batch &a, const Batch &b) noexcept {
            return /* SIMD operation */;
        }
    };
    ```

2. **Use in binary_kernel**:

    ```cpp
    binary_kernel<T, NewOp>(lhs, rhs, out, n);
    ```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run tests with debug build
make test-debug

# Quick test (no rebuild)
make quick-test

# Python module test
make python-test
```

The Python module test will:

1. Check that the CPU kernel module can be imported
2. Verify that a basic Buffer object can be created
3. Test functionality with a sample operation

### Test Structure

- **`tests/kernels/test_cpu_kernel.cpp`**: Buffer class tests
- **`tests/kernels/test_binary_operations.cpp`**: Mathematical operations
- **`tests/kernels/test_vecbuffer.cpp`**: VecBuffer functionality
- **`tests/kernels/test_dtype_enum.cpp`**: Data type handling

### Writing New Tests

1. **Create test file** in `tests/kernels/`
2. **Add to CMakeLists.txt** in `tests/kernels/CMakeLists.txt`
3. **Follow GoogleTest conventions**:

    ```cpp
    #include <gtest/gtest.h>

    TEST(BufferTest, NewFeature) {
        // Test implementation
    }
    ```

## Troubleshooting

### Common Issues

#### 1. CMake Configuration Fails

**Problem**: CMake can't find dependencies
**Solution**:

```bash
# Set vcpkg toolchain
export CMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

# Or install dependencies
./build.sh --install-deps
```

#### 2. Python Module Import Error

**Problem**: `ImportError: No module named 'cpu_kernel'`
**Solution**:

```bash
# Rebuild the module
make clean && make build

# Check if module exists
ls build/*.so build/*.dylib build/*.dll

# Install the module
make install

# Test the module
make python-test
```

#### 3. SIMD Compilation Errors

**Problem**: xsimd-related compilation errors
**Solution**:

```bash
# Check xsimd installation
vcpkg list | grep xsimd

# Reinstall if needed
vcpkg remove xsimd && vcpkg install xsimd
```

#### 4. Performance Issues

**Problem**: SIMD operations not working optimally
**Solution**:

```bash
# Check compiler flags
make VERBOSE=1

# Ensure proper alignment
# Check VecBuffer alignment in vecbuffer.h
```

### Debugging Tips

1. **Use debug build** for development:

    ```bash
    make debug
    ```

2. **Enable verbose output**:

    ```bash
    ./build.sh --verbose
    ```

3. **Check build artifacts**:

    ```bash
    make artifacts
    ```

4. **Test Python integration**:

    ```bash
    make python-test
    ```

5. **Check environment setup**:

    ```bash
    make check
    ```

6. **Use compile_commands.json** (generated with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`):
    - Import in VSCode, CLion, or other IDEs
    - Use with clangd for better code navigation and diagnostics

### Performance Optimization

1. **Use Release builds** for performance testing
2. **Check SIMD alignment** in VecBuffer
3. **Profile with tools** like `perf` or `gprof`
4. **Monitor memory alignment** for optimal SIMD performance

## Best Practices

### Code Style

1. **Follow existing patterns** in the codebase
2. **Use const correctness** where appropriate
3. **Add noexcept** to functions that don't throw
4. **Use template specialization** for type-specific optimizations
5. **Format code consistently** using:

    ```bash
    make format  # Uses clang-format
    ```

6. **Check code quality** with:

    ```bash
    make lint  # Uses cpplint
    ```

### Performance

1. **Use SIMD operations** for bulk data processing
2. **Ensure proper memory alignment** for SIMD
3. **Avoid unnecessary copies** in hot paths
4. **Profile before optimizing**

### Testing

1. **Write tests for new features**
2. **Test edge cases** (empty buffers, different dtypes)
3. **Test performance** with realistic data sizes
4. **Test Python bindings** thoroughly

## Integration with Python

### Using the C++ Kernel in Python

```python
import cpu_kernel

# Create buffers
a = cpu_kernel.Buffer([1, 2, 3], "float32")
b = cpu_kernel.Buffer([4, 5, 6], "float32")

# Perform operations (when implemented)
# result = cpu_kernel.add(a, b, 'float32')
```

### Installation in Python Projects

To install the compiled module in your Python environment:

```bash
# Install in development mode
make install

# Uninstall if needed
make uninstall
```

The installation process copies the compiled library to the `grad/kernels/` directory and uses pip's editable install feature (`pip install -e .`).

### Extending Python Interface

1. **Add methods to Buffer class** in `cpu_kernel.cpp`
2. **Use pybind11 decorators** for proper Python integration
3. **Handle Python exceptions** appropriately
4. **Test with Python test suite**

## Conclusion

This build system provides a robust foundation for developing high-performance C++ kernels with Python integration. The modular design allows for easy extension and maintenance while maintaining optimal performance through SIMD optimizations.

The makefile provides comprehensive targets for all aspects of development:

- Building and compiling (`build`, `debug`, `release`)
- Testing (`test`, `test-debug`, `quick-test`, `python-test`)
- Code quality (`format`, `lint`)
- Dependency management (`install-deps`, `check`)
- Python integration (`install`, `uninstall`, `python-test`)

Platform-specific details are automatically handled, with support for Linux, macOS, and Windows environments.

For questions or issues, refer to the troubleshooting section or create an issue in the project repository.
