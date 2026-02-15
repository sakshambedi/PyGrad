# PyGrad Build System Quick Reference

This is a quick reference for the PyGrad C++ kernel build system.

## Quick Start

### 1. First Time Setup

```bash
# Install dependencies (requires vcpkg)
./build.sh --install-deps

# Build and test
./build.sh --debug --tests
```

### 2. Daily Development

```bash
# Quick development build and test (Debug)
./build.sh test

# Or using make
make dev
```

## Build Scripts

| Script       | Purpose                             | Platform         |
| ------------ | ----------------------------------- | ---------------- |
| `./build.sh` | Unified build & development utility | Unix/Linux/macOS |
| `build.bat`  | Unified build & development utility | Windows          |
| `make`       | Makefile targets                    | Cross-platform   |

## Common Commands

### Build Commands

```bash
# Standard release build
./build.sh release
make build

# Debug build
./build.sh build
make debug

# Clean build
./build.sh clean
make clean
```

### Test Commands

```bash
# Build and run tests (Debug)
./build.sh test

# Quick test (no rebuild)
make quick-test

# Python module test
make python-test
./build.sh python
```

### Development Commands

```bash
# Development workflow (debug + tests)
make dev

# Watch mode (auto-rebuild on changes)
./build.sh watch

# Check dependencies
make check
```

## Project Structure

```
pygrad/
├── kernels/                 # C++ kernel source
│   ├── cpu_kernel.h/cpp    # Python bindings
│   ├── vecbuffer.h/cpp     # SIMD vector buffer
│   └── operations.h/cpp    # Mathematical operations
├── tests/kernels/          # C++ unit tests
├── examples/               # Python usage examples
├── build/                  # Build output (generated)
├── build.sh               # Build & development script
├── build.bat              # Windows build script
├── Makefile               # Make targets
├── CMakeLists.txt         # CMake configuration
└── Agent.md               # Detailed documentation
```
