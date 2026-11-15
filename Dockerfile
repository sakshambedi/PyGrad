# Builds the native extension on Linux and runs both test suites during the
# image build, so `docker build .` fails if any test fails.
#
#   docker build -t pygrad .
#   docker run --rm pygrad            # re-runs the Python tests
#
# Python 3.11 matches the version CI tests.
FROM python:3.11-slim

# Compiler toolchain, CMake, and the header-only libraries the kernels include.
# git is needed because CMake fetches pybind11 and GoogleTest at configure time.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libeigen3-dev \
        libxsimd-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Runtime dependency (numpy), build tooling, and the test dependencies.
RUN pip install --no-cache-dir numpy setuptools wheel pybind11 "pytest>=7.4.4"

# Build the two pybind11 extension modules in place (same command as CI).
RUN python setup.py build_ext --inplace

# Python tests. tests/test_torch_parity.py is skipped: PyTorch is not installed.
RUN pytest -q

# C++ GoogleTest suite.
RUN cmake -S . -B build/ctest \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYGRAD_BUILD_KERNEL_TESTS=ON \
        -Wno-dev && \
    cmake --build build/ctest --parallel && \
    ctest --test-dir build/ctest --output-on-failure --timeout 120

CMD ["pytest", "-q"]
