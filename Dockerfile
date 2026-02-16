FROM python:3.13-slim

# Install minimal tooling for setup script
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    sudo \
    bash \
    curl \
    zip \
    unzip \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code and helper scripts
COPY . /app

# Ensure the prerequisite installer is executable and run it
RUN chmod +x /app/scripts/setup_prerequisite.sh && \
    /app/scripts/setup_prerequisite.sh

# Expose vcpkg and include paths for downstream builds
ENV VCPKG_ROOT=/usr/local/vcpkg \
    XSIMD_INCLUDE_DIR=/usr/local/vcpkg/installed/x64-linux/include \
    EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    CMAKE_TOOLCHAIN_FILE=/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake

# Install Python deps, build & install the package, and run C++ tests
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf build && chmod +x run_cpp_tests.sh
RUN ./run_cpp_tests.sh

# Default command: run Python tests
# CMD ["pytest", "-v"]
