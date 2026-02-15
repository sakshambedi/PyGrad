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

RUN chmod +x  /app/scripts/setup_prerequisite.sh && /app/scripts/setup_prerequisite.sh

RUN if [ "$(uname -m)" = "aarch64" ]; then TRIPLET=arm64-linux; else TRIPLET=x64-linux; fi && \
    ln -s /usr/local/vcpkg/installed/$TRIPLET/include /usr/local/vcpkg/installed/current

ENV VCPKG_ROOT=/usr/local/vcpkg \
    XSIMD_INCLUDE_DIR=/usr/local/vcpkg/installed/current \
    EIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    CMAKE_TOOLCHAIN_FILE=/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake

RUN pip install --upgrade pip
RUN pip install --no-cache-dir ".[dev]"
RUN rm -rf build # && make test
