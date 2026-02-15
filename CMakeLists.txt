cmake_minimum_required(VERSION 3.14)
project(pygrad)

set(CMAKE_CXX_STANDARD 17)

# Enable SIMD optimizations while maintaining IEEE 754 compliance
# Set common flags for both Release and Debug modes to ensure consistent behavior
set(COMMON_FLAGS "-fno-math-errno -freciprocal-math -fno-fast-math -ffp-contract=on")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(COMMON_FLAGS "${COMMON_FLAGS} -march=native -mtune=native")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(COMMON_FLAGS "${COMMON_FLAGS} -mavx2 -mfma")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(COMMON_FLAGS "${COMMON_FLAGS} /arch:AVX2 /fp:strict")
endif()

# Set debug and release flags to have consistent behavior for float-to-int casting
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

# Set up vcpkg toolchain if not already set
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE
            "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING
            "Vcpkg toolchain file")
    elseif(WIN32 AND EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    elseif(UNIX AND NOT APPLE AND EXISTS
           "/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE
            "/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING
            "Vcpkg toolchain file")
    elseif(APPLE AND EXISTS
    "$ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE
            "$ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

include(FetchContent)

# Fetch Eigen
# DO NOT REVOVE THE NEXT TWO LINES, IF REMOVED IT STARTS EIGEN TEST SUITES
set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests")
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)
include_directories(${eigen_SOURCE_DIR})

# Fetch and setup pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Build tests only when explicitly requested.
option(PYGRAD_BUILD_TESTS "Build pygrad kernel tests" OFF)

# Detect and configure xsimd include directories
if(DEFINED ENV{XSIMD_INCLUDE_DIR})
    set(XSIMD_INCLUDE_DIR $ENV{XSIMD_INCLUDE_DIR})
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(XSIMD_INCLUDE_DIR "$ENV{HOME}/vcpkg/installed/arm64-osx/include")
    else()
        set(XSIMD_INCLUDE_DIR "$ENV{HOME}/vcpkg/installed/x64-osx/include")
    endif()
    if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
        set(XSIMD_INCLUDE_DIR "/opt/homebrew/include")
    endif()
    if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
        set(XSIMD_INCLUDE_DIR "/usr/local/include")
    endif()
elseif(WIN32)
    set(XSIMD_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
else()
    set(XSIMD_INCLUDE_DIR "/usr/local/vcpkg/installed/x64-linux/include")
    if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
        set(XSIMD_INCLUDE_DIR "$ENV{HOME}/vcpkg/installed/x64-linux/include")
    endif()
    if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
        set(XSIMD_INCLUDE_DIR "/usr/include")
    endif()
endif()

if(NOT EXISTS "${XSIMD_INCLUDE_DIR}/xsimd/xsimd.hpp")
    message(WARNING "xsimd headers not found at ${XSIMD_INCLUDE_DIR}. You may need to install xsimd with vcpkg or specify XSIMD_INCLUDE_DIR manually.")
endif()

# Create a static library for testing
add_library(cpu_kernel_lib STATIC
    kernels/cpu_kernel.cpp
    kernels/vecbuffer.cpp
    kernels/operations.h
    kernels/operations.cpp)
target_include_directories(cpu_kernel_lib PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_SOURCE_DIR}/include
    ${XSIMD_INCLUDE_DIR}
)

# Create Python module
pybind11_add_module(cpu_kernel
    kernels/cpu_kernel.cpp
    kernels/vecbuffer.cpp
    kernels/operations.h
    kernels/operations.cpp)
target_include_directories(cpu_kernel PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${XSIMD_INCLUDE_DIR}
)

if(PYGRAD_BUILD_TESTS)
    # Fetch GoogleTest
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)

    # Enable testing
    enable_testing()
    add_subdirectory(tests/kernels)
endif()
