#!/usr/bin/env python3
"""
Cross-platform build script for pygrad project.
This script handles building the C++ components and Python extensions.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def detect_platform():
    """Detect the platform and set appropriate variables."""
    system = platform.system().lower()
    architecture = platform.machine().lower()

    if system == "darwin":
        if "arm" in architecture:
            return "macos-arm64"
        else:
            return "macos-x64"
    elif system == "windows":
        return "windows-x64"
    elif system == "linux":
        return "linux-x64"
    else:
        print(f"Warning: Unsupported platform detected: {system} on {architecture}")
        return f"{system}-{architecture}"


def get_vcpkg_path():
    """Find the vcpkg installation path based on platform."""
    # Check environment variable first
    vcpkg_root = os.environ.get("VCPKG_ROOT")
    if vcpkg_root and os.path.exists(vcpkg_root):
        return vcpkg_root

    # Check common locations based on platform
    if platform.system().lower() == "windows":
        paths = ["C:/vcpkg"]
    elif platform.system().lower() == "darwin":
        paths = [os.path.expanduser("~/vcpkg"), "/opt/vcpkg"]
    else:  # Linux
        paths = [os.path.expanduser("~/vcpkg"), "/usr/local/vcpkg"]

    # Return the first existing path
    for path in paths:
        if os.path.exists(path):
            return path

    return None


def setup_environment():
    """Set up build environment variables based on the platform."""
    env = os.environ.copy()

    # Detect platform
    platform_name = detect_platform()
    print(f"Detected platform: {platform_name}")

    # Set environment variables
    vcpkg_root = get_vcpkg_path()
    if vcpkg_root:
        env["VCPKG_ROOT"] = vcpkg_root
        print(f"VCPKG_ROOT set to {vcpkg_root}")

        # Set toolchain file
        env["CMAKE_TOOLCHAIN_FILE"] = os.path.join(
            vcpkg_root, "scripts", "buildsystems", "vcpkg.cmake"
        )
    else:
        print("WARNING: vcpkg not found. Some dependencies may not be resolved.")

    # Set compiler flags based on platform
    if "windows" in platform_name:
        # Windows-specific settings
        env["CMAKE_GENERATOR"] = "Visual Studio 17 2022"
        env["CMAKE_GENERATOR_PLATFORM"] = "x64"
        if vcpkg_root:
            env["XSIMD_INCLUDE_DIR"] = os.path.join(
                vcpkg_root, "installed", "x64-windows", "include"
            )
    elif "macos" in platform_name:
        # macOS-specific settings
        if "arm64" in platform_name:
            triplet = "arm64-osx"
        else:
            triplet = "x64-osx"
        if vcpkg_root:
            env["XSIMD_INCLUDE_DIR"] = os.path.join(vcpkg_root, "installed", triplet, "include")
        env["EIGEN3_INCLUDE_DIR"] = "/opt/homebrew/include/eigen3"
        if not os.path.exists(env.get("EIGEN3_INCLUDE_DIR", "")):
            env["EIGEN3_INCLUDE_DIR"] = "/usr/local/include/eigen3"
    else:
        # Linux-specific settings
        if vcpkg_root:
            env["XSIMD_INCLUDE_DIR"] = os.path.join(vcpkg_root, "installed", "x64-linux", "include")

        # Find Eigen include directory
        eigen_locations = ["/usr/include/eigen3", "/usr/local/include/eigen3"]
        for loc in eigen_locations:
            if os.path.exists(loc):
                env["EIGEN3_INCLUDE_DIR"] = loc
                break

    return env


def cmake_build(build_dir, env, build_type="Release", clean=False):
    """Build the project with CMake."""
    build_path = Path(build_dir)

    # Clean build directory if requested
    if clean and build_path.exists():
        import shutil

        print(f"Cleaning build directory: {build_path}")
        shutil.rmtree(build_path)

    # Create build directory
    build_path.mkdir(exist_ok=True, parents=True)

    # Configure with CMake
    print("\nConfiguring with CMake...")

    cmake_args = ["cmake", ".."]

    # Add toolchain file if available
    if "CMAKE_TOOLCHAIN_FILE" in env:
        cmake_args.extend([f"-DCMAKE_TOOLCHAIN_FILE={env['CMAKE_TOOLCHAIN_FILE']}"])

    # Add build type
    cmake_args.extend([f"-DCMAKE_BUILD_TYPE={build_type}"])

    # Execute CMake configure
    try:
        subprocess.run(cmake_args, cwd=build_path, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CMake configuration failed with error code {e.returncode}")
        return False

    # Build with CMake
    print("\nBuilding with CMake...")

    if platform.system().lower() == "windows":
        # On Windows, we need to specify the config
        build_args = ["cmake", "--build", ".", "--config", build_type]
    else:
        # On Unix systems
        build_args = ["cmake", "--build", "."]

    try:
        subprocess.run(build_args, cwd=build_path, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CMake build failed with error code {e.returncode}")
        return False

    return True


def python_build(env, develop=False, clean=False):
    """Build Python package using setup.py."""
    print("\nBuilding Python package...")

    # Clean Python build artifacts if requested
    if clean:
        for pattern in ["build", "*.egg-info", "dist"]:
            import glob

            for path in glob.glob(pattern):
                import shutil

                print(f"Removing {path}")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    # Build the Python package
    cmd = [sys.executable, "setup.py"]

    if develop:
        cmd.append("develop")
    else:
        cmd.extend(["build_ext", "--inplace"])

    try:
        subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Python build failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build script for pygrad project")
    parser.add_argument(
        "--clean", action="store_true", help="Clean build directories before building"
    )
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument(
        "--develop", action="store_true", help="Install Python package in development mode"
    )
    parser.add_argument(
        "--cmake-only", action="store_true", help="Only build with CMake, skip Python setup"
    )
    parser.add_argument(
        "--python-only", action="store_true", help="Only build Python package, skip CMake"
    )
    args = parser.parse_args()

    # Setup environment
    env = setup_environment()
    build_type = "Debug" if args.debug else "Release"

    # Build with CMake if not python-only
    if not args.python_only:
        build_dir = "build_debug" if args.debug else "build"
        if not cmake_build(build_dir, env, build_type, args.clean):
            sys.exit(1)

    # Build Python package if not cmake-only
    if not args.cmake_only:
        if not python_build(env, args.develop, args.clean):
            sys.exit(1)

    print("\nBuild completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
