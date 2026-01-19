#!/usr/bin/env bash

set -e

# Function to check and install vcpkg
install_vcpkg() {
    local vcpkg_dir="$1"

    if [[ ! -d "$vcpkg_dir" ]]; then
        echo "Installing vcpkg to $vcpkg_dir..."
        git clone https://github.com/microsoft/vcpkg.git "$vcpkg_dir"

        # Bootstrap vcpkg
        if [[ "$OS" == "Windows_NT" ]]; then
            pushd "$vcpkg_dir"
            ./bootstrap-vcpkg.bat
            popd
        else
            pushd "$vcpkg_dir"
            ./bootstrap-vcpkg.sh
            popd
        fi

        echo "vcpkg installed successfully."
    else
        echo "vcpkg already installed at $vcpkg_dir"
    fi

    # Install xsimd
    echo "Installing xsimd with vcpkg..."
    if [[ "$OS" == "Windows_NT" ]]; then
        "$vcpkg_dir/vcpkg" install xsimd:x64-windows
    elif [[ "$OS" == "Darwin" ]]; then
        "$vcpkg_dir/vcpkg" install xsimd:arm64-osx
    else
        "$vcpkg_dir/vcpkg" install xsimd:x64-linux
    fi
}

# Detect OS
echo "Detecting OS..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    OS="Windows_NT"
else
    OS="$(uname)"
fi
echo "OS detected: $OS"

if [[ "$OS" == "Darwin" ]]; then
    # macOS
    echo "Installing prerequisites for macOS..."
    # Install Homebrew if not present
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for the current session if it was just installed
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -f "/usr/local/bin/brew" ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
        echo "Homebrew installed and configured."
    fi

    echo "Updating Homebrew and installing packages..."
    brew update
    brew install cmake eigen python@3 git pybind11

    # Install vcpkg for xsimd
    VCPKG_DIR="$HOME/vcpkg"
    install_vcpkg "$VCPKG_DIR"

    # Set environment variables for later use
    echo "export VCPKG_ROOT=$VCPKG_DIR" >> ~/.bash_profile
    echo "export XSIMD_INCLUDE_DIR=$VCPKG_DIR/installed/arm64-osx/include" >> ~/.bash_profile
    echo "export EIGEN3_INCLUDE_DIR=/opt/homebrew/include/eigen3" >> ~/.bash_profile

elif [[ "$OS" == "Linux" ]]; then
    # Linux
    echo "Installing prerequisites for Linux..."

    # Check if we can use apt-get (Debian/Ubuntu)
    if command -v apt-get &>/dev/null; then
        echo "Detected Debian/Ubuntu-based distribution"
        sudo apt-get update
        sudo apt-get install -y cmake g++ python3 python3-dev python3-pip git libeigen3-dev pybind11-dev
    # Check if we can use dnf (Fedora/RHEL)
    elif command -v dnf &>/dev/null; then
        echo "Detected Fedora/RHEL-based distribution"
        sudo dnf update -y
        sudo dnf install -y cmake gcc-c++ python3 python3-devel python3-pip git eigen3-devel pybind11-devel
    # Check if we can use pacman (Arch Linux)
    elif command -v pacman &>/dev/null; then
        echo "Detected Arch-based distribution"
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm cmake gcc python python-pip git eigen python-pybind11
    else
        echo "Unsupported Linux distribution. Please install the following packages manually:"
        echo "- cmake"
        echo "- C++ compiler (g++ or clang++)"
        echo "- Python 3 with development headers"
        echo "- pip for Python 3"
        echo "- git"
        echo "- Eigen 3 library"
        echo "- pybind11"
        exit 1
    fi

    # Install vcpkg for xsimd
    VCPKG_DIR="/usr/local/vcpkg"
    if [[ ! -w "/usr/local" ]]; then
        VCPKG_DIR="$HOME/vcpkg"
    fi
    install_vcpkg "$VCPKG_DIR"

    # Set environment variables
    echo "export VCPKG_ROOT=$VCPKG_DIR" >> ~/.bashrc
    echo "export XSIMD_INCLUDE_DIR=$VCPKG_DIR/installed/x64-linux/include" >> ~/.bashrc
    echo "export EIGEN3_INCLUDE_DIR=/usr/include/eigen3" >> ~/.bashrc

elif [[ "$OS" == "Windows_NT" ]]; then
    # Windows
    echo "Installing prerequisites for Windows..."

    if command -v choco &>/dev/null; then
        echo "Chocolatey found. Installing packages..."
        choco install -y cmake python git
    else
        echo "Please install the following packages manually:"
        echo "1. CMake (https://cmake.org/download/)"
        echo "2. Python 3 (https://www.python.org/downloads/windows/)"
        echo "3. Git (https://git-scm.com/download/win)"
        echo "4. Visual Studio Build Tools with C++ workload (https://visualstudio.microsoft.com/downloads/)"
    fi

    # Install vcpkg
    VCPKG_DIR="C:/vcpkg"
    if [[ ! -d "$VCPKG_DIR" ]]; then
        echo "Installing vcpkg..."
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"
        pushd "$VCPKG_DIR"
        ./bootstrap-vcpkg.bat
        ./vcpkg integrate install
        ./vcpkg install eigen3:x64-windows xsimd:x64-windows pybind11:x64-windows
        popd

        # Set environment variables
        setx VCPKG_ROOT "$VCPKG_DIR"
        setx CMAKE_TOOLCHAIN_FILE "$VCPKG_DIR/scripts/buildsystems/vcpkg.cmake"
        setx XSIMD_INCLUDE_DIR "$VCPKG_DIR/installed/x64-windows/include"
    else
        echo "vcpkg already installed at $VCPKG_DIR"
        pushd "$VCPKG_DIR"
        ./vcpkg install eigen3:x64-windows xsimd:x64-windows pybind11:x64-windows
        popd
    fi

    echo "Please restart your terminal or command prompt for environment variables to take effect."

else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Verify installations
echo "Verifying installations..."

if ! command -v cmake &>/dev/null; then
    echo "WARNING: cmake not found in PATH. Installation may have failed."
else
    CMAKE_VERSION=$(cmake --version | head -n 1)
    echo "✓ $CMAKE_VERSION"
fi

if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
elif command -v python &>/dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "WARNING: Python not found in PATH. Installation may have failed."
fi

if ! command -v git &>/dev/null; then
    echo "WARNING: git not found in PATH. Installation may have failed."
else
    GIT_VERSION=$(git --version)
    echo "✓ $GIT_VERSION"
fi

echo ""
echo "All prerequisites installed."
echo "You can now build the project with:"
if [[ "$OS" == "Windows_NT" ]]; then
    echo "  mkdir build && cd build"
    echo "  cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake .."
    echo "  cmake --build . --config Release"
else
    echo "  mkdir -p build && cd build"
    echo "  cmake .."
    echo "  make"
fi
