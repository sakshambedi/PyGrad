@echo off
setlocal enabledelayedexpansion

echo Detecting OS...
echo OS detected: Windows

:: Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Not running with administrator privileges.
    echo Some installations may fail. Consider running as administrator.
)

:: Check for package managers
set PACKAGE_MANAGER=""
where choco >nul 2>&1
if %errorLevel% equ 0 (
    set PACKAGE_MANAGER=choco
    echo Chocolatey detected, will use for package installation.
) else (
    where winget >nul 2>&1
    if %errorLevel% equ 0 (
        set PACKAGE_MANAGER=winget
        echo Winget detected, will use for package installation.
    ) else (
        echo No package manager detected. Will attempt manual installations.
    )
)

:: Install prerequisites
echo Installing prerequisites for Windows...

if "%PACKAGE_MANAGER%"=="choco" (
    echo Installing packages with Chocolatey...
    choco install -y cmake
    choco install -y python
    choco install -y git
    choco install -y visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
) else if "%PACKAGE_MANAGER%"=="winget" (
    echo Installing packages with Winget...
    winget install -e --id Kitware.CMake
    winget install -e --id Python.Python.3
    winget install -e --id Git.Git
    winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
) else (
    echo.
    echo Please install the following packages manually:
    echo 1. CMake (https://cmake.org/download/)
    echo 2. Python 3 (https://www.python.org/downloads/windows/)
    echo 3. Git (https://git-scm.com/download/win)
    echo 4. Visual Studio Build Tools with C++ workload (https://visualstudio.microsoft.com/downloads/)
    echo.
    echo Press any key when you have installed these prerequisites...
    pause > nul
)

:: Install vcpkg
set "VCPKG_DIR=C:\vcpkg"
if not exist "%VCPKG_DIR%" (
    echo Installing vcpkg...
    git clone https://github.com/microsoft/vcpkg.git "%VCPKG_DIR%"
    cd /d "%VCPKG_DIR%"
    call bootstrap-vcpkg.bat
    vcpkg integrate install

    echo Installing required libraries...
    vcpkg install eigen3:x64-windows
    vcpkg install xsimd:x64-windows
    vcpkg install pybind11:x64-windows
) else (
    echo vcpkg already installed at %VCPKG_DIR%
    cd /d "%VCPKG_DIR%"
    vcpkg update

    echo Installing/Updating required libraries...
    vcpkg install eigen3:x64-windows
    vcpkg install xsimd:x64-windows
    vcpkg install pybind11:x64-windows
)

:: Set environment variables
echo Setting environment variables...
setx VCPKG_ROOT "%VCPKG_DIR%"
setx CMAKE_TOOLCHAIN_FILE "%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake"
setx XSIMD_INCLUDE_DIR "%VCPKG_DIR%\installed\x64-windows\include"
setx EIGEN3_INCLUDE_DIR "%VCPKG_DIR%\installed\x64-windows\include\eigen3"

:: Verify installations
echo.
echo Verifying installations...

where cmake >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: cmake not found in PATH. Installation may have failed.
) else (
    for /f "tokens=*" %%i in ('cmake --version ^| findstr /B /C:"cmake version"') do (
        echo ✓ %%i
    )
)

where python >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: python not found in PATH. Installation may have failed.
) else (
    for /f "tokens=*" %%i in ('python --version') do (
        echo ✓ %%i
    )
)

where git >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: git not found in PATH. Installation may have failed.
) else (
    for /f "tokens=*" %%i in ('git --version') do (
        echo ✓ %%i
    )
)

echo.
echo All prerequisites installed.
echo You may need to restart your terminal or command prompt for environment variables to take effect.
echo.
echo You can now build the project with:
echo   mkdir build ^&^& cd build
echo   cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..
echo   cmake --build . --config Release

endlocal
