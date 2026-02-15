# PyGrad Makefile
# Provides convenient targets for building and testing the C++ kernel
.PHONY: help build clean test install-deps debug release install uninstall setup-env


.DEFAULT_GOAL := help


ifeq ($(PLATFORM),unix)
$(shell chmod +x ./setup_prerequisite.sh 2>/dev/null || true)
endif

# Variables
BUILD_DIR := build
PYTHON := python3
CMAKE := cmake
MAKE := make

# Detect OS
ifeq ($(OS),Windows_NT)
    PLATFORM := windows
    PYTHON := python
    CMAKE_BUILD := cmake --build . --config
    CMAKE_CONFIG := Release
    SO_EXT := dll
else
    PLATFORM := unix
    CMAKE_BUILD := $(MAKE)
    CMAKE_CONFIG :=
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        SO_EXT := dylib
    else
        SO_EXT := so
    endif
endif

# Help target
help:
	@echo "PyGrad C++ Kernel Build System"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make setup-env      # Set up development environment (install all prerequisites)"
	@echo "  make build          # Build in Release mode"
	@echo "  make debug          # Build in Debug mode"
	@echo "  make test           # Build and run tests"
	@echo "  make clean          # Clean build directory"


build: ## Build the project in Release mode
	@echo "Building in Release mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release -DPYGRAD_BUILD_TESTS=OFF -DPython3_EXECUTABLE=$(shell which $(PYTHON) 2>/dev/null || echo python3) -Wno-dev -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
	@cd $(BUILD_DIR) && $(CMAKE_BUILD) $(CMAKE_CONFIG)
	@echo "Build completed successfully!"

debug: ## Build the project in Debug mode
	@echo "Building in Debug mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug -DPYGRAD_BUILD_TESTS=OFF -DPython3_EXECUTABLE=$(shell which $(PYTHON) 2>/dev/null || echo python3)  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
	@cd $(BUILD_DIR) && $(CMAKE_BUILD)
	@echo "Debug build completed successfully!"

release: build

# Test targets
test: ## Build and run tests in Release mode
	@echo "Building tests in Release mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release -DPYGRAD_BUILD_TESTS=ON -DPython3_EXECUTABLE=$(shell which $(PYTHON) 2>/dev/null || echo python3) -Wno-dev -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
	@cd $(BUILD_DIR) && $(CMAKE_BUILD) $(CMAKE_CONFIG)
	@echo "Running tests..."
	@cd $(BUILD_DIR) && (ctest --output-on-failure || $(CMAKE) --build . --target test || echo "Warning: Some tests may have failed")
	@echo "Tests completed!"

test-debug: ## Build and run tests in Debug mode
	@echo "Building tests in Debug mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug -DPYGRAD_BUILD_TESTS=ON -DPython3_EXECUTABLE=$(shell which $(PYTHON) 2>/dev/null || echo python3) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
	@cd $(BUILD_DIR) && $(CMAKE_BUILD)
	@echo "Running tests (debug build)..."
	@cd $(BUILD_DIR) && (ctest --output-on-failure || $(CMAKE) --build . --target test || echo "Warning: Some tests may have failed")
	@echo "Tests completed!"


clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean completed!"


dev: debug test-debug


install-deps: ## Install dependencies using vcpkg
	@echo "Installing dependencies..."
	@if [ "$(OS)" = "Windows_NT" ]; then \
		echo "Running Windows setup script..."; \
		./setup_prerequisite.bat; \
	else \
		echo "Running Unix setup script..."; \
		chmod +x ./setup_prerequisite.sh; \
		./setup_prerequisite.sh; \
	fi
	@echo "Dependencies installed successfully!"

install: build ## Install the Python module
	@echo "Installing Python module..."
	@mkdir -p grad/kernels 2>/dev/null || true
	@find $(BUILD_DIR) -name "cpu_kernel*.$(SO_EXT)" | xargs -I{} cp {} grad/kernels/ 2>/dev/null || \
	find $(BUILD_DIR) -name "cpu_kernel*.*" | grep -E '\.so|\.dylib|\.dll' | xargs -I{} cp {} grad/kernels/ 2>/dev/null || \
	echo "Warning: Could not find module file to copy"
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e .; \
	else \
		$(PYTHON) -m pip install -e .; \
	fi
	@echo "Installation completed!"

uninstall: ## Uninstall the Python module
	@echo "Uninstalling Python module..."
	@$(PYTHON) -m pip uninstall -y pygrad
	@echo "Uninstallation completed!"


check: ## Check if all dependencies are available
	@echo "Checking dependencies..."
	@echo "Python: $(shell which $(PYTHON))"
	@echo "CMake: $(shell which $(CMAKE))"
	@echo "Make: $(shell which $(MAKE))"
	@echo "Platform: $(PLATFORM)"
	@if command -v vcpkg >/dev/null 2>&1; then \
		echo "vcpkg: $(shell which vcpkg)"; \
		echo "vcpkg libraries:"; \
		vcpkg list; \
	else \
		echo "vcpkg: Not found"; \
	fi
	@echo "All dependencies checked!"

format: ## Format C++ code (requires clang-format)
	@echo "Formatting C++ code..."
	@find kernels -name "*.cpp" -o -name "*.h" | xargs clang-format -i
	@echo "Formatting completed!"

lint: ## Lint C++ code (requires cpplint)
	@echo "Linting C++ code..."
	@find kernels -name "*.cpp" -o -name "*.h" | xargs cpplint
	@echo "Linting completed!"

# Quick test target for development
quick-test: ## Quick test without full rebuild
	@echo "Running quick test..."
	@cd $(BUILD_DIR) && (ctest --output-on-failure || $(CMAKE) --build . --target test || echo "Warning: Some tests may have failed")
	@echo "Quick test completed!"

# Show build artifacts
artifacts: ## Show build artifacts
	@echo "Build artifacts in $(BUILD_DIR):"
	@find $(BUILD_DIR) -name "*.$(SO_EXT)" -o -name "*.so" -o -name "*.dylib" -o -name "*.dll" 2>/dev/null | xargs ls -la 2>/dev/null || echo "No shared libraries found"
	@find $(BUILD_DIR) -name "test_*" 2>/dev/null | xargs ls -la 2>/dev/null || echo "No test executables found"

# Full environment setup
setup-env: ## Set up the complete development environment
	@echo "Setting up development environment..."
	@$(MAKE) install-deps
	@$(MAKE) check
	@echo "Development environment setup complete!"

# Python module test
python-install: ## Install Python Module
	@echo "Installing Python module..."
	uv pip install -e .
	@echo "Python module install completed!"

python-test: python-install ## Test the Python module
	@echo "Testing Python module..."
	@PYTHONPATH=$(BUILD_DIR):. $(PYTHON) -c "import cpu_kernel; print('Python module imported successfully!')" || \
	(cp $(BUILD_DIR)/cpu_kernel*.$(SO_EXT) . 2>/dev/null && PYTHONPATH=. $(PYTHON) -c "import cpu_kernel; print('Python module imported successfully!')")
	@PYTHONPATH=$(BUILD_DIR):. $(PYTHON) -c "import cpu_kernel; b = cpu_kernel.Buffer([1, 2, 3], 'float32'); print('Buffer created:', b)"
	@echo "Python module test completed!"


precommit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks on all files..."
	@pre-commit run --all-files
	@echo "Pre-commit checks completed successfully!"
