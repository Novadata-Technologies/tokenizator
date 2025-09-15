.PHONY: help quick install-deps test build-dev build-release lint format check clean dev-setup rebuild
PYTHON_VERSION ?= 3.12

help:   ## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

quick: format check-python ## Run quick checks

install-deps:	## Install required dependencies
	@echo "Installing Rust dependencies..."
	cargo fetch
	@echo "Installing Python build tools..."
	uv venv --python $(PYTHON_VERSION)
	uv add --dev maturin pytest cffi

test: build-dev ## Run minimal tests (most require internet connection)
	@echo "Running Python tests (most skipped by default)..."
	uv run python -m pytest tests/ -s

build-dev: clean-cache	## Build development version
	@echo "Installing cffi if needed..."
	@uv add cffi || echo "cffi already installed"
	@echo "Uninstalling old version to avoid cache issues..."
	@uv pip uninstall tokenizator -q 2>/dev/null || true
	@echo "Building development version..."
	uv run maturin develop --release

build-release:	## Build optimized release version
	@echo "Uninstalling old version to avoid cache issues..."
	@uv pip uninstall tokenizator -q 2>/dev/null || true
	@echo "Building release version..."
	uv run maturin build --release

lint:	## Lint code
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

format:	## Format code
	@echo "Formatting Rust code..."
	cargo fmt

check:	## Check code without building
	@echo "Checking Rust code..."
	cargo check --all-targets --features python

check-python: ## Check Python code
	@echo "Checking Python code..."
	uv run python -c "import tokenizator; print('Python import successful')" || echo "Build required first"

clean:	## Clean build artifacts
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf *.so
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .venv

clean-cache:	## Clean only Python/uv caches without removing venv
	@echo "Cleaning Python caches..."
	@uv pip uninstall tokenizator -q 2>/dev/null || true
	@rm -rf .venv/lib/python*/site-packages/tokenizator*
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "Cache cleaned, venv preserved"

dev-setup: install-deps build-dev ## Development setup
	@echo "Development environment ready!"
	@echo "Note: Using uv for Python package management"
	@echo "To activate the environment manually: source .venv/bin/activate"
	@echo "To run commands in the environment: uv run <command>"

rebuild: ## Force rebuild clearing uv cache (use when Rust changes aren't reflected)
	@echo "=== Clearing uv cache to ensure fresh build ==="
	@# Kill Python processes
	@pkill -f "python.*tokenizator" 2>/dev/null || true
	@# Remove uv cache (THE CRITICAL PART for reflecting Rust changes)
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		echo "Removing macOS uv caches..."; \
		rm -rf ~/Library/Caches/uv 2>/dev/null || true; \
	fi
	@rm -rf ~/.cache/uv 2>/dev/null || true
	@rm -rf "$${XDG_CACHE_HOME:-$$HOME/.cache}/uv" 2>/dev/null || true
	@# Remove maturin cache
	@rm -rf ~/.cache/maturin 2>/dev/null || true
	@rm -rf ~/Library/Caches/maturin 2>/dev/null || true
	@# Clean Rust/Cargo
	@cargo clean
	@rm -rf target/
	@# Remove all Python artifacts
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@find . -name "*.so" -delete 2>/dev/null || true
	@find . -name "*.pyd" -delete 2>/dev/null || true
	@# Uninstall tokenizator
	@uv pip uninstall tokenizator -y 2>/dev/null || true
	@rm -rf .venv/lib/python*/site-packages/tokenizator* 2>/dev/null || true
	@# Recreate clean environment
	@rm -rf .venv
	@rm -f uv.lock
	@uv venv --python $(PYTHON_VERSION)
	@uv add --dev maturin pytest cffi numpy
	@# Touch source files to force recompilation
	@find src -name "*.rs" -exec touch {} \;
	@touch Cargo.toml
	@echo "Cache cleared! Building fresh version..."
	@uv run maturin develop --release
	@echo "✅ Rebuild complete! Your changes should now be reflected."
