.PHONY: help quick install-deps test build-dev build-release lint format check clean dev-setup

help:   ## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

quick: format check-python ## Run quick checks

install-deps:	## Install required dependencies
	@echo "Installing Rust dependencies..."
	cargo fetch
	@echo "Installing Python build tools..."
	uv add --dev maturin pytest cffi

test: build-dev ## Run minimal tests (most require internet connection)
	@echo "Running Python tests (most skipped by default)..."
	python -m pytest tests/ -s

build-dev:	## Build development version for Python
	@echo "Installing cffi if needed..."
	@uv add cffi || echo "cffi already installed"
	@echo "Building development version..."
	maturin develop --release

build-release:	## Build optimized release version
	@echo "Building release version..."
	maturin build --release

lint:	## Lint code
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

format:	## Format code
	@echo "Formatting Rust code..."
	cargo fmt

check:	## Check code without building
	@echo "Checking Rust code..."
	cargo check --all-targets --features python

clean:	## Clean build artifacts
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf *.so
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

dev-setup: install-deps build-dev ## Development setup
	@echo "Development environment ready!"
	@echo "Note: Using uv for Python package management"
