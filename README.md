# TokenizationEstimator

A fast Rust-based Python extension for estimating tokenization dimensions and memory usage in ColBERT-style embedding models.

## Quick Start

### Installation

```bash
# Install dependencies (using uv package manager)
uv add --dev maturin cffi

# Or with pip
pip install maturin cffi

# Build and install
maturin develop --release
```

### Usage

```python
from tokenizator import TokenizationEstimator

# Load tokenizer from HuggingFace Hub (requires internet connection)
# Note: Use a model that has a tokenizer.json file available
estimator = TokenizationEstimator.from_pretrained(
    repo_id="lightonai/GTE-ModernColBERT-v1",  # Example model
    query_length=128,
    document_length=512
)

# Estimate dimensions
queries = ["What is machine learning?"]
batch_size, seq_len = estimator.estimate_dimensions(queries, is_query=True)
print(f"Batch: {batch_size}, Sequence: {seq_len}")

# Estimate memory usage
token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
    queries, is_query=True
)
print(f"Memory needed: {total_mem / 1024**2:.2f} MB")


## API Reference

### TokenizationEstimator.from_pretrained()

```python
estimator = TokenizationEstimator.from_pretrained(
    repo_id: str,
    query_length: int = 128,
    document_length: int = 512,
)
```

### Methods

- `estimate_dimensions(texts, is_query)` → `(batch_size, sequence_length)`
- `estimate_memory_usage(texts, is_query)` → `(token_memory, intermediate_memory, total_memory)`
- `can_fit_in_memory(texts, is_query, available_bytes)` → `bool`
- `split_into_batches(texts, is_query, available_bytes)` → `List[List[str]]`

## Release
- Bump version on `pyproject.toml` and `Cargo.toml`
- Create new tag with `git tag v{version}`
- Push tag to GitHub with `git push origin v{version}`
- Push repository to GitHub with `git push origin main` or merge the PR

## Development

# Build & Test

```bash
# Quick development iteration
make quick          # Format + check code

# Build for testing
make build-dev      # Build Python extension

# Run full tests
make test

# Clean build artifacts
make clean

# Rebuild project (slow, use if problems with python caching modules)
make rebuild
```

## Features

- **Fast Estimation**: Predict dimensions without full tokenization
- **Memory Planning**: Calculate memory needs before processing
- **Batch Optimization**: Split datasets into memory-efficient batches
- **HuggingFace Integration**: Load any tokenizer from the Hub

## Notes

- **Module name**: Import as `from tokenizator import TokenizationEstimator`
- **Internet required**: `from_pretrained()` downloads from HuggingFace Hub
- **Model compatibility**: Use models that have `tokenizer.json` available
- **Padding strategy**: Queries use fixed-length, documents use variable-length
- **Memory overhead**: Estimates include ~3x overhead, deduced with empirical trials
- **uv package manager**: Examples use `uv` instead of `pip`
