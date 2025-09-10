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
    repo_id="sentence-transformers/all-MiniLM-L6-v2",  # Example model
    device="cpu",
    query_length=32,
    document_length=512
)

# Estimate dimensions
queries = ["What is machine learning?"]
batch_size, seq_len = estimator.estimate_dimensions(queries, is_query=True)
print(f"Batch: {batch_size}, Sequence: {seq_len}")

# Estimate memory usage
token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
    queries, is_query=True, embedding_dim=128, bytes_per_token=4
)
print(f"Memory needed: {total_mem / 1024**2:.2f} MB")

# Get optimal batch size for available memory
optimal_batch = estimator.get_optimal_batch_size(
    queries[0], is_query=True,
    available_bytes=1024**3,  # 1GB
    embedding_dim=128
)
print(f"Optimal batch size: {optimal_batch}")
```

## API Reference

### TokenizationEstimator.from_pretrained()

```python
estimator = TokenizationEstimator.from_pretrained(
    repo_id: str,
    device: str = "cpu",              # "cpu", "cuda", "cuda:0", "mps"
    query_length: int = 32,
    document_length: int = 512,
    query_prefix: str = "Query: ",
    document_prefix: str = "Document: ",
    mask_token: str = "[MASK]"
)
```

### Methods

- `estimate_dimensions(texts, is_query)` → `(batch_size, sequence_length)`
- `estimate_memory_usage(texts, is_query, embedding_dim, bytes_per_token)` → `(token_memory, intermediate_memory, total_memory)`
- `can_fit_in_memory(texts, is_query, available_bytes, embedding_dim)` → `bool`
- `get_optimal_batch_size(sample_text, is_query, available_bytes, embedding_dim)` → `int`
- `split_into_batches(texts, is_query, available_bytes, embedding_dim)` → `List[List[str]]`

## Development

# Build & Test

```bash
# Quick development iteration (no network needed)
make quick          # Format + check code

# Build for testing
make build-dev      # Build Python extension

# Quick import test
make test-import    # Test basic import

# Run full tests (most require internet)
make test

# Clean build artifacts
make clean
```

### Example

```bash
# Run example (requires internet connection)
python examples/usage_example.py
```

## Features

- **Fast Estimation**: Predict dimensions without full tokenization
- **Memory Planning**: Calculate memory needs before processing
- **Batch Optimization**: Split datasets into memory-efficient batches
- **HuggingFace Integration**: Load any tokenizer from the Hub
- **Multi-Device**: CPU, CUDA, and Metal support

## Notes

- **Module name**: Import as `from tokenizator import TokenizationEstimator`
- **Internet required**: `from_pretrained()` downloads from HuggingFace Hub
- **Model compatibility**: Use models that have `tokenizer.json` available
- **Padding strategy**: Queries use fixed-length, documents use variable-length
- **Memory overhead**: Estimates include ~50% for intermediate tensors
- **uv package manager**: Examples use `uv` instead of `pip`

## Rapid Development Workflow

```bash
# 1. Quick check (fastest, no network needed)
make quick

# 2. Build and test import
make test-import

# 3. Run with real model (requires internet)
python examples/usage_example.py
```
