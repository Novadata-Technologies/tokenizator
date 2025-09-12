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
    document_length=512
)

# Estimate dimensions
queries = ["What is machine learning?"]
batch_size, seq_len = estimator.estimate_dimensions(queries, is_query=True)
print(f"Batch: {batch_size}, Sequence: {seq_len}")

# Estimate memory usage
token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
    queries, is_query=True, embedding_dim=128, bytes_per_token=4, num_attention_heads=12
)
print(f"Memory needed: {total_mem / 1024**2:.2f} MB")


## API Reference

### TokenizationEstimator.from_pretrained()

```python
estimator = TokenizationEstimator.from_pretrained(
    repo_id: str,
    document_length: int = 512,
    query_prefix: str = "[Q]",
    document_prefix: str = "[D]",
    mask_token: str = "[MASK]"
)
```

### Methods

- `estimate_dimensions(texts, is_query)` → `(batch_size, sequence_length)`
- `estimate_memory_usage(texts, is_query, embedding_dim, bytes_per_token, num_attention_heads)` → `(token_memory, intermediate_memory, total_memory)`
- `can_fit_in_memory(texts, is_query, available_bytes, embedding_dim, num_attention_heads)` → `bool`
- `split_into_batches(texts, is_query, available_bytes, embedding_dim)` → `List[List[str]]`
- `split_into_batches(texts, is_query, available_bytes, embedding_dim)` → `List[List[str]]`

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
- **Memory overhead**: Estimates include ~20% for intermediate tensors
- **uv package manager**: Examples use `uv` instead of `pip`
