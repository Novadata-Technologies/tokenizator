#!/usr/bin/env python3
"""
Minimal Python tests for the tokenization_estimator module.

Run these tests after building the Rust extension:
    maturin develop
    python -m pytest tests/test_python_interface.py -v
"""

import pytest

# Import the compiled module
try:
    from tokenizator import TokenizationEstimator, TokenizationError
except ImportError:
    pytest.skip("tokenizator module not built. Run 'maturin develop' first.", allow_module_level=True)


class TestTokenizationEstimator:
    """Minimal test suite for TokenizationEstimator."""

    def test_basic_query(self):
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            document_length=1024
        )

        queries = ["What is machine learning?", "How do neural networks work?"]

        # Estimate dimensions
        q_batch, q_seq = estimator.estimate_dimensions(queries, is_query=True)

        print(f"\nQuery batch: {q_batch} samples, {q_seq} tokens each")

        assert q_seq == 1024

        # Estimate memory usage
        print("\nTesting memory estimation for queries...")
        token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
            queries, is_query=True, embedding_dim=128, bytes_per_token=4
        )

        print(f"Memory needed: {total_mem / 1024**2:.2f} MB")

        assert total_mem / 1024**2 > 2, f"Expected memory usage to be greater than 2MB, but got {total_mem / 1024**2}MB"

        # Test batch optimization
        print("\nTesting batch optimization...")
        available_memory = 1024 * 1024 * 1024  # 1GB

        optimal_size = estimator.get_optimal_batch_size(
            queries[0], is_query=True,
            available_bytes=available_memory,
            embedding_dim=128,
            bytes_per_token=4
        )

        print(f"Optimal batch size for 1GB: {optimal_size}")

        assert optimal_size == 674, f"Expected optimal size to be 674, but got {optimal_size}"

    def test_basic_document(self):
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            document_length=8192
        )

        documents = ["ML is a field of AI...", "Neural networks are computing systems..."]

        # Estimate dimensions
        d_batch, d_seq = estimator.estimate_dimensions(documents, is_query=False)

        print(f"\nDocument batch: {d_batch} samples, {d_seq} tokens each")

        # Estimate memory usage
        print("\nTesting memory estimation for documents...")
        token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
            documents, is_query=False, embedding_dim=128, bytes_per_token=4
        )

        print(f"Memory needed: {total_mem / 1024**2:.2f} MB")

        assert total_mem / 1024**2 > 0, f"Expected memory usage to be greater than 0MB, but got {total_mem / 1024**2}MB"

        # Test batch optimization
        print("\nTesting batch optimization...")
        available_memory = 1024 * 1024 * 1024  # 1GB

        optimal_size = estimator.get_optimal_batch_size(
            documents[0], is_query=False,
            available_bytes=available_memory,
            embedding_dim=128,
            bytes_per_token=4
        )

        print(f"Optimal batch size for 1GB: {optimal_size}")

        assert optimal_size == 62814, f"Expected optimal size to be 62814, but got {optimal_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
