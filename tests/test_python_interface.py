#!/usr/bin/env python3

import pytest
import random
import string

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
            query_length=256,
            document_length=1024
        )

        queries = ["What is machine learning?", "How do neural networks work?"]

        # Estimate dimensions
        q_batch, q_seq = estimator.estimate_dimensions(queries, is_query=True)

        print(f"\nQuery batch: {q_batch} samples, {q_seq} tokens each")

        assert q_seq == 256

        # Estimate memory usage
        print("\nTesting memory estimation for queries...")
        token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
            queries, is_query=True
        )

        print(f"Memory needed: {total_mem / 1024**2:.2f} MB")

        assert total_mem / 1024**2 > 2, f"Expected memory usage to be greater than 2MB, but got {total_mem / 1024**2}MB"


    def test_basic_document(self):
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=256,
            document_length=8192
        )

        documents = ["ML is a field of AI...", "Neural networks are computing systems..."]

        # Estimate dimensions
        d_batch, d_seq = estimator.estimate_dimensions(documents, is_query=False)

        print(f"\nDocument batch: {d_batch} samples, {d_seq} tokens each")

        # Estimate memory usage
        print("\nTesting memory estimation for documents...")
        token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
            documents, is_query=False
        )

        print(f"Memory needed: {total_mem / 1024**2:.2f} MB")

        assert total_mem / 1024**2 > 0, f"Expected memory usage to be greater than 0MB, but got {total_mem / 1024**2}MB"


    def test_split_into_batches_single_batch_fits(self):
        """Test that when all texts fit in memory, they return as a single batch."""
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=128,
            document_length=512
        )

        # Create 8 simple texts
        texts = [f"Simple text {i}" for i in range(8)]

        # Set available_bytes high enough that all 8 texts fit in one batch
        available_bytes = 100 * 1024 * 1024  # 100MB - plenty of memory

        batches = estimator.split_into_batches(
            texts,
            is_query=False,
            available_bytes=available_bytes
        )

        print(f"\nSingle batch test - Number of batches: {len(batches)}")
        print(f"Batch sizes: {[len(batch) for batch in batches]}")

        # Should return a single batch with all 8 texts
        assert len(batches) == 1, f"Expected 1 batch, got {len(batches)}"
        assert len(batches[0]) == 8, f"Expected 8 texts in batch, got {len(batches[0])}"
        assert batches[0] == texts, "Batch should contain all original texts"

    def test_split_into_batches_needs_splitting(self):
        """Test that when texts don't fit together, they get split appropriately."""
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=128,
            document_length=512
        )

        # Create 8 texts
        texts = [f"Text number {i}" for i in range(8)]

        # Set available_bytes low enough to force splitting
        available_bytes = 50 * 1024 * 10  # 50KB - should force splitting, added 10X multiplier to account for overhead

        batches = estimator.split_into_batches(
            texts,
            is_query=False,
            available_bytes=available_bytes
        )

        print(f"\nSplitting test - Number of batches: {len(batches)}")
        print(f"Batch sizes: {[len(batch) for batch in batches]}")

        # Should be split into multiple batches
        assert len(batches) > 1, f"Expected multiple batches, got {len(batches)}"

        # Each batch should be smaller than the original
        for i, batch in enumerate(batches):
            assert len(batch) < 8, f"Batch {i} should be smaller than 8, got {len(batch)}"
            assert len(batch) >= 1, f"Batch {i} should have at least 1 text, got {len(batch)}"

        # All texts should be present across batches
        all_texts = []
        for batch in batches:
            all_texts.extend(batch)

        assert sorted(all_texts) == sorted(texts), "All original texts should be preserved across batches"

    def test_split_into_batches_single_text_too_large(self):
        """Test that when a single text is too large, an error is raised."""
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=128,
            document_length=512
        )

        # Create texts with one very large text
        texts = [
            "Small text 1",
            "Small text 2",
            "Very large text " * 1000,  # This should be too large
            "Small text 4"
        ]

        # Set available_bytes very low
        available_bytes = 1024  # 1KB - extremely small

        # Should raise an error
        with pytest.raises(Exception) as exc_info:
            estimator.split_into_batches(
                texts,
                is_query=False,
                available_bytes=available_bytes
            )

        print(f"\nError test - Exception: {exc_info.value}")
        assert "too large to fit in available memory" in str(exc_info.value)

    def test_split_into_batches_empty_input(self):
        """Test that empty input returns empty batches."""
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=128,
            document_length=512
        )

        batches = estimator.split_into_batches(
            [],  # empty texts
            is_query=False,
            available_bytes=1024
        )

        print(f"\nEmpty input test - Number of batches: {len(batches)}")
        assert len(batches) == 0, f"Expected 0 batches for empty input, got {len(batches)}"

    def test_split_into_batches_power_of_2_splitting(self):
        """Test that the power-of-2 splitting works correctly with larger input."""
        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=128,
            document_length=256  # Smaller to make memory calculations more predictable
        )

        # Create 16 texts to test power-of-2 splitting
        texts = [f"Test document {i}" for i in range(16)]

        # Set memory so that 16 doesn't fit, but smaller batches do
        available_bytes = 20 * 1024 * 100  # 20KB, added 100X multiplier to account for overhead

        batches = estimator.split_into_batches(
            texts,
            is_query=False,
            available_bytes=available_bytes
        )

        print(f"\nPower of 2 test - Number of batches: {len(batches)}")
        print(f"Batch sizes: {[len(batch) for batch in batches]}")

        # Should have multiple batches
        assert len(batches) > 1, f"Expected multiple batches, got {len(batches)}"

        # Verify all original texts are preserved
        all_texts = []
        for batch in batches:
            all_texts.extend(batch)

        assert sorted(all_texts) == sorted(texts), "All original texts should be preserved"

        # Verify that each batch should fit in memory
        for i, batch in enumerate(batches):
            fits = estimator.can_fit_in_memory(
                batch,
                is_query=False,
                available_bytes=available_bytes
            )
            assert fits, f"Batch {i} should fit in available memory but doesn't"

    def test_very_large_batch(self):

        estimator = TokenizationEstimator.from_pretrained(
            repo_id="lightonai/GTE-ModernColBERT-v1",
            query_length=512,
            document_length=8192
        )

        characters = string.ascii_letters + string.digits

        # Generate random strings
        documents_list = [
            "".join(random.choices(characters, k=1000)) for _ in range(250)
        ]

        token_mem, inter_mem, total_mem = estimator.estimate_memory_usage(
            documents_list, is_query=False,
        )
        print(token_mem/1e9, inter_mem/1e9, total_mem/1e9)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
