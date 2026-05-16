"""
Unit tests for data loading utilities.

This module contains tests for:
- harp_dataset.py: load_harp_dataset

Run with:
    pytest tests/test_data.py -v

Or run all tests:
    pytest tests/ -v
"""

import pytest
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Tests for harp_dataset.py
# =============================================================================


class TestLoadHarpDataset:
    """Tests for the load_harp_dataset function."""

    @pytest.fixture(scope="class")
    def check_data_exists(self):
        """Check if HARP data files exist."""
        data_path = PROJECT_ROOT / "data" / "HARP.jsonl"
        if not data_path.exists():
            pytest.skip("HARP dataset not available")
        return True

    def test_load_standard_split(self, check_data_exists):
        """Test loading the standard split."""
        from harp_dataset import load_harp_dataset

        dataset = load_harp_dataset(split="standard")

        assert dataset is not None
        assert len(dataset) > 0
        # Check expected columns exist
        assert "problem" in dataset.column_names
        assert "answer" in dataset.column_names

    def test_load_mcq_split(self, check_data_exists):
        """Test loading the MCQ split."""
        from harp_dataset import load_harp_dataset

        # Check if MCQ file exists
        mcq_path = PROJECT_ROOT / "data" / "HARP_mcq.jsonl"
        if not mcq_path.exists():
            pytest.skip("HARP MCQ dataset not available")

        dataset = load_harp_dataset(split="mcq")

        assert dataset is not None
        assert len(dataset) > 0
        # MCQ split should have choices embedded in problem
        first_problem = dataset[0]["problem"]
        assert "(" in first_problem  # Should contain choice markers

    def test_load_with_difficulty_filter(self, check_data_exists):
        """Test loading with difficulty filter."""
        from harp_dataset import load_harp_dataset

        dataset_full = load_harp_dataset(split="standard")
        dataset_filtered = load_harp_dataset(split="standard", difficulty=3)

        assert len(dataset_filtered) <= len(dataset_full)
        # All entries should have level = 3
        for entry in dataset_filtered:
            assert entry["level"] == 3

    def test_difficulty_range(self, check_data_exists):
        """Test that difficulty levels 1-6 are valid."""
        from harp_dataset import load_harp_dataset

        for difficulty in [1, 2, 3, 4, 5, 6]:
            dataset = load_harp_dataset(split="standard", difficulty=difficulty)
            # Should either have entries or be empty (but not error)
            assert dataset is not None

    def test_invalid_difficulty_raises(self, check_data_exists):
        """Test that invalid difficulty raises assertion error."""
        from harp_dataset import load_harp_dataset

        with pytest.raises(AssertionError):
            load_harp_dataset(split="standard", difficulty=0)

        with pytest.raises(AssertionError):
            load_harp_dataset(split="standard", difficulty=7)

    def test_invalid_split_raises(self, check_data_exists):
        """Test that invalid split raises assertion error."""
        from harp_dataset import load_harp_dataset

        with pytest.raises(AssertionError):
            load_harp_dataset(split="invalid_split")

    def test_dataset_has_required_fields(self, check_data_exists):
        """Test that dataset entries have required fields."""
        from harp_dataset import load_harp_dataset

        dataset = load_harp_dataset(split="standard", difficulty=1)

        if len(dataset) > 0:
            entry = dataset[0]
            # Check required fields
            assert "problem" in entry
            assert "answer" in entry
            assert "level" in entry

    def test_mcq_formatting(self, check_data_exists):
        """Test that MCQ problems are properly formatted."""
        from harp_dataset import load_harp_dataset

        mcq_path = PROJECT_ROOT / "data" / "HARP_mcq.jsonl"
        if not mcq_path.exists():
            pytest.skip("HARP MCQ dataset not available")

        dataset = load_harp_dataset(split="mcq")

        if len(dataset) > 0:
            problem = dataset[0]["problem"]
            # MCQ should have lettered choices
            has_choices = any(f"({letter})" in problem for letter in "ABCDE")
            assert has_choices, "MCQ problem should contain lettered choices"


# =============================================================================
# Tests for loading_utils.py (if it exists)
# =============================================================================


class TestLoadingUtils:
    """Tests for loading_utils functions."""

    def test_load_pruning_results_interface(self):
        """Test load_pruning_results function exists and has correct interface."""
        try:
            from loading_utils import load_pruning_results
            import inspect

            sig = inspect.signature(load_pruning_results)
            params = list(sig.parameters.keys())
            assert "model_name" in params
        except ImportError:
            pytest.skip("loading_utils not available")

    def test_load_attribution_entries_interface(self):
        """Test load_attribution_entries function exists and has correct interface."""
        try:
            from loading_utils import load_attribution_entries
            import inspect

            sig = inspect.signature(load_attribution_entries)
            params = list(sig.parameters.keys())
            assert "model_name" in params
        except ImportError:
            pytest.skip("loading_utils not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
