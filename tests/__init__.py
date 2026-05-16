"""
Unit tests for reasoning-interp project.

Test modules:
- test_cot_utils.py: Tests for CoT utilities (sentence splitting, answer extraction, etc.)
- test_evaluation.py: Tests for evaluation functions (sufficiency, perplexity, etc.)
- test_data.py: Tests for data loading utilities (HARP dataset, loading utils)

To run all tests:
    pytest tests/ -v

To run specific test file:
    pytest tests/test_cot_utils.py -v

To run tests without model requirement:
    pytest tests/ -v -m "not requires_model"
"""
