"""
Pytest configuration and shared fixtures for the test suite.

This file is automatically loaded by pytest and provides:
- Custom markers configuration
- Shared fixtures for tests
- Test environment setup

Run all tests:
    pytest tests/ -v

Run without model-dependent tests:
    pytest tests/ -v -m "not requires_model"

Run with coverage:
    pytest tests/ -v --cov=. --cov-report=html
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_model: mark test as requiring a loaded model (may be slow)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Return the data directory path."""
    return PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def results_dir():
    """Return the results directory path."""
    return PROJECT_ROOT / "results"


@pytest.fixture
def sample_cot_gpt_oss():
    """Sample Chain-of-Thought output from GPT-OSS model."""
    return """<|start|>analysis<|message|>Let me think about this step by step.

First, I need to understand the problem. We have a ratio of 2:3 for boys to girls.

The total number of parts is 2 + 3 = 5 parts.

If there are 30 students total, each part represents 30/5 = 6 students.

Therefore, boys = 2 * 6 = 12 and girls = 3 * 6 = 18.

The difference is 18 - 12 = 6.<|end|><|start|>final<|message|>The answer is \\boxed{6}."""


@pytest.fixture
def sample_cot_deepseek():
    """Sample Chain-of-Thought output from DeepSeek model."""
    return """<think>Let me solve this problem step by step.

The ratio of boys to girls is 2:3.
Total parts = 2 + 3 = 5.
Total students = 30.
Each part = 30/5 = 6 students.

Boys = 2 * 6 = 12.
Girls = 3 * 6 = 18.

Difference = 18 - 12 = 6.
</think>

The answer is \\boxed{6}."""


@pytest.fixture
def sample_sentences():
    """Sample list of sentences for testing."""
    return [
        "Let me think about this step by step.",
        " First, I need to understand the problem.",
        " The ratio is 2:3.",
        " Total parts is 5.",
        " Each part is 6 students.",
        " Therefore, \\boxed{6}."
    ]


@pytest.fixture
def sample_grad_matrix():
    """Sample gradient matrix for testing."""
    import torch

    # Create a realistic-looking lower triangular gradient matrix
    # where earlier sentences have some influence on later ones
    n = 6
    matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i):
            # Simulate decreasing influence with distance
            matrix[i, j] = 0.5 ** (i - j) + 0.1 * torch.rand(1).item()

    return matrix


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    from unittest.mock import Mock
    import torch

    tokenizer = Mock()
    tokenizer.name_or_path = "mock-model"

    def mock_call(text, **kwargs):
        # Simple token count based on word count
        words = text.split()
        ids = list(range(len(words)))
        return {"input_ids": torch.tensor([ids])}

    def mock_encode(text, **kwargs):
        words = text.split()
        return list(range(len(words)))

    def mock_decode(ids, **kwargs):
        return " ".join(f"token{i}" for i in ids)

    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    tokenizer.encode = mock_encode
    tokenizer.decode = mock_decode

    return tokenizer


# Skip markers for conditional test execution
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle conditional skipping."""
    import torch

    # Check if CUDA is available for model tests
    cuda_available = torch.cuda.is_available()

    for item in items:
        # Skip model tests if no CUDA
        if "requires_model" in item.keywords and not cuda_available:
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        # Check if data files exist for data tests
        if "check_data_exists" in item.fixturenames:
            data_path = PROJECT_ROOT / "data" / "HARP.jsonl"
            if not data_path.exists():
                item.add_marker(pytest.mark.skip(reason="HARP data not available"))
