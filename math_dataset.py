from datasets import load_dataset, Dataset


def load_math_dataset() -> Dataset:
    """Load the MATH-500 dataset from HuggingFace.

    Returns a HuggingFace Dataset with columns "problem" and "answer".
    """
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds
