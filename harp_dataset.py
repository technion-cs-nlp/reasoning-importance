import json
from datasets import Dataset
from typing import Optional


def load_harp_dataset(split: str = "standard", difficulty: Optional[int] = None):
    json_paths = {
        "standard": "./data/HARP.jsonl",
        "mcq": "./data/HARP_mcq.jsonl",
        "proof_based": "./data/HARP_proof-based.jsonl",
        "raw": "./data/HARP_raw.jsonl",
    }
    assert split in json_paths
    assert split in [
        "standard",
        "mcq",
    ], "Currently only these two splits are needed and supported"

    raw_data_path = json_paths[split]
    with open(raw_data_path, "rb") as f:
        try:
            data = [json.loads(line) for line in f.readlines() if line]
        except Exception as e:
            print(f"Error loading dataset from {raw_data_path}: {e}")
            print(
                f"Did you download the HARP dataset? If not, visit: https://github.com/aadityasingh/HARP"
            )
            raise

    if split == "standard":
        # Problem field is already as-is
        pass
    if split == "mcq":
        for entry in data:
            entry["problem"] = f"{entry['problem']}\n" + "\n".join(
                f"({letter}) {choice}" for letter, choice in entry["choices"].items()
            )

    if difficulty is not None:
        assert 1 <= difficulty <= 6
        data = [entry for entry in data if entry["level"] == difficulty]

    # Convert to huggingface dataset
    data = Dataset.from_list(data)
    return data
