import glob
import json
import logging
import os
from typing import Optional, Any, Dict


def _load_results(
    base_filename: str,
    model_name: str,
    difficulty: Optional[int] = None,
    dataset: str = "harp-standard",
    suffix: str = "",
    seed: int = 42,
    subdir: str = "",
):
    results_dir = os.path.join("results", model_name, subdir) if subdir else os.path.join("results", model_name)

    if difficulty is not None:
        files = [
            os.path.join(
                results_dir,
                f"{dataset}_{base_filename}_difficulty={difficulty}_seed={seed}{suffix}.json",
            )
        ]
    else:
        pattern = os.path.join(
            results_dir,
            f"{dataset}_{base_filename}_difficulty=*_seed={seed}{suffix}.json",
        )
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                results_dir, f"{dataset}_{base_filename}_seed={seed}{suffix}.json"
            )
            files = sorted(glob.glob(pattern))

    all_results = {}
    all_metadata = {}
    for fp in files:
        diff_results, metadata = json.load(open(fp, "r"))
        all_results.update(diff_results)
        all_metadata[fp] = metadata
    return all_results, all_metadata


def load_generations(model_name: str, dataset: str = "harp-standard") -> Dict[str, Any]:
    """
    Load generation results from JSON files.

    Args:
        model_name: Name of the model directory
        difficulty: Optional difficulty level to filter by
        dataset: Dataset identifier (default: "harp-standard")

    Returns:
        Tuple of (results dict, metadata dict)
    """
    results_dir = os.path.join("results", model_name)

    # Find generation files
    pattern = f"{dataset}_generations_*.json"
    gen_files = sorted(glob.glob(os.path.join(results_dir, pattern)))

    if not gen_files:
        raise FileNotFoundError(
            f"No generation files found in {results_dir} matching pattern {pattern}. "
            "Run generate_reasoning_chains.py first."
        )

    # Load and merge results from all matching files
    all_results = {}
    all_metadata = {}
    for fp in gen_files:
        logging.info(f"Loading generations from {fp}")
        with open(fp, "r") as f:
            data = json.load(f)
            if isinstance(data, (list, tuple)) and len(data) == 2:
                results, metadata = data
            else:
                results = data
                metadata = {}
            # Store with file path prefix to avoid key collisions
            for idx, (key, entry) in enumerate(results.items()):
                entry_key = f"{fp}:{f'{idx}'.zfill(5)}"
                all_results[entry_key] = entry
                all_results[entry_key]["source_file"] = fp
            all_metadata[fp] = metadata

    logging.info(f"Loaded {len(all_results)} total entries from {len(gen_files)} files")
    return all_results, all_metadata


def load_single_step_results(
    model_name: str,
    difficulty: Optional[int] = None,
    seed: int = 42,
    dataset: str = "harp-standard",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return _load_results(
        "processed_sentences", model_name, difficulty, dataset, seed=seed
    )


def load_attribution_pruning_results(
    model_name: str,
    difficulty: Optional[int] = None,
    dataset: str = "harp-standard",
    suffix: str = "",
):
    return _load_results(
        "attribution_pruning", model_name, difficulty, dataset, suffix=suffix
    )


def load_llm_pruning_results(
    model_name: str, difficulty: Optional[int] = None, dataset: str = "harp-standard"
):
    return _load_results("llm_pruning", model_name, difficulty, dataset)


def load_random_pruning_results(
    model_name: str, difficulty: Optional[int] = None, dataset: str = "harp-standard"
):
    return _load_results("random_pruning", model_name, difficulty, dataset)

