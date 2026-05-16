"""
Utility functions for experiments on reasoning step importance and removability.
"""

import logging
from typing import List, Dict, Optional, Tuple

import torch

from consts import (
    GREEDY_KEY,
    OBTAINABILITY_THRESHOLD,
    SUFFICIENCY_THRESHOLD,
    VALID_PREFIX_SENT_LIMITS,
    THRESHOLDS_KEY,
)
from cot_evaluations import is_good_prune
from cot_utils import apply_step_mask_to_cot


def get_sufficient_prefix_data(
    entry_key: str,
    processed_sentences: Dict,
    attribution_pruning_results: Dict,
    model_name,
) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    """
    Get relevant data on the sufficient prefix for a given entry.
    Get the sufficient prefix prompt, sentences, and token borders for an entry.

    Returns:
        - prefix_length (in reasoning steps)
        - prefix_prompt (the prompt corresponding to the sufficient prefix, all prefix steps concatenated)
        - prefix_sentences (list of steps in the sufficient prefix)
        - prefix_token_borders (list of (start_token_idx, end_token_idx) for each step in the sufficient prefix, starting at 0 for the first step in the prefix)
    """
    sent_entry = processed_sentences[entry_key]
    attr_entry = attribution_pruning_results[entry_key]

    prefix_length = attr_entry["prefix_length"]
    clean_prompt = sent_entry["post_removal"]["full_prompt"]
    clean_sentences = sent_entry["post_removal"]["sentences"]
    clean_token_borders = sent_entry["post_removal"]["token_borders"]

    # Build prefix prompt
    prefix_mask = torch.zeros(len(clean_sentences), dtype=torch.bool)
    prefix_mask[:prefix_length] = True
    prefix_prompt = apply_step_mask_to_cot(
        clean_prompt,
        clean_sentences,
        prefix_mask,
        model_name,
        clean_whitespace=False,
    )

    # Get sentences and token borders for prefix
    prefix_sentences = clean_sentences[:prefix_length]
    prefix_token_borders = clean_token_borders[:prefix_length]
    assert len(prefix_sentences) == len(prefix_token_borders)

    return prefix_length, prefix_prompt, prefix_sentences, prefix_token_borders


def get_valid_entries(
    generations: Dict,
    step_results: Dict,
    attr_pruning_results: Dict,
    sufficiency_threshold: float = SUFFICIENCY_THRESHOLD,
    prefix_step_limits: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Get entry keys that have valid sufficient prefixes and removability results."""
    valid_keys = []

    # Filter entries that have all types of compute results.
    keys_iter = (
        set(generations.keys())
        & set(step_results.keys())
        & set(attr_pruning_results.keys())
    )

    # Extra validation for entries
    for entry_key in keys_iter:
        pr = step_results[entry_key]
        ar = attr_pruning_results[entry_key]

        # Make sure the entry wasn't skipped in the pruning stage (e.g. due to errors or un-neccesary CoT)
        if pr.get("skipped", False):
            continue

        # Validate the CoT to be sufficient after removal of shortcut steps
        if pr.get("post_removal_full_cot_suff", 0.0) < sufficiency_threshold:
            continue

        # Validate the CoT still has an answer sentence after removal of shortcut steps
        if pr.get("post_removal", {}).get("answer_sentence_idx", -1) < 0:
            continue

        # Make sure the prefix length is within limits (if specified)
        if ar.get("prefix_length") is None:
            continue
        if prefix_step_limits:
            plen = ar["prefix_length"]
            if plen < prefix_step_limits[0] or plen > prefix_step_limits[1]:
                continue

        valid_keys.append(entry_key)

    return valid_keys


def extract_removable_nonremovable_entry_keys(
    generations: Dict,
    attr_pruning_results: Dict,
    step_results: Dict,
    suff_threshold: Optional[float] = SUFFICIENCY_THRESHOLD,
    obtn_threshold: Optional[float] = OBTAINABILITY_THRESHOLD,
) -> Dict[str, Dict[int, bool]]:
    """
    Extract reasoning step removability labels for all valid entries.

    Note that because of our chosen definition for step removability, not all steps will neccesarily get a
    true / false label, and thus might be missing from this dictionary. See paper appendix for discussion.

    Returns:
        - A dict of:
            entry_key -> {step_idx -> is_removable (True for removable, False for non-removable)}
    """
    valid_keys = get_valid_entries(
        generations,
        step_results,
        attr_pruning_results,
        suff_threshold,
        prefix_step_limits=VALID_PREFIX_SENT_LIMITS,
    )
    results = {}

    for entry_key in valid_keys:
        attr_pruning_entry = attr_pruning_results[entry_key]
        processed_entry = step_results[entry_key]

        prefix_len = attr_pruning_entry["prefix_length"]
        sentences = processed_entry.get("post_removal", {}).get("sentences", [])

        is_removable_labels = {}

        # Sentences are marked as removable only if they are dropped in the thresholding stage
        threshold_results = attr_pruning_entry.get(THRESHOLDS_KEY, [])
        if not threshold_results:
            logging.debug(f"No removability threshold results found for entry")
            continue

        threshold_mask = None
        for result in threshold_results:
            if is_good_prune(
                result["suff"], result["obtn"], suff_threshold, obtn_threshold
            ):
                # Exclude last sentence (answer suffix) from mask
                threshold_mask = result["mask"][:-1]
                break
        if threshold_mask is not None:
            for idx in range(min(prefix_len, len(sentences))):
                if threshold_mask[idx] == 0:
                    # Sentence is labeled as removable by the thresholding stage
                    is_removable_labels[idx] = True
                    pass

        # Sentences are marked as non-removable only if they are kept in the end of the iterative stage
        greedy_pruning_results = attr_pruning_entry.get(GREEDY_KEY, [])
        if not greedy_pruning_results:
            logging.debug(f"No greedy pruning results found for entry")
            continue

        pruning_mask = None
        for result in greedy_pruning_results[::-1]:
            if is_good_prune(
                result["suff"], result["obtn"], suff_threshold, obtn_threshold
            ):
                # Exclude last sentence (answer suffix) from mask
                pruning_mask = result["mask"][:-1]
                break
        if pruning_mask is not None:
            for idx in range(min(prefix_len, len(sentences))):
                if pruning_mask[idx] == 1:
                    # Sentence is labeled as non-removable by the end of the iterative stage
                    is_removable_labels[idx] = False
                # else:
                # is_removable_labels[idx] = True

        if is_removable_labels:
            results[entry_key] = is_removable_labels

    logging.info(
        f"Extracted removability labels for {len(results)} entries "
        f"({sum(sum(1 for v in d.values() if v) for d in results.values())} removable, "
        f"{sum(sum(1 for v in d.values() if not v) for d in results.values())} non-removable)"
    )
    return results


def get_min_passing_n_kept(
    grouped_results, iterative_results, prefix_length, suff_threshold, obtn_threshold
):
    """
    Get the minimal number of steps required to cross sufficiency and attainability thresholds,
    based on previously-computed pruning results (grouped_results & iterative results, each corresponding
    to one of the phases in Section 3.1 in the paper).

    Args:
    - grouped_results: list of pruning results from the grouped pruning stage (First phase in 3.1), ordered from most pruned to least pruned.
    - iterative_results: list of pruning results from the greedy pruning stage (second phase in 3.1), ordered from least pruned to most pruned.
    - prefix_length: the original prefix length (number of reasoning steps)
    - suff_threshold: the sufficiency threshold to determine if a prune is good
    - obtn_threshold: the obtainability threshold to determine if a prune is good
    """
    # First iterate over the iterative results, in reversed order (from most pruned to least pruned)
    for res in reversed(iterative_results):
        if is_good_prune(res["suff"], res["obtn"], suff_threshold, obtn_threshold):
            return res["n_kept"]

    # If no iterative result passes, check the grouped results (which are more aggressive pruning steps)
    # Note the grouped results aren't reversed: they start from the most pruned and go towards less pruned.
    n_kept_grouped = [res["n_kept"] for res in grouped_results]
    assert n_kept_grouped == sorted(n_kept_grouped)
    for res in grouped_results:
        if is_good_prune(res["suff"], res["obtn"], suff_threshold, obtn_threshold):
            return res["n_kept"]

    # No prune passes the thresholds, so we return the original prefix length (no pruning)
    return prefix_length
