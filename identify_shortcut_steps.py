"""
Process reasoning chains to identify and remove shortcut steps.

This script:
1. Loads generation results from generate_reasoning_chains.py
2. Splits outputs into sentences (steps) and computes token borders
3. Identifies shortcut sentences (those that alone can produce the correct answer)
4. Removes shortcut sentences
5. Saves both pre- and post-removal data (prompt, sentences, token borders)
"""

import sys
import argparse
import glob
import json
import logging
import os
import torch
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_utils import (
    AnswerError,
    apply_step_mask_to_cot,
    cot_sentences_and_token_borders,
    extract_cot_from_output,
    search_answer,
    sentence_states_answer,
    get_prefill_ending,
)
from loading_utils import load_generations
from cot_evaluations import eval_sufficiency, eval_cot_neccesity
from general_utils import (
    save_results,
    load_model,
    perplexity,
    set_deterministic,
)
from consts import (
    MODEL_NAME_TO_ID_PATH,
    SUFFICIENCY_THRESHOLD,
    NECCESITY_THRESHOLD,
    N_SUFFICINECY_EVAL_RESAMPLES,
)

torch.set_grad_enabled(False)


def find_answer_sentence(
    sentences: List[str], gt_answer: str, verbose: bool = True
) -> int:
    """
    Find the index of the answer sentence in the list of sentences.
    Searches for boxed answers (\\boxed{XXX} format) first, then falls back to plain text search.
    """
    if verbose:
        logging.info("Locating answer sentence in the output...")
    answer_sentence_idx, answer_sentence = search_answer(
        sentences, gt_answer, boxed=True
    )

    if answer_sentence_idx == AnswerError.WRONG_ANSWER:
        if verbose:
            logging.info(
                f"WRONG answer found: GT ({gt_answer}) != Found ({answer_sentence})"
            )
        return -1
    elif answer_sentence_idx == AnswerError.ANSWER_NOT_FOUND:
        if verbose:
            logging.info(
                "Failed to find boxed answer, searching for plain text answer..."
            )
        answer_sentence_idx, answer_sentence = search_answer(
            sentences, gt_answer, boxed=False
        )

    if verbose:
        if answer_sentence_idx == AnswerError.ANSWER_NOT_FOUND:
            logging.info("Failed to find answer in the output.")
        else:
            logging.info(
                f"Identified answer ({gt_answer=}) sentence ({answer_sentence_idx}): {answer_sentence}"
            )
    return answer_sentence_idx


def find_shortcut_steps(
    full_prompt: str,
    sentences: List[str],
    answer_sentence_idx: int,
    gt_answer: str,
    model,
    tokenizer,
    seed: int,
    n_resamples: int = N_SUFFICINECY_EVAL_RESAMPLES,
    sufficiency_threshold: float = SUFFICIENCY_THRESHOLD,
    verbose: bool = True,
) -> List[int]:
    """
    Find shotcut steps.

    A sentence (or step) is a shortcut if:
    1. It directly states the answer (heuristic check), OR
    2. The model can produce the correct answer using only that sentence (empirical check)

    Args:
        full_prompt: The full prompt (input + CoT)
        sentences: List of sentences in the CoT
        answer_sentence_idx: Index of the actual answer sentence
        gt_answer: Ground truth answer
        model: Language model
        tokenizer: Tokenizer
        seed: Random seed for reproducibility
        n_resamples: Number of resamples for sufficiency evaluation
        sufficiency_threshold: Threshold for sufficiency success
        verbose: Whether to log verbose output

    Returns:
        List of indices of shortcut sentences
    """
    sufficient_sentences = []

    for sentence_idx in range(len(sentences)):
        if sentence_idx == answer_sentence_idx:
            # Skip the actual answer sentence
            continue

        # Heuristic check: does the sentence directly state the answer?
        if sentence_states_answer(sentences[sentence_idx], gt_answer):
            if verbose:
                logging.info(
                    f"Sentence {sentence_idx} is heuristically sufficient: "
                    f"{sentences[sentence_idx][:100]}..."
                )
            sufficient_sentences.append(sentence_idx)
            continue

        # Empirical check: can the model produce the correct answer from this sentence alone?
        single_sentence_mask = torch.zeros(len(sentences), dtype=torch.bool)
        single_sentence_mask[sentence_idx] = True

        suff = eval_sufficiency(
            full_prompt,
            sentences,
            single_sentence_mask,
            answer_sentence_idx,
            gt_answer,
            model,
            tokenizer,
            n_resamples=n_resamples,
            seed=seed,
            verbose=False,
        )[0]

        if suff >= sufficiency_threshold:
            if verbose:
                logging.info(
                    f"Sentence {sentence_idx} is empirically sufficient (suff={suff:.2f}): "
                    f"{sentences[sentence_idx][:100]}..."
                )
            sufficient_sentences.append(sentence_idx)

    return sufficient_sentences


def remove_shortcut_steps(
    full_prompt: str,
    sentences: List[str],
    token_borders: List[tuple],
    answer_sentence_idx: int,
    sufficient_sentences: List[int],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """
    Remove shortcut sentences and return the cleaned data.

    Args:
        full_prompt: The full prompt (input + CoT)
        sentences: List of sentences in the CoT
        token_borders: Token border indices for each sentence
        answer_sentence_idx: Index of the answer sentence
        sufficient_sentences: Indices of shortcut sentences to remove
        model_name: Name of the model (for prompt formatting)

    Returns:
        Tuple of cleaned prompt, sentences, token_borders, and answer_sentence_idx.
    """
    if not sufficient_sentences:
        # Nothing to remove
        return full_prompt, sentences, token_borders, answer_sentence_idx

    # Create mask for remaining sentences
    remaining_indices = [
        i for i in range(len(sentences)) if i not in sufficient_sentences
    ]
    remaining_mask = torch.zeros(len(sentences))
    remaining_mask[remaining_indices] = 1

    # Apply mask to prompt
    cleaned_prompt = apply_step_mask_to_cot(
        full_prompt, sentences, remaining_mask, model.model_name, clean_whitespace=False
    )

    # Get new sentences and token borders
    cleaned_sentences, cleaned_token_borders = cot_sentences_and_token_borders(
        cleaned_prompt, model, tokenizer
    )
    assert cleaned_token_borders[0][0] != 0, "Token borders extraction failed..."

    # Update answer sentence index
    if sentences[answer_sentence_idx] in cleaned_sentences:
        cleaned_answer_idx = cleaned_sentences.index(sentences[answer_sentence_idx])
    else:
        # Fallback - strip and search again
        stripped_cleaned_sentences = [s.strip() for s in cleaned_sentences]
        stripped_answer_sentence = sentences[answer_sentence_idx].strip()
        if stripped_answer_sentence in stripped_cleaned_sentences:
            cleaned_answer_idx = stripped_cleaned_sentences.index(
                stripped_answer_sentence
            )
        else:
            # Fallback #2 - search if answer sentence is CONTAINED within one of the cleaned sentences (might happen if a sentence was merged with answer sentence)
            for idx, sent in enumerate(reversed(stripped_cleaned_sentences)):
                if stripped_answer_sentence in sent:
                    cleaned_answer_idx = len(stripped_cleaned_sentences) - 1 - idx
                    break

    return cleaned_prompt, cleaned_sentences, cleaned_token_borders, cleaned_answer_idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process reasoning chains to identify and remove shortcut sentences"
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset identifier",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=None,
        help="Difficulty level to filter (if applicable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=N_SUFFICINECY_EVAL_RESAMPLES,
        help="Number of resamples for sufficiency evaluation",
    )
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=SUFFICIENCY_THRESHOLD,
        help="Threshold for sufficiency success",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model_id, model_path = MODEL_NAME_TO_ID_PATH[args.model_name]
    logging.info(f"Loading model {model_id} from {model_path}")
    model, tokenizer = load_model(
        model_path,
        dtype=(torch.bfloat16 if "gpt" not in model_id.lower() else None),
        device_map="auto",
    )
    model.model_name = args.model_name

    # Load generation results
    logging.info("Loading generation results...")
    ds_name = args.dataset.split("/")[-1]
    generations, gen_metadata = load_generations(
        args.model_name, args.difficulty, dataset=args.dataset
    )

    # Prepare output
    output_filename = f"{ds_name}{f'_difficulty={args.difficulty}' if args.difficulty else ''}_seed={args.seed}_processed_sentences.json"
    output_path = os.path.join(
        "results",
        args.model_name,
        output_filename,
    )

    metadata = {
        "model_id": model_id,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "sufficiency_threshold": args.sufficiency_threshold,
        "generation_metadata": gen_metadata,
    }

    # Load existing results if available
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        with open(output_path, "r") as f:
            data = json.load(f)
            if isinstance(data, (list, tuple)) and len(data) == 2:
                outputs, cached_metadata = data
                assert cached_metadata == metadata, "Metadata mismatch"
            else:
                outputs = {}
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        outputs = {}

    # Process each generation
    verbose = not args.quiet
    processed_count = 0
    error_count = 0
    for entry_key, entry in tqdm(generations.items(), desc="Processing entries"):
        # Skip if already processed
        if entry_key in outputs:
            if verbose:
                logging.info(f"Entry {entry_key} already processed, skipping")
            continue

        # Extract entry data
        output = entry["output"]
        full_prompt = entry["full_prompt"]
        full_input = entry["full_input"]
        gt_answer = entry["gt_answer"]
        is_cot_complete = entry.get("is_cot_complete", True)

        if gt_answer is None:
            logging.warning(f"Entry {entry_key}: No ground truth answer, skipping")
            outputs[entry_key] = {"skipped": True, "reason": "no_gt_answer"}
            continue

        if not is_cot_complete:
            logging.warning(f"Entry {entry_key}: CoT incomplete, skipping")
            outputs[entry_key] = {"skipped": True, "reason": "cot_incomplete"}
            continue

        cot_neccesity = eval_cot_neccesity(full_input, model, tokenizer, gt_answer)
        logging.info(f"Entry {entry_key}: CoT Neccesity: {cot_neccesity :.2f}")
        if cot_neccesity < NECCESITY_THRESHOLD:
            if verbose:
                logging.warning(
                    f"Entry {entry_key}: Skipping because CoT neccesity {cot_neccesity :.2f} < {NECCESITY_THRESHOLD}"
                )
            outputs[entry_key] = {
                "skipped": True,
                "reason": "cot_neccesity_low",
                "neccesity": cot_neccesity,
            }
            continue

        # Split output into sentences and get token borders
        sentences, token_borders = cot_sentences_and_token_borders(
            full_prompt, model, tokenizer
        )

        if len(sentences) == 0:
            logging.warning(f"Entry {entry_key}: No sentences found, skipping")
            outputs[entry_key] = {"skipped": True, "reason": "no_sentences"}
            continue

        # Find the answer sentence
        answer_sentence_idx = find_answer_sentence(sentences, gt_answer, verbose)

        if answer_sentence_idx == -1:
            logging.warning(
                f"Entry {entry_key}: Answer not found in sentences, skipping"
            )
            outputs[entry_key] = {"skipped": True, "reason": "answer_not_found"}
            continue

        # Store original data (before removal)
        original_data = {
            "full_prompt": full_prompt,
            "sentences": sentences,
            "token_borders": token_borders,
            "answer_sentence_idx": answer_sentence_idx,
        }

        # Find shortcut sentences
        if verbose:
            logging.info(
                f"Entry {entry_key}: Finding shortcut sentences (out of {len(sentences)} sentences)..."
            )

        set_deterministic(args.seed)
        sufficient_sentences = find_shortcut_steps(
            full_prompt=full_prompt,
            sentences=sentences,
            answer_sentence_idx=answer_sentence_idx,
            gt_answer=gt_answer,
            model=model,
            tokenizer=tokenizer,
            seed=args.seed,
            n_resamples=args.n_resamples,
            sufficiency_threshold=args.sufficiency_threshold,
            verbose=verbose,
        )

        logging.info(
            f"Entry {entry_key}: Found {len(sufficient_sentences)}/{len(sentences)} shortcut sentences"
        )

        # Remove shortcut sentences
        (
            cleaned_full_prompt,
            cleaned_sentences,
            cleaned_token_borders,
            cleaned_answer_idx,
        ) = remove_shortcut_steps(
            full_prompt=full_prompt,
            sentences=sentences,
            token_borders=token_borders,
            answer_sentence_idx=answer_sentence_idx,
            sufficient_sentences=sufficient_sentences,
            model=model,
            tokenizer=tokenizer,
        )

        # Eval sufficiency post-removal
        post_removal_full_cot_suff = eval_sufficiency(
            cleaned_full_prompt,
            cleaned_sentences,
            torch.ones(len(cleaned_sentences), dtype=torch.bool),
            cleaned_answer_idx,
            gt_answer,
            model,
            tokenizer,
            n_resamples=args.n_resamples,
            seed=args.seed,
            verbose=False,
        )[0]

        # Eval perplexities for full cot and empty cot
        cot_start_idx = extract_cot_from_output(cleaned_full_prompt, args.model_name)[1]
        no_cot_direct_answer_prompt = (
            cleaned_full_prompt[:cot_start_idx]
            + get_prefill_ending(model.model_name)
            + gt_answer
            + "}"
        )
        full_cot_ppl = perplexity(full_prompt, model, tokenizer)
        cleaned_full_cot_ppl = perplexity(cleaned_full_prompt, model, tokenizer)
        empty_cot_ppl = perplexity(no_cot_direct_answer_prompt, model, tokenizer)

        # Save output entry
        output_entry = {
            "input_str": entry["input_str"],
            "gt_answer": gt_answer,
            "pre_removal": original_data,  # full prompt, sentences, token borders, answer idx prior to removal of self-sufficient sentences;
            "post_removal": {
                "full_prompt": cleaned_full_prompt,
                "sentences": cleaned_sentences,
                "token_borders": cleaned_token_borders,
                "answer_sentence_idx": cleaned_answer_idx,
            },
            "sufficient_sentence_indices": sufficient_sentences,
            "n_original_sentences": len(sentences),
            "n_cleaned_sentences": len(cleaned_sentences),
            "n_removed_sentences": len(sufficient_sentences),
            "neccesity": round(cot_neccesity, 4),
            "pre_removal_full_cot_ppl": round(full_cot_ppl, 4),
            "post_removal_full_cot_ppl": round(cleaned_full_cot_ppl, 4),
            "empty_cot_ppl": round(empty_cot_ppl, 4),
            "post_removal_full_cot_suff": round(post_removal_full_cot_suff, 4),
        }

        outputs[entry_key] = output_entry
        processed_count += 1

        # Save incrementally
        save_results(outputs, metadata, output_path)

    # Save once in the end
    save_results(outputs, metadata, output_path)

    # Print summary
    logging.info(f"\n=== Summary ===")
    logging.info(f"Processed: {processed_count}")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Total in output: {len(outputs)}")
    logging.info(f"Results saved to: {output_path}")

    # Print statistics on shortcut sentences
    valid_entries = [
        e
        for e in outputs.values()
        if not e.get("skipped", False) and "n_removed_sentences" in e
    ]
    if valid_entries:
        total_original = sum(e["n_original_sentences"] for e in valid_entries)
        total_removed = sum(e["n_removed_sentences"] for e in valid_entries)
        entries_with_removal = sum(
            1 for e in valid_entries if e["n_removed_sentences"] > 0
        )

        logging.info(f"\n=== Shortcut Sentence Statistics ===")
        logging.info(f"Total sentences processed: {total_original}")
        logging.info(
            f"Total sentences removed: {total_removed} ({100*total_removed/total_original:.1f}%)"
        )
        logging.info(
            f"Entries with at least one removal: {entries_with_removal}/{len(valid_entries)} "
            f"({100*entries_with_removal/len(valid_entries):.1f}%)"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    main()
