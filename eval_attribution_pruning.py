"""
Attribution-based greedy pruning evaluation.

This script has two parts:
    1. Finding the sufficient minimal prefix: we iteratively evaluates sufficiency and obtainability as
    sequential sentences are incrementally added the CoT, identifying the first point at which the model
    can produce the correct answer. This point (including all sentences up to it) is considered the "sufficient minimal prefix".

    2. Finding which parts of the prefix can be removed without harming the sufficiency and obtainability of the prefix.
    We do this in several steps:
        2.0. Calculate influence scores from each sentence i to each sentence j (j > i).
        2.1. We find (based on gradient attribution scores) sentences that directly influence the answer
             sentence beyond some threshold, and include them in the mask. We repeat this to find sentences
             that affect any other sentences in the mask, and repeat this recursively. This is done in "prune_cot".
        2.2. We measure the sufficiency and obtainability of the CoT only with the chosen sentences above,
             and repeat the process for increasing thresholds until either all sentences are included or a successful threshold
             is found (i.e. sufficiency and obtainability are above the thresholds).
        2.3. We perform a greedy queue-based removal ordered by attribution (least influential first).
             Starts from the best passing threshold mask. Measure sufficiency and obtainability at
             each step, until the results drop below the thresholds.

Each output entry includes:
    1. The sufficient minimal prefix analysis results ("prefix_length" and "incremental_prefix_results").
    2. The ranking of sentences based on attribution scores ("attribution_ranking")
    3. Results of step 2_1 + 2_2 ("threshold_results")
    5. The number of remaining sentences prior to the greedy removal ("starting_n_kept"),
       number of sentence post greedy removal ("final_n_kept"),
       and the sufficiency and obtainability scores for each greedy removal ("greedy_results").
"""

import json
import os

import sys
import logging
import argparse
from collections import deque
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cot_utils import (
    apply_step_mask_to_cot,
    extract_cot_from_output,
    normalize_ppl,
    get_prefill_ending,
)
from general_utils import (
    load_model,
    save_results,
    set_deterministic,
    perplexity,
    len_tokens,
)
from cot_evaluations import eval_sufficiency, is_good_prune
from consts import (
    GREEDY_KEY,
    MODEL_NAME_TO_ID_PATH,
    OBTAINABILITY_THRESHOLD,
    PREFIX_ITERATION_KEY,
    SUFFICIENCY_THRESHOLD,
    N_SUFFICINECY_EVAL_RESAMPLES,
    THRESHOLDS_KEY,
    PRUNING_THRESHOLDS,
)
from cot_utils import prune_cot
from loading_utils import (
    load_generations,
    load_single_step_results,
)
from gradient_attribution import calculate_output_to_output_gradient_matrix

torch.set_grad_enabled(False)


def compute_sufficient_prefix_results(
    full_prompt,
    sentences,
    answer_sentence_idx,
    gt_answer,
    full_cot_ppl,
    empty_cot_ppl,
    entry_key,
    model,
    tokenizer,
    args,
):
    """
    Iteratively evaluate sufficiency and obtainability for increasing reasoning prefix lengths (in steps).

    Returns:
        incremental_results: List of dicts with sufficiency and obtainability for each prefix length
        prefix_len: The length of the sufficient minimal prefix (in sentences)
    """
    incremental_results = []
    prefix_len = None

    # Iterate over prefix lengths, measuring sufficiency at each step
    for candidate_prefix in range(1, len(sentences) + 1):
        # Evaluate sufficiency with this prefix
        sentence_mask = torch.zeros(len(sentences), dtype=torch.bool)
        sentence_mask[:candidate_prefix] = True
        suff, pruned_prompt, sample_ans = eval_sufficiency(
            full_prompt=full_prompt,
            sentences=sentences,
            sentence_mask=sentence_mask,
            answer_sentence_idx=answer_sentence_idx,
            gt_answer=gt_answer,
            model=model,
            tokenizer=tokenizer,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )

        # Evaluate obtainability (which is 1 - normalized ppl)
        pruned_ppl = perplexity(pruned_prompt, model, tokenizer)
        norm_ppl = normalize_ppl(pruned_ppl, full_cot_ppl, empty_cot_ppl)

        incremental_results.append(
            {
                "prefix_length": candidate_prefix,
                "sufficiency": round(suff, 4),
                "normalized_ppl": round(norm_ppl, 4),
                "obtainability": round(1 - norm_ppl, 4),
                "sample_answer": sample_ans,
            }
        )

        logging.info(
            f"Entry {entry_key} | Prefix {candidate_prefix}/{len(sentences)} | "
            f"Sufficiency: {suff:.2f} | Obtainability: {1 - norm_ppl:.2f}"
        )

        # Mark first sufficient index
        # No need to measure obtainability since the prefix is obtainable by definition
        if prefix_len is None and suff >= args.suff_threshold:
            prefix_len = candidate_prefix
            logging.info(
                f"Entry {entry_key}: First sufficient prefix at {prefix_len}/{len(sentences)} sentences"
            )
            logging.info(f"Sufficient CoT prefix: {pruned_prompt}".replace("\n", "\\n"))
            break

    if not prefix_len:
        logging.info(
            f"Entry {entry_key}: No sufficient prefix found, using full prompt (setting prefix_len={len(sentences)} sentences)"
        )
        prefix_len = len(sentences)

    return incremental_results, prefix_len


def compute_attribution_ranking(
    full_prefix_prompt: str,
    prefix_token_borders: List[Tuple[int, int]],
    cot_start_tok_idx: int,
    model,
    tokenizer,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute gradient attribution matrix and the removal ordering (based on average attribution score
    of each reasoning step on subsequent steps).

    Returns:
        grad_matrix: the sentence-to-sentence influence matrix, where entry (i, j) is the influence of sentence i on sentence j.
        removal_order: list of sentence indices from least to most influential.
    """
    # Adjust token borders relative to CoT start
    adjusted_borders = [
        (s - cot_start_tok_idx, e - cot_start_tok_idx)
        for (s, e) in prefix_token_borders
    ]

    assert (
        adjusted_borders[0][0] == 0
    ), f"First sentence should start at token 0 (got {adjusted_borders[0][0]})"

    logging.info("Computing gradient attribution matrix for prefix...")
    with torch.enable_grad():
        grad_matrix = calculate_output_to_output_gradient_matrix(
            model,
            tokenizer,
            full_prefix_prompt,
            cot_start_tok_idx,
            sentence_token_borders=adjusted_borders,
        )
        grad_matrix = grad_matrix - torch.diag(torch.diag(grad_matrix))

    # Compute mean influence per sentence (column-wise) and rank least to most
    nonzero_counts = (grad_matrix != 0).sum(dim=0).float()
    nonzero_counts[nonzero_counts == 0] = 1  # avoid division by zero
    mean_influence = grad_matrix.sum(dim=0) / nonzero_counts
    removal_order = mean_influence.argsort(descending=False).tolist()

    return grad_matrix, removal_order


def compute_threshold_results(
    influence_matrix: torch.tensor,
    full_prefix_prompt: str,
    prefix_sentences: List[str],
    answer_sentence_idx: int,
    gt_answer: str,
    full_cot_ppl: float,
    empty_cot_ppl: float,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    args,
):
    """
    Iterate over influence thresholds to find sentences to prune, and evaluate sufficiency and obtainability for each threshold.

    Returns:
        threshold_results: List of dicts with threshold, mask, sufficiency, and obtainability for each threshold.
    """
    # Iterate over thresholds until one is succesful
    threshold_results = []
    for threshold in PRUNING_THRESHOLDS:
        # Prune using direct-recursive influence calculation
        mask = prune_cot(
            influence_matrix,
            threshold,
            answer_sentence_idx,
            lambda mat, idx: mat[idx],
            is_recursive=True,
        )

        # Evaluate sufficiency
        set_deterministic(args.seed)
        suff = eval_sufficiency(
            full_prefix_prompt,
            prefix_sentences,
            mask,
            answer_sentence_idx,
            gt_answer,
            model,
            tokenizer,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )[0]

        # Evaluate obtainability (1 - normalized perplexity)
        pruned_prompt = apply_step_mask_to_cot(
            full_prefix_prompt, prefix_sentences, mask, model.model_name
        )
        pruned_ppl = perplexity(pruned_prompt, model, tokenizer)
        obtn = 1 - normalize_ppl(pruned_ppl, full_cot_ppl, empty_cot_ppl)

        threshold_results.append(
            {
                "threshold": threshold,
                "mask": mask.tolist(),
                "n_kept": int(mask.sum()),
                "suff": round(suff, 4),
                "obtn": round(obtn, 4),
            }
        )

        logging.info(
            f"Threshold {threshold:.2f}: kept {mask.sum()}/{len(mask)} sentences, suff={suff:.3f}, obtn={obtn:.3f}"
        )

        # Stop if successful threshold is found or all sentences are kept
        if is_good_prune(
            suff, obtn, args.suff_threshold, args.obtn_threshold
        ) or torch.all(mask):
            logging.info(
                f"Breaking after either finding successful threshold or keeping all sentences"
            )
            break

    return threshold_results


def get_starting_mask_from_thresholds(
    threshold_results: List[Dict],
    suff_threshold: float,
    obtn_threshold: float,
    n_sentences: int,
) -> List[int]:
    """
    Get the minimal (in term of number of included reasoning steps) threshold-induced mask of reasoning steps
    that reaches better sufficiency and obtainability.
    """
    for result in threshold_results:
        if is_good_prune(
            result["suff"], result["obtn"], suff_threshold, obtn_threshold
        ):
            return result["mask"]

    # No passing mask — return all-ones
    return [1] * n_sentences


def greedy_pruning(
    prefix_prompt: str,
    prefix_sentences: List[str],
    answer_sentence_idx: int,
    starting_mask: List[int],
    removal_order: List[int],
    gt_answer: str,
    full_cot_ppl: float,
    empty_cot_ppl: float,
    model,
    tokenizer,
    n_resamples: int,
    eval_seed: int,
    suff_threshold: float,
    obtn_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Greedy queue-based pruning following attribution ranking order.

    Removes sentences from least influential first. If removal fails (suff or obtn
    drops below threshold), reverts and re-queues. Stops when a full pass produces
    no removals.
    """
    n_sentences = len(prefix_sentences)
    mask = torch.tensor(starting_mask, dtype=torch.long)
    protected = {answer_sentence_idx, answer_sentence_idx - 1}

    queue = deque(
        idx for idx in removal_order if mask[idx] == 1 and idx not in protected
    )

    removal_steps = []
    failures_since_last_removal = 0

    while queue and failures_since_last_removal < len(queue):
        sent_idx = queue.popleft()

        if mask[sent_idx] == 0:
            continue

        # Tentatively remove
        mask[sent_idx] = 0

        set_deterministic(eval_seed)
        suff = eval_sufficiency(
            prefix_prompt,
            prefix_sentences,
            mask,
            answer_sentence_idx,
            gt_answer,
            model,
            tokenizer,
            n_resamples=n_resamples,
            seed=eval_seed,
        )[0]

        pruned_prompt = apply_step_mask_to_cot(
            prefix_prompt, prefix_sentences, mask, model.model_name
        )
        pruned_ppl = perplexity(pruned_prompt, model, tokenizer)
        obtn = 1 - normalize_ppl(pruned_ppl, full_cot_ppl, empty_cot_ppl)

        if is_good_prune(suff, obtn, suff_threshold, obtn_threshold):
            removal_steps.append(
                {
                    "mask": mask.tolist(),
                    "removed_sentence_idx": sent_idx,
                    "n_kept": int(mask.sum()),
                    "suff": round(suff, 4),
                    "obtn": round(obtn, 4),
                }
            )
            failures_since_last_removal = 0
            logging.info(
                f"  Removed sent {sent_idx}: kept {mask.sum().item()}/{n_sentences}, "
                f"suff={suff:.3f}, obtn={obtn:.3f}"
            )
        else:
            # Revert and re-queue
            mask[sent_idx] = 1
            queue.append(sent_idx)
            failures_since_last_removal += 1
            logging.debug(
                f"  Failed to remove sent {sent_idx}: "
                f"suff={suff:.3f}, obtn={obtn:.3f} — re-queued"
            )

    return removal_steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attribution-based greedy pruning evaluation"
    )
    parser.add_argument("--model-name", type=str, required=True)
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
        help="Dataset difficulty level (if applicable). If not specified, uses all difficulties.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sufficiency evaluation.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=N_SUFFICINECY_EVAL_RESAMPLES,
        help="Number of resamples for sufficiency evaluation.",
    )
    parser.add_argument(
        "--suff-threshold",
        type=float,
        default=SUFFICIENCY_THRESHOLD,
        help="Sufficiency threshold.",
    )
    parser.add_argument(
        "--obtn-threshold",
        type=float,
        default=OBTAINABILITY_THRESHOLD,
        help="Obtainability threshold.",
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
    model_name = args.model_name
    model.model_name = model_name

    # Load data
    logging.info("Loading prior results...")
    generations, _ = load_generations(model_name, args.difficulty, dataset=args.dataset)
    step_results, _ = load_single_step_results(
        model_name, args.difficulty, dataset=args.dataset
    )

    # Output path
    ds_name = args.dataset.split("/")[-1]
    suffix = (
        f"-suff{args.suff_threshold}-atnb{args.obtn_threshold}"
        if (
            args.suff_threshold != SUFFICIENCY_THRESHOLD
            or args.obtn_threshold != OBTAINABILITY_THRESHOLD
        )
        else ""
    )
    output_filename = f"{ds_name}{f'_difficulty={args.difficulty}' if args.difficulty else ''}_seed={args.seed}{suffix}_attribution_pruning.json"
    output_path = os.path.join("results", model_name, output_filename)

    metadata = {
        "model_id": model_id,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "suff_threshold": args.suff_threshold,
        "obtn_threshold": args.obtn_threshold,
    }

    # Load existing results
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        with open(output_path, "r") as f:
            outputs, cached_metadata = json.load(f)
            assert cached_metadata == metadata
    else:
        outputs = {}

    # Filter only valid entries
    valid_entries = []
    keys_iter = set(generations.keys()) & set(step_results.keys())
    for entry_key in keys_iter:
        sr = step_results[entry_key]
        if (
            sr.get("skipped", False)
            or sr.get("post_removal_full_cot_suff", 0.0) < args.suff_threshold
            or sr.get("post_removal", {}).get("answer_sentence_idx", -1) < 0
        ):
            continue
        valid_entries.append(entry_key)
    logging.info(f"Total valid entries to process: {len(valid_entries)}")

    iterator = tqdm(valid_entries)
    for entry_key in iterator:
        gen_entry = generations[entry_key]
        step_entry = step_results[entry_key]

        # Skip if already processed
        if entry_key in outputs and GREEDY_KEY in outputs[entry_key]:
            logging.info(f"Entry {entry_key} already processed, skipping")
            continue
        elif entry_key not in outputs:
            outputs[entry_key] = {}

        # Extract necessary data
        gt_answer = gen_entry["gt_answer"]
        full_cot_ppl = step_entry["post_removal_full_cot_ppl"]
        empty_cot_ppl = step_entry["empty_cot_ppl"]
        clean_full_prompt = step_entry["post_removal"]["full_prompt"]
        clean_sentences = step_entry["post_removal"]["sentences"]
        clean_token_borders = step_entry["post_removal"]["token_borders"]
        clean_answer_idx = step_entry["post_removal"]["answer_sentence_idx"]

        #
        # Step 1 - Find sufficient minimal prefix
        #
        if PREFIX_ITERATION_KEY not in outputs[entry_key]:
            incremental_results, prefix_len = compute_sufficient_prefix_results(
                clean_full_prompt,
                clean_sentences,
                clean_answer_idx,
                gt_answer,
                full_cot_ppl,
                empty_cot_ppl,
                entry_key,
                model,
                tokenizer,
                args,
            )
            outputs[entry_key][PREFIX_ITERATION_KEY] = incremental_results
            outputs[entry_key]["prefix_length"] = prefix_len
            save_results(outputs, metadata, output_path)
        else:
            logging.info(f"Entry {entry_key}: Loading cached incremental results")
            incremental_results = outputs[entry_key][PREFIX_ITERATION_KEY]
            prefix_len = outputs[entry_key]["prefix_length"]

        #
        # Step 2 - Calculate attribution scores and threshold results
        #
        # Build prefix prompt
        prefix_mask = torch.zeros(len(clean_sentences), dtype=torch.bool)
        prefix_mask[:prefix_len] = True
        prefix_prompt = apply_step_mask_to_cot(
            clean_full_prompt,
            clean_sentences,
            prefix_mask,
            model_name,
            clean_whitespace=False,
        )

        # Get sentences and token borders for prefix
        prefix_sentences = clean_sentences[:prefix_len]
        prefix_token_borders = clean_token_borders[:prefix_len]
        assert len(prefix_sentences) == len(prefix_token_borders)

        # Append answer suffix sentence
        suffix_sent = get_prefill_ending(model.model_name) + gt_answer + "}"
        full_prefix_prompt = prefix_prompt + suffix_sent
        prefix_sentences = prefix_sentences + [suffix_sent]
        prefix_token_borders = prefix_token_borders + [
            (
                prefix_token_borders[-1][1],
                len_tokens(full_prefix_prompt, tokenizer),
            )
        ]
        answer_sentence_idx = len(prefix_sentences) - 1

        if THRESHOLDS_KEY not in outputs[entry_key]:
            # Compute gradient attribution matrix and ranking
            cot_start_idx = extract_cot_from_output(clean_full_prompt, model_name)[1]
            cot_start_tok_idx = len_tokens(clean_full_prompt[:cot_start_idx], tokenizer)
            influence_matrix, removal_order = compute_attribution_ranking(
                full_prefix_prompt,
                prefix_token_borders,
                cot_start_tok_idx,
                model,
                tokenizer,
            )

            threshold_results = compute_threshold_results(
                influence_matrix,
                full_prefix_prompt,
                prefix_sentences,
                answer_sentence_idx,
                gt_answer,
                full_cot_ppl,
                empty_cot_ppl,
                model,
                tokenizer,
                args,
            )
            outputs[entry_key]["attribution_ranking"] = removal_order
            outputs[entry_key][THRESHOLDS_KEY] = threshold_results
            save_results(outputs, metadata, output_path)
        else:
            logging.info(f"Entry {entry_key}: Loading cached threshold results")
            removal_order = outputs[entry_key]["attribution_ranking"]
            threshold_results = outputs[entry_key][THRESHOLDS_KEY]

        #
        # Step 3 - Find best threshold mask and run greedy pruning
        #
        # Get starting mask from best threshold result
        starting_mask = get_starting_mask_from_thresholds(
            threshold_results,
            args.suff_threshold,
            args.obtn_threshold,
            len(prefix_sentences),
        )
        starting_n_kept = sum(starting_mask)
        logging.info(
            f"Entry {entry_key}: starting mask has {starting_n_kept}/{len(prefix_sentences)} kept"
        )

        # Run greedy attribution-based pruning
        greedy_results = greedy_pruning(
            prefix_prompt=full_prefix_prompt,
            prefix_sentences=prefix_sentences,
            answer_sentence_idx=answer_sentence_idx,
            starting_mask=starting_mask,
            removal_order=removal_order,
            gt_answer=gt_answer,
            full_cot_ppl=full_cot_ppl,
            empty_cot_ppl=empty_cot_ppl,
            model=model,
            tokenizer=tokenizer,
            n_resamples=args.n_resamples,
            eval_seed=args.seed,
            suff_threshold=args.suff_threshold,
            obtn_threshold=args.obtn_threshold,
        )

        # Save entry results
        entry_results = {
            "prefix_length": prefix_len,
            PREFIX_ITERATION_KEY: incremental_results,
            "attribution_ranking": removal_order,
            THRESHOLDS_KEY: threshold_results,
            "starting_n_kept": starting_n_kept,
            GREEDY_KEY: greedy_results,
            "final_n_kept": (
                greedy_results[-1]["n_kept"] if greedy_results else starting_n_kept
            ),
        }
        outputs[entry_key] = entry_results
        save_results(outputs, metadata, output_path)

    logging.info(f"Results saved to {output_path}")

    # Summary
    n_processed = sum(1 for v in outputs.values() if GREEDY_KEY in v)
    logging.info(f"Total entries processed: {n_processed}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
