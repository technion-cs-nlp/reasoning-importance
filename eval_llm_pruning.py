"""
LLM-based pruning evaluation.

This script asks an external LLM for an initial mask of sentences (from a previously-calculated
 sufficient prefix) that don't contribute directly or indirectly to the answer, then performs greedy
queue-based removal ordered by the LLM importance ranking.

Each output entry includes:
    1. The prefix length, previously calculated.
    2. The masking results: the supposedely unimportant steps ("llm_initial_mask"), their amount ("starting_n_kept"),
       and the sufficiency and obtainability scores for the initial LLM mask ("initial_mask_eval").
    3. The greedy removal results: the order of removal ("removal order"), the sufficiency and obtainability results
       ("greedy_results") of each removed step, and the number of sentence post greedy removal ("final_n_kept").
"""

import json
import re
import os
import sys
import logging
import argparse
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import torch
from tqdm import tqdm

from cot_utils import (
    apply_step_mask_to_cot,
    normalize_ppl,
    get_prefill_ending,
)
from eval_attribution_pruning import greedy_pruning
from general_utils import (
    call_llm_api,
    load_model,
    set_deterministic,
    perplexity,
    LLMProvider,
)
from cot_evaluations import eval_sufficiency, is_good_prune
from consts import (
    GREEDY_KEY,
    MODEL_NAME_TO_ID_PATH,
    OBTAINABILITY_THRESHOLD,
    SUFFICIENCY_THRESHOLD,
    N_SUFFICINECY_EVAL_RESAMPLES,
)
from loading_utils import (
    load_attribution_pruning_results,
    load_generations,
    load_single_step_results,
)
from removability_utils import (
    get_sufficient_prefix_data,
    get_valid_entries,
)

torch.set_grad_enabled(False)

RANKING_PROMPT = """\
You are analyzing a mathematical chain-of-thought (CoT) reasoning trace. \
You are given the input question and the minimal reasoning trace required to answer,\
split to steps.

Your task is to rank ALL reasoning steps by their importance in producing the final answer. \
A step is important if it directly contributes to the final answer, or if it indirectly \
influences another step that recursively affects the final answer. A step is unimportant \
if removing it would not affect the reasoning chain's ability to reach the correct answer.

Think about which steps are load-bearing for the argument and which are \
redundant, restatements, or tangential. 

Don't rank the last sentence.

The question:
<question>
{question}
</question>

The reasoning steps:
{sentence_lines}

After thinking, output your final ranking as a JSON list of step indices ordered from \
MOST important to LEAST important. The list must contain exactly the indices {valid_indices}. \
Output the ranking inside a ```json code block, like:
```json
[most_important_idx, ..., least_important_idx]
```"""

MASK_PROMPT = """\
You are analyzing a mathematical chain-of-thought (CoT) reasoning trace. \
You are given the input question and the minimal reasoning trace required to answer, \
split to steps.

Your task is to identify which reasoning steps do NOT contribute directly or indirectly \
to producing the final answer. \
A step contributes directly if it is used in reaching the final answer. \
A step contributes indirectly if it influences another step that recursively affects \
the final answer. \
A step does NOT contribute if removing it would not affect the reasoning chain's \
ability to reach the correct answer — for example, steps that are redundant, \
restatements, or tangential.

Don't consider the last sentence (the answer step).

The question:
<question>
{question}
</question>

The reasoning steps:
{sentence_lines}

After thinking, output your final answer as a JSON list of step indices that do NOT \
contribute to the answer and can be safely removed. Output the list inside a \
```json code block, like:
```json
[removable_idx_1, removable_idx_2, ...]
```"""


def parse_mask_response(
    raw_response: str, valid_indices: List[int], n_sentences: int
) -> Optional[List[int]]:
    """
    Parse the LLM mask response into a binary mask.
    Returns a list of 0/1 of length n_sentences, or None if unparseable.
    """
    match = re.search(r"```json\s*\n?(.*?)\n?\s*```", raw_response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(r"\[[\d,\s]*\]", raw_response)
        if match:
            json_str = match.group(0)
        else:
            logging.warning(
                f"Failed to find mask JSON in response: {raw_response[:200]!r}"
            )
            return None

    try:
        removable_indices = json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse mask JSON: {json_str!r}")
        return None

    if not isinstance(removable_indices, list):
        logging.warning(f"Mask is not a list: {removable_indices}")
        return None

    # Filter to valid indices only
    removable_set = set(removable_indices) & set(valid_indices)

    # Build mask: 1 = kept, 0 = removed
    mask = [1] * n_sentences
    for idx in removable_set:
        mask[idx] = 0

    return mask


def get_llm_initial_mask(
    prefix_sentences: List[str],
    answer_sentence_idx: int,
    question: str,
    external_model: str,
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Ask an external LLM which sentences can be removed.
    Returns (mask, raw_response) where mask is a list of 0/1.
    """
    # Indices that can be removed (exclude answer and prev-to-answer)
    valid_indices = [
        i
        for i in range(len(prefix_sentences))
        if i != answer_sentence_idx and i != answer_sentence_idx - 1
    ]

    sentence_lines = []
    for i, sent in enumerate(prefix_sentences):
        if i == answer_sentence_idx:
            sentence_lines.append(f"[{i}] [FINAL ANSWER STEP - DON'T CONSIDER] {sent}")
        else:
            sentence_lines.append(f"[{i}] {sent}")

    prompt = MASK_PROMPT.format(
        question=question,
        sentence_lines="\n".join(sentence_lines),
    )

    try:
        raw_response = call_llm_api(
            prompt,
            LLMProvider.GEMINI,
            model=external_model,
            max_tokens=24576,
            temperature=0.0,
        )
        logging.info(f"Got mask response: {raw_response}".replace("\n", "\\n"))
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        return None, None

    mask = parse_mask_response(raw_response, valid_indices, len(prefix_sentences))
    return mask, raw_response


def parse_ranking_response(
    raw_response: str, valid_indices: List[int]
) -> Optional[List[int]]:
    """
    Parse the LLM ranking response into a list of sentence indices.
    Returns a list ordered from most important to least important, or None if unparseable.
    """
    match = re.search(r"```json\s*\n?(.*?)\n?\s*```", raw_response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(r"\[[\d,\s]+\]", raw_response)
        if match:
            json_str = match.group(0)
        else:
            logging.warning(
                f"Failed to find ranking JSON in response: {raw_response[:200]!r}"
            )
            return None

    try:
        ranking = json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse ranking JSON: {json_str!r}")
        return None

    if not isinstance(ranking, list):
        logging.warning(f"Ranking is not a list: {ranking}")
        return None

    # Validate: must contain exactly the valid indices
    ranking_set = set(ranking)
    valid_set = set(valid_indices)

    if ranking_set != valid_set:
        # Try to salvage: filter to valid indices and append missing ones at the end
        filtered = [idx for idx in ranking if idx in valid_set]
        missing = [idx for idx in valid_indices if idx not in ranking_set]
        ranking = filtered + missing
        if set(ranking) != valid_set:
            logging.warning(
                f"Could not reconcile ranking {ranking_set} with valid indices {valid_set}"
            )
            return None

    return ranking


def get_importance_ranking(
    prefix_sentences: List[str],
    answer_sentence_idx: int,
    question: str,
    gt_answer: str,
    external_model: str,
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Ask an external LLM to rank prefix sentences by importance.

    Returns (ranking, raw_response) where ranking is a list of sentence indices
    ordered from most important to least important, or (None, raw_response) on failure.
    """
    # Indices that can be ranked/removed (exclude answer)
    rankable_indices = [
        i for i in range(len(prefix_sentences)) if i != answer_sentence_idx
    ]

    sentence_lines = []
    for i, sent in enumerate(prefix_sentences):
        if i == answer_sentence_idx:
            sentence_lines.append(f"[{i}] [FINAL ANSWER STEP - DON'T RANK]  {sent}")
        else:
            sentence_lines.append(f"[{i}] {sent}")
    sentence_lines_str = "\n".join(sentence_lines)

    valid_indices_str = str(rankable_indices)
    prompt = RANKING_PROMPT.format(
        question=question,
        answer=gt_answer,
        sentence_lines=sentence_lines_str,
        valid_indices=valid_indices_str,
    )

    try:
        raw_response = call_llm_api(
            prompt,
            LLMProvider.GEMINI,
            model=external_model,
            max_tokens=24576,
            temperature=0.0,
        )
        logging.info(f"Got ranking response: {raw_response}".replace("\n", "\\n"))
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        return None, None

    ranking = parse_ranking_response(raw_response, rankable_indices)
    return ranking, raw_response


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based greedy pruning evaluation")
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
        "--external-model",
        type=str,
        default="gemini-2.5-pro",
        help="External LLM model name for mask generation (default: gemini-2.5-pro)",
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
    generations, _ = load_generations(model_name, args.difficulty, dataset=args.dataset)
    step_results, _ = load_single_step_results(
        model_name, args.difficulty, dataset=args.dataset
    )
    attr_pruning_results, _ = load_attribution_pruning_results(
        model_name, args.difficulty, dataset=args.dataset
    )
    llm_ranking_results = {}

    # Output path
    ds_name = args.dataset.split("/")[-1]
    output_filename = f"{ds_name}{f'_difficulty={args.difficulty}' if args.difficulty else ''}_seed={args.seed}_llm_pruning.json"
    output_path = os.path.join("results", model_name, output_filename)

    metadata = {
        "model_id": model_id,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "external_model": args.external_model,
        "suff_threshold": args.suff_threshold,
        "obtn_threshold": args.obtn_threshold,
    }

    # Load existing results
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        with open(output_path, "r") as f:
            outputs, cached_metadata = json.load(f)
            assert (
                cached_metadata == metadata
            ), f"Metadata mismatch: {cached_metadata} vs {metadata}"
    else:
        outputs = {}

    # Filter valid entries
    valid_entries = get_valid_entries(
        generations,
        step_results,
        attr_pruning_results,
        sufficiency_threshold=args.suff_threshold,
    )
    logging.info(f"Total valid entries to process: {len(valid_entries)}")

    iterator = tqdm(valid_entries)
    for entry_key in iterator:
        # Skip if already fully processed
        if entry_key in outputs and GREEDY_KEY in outputs[entry_key]:
            logging.info(f"Entry {entry_key} already processed, skipping")
            continue

        # Extract necessary data
        gt_answer = generations[entry_key]["gt_answer"]
        sent_entry = step_results[entry_key]
        full_cot_ppl = sent_entry["post_removal_full_cot_ppl"]
        empty_cot_ppl = sent_entry["empty_cot_ppl"]

        prefix_len, prefix_prompt, prefix_sentences, _ = get_sufficient_prefix_data(
            entry_key, step_results, attr_pruning_results, model_name
        )

        # Append answer suffix sentence
        suffix_sent = get_prefill_ending(model.model_name) + gt_answer + "}"
        full_prefix_prompt = prefix_prompt + suffix_sent
        prefix_sentences = prefix_sentences + [suffix_sent]
        answer_sentence_idx = len(prefix_sentences) - 1

        # prefix_len = attr_pruning_results[entry_key]["prefix_length"]

        logging.info(
            f"Entry {entry_key}: {len(prefix_sentences)} sentences (prefix_len={prefix_len})"
        )

        # Get or compute LLM ranking (stored as least-to-most important, matching
        # the attribution_ranking convention in eval_attribution_pruning)
        entry_results = outputs.get(entry_key, {})
        question = sent_entry.get("input_str", "")

        if "llm_ranking" in entry_results and entry_results["llm_ranking"] is not None:
            removal_order = entry_results["llm_ranking"]
        elif (
            entry_key in llm_ranking_results
            and llm_ranking_results[entry_key].get("ranking") is not None
        ):
            # Existing results are most-to-least; reverse to least-to-most
            removal_order = list(reversed(llm_ranking_results[entry_key]["ranking"]))
            logging.info(f"Entry {entry_key}: using existing LLM ranking")
        else:
            logging.info(
                f"Entry {entry_key}: no existing ranking, requesting from {args.external_model}"
            )
            ranking, ranking_raw_response = get_importance_ranking(
                prefix_sentences=prefix_sentences,
                answer_sentence_idx=answer_sentence_idx,
                question=question,
                gt_answer=gt_answer,
                external_model=args.external_model,
            )
            if ranking is None:
                logging.warning(f"Entry {entry_key}: failed to get ranking, skipping")
                entry_results["llm_ranking"] = None
                entry_results["ranking_raw_response"] = ranking_raw_response
                outputs[entry_key] = entry_results
                with open(output_path, "w") as f:
                    json.dump((outputs, metadata), f, indent=4, sort_keys=True)
                continue
            # Store as least-to-most important
            removal_order = list(reversed(ranking))

        # Get or compute LLM initial mask
        if "llm_initial_mask" not in entry_results:
            logging.info(
                f"Entry {entry_key}: requesting initial mask from {args.external_model}"
            )
            llm_mask, mask_raw_response = get_llm_initial_mask(
                prefix_sentences,
                answer_sentence_idx,
                question,
                args.external_model,
            )

            if llm_mask is None:
                logging.warning(
                    f"Entry {entry_key}: failed to get LLM mask, using all-ones"
                )
                entry_results["mask_raw_response"] = mask_raw_response
                llm_mask = [1] * len(prefix_sentences)

            entry_results["llm_initial_mask"] = llm_mask
        else:
            llm_mask = entry_results["llm_initial_mask"]

        # Always keep answer and prev-to-answer in starting mask
        llm_mask[answer_sentence_idx] = 1
        llm_mask[answer_sentence_idx - 1] = 1

        starting_n_kept = sum(llm_mask)
        logging.info(
            f"Entry {entry_key}: LLM mask keeps {starting_n_kept}/{len(prefix_sentences)}"
        )

        # Evaluate the initial LLM mask before greedy pruning
        if "initial_mask_eval" not in entry_results:
            mask_tensor = torch.tensor(llm_mask, dtype=torch.long)
            set_deterministic(args.seed)
            init_suff = eval_sufficiency(
                full_prefix_prompt,
                prefix_sentences,
                mask_tensor,
                answer_sentence_idx,
                gt_answer,
                model,
                tokenizer,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )[0]
            pruned_prompt = apply_step_mask_to_cot(
                full_prefix_prompt, prefix_sentences, mask_tensor, model.model_name
            )
            pruned_ppl = perplexity(pruned_prompt, model, tokenizer)
            init_obtn = 1 - normalize_ppl(pruned_ppl, full_cot_ppl, empty_cot_ppl)
            entry_results["initial_mask_eval"] = [
                {
                    "suff": round(init_suff, 4),
                    "obtn": round(init_obtn, 4),
                    "n_kept": starting_n_kept,
                }
            ]
            logging.info(
                f"Entry {entry_key}: initial mask eval — "
                f"suff={init_suff:.3f}, obtn={init_obtn:.3f}"
            )

        # Run greedy LLM-based pruning
        if GREEDY_KEY not in entry_results:
            logging.info(f"Entry {entry_key}: running greedy LLM-based pruning")
            greedy_results = greedy_pruning(
                prefix_prompt=full_prefix_prompt,
                prefix_sentences=prefix_sentences,
                answer_sentence_idx=answer_sentence_idx,
                starting_mask=llm_mask,
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
            for res in greedy_results:
                del res["mask"]  # Unneccesary, cumbersome

            entry_results[GREEDY_KEY] = greedy_results

        entry_results["prefix_length"] = prefix_len
        entry_results["removal_order"] = removal_order
        entry_results["starting_n_kept"] = starting_n_kept
        entry_results["final_n_kept"] = (
            entry_results[GREEDY_KEY][-1]["n_kept"]
            if entry_results[GREEDY_KEY]
            else starting_n_kept
        )

        outputs[entry_key] = entry_results

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump((outputs, metadata), f, indent=4, sort_keys=True)

    logging.info(f"Results saved to {output_path}")

    # Summary
    n_processed = sum(1 for v in outputs.values() if v.get(GREEDY_KEY) is not None)
    n_failed_mask = sum(
        1 for v in outputs.values() if v.get("llm_initial_mask") is None or not v
    )
    logging.info(
        f"Total entries processed: {n_processed}, failed mask: {n_failed_mask}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
