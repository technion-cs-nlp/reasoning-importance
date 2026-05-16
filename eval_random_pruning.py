"""
Random greedy pruning evaluation (strong baseline for optimality of attribution pruning).

This script randomly ranks reasoning steps (from a previously-calculated sufficient prefix)
and greedy removes them. If, after a removal, the sufficiency and obtainability of the
remaining chain remain above thresholds, the removal is kept. Otherwise the sentence is pushed
to the back of the queue for retry after other removals. The process terminates when no sentence
can be removed in a full pass.

The removal order is randomly shuffled (with a configurable seed), and multiple
random seeds are run per entry.

Each output entry includes:
    1. The removal evaluation results per seed. Each seed has a list of removed steps, showing the
       removed sentence index ("removed_sentence_idx"), number of remaining steps ("n_kept"), and the
       resulting metrics ("suff" and "obtn") at that step.
    2. The prefix length in steps ("starting_n_kept"), and the average and std of final steps kept after
       the random greedy removal ("avg_n_kept" and "std_n_kept").
"""

import json
import math
import os
import sys
import logging
import argparse
import random
from collections import deque
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from cot_utils import (
    apply_step_mask_to_cot,
    normalize_ppl,
    get_prefill_ending,
)
from general_utils import (
    load_model,
    set_deterministic,
    perplexity,
)
from cot_evaluations import eval_sufficiency, is_good_prune
from consts import (
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
from removability_utils import get_sufficient_prefix_data, get_valid_entries

torch.set_grad_enabled(False)

N_RANDOM_SEEDS = 3


def random_iterative_pruning(
    prefix_prompt: str,
    prefix_sentences: List[str],
    answer_sentence_idx: int,
    starting_mask: List[int],
    gt_answer: str,
    full_cot_ppl: float,
    empty_cot_ppl: float,
    model,
    tokenizer,
    n_resamples: int,
    eval_seed: int,
    random_seed: int,
    suff_threshold: float,
    obtn_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Random pruning starting from a given mask.

    Uses a queue of candidate sentence indices. For each candidate:
    - Tentatively remove it from the mask
    - Evaluate suff and obtn
    - If both pass: keep the removal, record the step
    - If either fails: revert, push to back of queue

    Stops when a full pass over the queue produces no removals.
    """
    n_sentences = len(prefix_sentences)
    mask = torch.tensor(starting_mask, dtype=torch.long)

    # Protected indices: answer and prev-to-answer
    protected = {answer_sentence_idx, answer_sentence_idx - 1}

    # Build initial candidate queue: currently-kept, non-protected sentences
    candidates = [i for i in range(n_sentences) if mask[i] == 1 and i not in protected]

    rng = random.Random(random_seed)
    rng.shuffle(candidates)
    queue = deque(candidates)

    removal_steps = []
    failures_since_last_removal = 0

    while queue and failures_since_last_removal < len(queue):
        sent_idx = queue.popleft()

        # Skip if already removed (shouldn't happen, but be safe)
        if mask[sent_idx] == 0:
            continue

        # Tentatively remove
        mask[sent_idx] = 0

        # Evaluate sufficiency
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

        # Evaluate obtainability
        pruned_prompt = apply_step_mask_to_cot(
            prefix_prompt, prefix_sentences, mask, model.model_name
        )
        pruned_ppl = perplexity(pruned_prompt, model, tokenizer)
        obtn = 1 - normalize_ppl(pruned_ppl, full_cot_ppl, empty_cot_ppl)

        if is_good_prune(suff, obtn, suff_threshold, obtn_threshold):
            # Removal succeeded — keep it
            removal_steps.append(
                {
                    "removed_sentence_idx": sent_idx,
                    "n_kept": int(mask.sum()),
                    "suff": round(suff, 4),
                    "obtn": round(obtn, 4),
                }
            )
            failures_since_last_removal = 0

            logging.info(
                f"  Seed {random_seed}: removed sent {sent_idx}, "
                f"kept {mask.sum().item()}/{n_sentences}, suff={suff:.3f}, obtn={obtn:.3f}"
            )
        else:
            # Removal failed — revert and push to back of queue
            mask[sent_idx] = 1
            queue.append(sent_idx)
            failures_since_last_removal += 1

            logging.debug(
                f"  Seed {random_seed}: failed to remove sent {sent_idx}, "
                f"suff={suff:.3f}, obtn={obtn:.3f} — re-queued"
            )

    return removal_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Random pruning optimality evaluation")
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
        "--n-random-seeds",
        type=int,
        default=N_RANDOM_SEEDS,
        help="Number of random seeds for randomization repeats.",
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

    # Prepare output path
    ds_name = args.dataset.split("/")[-1]
    output_filename = f"{ds_name}{f'_difficulty={args.difficulty}' if args.difficulty else ''}_seed={args.seed}_random_pruning.json"
    output_path = os.path.join("results", model_name, output_filename)

    metadata = {
        "model_id": model_id,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "n_random_seeds": args.n_random_seeds,
        "suff_threshold": args.suff_threshold,
        "obtn_threshold": args.obtn_threshold,
    }

    # Load existing results if available
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        with open(output_path, "r") as f:
            outputs, cached_metadata = json.load(f)
            # assert cached_metadata == metadata
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
        if entry_key in outputs:
            existing = outputs[entry_key]
            seed_keys = [k for k in existing if k.isdigit()]
            if len(seed_keys) >= args.n_random_seeds and "avg_n_kept" in existing:
                logging.info(f"Entry {entry_key} already fully processed, skipping")
                continue

        # Extract necessary data
        gt_answer = generations[entry_key]["gt_answer"]
        full_cot_ppl = step_results[entry_key]["post_removal_full_cot_ppl"]
        empty_cot_ppl = step_results[entry_key]["empty_cot_ppl"]

        # Get prefix length and build prefix prompt
        prefix_len, prefix_prompt, prefix_sentences, _ = get_sufficient_prefix_data(
            entry_key, step_results, attr_pruning_results, model_name
        )

        # Append the answer suffix sentence
        suffix_sent = get_prefill_ending(model.model_name) + gt_answer + "}"
        full_prefix_prompt = prefix_prompt + suffix_sent
        prefix_sentences = prefix_sentences + [suffix_sent]
        answer_sentence_idx = len(prefix_sentences) - 1

        logging.info(
            f"Entry {entry_key}: {len(prefix_sentences)} prefix sentences "
            f"(prefix_len={prefix_len})"
        )

        starting_mask = [1] * len(prefix_sentences)
        starting_n_kept = sum(starting_mask)
        logging.info(
            f"Entry {entry_key}: starting from mask with {starting_n_kept}/{len(prefix_sentences)} kept"
        )

        # Run random pruning with multiple random seeds
        entry_results = outputs.get(entry_key, {})
        for rand_seed in range(args.seed, args.seed + args.n_random_seeds):
            seed_key = str(rand_seed)
            if seed_key in entry_results:
                logging.info(
                    f"Entry {entry_key}: seed {rand_seed} already processed, skipping"
                )
                continue

            logging.info(
                f"Entry {entry_key}: running random pruning with seed {rand_seed}"
            )
            seed_results = random_iterative_pruning(
                prefix_prompt=full_prefix_prompt,
                prefix_sentences=prefix_sentences,
                answer_sentence_idx=answer_sentence_idx,
                starting_mask=starting_mask,
                gt_answer=gt_answer,
                full_cot_ppl=full_cot_ppl,
                empty_cot_ppl=empty_cot_ppl,
                model=model,
                tokenizer=tokenizer,
                n_resamples=args.n_resamples,
                eval_seed=args.seed,
                random_seed=rand_seed,
                suff_threshold=args.suff_threshold,
                obtn_threshold=args.obtn_threshold,
            )
            entry_results[seed_key] = seed_results

        # Compute summary stats: final n_kept after random pruning
        final_n_kept = []
        for seed_key in range(args.seed, args.seed + args.n_random_seeds):
            seed_data = entry_results.get(str(seed_key), [])
            if seed_data:
                final_n_kept.append(seed_data[-1]["n_kept"])
            else:
                # No removals were possible — n_kept is the starting mask
                final_n_kept.append(starting_n_kept)

        if final_n_kept:
            avg = sum(final_n_kept) / len(final_n_kept)
            std = math.sqrt(
                sum((x - avg) ** 2 for x in final_n_kept) / len(final_n_kept)
            )
            entry_results["starting_n_kept"] = starting_n_kept
            entry_results["avg_n_kept"] = round(avg, 4)
            entry_results["std_n_kept"] = round(std, 4)
            logging.info(
                f"Entry {entry_key}: starting={starting_n_kept}, "
                f"avg_final={avg:.2f}, std={std:.2f}"
            )

        outputs[entry_key] = entry_results

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump((outputs, metadata), f, indent=4, sort_keys=True)

    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    main()
