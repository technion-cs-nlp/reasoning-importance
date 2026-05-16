import json
import os
import sys
import gc
import logging
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from cot_utils import (
    apply_step_mask_to_cot,
    extract_cot_from_output,
    normalize_ppl,
    get_prefill_ending,
)
from general_utils import (
    load_model,
    set_deterministic,
    perplexity,
    len_tokens,
    save_results,
)
from cot_evaluations import eval_sufficiency, is_good_prune
from collections import deque
from consts import (
    SUFFICIENCY_THRESHOLD,
    OBTAINABILITY_THRESHOLD,
    N_SUFFICINECY_EVAL_RESAMPLES,
    PRUNING_THRESHOLDS,
    GREEDY_KEY,
    THRESHOLDS_KEY,
    MODEL_NAME_TO_ID_PATH,
)
from cot_utils import prune_cot
from loading_utils import (
    load_attribution_pruning_results,
    load_generations,
    load_single_step_results,
)

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Attribution functions (copied from gradient_attribution.py and modified)
# ---------------------------------------------------------------------------


@torch.enable_grad()
def calculate_grad_x_input_matrix(
    model,
    tokenizer,
    prompt: str,
    cot_start_idx: int,
    use_checkpointing: bool = True,
    sentence_token_borders: Optional[List[Tuple[int, int]]] = None,
    are_token_borders_absolute: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute sentence-level attribution using Gradient x Input.

    For each target sentence k, computes the gradient of the sum of target-token
    probabilities w.r.t. the output embeddings, then for each source sentence j
    returns ||grad * embed||.
    """
    torch.cuda.empty_cache()
    gc.collect()

    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    token_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    output_length = len(token_ids[0]) - cot_start_idx - 1

    embeddings = model.get_input_embeddings()(token_ids)
    input_embeds, output_embeds = (
        embeddings[:, :-output_length, :],
        embeddings[:, -output_length:, :],
    )
    embeddings = torch.cat((input_embeds, output_embeds), dim=1)
    embeddings = embeddings.contiguous().to(model.device).to(model.dtype)
    embeddings.requires_grad_(True)

    if use_checkpointing:

        def checkpointed_forward(embeddings):
            return model(inputs_embeds=embeddings).logits

        logits = checkpoint(checkpointed_forward, embeddings, use_reentrant=False)[0]
    else:
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits[0]

    assert sentence_token_borders is not None, "Sentence token borders are required"

    influence_matrix = torch.zeros(
        (len(sentence_token_borders), len(sentence_token_borders)),
        device="cpu",
    )
    iter_borders = (
        sentence_token_borders if not verbose else tqdm(sentence_token_borders)
    )
    for k, (sent_start_k, sent_end_k) in enumerate(iter_borders):
        offset = 0 if are_token_borders_absolute else cot_start_idx
        output_positions = torch.tensor(
            range(offset + sent_start_k, offset + sent_end_k)
        )
        target_tokens = token_ids[0, output_positions]

        probs = F.softmax(logits[output_positions - 1], dim=-1)
        target_probs = probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze()
        total_target_prob = target_probs.sum()

        grads = torch.autograd.grad(
            outputs=total_target_prob,
            inputs=output_embeds,
            grad_outputs=torch.ones_like(total_target_prob),
            retain_graph=k < len(sentence_token_borders) - 1,
            create_graph=False,
        )[0]

        for j, (src_sent_start, src_sent_end) in enumerate(sentence_token_borders):
            sent_grads = grads[0, src_sent_start:src_sent_end]
            sent_embeds = output_embeds[0, src_sent_start:src_sent_end].detach()
            influence_matrix[k, j] = (sent_grads * sent_embeds).norm().detach()

        del grads, probs
        torch.cuda.empty_cache()
        gc.collect()

    return influence_matrix.to(torch.float32)


@torch.enable_grad()
def calculate_integrated_gradients_matrix(
    model,
    tokenizer,
    prompt: str,
    cot_start_idx: int,
    n_steps: int = 5,
    use_checkpointing: bool = True,
    sentence_token_borders: Optional[List[Tuple[int, int]]] = None,
    are_token_borders_absolute: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute sentence-level attribution using Integrated Gradients.

    Baseline: mean output-token embedding (broadcast to every position).
    Integrates gradients along a linear path from baseline to actual embeddings
    using ``n_steps`` evenly-spaced interpolation points (Riemann right-sum).

    IG_j = (embed_j - baseline) * (1/n_steps) * sum_{alpha} grad_alpha_j

    The sentence-level score is the norm of IG over all tokens in the sentence.
    """
    torch.cuda.empty_cache()
    gc.collect()

    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    token_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    output_length = len(token_ids[0]) - cot_start_idx - 1

    # Get the original embeddings
    with torch.no_grad():
        all_embeddings = model.get_input_embeddings()(token_ids)
        input_embeds = all_embeddings[:, :-output_length, :].clone()
        output_embeds_orig = all_embeddings[:, -output_length:, :].clone()

    # Baseline: mean output embedding broadcast to every output position, s.t. shape: (1, output_length, d_model)
    baseline = (
        output_embeds_orig[0]
        .mean(dim=0)
        .unsqueeze(0)
        .expand_as(output_embeds_orig[0])
        .unsqueeze(0)
    )

    assert sentence_token_borders is not None, "Sentence token borders are required"

    influence_matrix = torch.zeros(
        (len(sentence_token_borders), len(sentence_token_borders)),
        device="cpu",
    )

    # Alphas for Riemann right-sum: [1/n, 2/n, ..., 1]
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps)

    # For each target sentence, accumulate gradients over interpolation steps
    iter_borders = (
        sentence_token_borders if not verbose else tqdm(sentence_token_borders)
    )
    for k, (sent_start_k, sent_end_k) in enumerate(iter_borders):
        offset = 0 if are_token_borders_absolute else cot_start_idx

        # Accumulate gradients across interpolation steps
        accumulated_grads = torch.zeros_like(output_embeds_orig)

        for step_idx, alpha in enumerate(alphas):
            # Interpolated output embeddings
            interpolated_output = baseline + alpha * (output_embeds_orig - baseline)
            interpolated_output = (
                interpolated_output.contiguous().to(model.device).to(model.dtype)
            )
            interpolated_output.requires_grad_(True)

            # Build full embeddings (input unchanged, output interpolated)
            full_embeds = torch.cat((input_embeds, interpolated_output), dim=1)
            full_embeds = full_embeds.contiguous().to(model.device).to(model.dtype)

            if use_checkpointing:

                def checkpointed_forward(embeds):
                    return model(inputs_embeds=embeds).logits

                logits = checkpoint(
                    checkpointed_forward, full_embeds, use_reentrant=False
                )[0]
            else:
                logits = model(inputs_embeds=full_embeds).logits[0]

            output_positions = torch.tensor(
                range(offset + sent_start_k, offset + sent_end_k)
            )
            target_tokens = token_ids[0, output_positions]

            probs = F.softmax(logits[output_positions - 1], dim=-1)
            target_probs = probs.gather(
                dim=-1, index=target_tokens.unsqueeze(-1)
            ).squeeze()
            total_target_prob = target_probs.sum()

            grads = torch.autograd.grad(
                outputs=total_target_prob,
                inputs=interpolated_output,
                grad_outputs=torch.ones_like(total_target_prob),
                retain_graph=False,
                create_graph=False,
            )[0]

            accumulated_grads += grads.detach()  # .cpu()

            del grads, probs, logits, full_embeds, interpolated_output
            torch.cuda.empty_cache()
            gc.collect()

        # Average gradients and multiply by (input - baseline)
        avg_grads = accumulated_grads / n_steps  # (1, output_length, d_model)
        delta = (output_embeds_orig - baseline).detach()  # .cpu()
        ig_attributions = avg_grads * delta  # (1, output_length, d_model)

        for j, (src_sent_start, src_sent_end) in enumerate(sentence_token_borders):
            sent_ig = ig_attributions[0, src_sent_start:src_sent_end]
            influence_matrix[k, j] = sent_ig.norm().item()

        del accumulated_grads, avg_grads, delta, ig_attributions
        torch.cuda.empty_cache()
        gc.collect()

    return influence_matrix.to(torch.float32)


def compute_removal_order(
    influence_matrix: torch.Tensor,
) -> List[int]:
    """
    Given an influence matrix, remove diagonal, compute mean column influence,
    and return sentence indices sorted from least to most influential.
    """
    nonzero_counts = (influence_matrix != 0).sum(dim=0).float()
    nonzero_counts[nonzero_counts == 0] = 1
    mean_influence = influence_matrix.sum(dim=0) / nonzero_counts
    removal_order = mean_influence.argsort(descending=False).tolist()
    return removal_order


def get_starting_mask_from_thresholds(
    threshold_results: List[dict],
    suff_threshold: float,
    obtn_threshold: float,
    n_sentences: int,
) -> List[int]:
    """Get the most-pruned passing threshold mask."""
    for result in threshold_results:
        if is_good_prune(
            result["suff"], result["obtn"], suff_threshold, obtn_threshold
        ):
            return result["mask"]
    return [1] * n_sentences


def greedy_attribution_pruning(
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
) -> List[dict]:
    """
    Greedy queue-based pruning following attribution ranking order.

    Removes sentences from least influential first. If removal fails (suff or obtn
    drops below threshold), reverts and re-queues. Stops when a full pass produces
    no removals.

    Identical logic to eval_attribution_pruning.greedy_attribution_pruning.
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


def compute_threshold_results(
    influence_matrix: torch.Tensor,
    full_prefix_prompt: str,
    prefix_sentences: List[str],
    answer_sentence_idx: int,
    gt_answer: str,
    full_cot_ppl: float,
    empty_cot_ppl: float,
    model,
    tokenizer,
    args,
):
    """
    Iterate over pruning thresholds, prune the CoT using the influence matrix,
    and evaluate sufficiency and obtainability at each threshold.
    Stops at the first successful threshold or when all sentences are kept.

    Identical logic to eval_attribution_pruning.compute_threshold_results.
    """
    threshold_results = []
    for threshold in PRUNING_THRESHOLDS:
        mask = prune_cot(
            influence_matrix,
            threshold,
            answer_sentence_idx,
            lambda mat, idx: mat[idx],
            is_recursive=True,
        )

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
            f"Threshold {threshold:.2f}: kept {mask.sum()}/{len(mask)} sentences, "
            f"suff={suff:.3f}, obtn={obtn:.3f}"
        )

        if is_good_prune(
            suff, obtn, args.suff_threshold, args.obtn_threshold
        ) or torch.all(mask):
            logging.info(
                "Breaking after finding successful threshold or keeping all sentences"
            )
            break

    return threshold_results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alternative attribution methods (grad*input, integrated gradients) on MATH-500"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=5,
        help="Number of interpolation steps for integrated gradients.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_deterministic(args.seed)
    dataset = "math-500"

    # Load model
    model_id, model_path = MODEL_NAME_TO_ID_PATH[args.model_name]
    logging.info(f"Loading model {model_id} from {model_path}")
    model, tokenizer = load_model(
        model_path,
        dtype=(torch.bfloat16 if "gpt" not in model_id.lower() else None),
        device_map="auto",
    )
    model_name = model_id.split("/")[-1]
    model.model_name = model_name

    # Load pre-generated data (same as eval_attribution_pruning.py)
    generations, _ = load_generations(model_name, dataset=dataset)
    step_results, _ = load_single_step_results(model_name, dataset=dataset)
    standard_attr_results, _ = load_attribution_pruning_results(
        model_name, dataset=dataset
    )

    # Output path
    output_path = os.path.join(
        "results",
        model_name,
        f"{dataset}_alternative_attributions_seed={args.seed}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metadata = {
        "model_id": model_id,
        "dataset": dataset,
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "suff_threshold": args.suff_threshold,
        "obtn_threshold": args.obtn_threshold,
        "ig_steps": args.ig_steps,
    }

    # Load existing results
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        with open(output_path, "r") as f:
            outputs, cached_metadata = json.load(f)
    else:
        outputs = {}

    # Filter valid entries (same logic as eval_attribution_pruning.py)
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

        # Skip if already fully processed (both methods have greedy results)
        if (
            entry_key in outputs
            and "grad_x_input" in outputs[entry_key]
            and GREEDY_KEY in outputs[entry_key].get("grad_x_input", {})
            and "integrated_gradients" in outputs[entry_key]
            and GREEDY_KEY in outputs[entry_key].get("integrated_gradients", {})
        ):
            continue

        if entry_key not in outputs:
            outputs[entry_key] = {}

        # Extract necessary data (same as eval_attribution_pruning.py)
        gt_answer = gen_entry["gt_answer"]
        full_cot_ppl = step_entry["post_removal_full_cot_ppl"]
        empty_cot_ppl = step_entry["empty_cot_ppl"]
        clean_full_prompt = step_entry["post_removal"]["full_prompt"]
        clean_sentences = step_entry["post_removal"]["sentences"]
        clean_token_borders = step_entry["post_removal"]["token_borders"]
        clean_answer_idx = step_entry["post_removal"]["answer_sentence_idx"]

        if len(clean_sentences) < 2:
            logging.warning(f"Entry {entry_key}: too few sentences, skipping")
            continue

        # Build prefix prompt with answer suffix (use full CoT as prefix)
        prefix_len = standard_attr_results[entry_key]["prefix_length"]
        prefix_mask = torch.zeros(len(clean_sentences), dtype=torch.bool)
        prefix_mask[:prefix_len] = True
        prefix_prompt = apply_step_mask_to_cot(
            clean_full_prompt,
            clean_sentences,
            prefix_mask,
            model_name,
            clean_whitespace=False,
        )
        prefix_sentences = clean_sentences[:prefix_len]
        prefix_token_borders = clean_token_borders[:prefix_len]
        outputs[entry_key]["prefix_length"] = prefix_len

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

        # Get CoT start token index
        _, cot_start_char_idx = extract_cot_from_output(clean_full_prompt, model_name)
        cot_start_tok_idx = len_tokens(
            clean_full_prompt[:cot_start_char_idx], tokenizer
        )

        # Adjust token borders relative to CoT start
        adjusted_borders = [
            (s - cot_start_tok_idx, e - cot_start_tok_idx)
            for (s, e) in prefix_token_borders
        ]
        if adjusted_borders[0][0] != 0:
            logging.warning(
                f"Entry {entry_key}: first sentence doesn't start at token 0 "
                f"(got {adjusted_borders[0][0]}), skipping"
            )
            continue

        # ===================================================================
        # Grad x Input: attribution → thresholds → ranking → greedy pruning
        # ===================================================================
        if "grad_x_input" not in outputs[entry_key]:
            outputs[entry_key]["grad_x_input"] = {}
        gxi_out = outputs[entry_key]["grad_x_input"]

        if THRESHOLDS_KEY not in gxi_out:
            logging.info(f"Entry {entry_key}: computing grad x input attribution...")
            with torch.enable_grad():
                gxi_matrix = calculate_grad_x_input_matrix(
                    model,
                    tokenizer,
                    full_prefix_prompt,
                    cot_start_tok_idx,
                    sentence_token_borders=adjusted_borders,
                )
                gxi_matrix = gxi_matrix - torch.diag(torch.diag(gxi_matrix))
            gxi_ranking = compute_removal_order(gxi_matrix)

            logging.info(f"Entry {entry_key}: threshold evaluation (grad x input)...")
            gxi_threshold_results = compute_threshold_results(
                gxi_matrix,
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
            gxi_out["influence_matrix"] = gxi_matrix.tolist()
            gxi_out["attribution_ranking"] = gxi_ranking
            gxi_out[THRESHOLDS_KEY] = gxi_threshold_results
            save_results(outputs, metadata, output_path)
        else:
            gxi_ranking = gxi_out["attribution_ranking"]
            gxi_threshold_results = gxi_out[THRESHOLDS_KEY]

        if GREEDY_KEY not in gxi_out:
            starting_mask = get_starting_mask_from_thresholds(
                gxi_threshold_results,
                args.suff_threshold,
                args.obtn_threshold,
                len(prefix_sentences),
            )
            starting_n_kept = sum(starting_mask)
            logging.info(
                f"Entry {entry_key} (gxi): starting mask has "
                f"{starting_n_kept}/{len(prefix_sentences)} kept"
            )

            greedy_results = greedy_attribution_pruning(
                prefix_prompt=full_prefix_prompt,
                prefix_sentences=prefix_sentences,
                answer_sentence_idx=answer_sentence_idx,
                starting_mask=starting_mask,
                removal_order=gxi_ranking,
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
            gxi_out["starting_n_kept"] = starting_n_kept
            gxi_out[GREEDY_KEY] = greedy_results
            gxi_out["final_n_kept"] = (
                greedy_results[-1]["n_kept"] if greedy_results else starting_n_kept
            )
            save_results(outputs, metadata, output_path)

        # ===================================================================
        # Integrated Gradients: attribution → thresholds → ranking → greedy
        # ===================================================================
        if "integrated_gradients" not in outputs[entry_key]:
            outputs[entry_key]["integrated_gradients"] = {}
        ig_out = outputs[entry_key]["integrated_gradients"]

        if THRESHOLDS_KEY not in ig_out:
            logging.info(
                f"Entry {entry_key}: computing integrated gradients attribution..."
            )
            with torch.enable_grad():
                ig_matrix = calculate_integrated_gradients_matrix(
                    model,
                    tokenizer,
                    full_prefix_prompt,
                    cot_start_tok_idx,
                    n_steps=args.ig_steps,
                    sentence_token_borders=adjusted_borders,
                )
                ig_matrix = ig_matrix - torch.diag(torch.diag(ig_matrix))
            ig_ranking = compute_removal_order(ig_matrix)

            logging.info(
                f"Entry {entry_key}: threshold evaluation (integrated gradients)..."
            )
            ig_threshold_results = compute_threshold_results(
                ig_matrix,
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
            ig_out["influence_matrix"] = ig_matrix.tolist()
            ig_out["attribution_ranking"] = ig_ranking
            ig_out[THRESHOLDS_KEY] = ig_threshold_results
            save_results(outputs, metadata, output_path)
        else:
            ig_ranking = ig_out["attribution_ranking"]
            ig_threshold_results = ig_out[THRESHOLDS_KEY]

        if GREEDY_KEY not in ig_out:
            starting_mask = get_starting_mask_from_thresholds(
                ig_threshold_results,
                args.suff_threshold,
                args.obtn_threshold,
                len(prefix_sentences),
            )
            starting_n_kept = sum(starting_mask)
            logging.info(
                f"Entry {entry_key} (ig): starting mask has "
                f"{starting_n_kept}/{len(prefix_sentences)} kept"
            )

            greedy_results = greedy_attribution_pruning(
                prefix_prompt=full_prefix_prompt,
                prefix_sentences=prefix_sentences,
                answer_sentence_idx=answer_sentence_idx,
                starting_mask=starting_mask,
                removal_order=ig_ranking,
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
            ig_out["starting_n_kept"] = starting_n_kept
            ig_out[GREEDY_KEY] = greedy_results
            ig_out["final_n_kept"] = (
                greedy_results[-1]["n_kept"] if greedy_results else starting_n_kept
            )
            save_results(outputs, metadata, output_path)

        # Store shared info
        outputs[entry_key]["n_sentences"] = len(prefix_sentences)
        outputs[entry_key]["answer_sentence_idx"] = answer_sentence_idx
        save_results(outputs, metadata, output_path)

    # Summary
    n_total = len(outputs)
    n_gxi = sum(1 for v in outputs.values() if "grad_x_input" in v)
    n_ig = sum(1 for v in outputs.values() if "integrated_gradients" in v)
    logging.info(f"Done. Total entries: {n_total}, grad*input: {n_gxi}, IG: {n_ig}")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
