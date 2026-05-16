"""
Predict removability of individual CoT reasoning steps using LLM-as-a-judge.
This script contains four LLM-as-a-judge setups:
1. label_external:              Single sentence -> external LLM, no question context.
2. label_external_full_context: Single sentence + question + full CoT -> external LLM.
3. label_local:                 Single sentence -> local LLM, no question context.
4. label_local_full_context:    Single sentence + question + full CoT -> local LLM.

Methods 3 and 4 are only run if --model-path is provided.

Output format: (results_dict, metadata_dict)
  results_dict[entry_key][str(sentence_idx)] = {
      "sentence": "...",
      "ground_truth_is_removable": true/false,  # true = removable
      "label_external": "removable" | "nonremovable",
      "label_external_full_context": "nonremovable",
      ...
  }
"""

import argparse
import json
import logging
import random
import sys
import os

import torch
from general_utils import (
    call_llm_api,
    generate_output,
    load_model,
    LLMProvider,
    save_results,
    set_deterministic,
)
from loading_utils import (
    load_attribution_pruning_results,
    load_generations,
    load_single_step_results,
)
from removability_utils import (
    extract_removable_nonremovable_entry_keys,
    get_sufficient_prefix_data,
)
from consts import (
    MAX_EXTERNAL_COMPLETION_TOKENS,
    MODEL_NAME_TO_ID_PATH,
    SUFFICIENCY_THRESHOLD,
    OBTAINABILITY_THRESHOLD,
)

EXTERNAL_LABELING_METHODS = ["label_external", "label_external_full_context"]
INTERNAL_LABELING_METHODS = ["label_local", "label_local_full_context"]
VALID_SETUPS = ["external", "local"]
VALID_LABELS = ["removable", "nonremovable"]

TAGGING_PROMPT = """\
You are labeling a sentence from a mathetmatical chain-of-thought (CoT) reasoning trace.

Your task is to predict whether this sentence is **removable** or **nonremovable**.

Definitions:
- **removable**: The sentence can be deleted from the trace such that all subsequent \
sentences remain logically supported and the final answer is still reachable from the \
remaining sentences alone.
- **nonremovable**: The sentence is essential to the reasoning chain. Without it, at \
least one subsequent sentence loses its logical basis, or the final answer can no longer \
be derived from the remaining sentences.

To make your judgment, consider whether the sentence:
- Introduces a new variable, equation, or fact that is used by a later sentence.
- Performs a calculation or derivation whose result is referenced downstream.
- Or instead restates the problem, expresses uncertainty, narrates the approach, \
summarizes prior steps, or makes a tangential observation.

Based only on the sentence text, predict whether this type of sentence is typically \
removable or nonremovable in a reasoning trace.

Think step by step about what role this sentence most likely plays, then on the last \
line output ONLY the word "removable" or "nonremovable".

Sentence: {sentence}
"""

FULL_CONTEXT_TAGGING_PROMPT = """\
You are labeling a sentence from a mathematical chain-of-thought (CoT) reasoning trace.

Your task is to predict whether this sentence is **removable** or **nonremovable**.

Definitions:
- **removable**: The sentence can be deleted from the trace such that all subsequent \
sentences remain logically supported and the final answer is still reachable from the \
remaining sentences alone.
- **nonremovable**: The sentence is essential to the reasoning chain. Without it, at \
least one subsequent sentence loses its logical basis, or the final answer can no longer \
be derived from the remaining sentences.

To make your judgment, consider whether the sentence:
- Introduces a new variable, equation, or fact that is used by a later sentence.
- Performs a calculation or derivation whose result is referenced downstream.
- Or instead restates the problem, expresses uncertainty, narrates the approach, \
summarizes prior steps, or makes a tangential observation.

You are given the input question and the full CoT reasoning trace with sentence indices. \
The target sentence to label is marked with ">>>". Read the full chain of thought, then \
assess whether the marked sentence's content is used or referenced by any subsequent \
sentence, or whether it provides information necessary to reach the final answer.

Think step by step about the sentence's role in the chain, then on the last line output \
ONLY the word "removable" or "nonremovable".

The question: <question>{question}</question>

The CoT sentences (the target sentence is marked with >>>):
{sentence_lines}

"""


def _parse_single_label(raw_response: str) -> str | None:
    """
    Parse a single-sentence LLM response into a valid label.
    Returns "removable", "nonremovable", or None if unparseable.
    """
    cleaned = raw_response.strip().lower()
    # Handle potential markdown or quotes
    cleaned = cleaned.strip("`\"'")

    if cleaned in VALID_LABELS:
        return cleaned

    # Try to find a valid label anywhere in the response
    # "Nonremovable" is searched first since it contains "removable"
    if "nonremovable" in cleaned or "non-removable" in cleaned:
        return "nonremovable"
    if "removable" in cleaned:
        return "removable"

    logging.warning(f"Failed to parse label from response: {raw_response!r}")
    return None


def tag_single_sentence_external(
    sentence: str,
    external_model: str = "gemini-2.5-pro",
) -> str | None:
    """
    Call external LLM to tag a single sentence (no context).
    Returns "removable", "nonremovable", or None.
    """
    prompt = TAGGING_PROMPT.format(sentence=sentence)
    raw_response = call_llm_api(
        prompt,
        LLMProvider.GEMINI,
        external_model,
        max_tokens=MAX_EXTERNAL_COMPLETION_TOKENS,
        temperature=0.0,
    )
    return _parse_single_label(raw_response)


def tag_single_sentence_external_full_context(
    sentence_idx: int,
    all_sentences: list[str],
    question: str,
    external_model: str = "gemini-2.5-pro",
) -> str | None:
    """
    Call external LLM to tag a single sentence with full CoT context.
    Returns "removable", "nonremovable", or None.
    """
    sentence_lines = []
    for i, sent in enumerate(all_sentences):
        prefix = ">>>" if i == sentence_idx else "   "
        sentence_lines.append(f"{prefix} [{i}] {sent}")
    sentence_lines_str = "\n".join(sentence_lines)

    prompt = FULL_CONTEXT_TAGGING_PROMPT.format(
        question=question,
        sentence_lines=sentence_lines_str,
    )
    raw_response = call_llm_api(
        prompt,
        LLMProvider.GEMINI,
        external_model,
        max_tokens=MAX_EXTERNAL_COMPLETION_TOKENS,
        temperature=0.0,
    )
    return _parse_single_label(raw_response)


def tag_single_sentence_local(
    sentence: str,
    model,
    tokenizer,
) -> str | None:
    """
    Use a local LLM to tag a single sentence (no context).
    Returns "removable", "nonremovable", or None.
    """
    prompt = TAGGING_PROMPT.format(sentence=sentence)
    raw_response = generate_output(model, tokenizer, prompt, new_toks=8192)
    return _parse_single_label(raw_response)


def tag_single_sentence_local_full_context(
    sentence_idx: int,
    all_sentences: list[str],
    question: str,
    model,
    tokenizer,
) -> str | None:
    """
    Use a local LLM to tag a single sentence with full CoT context.
    Returns "removable", "nonremovable", or None.
    """
    sentence_lines = []
    for i, sent in enumerate(all_sentences):
        prefix = ">>>" if i == sentence_idx else "   "
        sentence_lines.append(f"{prefix} [{i}] {sent}")
    sentence_lines_str = "\n".join(sentence_lines)

    prompt = FULL_CONTEXT_TAGGING_PROMPT.format(
        question=question,
        sentence_lines=sentence_lines_str,
    )
    raw_response = generate_output(model, tokenizer, prompt, new_toks=8192)
    return _parse_single_label(raw_response)


def build_sentence_list(
    entry_labels: dict[str, dict[int, bool]],
    step_results: dict,
    generations: dict,
    attr_results: dict,
    model_name: str,
) -> list[dict]:
    """
    Build two flat list of individual sentences to label, drawing from the
    removable/nonremovable entry keys.

    Each item in the returned list is a dict:
        {
            "entry_key": str,
            "sentence_idx": int,
            "sentence": str,
            "ground_truth_is_removable": bool,    # True = removable
            "all_sentences": list[str],  # all prefix sentences for context
            "question": str,
            "gt_answer": str,
        }
    """
    sentence_items = []

    for entry_key, labels in entry_labels.items():
        step_entry = step_results.get(entry_key)
        gen_entry = generations.get(entry_key)
        attr_entry = attr_results.get(entry_key)
        if step_entry is None or gen_entry is None or attr_entry is None:
            continue

        _, _, prefix_sentences, _ = get_sufficient_prefix_data(
            entry_key, step_results, attr_results, model_name
        )
        if not prefix_sentences:
            continue

        question = step_entry.get("input_str", "")
        gt_answer = gen_entry.get("gt_answer", "")

        for sent_idx, is_removable in labels.items():
            if sent_idx >= len(prefix_sentences):
                continue
            sentence = prefix_sentences[sent_idx].strip()
            if not sentence:
                continue

            sentence_items.append(
                {
                    "entry_key": entry_key,
                    "sentence_idx": sent_idx,
                    "sentence": sentence,
                    "ground_truth_is_removable": is_removable,
                    "all_sentences": prefix_sentences,
                    "question": question,
                    "gt_answer": gt_answer,
                }
            )

    removable_items = [s for s in sentence_items if s["ground_truth_is_removable"]]
    nonremovable_items = [
        s for s in sentence_items if not s["ground_truth_is_removable"]
    ]
    return removable_items, nonremovable_items


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tag individual CoT sentences with removability labels (single-sentence prompts)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=f"Model name (one of: {list(MODEL_NAME_TO_ID_PATH.keys())})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="harp-standard",
        help="Dataset identifier (default: harp-standard)",
    )
    parser.add_argument(
        "--setups",
        type=str,
        nargs="+",
        choices=VALID_SETUPS,
        required=True,
        help="Which labeling setups to run: 'external' (Gemini API) and/or 'local' (local LLM).",
    )
    parser.add_argument(
        "--external-model",
        type=str,
        default="gemini-2.5-pro",
        help="External LLM model name for Gemini API tagging (used when 'external' is in --setups).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of sentences to process (for debugging)",
    )
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=SUFFICIENCY_THRESHOLD,
        help="Minimum sufficiency threshold for valid masks",
    )
    parser.add_argument(
        "--obtainability-threshold",
        type=float,
        default=OBTAINABILITY_THRESHOLD,
        help="Minimum obtainability threshold for valid masks",
    )
    return parser.parse_args()


def main():
    torch.set_grad_enabled(False)

    args = parse_args()
    model_name = args.model_name
    if model_name not in MODEL_NAME_TO_ID_PATH:
        raise ValueError(
            f"Unknown model name {model_name!r}. Must be one of "
            f"{list(MODEL_NAME_TO_ID_PATH.keys())}"
        )
    model_id, model_path = MODEL_NAME_TO_ID_PATH[model_name]

    # Load data and previously-calculated GT removability labels
    generations, _ = load_generations(model_name, dataset=args.dataset)
    step_results, _ = load_single_step_results(model_name, dataset=args.dataset)
    attr_pruning_results, _ = load_attribution_pruning_results(
        model_name, dataset=args.dataset
    )
    entry_labels = extract_removable_nonremovable_entry_keys(
        generations,
        attr_pruning_results,
        step_results,
        suff_threshold=args.sufficiency_threshold,
        obtn_threshold=args.obtainability_threshold,
    )
    assert entry_labels

    # Build flat sentence list
    removable_items, nonremovable_items = build_sentence_list(
        entry_labels,
        step_results,
        generations,
        attr_pruning_results,
        model_name,
    )
    logging.info(
        f"Found {len(removable_items)} removable and {len(nonremovable_items)} nonremovable sentences"
    )

    # Shuffle and optionally limit
    set_deterministic(args.seed)
    random.shuffle(removable_items)
    random.shuffle(nonremovable_items)
    if args.max_samples:
        n_per_class = min(
            [args.max_samples // 2, len(removable_items), len(nonremovable_items)]
        )
        sentence_items = (
            removable_items[:n_per_class] + nonremovable_items[:n_per_class]
        )
        logging.info(f"Limited to {len(sentence_items)} sentences")
    else:
        sentence_items = removable_items + nonremovable_items

    # Determine methods from setups
    methods = []
    local_model = None
    local_tokenizer = None
    if "external" in args.setups:
        logging.info("Running external labeling methods")
        methods += EXTERNAL_LABELING_METHODS
    if "local" in args.setups:
        logging.info(f"Loading local model from {model_path}")
        local_model, local_tokenizer = load_model(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        methods += INTERNAL_LABELING_METHODS

    logging.info(f"Running tagging methods: {methods}")

    # Output path
    output_filename = f"{args.dataset}_sentence_removability_tags_seed={args.seed}.json"
    output_path = os.path.join(
        "results",
        model_name,
        output_filename,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metadata = {
        "model_name": model_name,
        "model_id": model_id,
        "setups": sorted(args.setups),
        "seed": args.seed,
        "external_model": args.external_model if "external" in args.setups else None,
        "sufficiency_threshold": args.sufficiency_threshold,
        "obtainability_threshold": args.obtainability_threshold,
    }

    # Load existing results for incremental processing
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        results, cached_metadata = json.load(open(output_path, "r"))
        assert (
            metadata == cached_metadata
        ), "Metadata mismatch with existing results file"
    else:
        results = {}

    # Tag individual reasoning steps, iterating one by one
    for item_idx, item in enumerate(sentence_items):
        logging.info(
            f"Tagging {item_idx}/{len(sentence_items)}: entry {item['entry_key']} sentence {item['sentence_idx']}"
        )
        entry_key = item["entry_key"]
        sent_idx = item["sentence_idx"]
        sentence = item["sentence"]
        sent_key = str(sent_idx)

        # Initialize entry/sentence result if needed
        if entry_key not in results:
            results[entry_key] = {}
        if sent_key not in results[entry_key]:
            results[entry_key][sent_key] = {
                "sentence": sentence,
                "ground_truth_is_removable": item["ground_truth_is_removable"],
            }
        sent_result = results[entry_key][sent_key]

        method_dispatch = {
            "label_external": lambda: tag_single_sentence_external(
                sentence, external_model=args.external_model
            ),
            "label_external_full_context": lambda: tag_single_sentence_external_full_context(
                sentence_idx=sent_idx,
                all_sentences=item["all_sentences"],
                question=item["question"],
                external_model=args.external_model,
            ),
            "label_local": lambda: tag_single_sentence_local(
                sentence, local_model, local_tokenizer
            ),
            "label_local_full_context": lambda: tag_single_sentence_local_full_context(
                sentence_idx=sent_idx,
                all_sentences=item["all_sentences"],
                question=item["question"],
                model=local_model,
                tokenizer=local_tokenizer,
            ),
        }

        for method in methods:
            if method in sent_result:
                continue
            logging.info(f"  Method: {method}")
            label = method_dispatch[method]()
            if label is not None:
                sent_result[method] = label

        # Save incrementally
        save_results(results, metadata, output_path)

    # Final save and summary
    save_results(results, metadata, output_path)

    total_sentences = sum(len(v) for v in results.values())
    logging.info(f"Results saved to {output_path}")
    logging.info(f"Total: {len(results)} entries, {total_sentences} sentences tagged")

    # Print accuracy summary per method
    for method in methods:
        correct = 0
        total = 0
        for entry_data in results.values():
            for sent_data in entry_data.values():
                if method in sent_data and "ground_truth_is_removable" in sent_data:
                    total += 1
                    predicted = sent_data[method]
                    gt = (
                        "removable"
                        if sent_data["ground_truth_is_removable"]
                        else "nonremovable"
                    )
                    if predicted == gt:
                        correct += 1
        if total > 0:
            logging.info(f"  {method}: {correct}/{total} correct ({correct/total:.1%})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
