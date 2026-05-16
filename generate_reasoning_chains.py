"""
Generate reasoning chains from a language model.
This script generates CoT outputs deterministically and saves them as JSON.
"""

import sys
import argparse
import json
import logging
import os

import torch

from cot_utils import (
    extract_cot_from_output,
    wrap_math_boxed_prompt,
    wrap_general_boxed_prompt,
)
from general_utils import (
    get_chat_template,
    load_model,
    set_deterministic,
    generate_output,
    len_tokens,
)
from harp_dataset import load_harp_dataset
from math_dataset import load_math_dataset
from datasets import load_dataset
from consts import DATASET_REGISTRY, MODEL_NAME_TO_ID_PATH


def load_data(ds_name, args):
    """Load dataset based on name and arguments."""
    if "harp" in ds_name.lower():
        harp_split = ds_name.lower().split("-")[-1]
        data = load_harp_dataset(harp_split, args.difficulty)
    elif "math-500" in ds_name.lower():
        data = load_math_dataset()
    else:
        dataset = load_dataset(args.dataset_id)
        if "train" in dataset:
            data = dataset["train"]
        else:
            available_splits = list(dataset.keys())
            data = dataset[available_splits[0]]
            logging.info(f"Using split: {available_splits[0]}")

    return data


def solve_prompt(
    input_text, model, tokenizer, max_toks, verbose=True, prompt_wrapper="math"
):
    """
    Generate a reasoning chain for the given input prompt.

    Returns:
        tuple: (full_input, output, input_token_len, out_token_len)
    """
    # Prepare input dictionary for chat template
    if prompt_wrapper == "general":
        wrapped = wrap_general_boxed_prompt(input_text)
    else:
        wrapped = wrap_math_boxed_prompt(input_text)
    input_dict = [{"role": "user", "content": wrapped}]

    full_input, input_token_len = get_chat_template(
        input_dict, tokenizer, return_as_str=True
    )
    if verbose:
        logging.info(
            f"Processing input (length: {input_token_len}): {input_text[:100]}...".replace(
                "\n", "\\n"
            )
        )

    # Generate output deterministically
    output = generate_output(model, tokenizer, input_dict, new_toks=max_toks)

    out_token_len = len_tokens(output, tokenizer)
    if verbose:
        out_str = f"{output[:100]}...{output[-100:]}" if len(output) > 200 else output
        logging.info(
            f"Generated output (length {out_token_len}): {out_str}".replace("\n", "\\n")
        )

    return full_input, output, input_token_len, out_token_len


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reasoning chains for CoT prompts (no attribution)"
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--difficulty",
        type=int,
        default=None,
        help="Difficulty level to filter (if applicable)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--max-toks-per-prompt",
        type=int,
        default=16380,
        help="Maximum number of tokens to generate per prompt",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set deterministic behavior
    set_deterministic(args.seed)
    torch.set_grad_enabled(False)

    # Load model and tokenizer
    model_id, model_path = MODEL_NAME_TO_ID_PATH[args.model_name]
    logging.info(f"Loading model {model_id} from {model_path}")
    model, tokenizer = load_model(
        model_path,
        dtype=(
            torch.bfloat16 if "gpt" not in model_id.lower() else None
        ),  # GPT-OSS works with MXFP4 by default
        device_map="auto",
    )
    model.model_name = args.model_name

    # Load dataset
    logging.info(f"Loading dataset: {args.dataset}")
    ds_name = args.dataset.split("/")[-1]
    data = load_data(ds_name, args)

    # Look up dataset config from registry
    ds_config = DATASET_REGISTRY.get(ds_name, {})
    text_column = ds_config.get("text_column", "problem")
    answer_column = ds_config.get("answer_column", "answer")
    prompt_wrapper = ds_config.get("prompt_wrapper", "math")

    # Prepare metadata
    metadata = {
        "model_id": model_id,
        "model_path": model_path,
        "dataset_id": args.dataset,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "max_toks_per_prompt": args.max_toks_per_prompt,
    }

    # Set up results path and load existing results if available
    output_path = os.path.join(
        "results",
        args.model_name,
        f"{ds_name}_generations{f'_difficulty={args.difficulty}' if args.difficulty else ''}_seed={args.seed}.json",
    )
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Loading existing results from {output_path}")
        with open(output_path, "r") as f:
            results, cached_metadata = json.load(f)
            assert (
                cached_metadata == metadata
            ), "Cached metadata does not match current configuration!"
        logging.info(f"Loaded {len(results)} cached entries")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = {}

    # Limit samples if specified
    if args.max_samples is not None:
        data = data.select(range(0, min(args.max_samples, len(data))))

    logging.info(f"Processing {len(data)} samples")

    # Process each sample
    processed_count = 0
    skipped_count = 0

    for i, sample in enumerate(data):
        try:
            input_sample = sample[text_column]
            gt_answer = sample.get(answer_column) if answer_column else None

            # Use input as key (same as original script)
            entry_key = input_sample

            # Skip if already processed
            if entry_key in results:
                if not args.quiet:
                    logging.info(f"Skipping already processed sample {i}")
                skipped_count += 1
                continue
            logging.info(f"Processing sample {i}:")

            # Generate output
            full_input, output, input_token_len, out_token_len = solve_prompt(
                input_sample,
                model,
                tokenizer,
                args.max_toks_per_prompt,
                not args.quiet,
                prompt_wrapper=prompt_wrapper,
            )
            full_prompt = full_input + output

            # Check if CoT was complete
            cot_output, cot_start_idx = extract_cot_from_output(
                output, model.model_name
            )
            cot_start_tok_idx = (
                0
                if cot_start_idx == 0
                else len_tokens(output[:cot_start_idx], tokenizer)
            )
            cot_token_len = len_tokens(cot_output, tokenizer)
            cot_complete = (
                cot_start_tok_idx + cot_token_len < args.max_toks_per_prompt - 1
                and cot_output != output
            )
            result = {
                "input_str": input_sample,
                "gt_answer": gt_answer,
                "full_input": full_input,
                "output": output,
                "cot_output": cot_output,
                "full_prompt": full_prompt,
                "input_token_len": input_token_len,
                "output_token_len": out_token_len,
                "cot_token_len": cot_token_len,
                "is_cot_complete": cot_complete,
            }

            # Save results immediately after each prompt
            results[entry_key] = result
            with open(output_path, "w") as f:
                json.dump((results, metadata), f, indent=2)
            processed_count += 1

        except Exception as e:
            logging.error(f"Error processing sample {i}: {str(e)}")
            raise e

    logging.info(f"Successfully processed {processed_count} new samples")
    logging.info(f"Skipped {skipped_count} cached samples")
    logging.info(f"Total samples in results: {len(results)}")
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    main()
