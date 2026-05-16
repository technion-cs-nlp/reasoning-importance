import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from consts import (
    N_SUFFICINECY_EVAL_RESAMPLES,
    OBTAINABILITY_THRESHOLD,
    SUFFICIENCY_THRESHOLD,
)
from cot_utils import (
    apply_step_mask_to_cot,
    are_answers_equivalent,
    extract_boxed_answers,
    get_prefill_ending,
    get_empty_cot_ending,
)
from general_utils import generate_output, set_deterministic


def eval_immediate_answer(
    prompt, model, tokenizer, gt_answer, n_resamples=N_SUFFICINECY_EVAL_RESAMPLES
):
    count_successes = 0
    sample_answer = None
    gt_answer_tok_len = len(tokenizer(gt_answer).input_ids)
    for i in range(n_resamples):
        forced_output = generate_output(
            model,
            tokenizer,
            prompt,
            new_toks=gt_answer_tok_len * 2,
            sampling_temp=0.6,
            sampling_top_p=0.95,
        )  # *2 to be safe

        found_answers = extract_boxed_answers(prompt + forced_output)
        if len(found_answers) > 0:
            found_answer = found_answers[-1]
            if sample_answer is None:
                sample_answer = (
                    found_answer  # Record a sample answer for logging purposes
                )

            if are_answers_equivalent(found_answer, gt_answer):
                count_successes += 1

    return count_successes / n_resamples, sample_answer


def eval_sufficiency(
    full_prompt: str,
    sentences: List[str],
    sentence_mask: torch.Tensor,
    answer_sentence_idx: int,
    gt_answer: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    n_resamples: int = N_SUFFICINECY_EVAL_RESAMPLES,
    seed: int = 42,
):
    """
    Checks the sufficiency of a subset of a reasoning chain (presented as a sentence mask).
    This is done by generating multiple outputs from the pruned prompt (with a proper suffix)
    and checking how many times the correct answer is produced.

    Additionally,

    Args:
        full_prompt (str): The original full prompt containing the CoT.
        sentences (list of str): List of sentences in the CoT (Not the whole input+output!).
        sentence_mask (list of bool): Boolean mask indicating which sentences to keep.
        gt_answer (str): The ground truth answer to compare against.
        model (AutoModel): The language model used for generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        n_resamples (int): The number of generations to sample for estimating sufficiency.
        seed (int): A random seed for reproducibility.
    Returns:
        float: The sufficiency score (proportion of correct answers).
        str: The pruned prompt used for generating the sufficiency score.
        str: A sample answer generated from the pruned prompt.
    """
    # Prepare pruned prompt
    sufficiency_mask = sentence_mask.clone()
    sufficiency_mask[answer_sentence_idx] = False
    pruned_prompt = apply_step_mask_to_cot(
        full_prompt,
        sentences,
        sufficiency_mask,
        model.model_name,
        clean_whitespace=False,
    )
    prompt_suffix = (
        get_prefill_ending(model.model_name)
        if sufficiency_mask.sum() > 0
        else get_empty_cot_ending(model.model_name)
    )
    pruned_prompt += prompt_suffix

    # Measure sufficiency
    set_deterministic(seed)
    sufficiency, sample_answer = eval_immediate_answer(
        pruned_prompt, model, tokenizer, gt_answer, n_resamples
    )
    return (
        sufficiency,
        pruned_prompt[: -len(prompt_suffix)],
        sample_answer,
    )


def eval_cot_neccesity(full_input, model, tokenizer, gt_answer):
    """
    Evaluate the necessity of a CoT by checking if the model can produce the correct answer
    without any CoT sentences. This is done by prompting for a direct answer with any CoT and
    measuring if the model is able to answer correctly.
    """
    no_cot_prompt = full_input
    if no_cot_prompt.endswith("<|end|>"):
        no_cot_prompt = no_cot_prompt[: -len("<|end|>")]
    no_cot_prompt += get_empty_cot_ending(model.model_name)
    cot_neccesity = (
        1 - eval_immediate_answer(no_cot_prompt, model, tokenizer, gt_answer)[0]
    )
    return cot_neccesity


def is_good_prune(
    sufficiency: float,
    obtainability: float,
    suff_threshold: float = SUFFICIENCY_THRESHOLD,
    obtain_threshold: float = OBTAINABILITY_THRESHOLD,
) -> bool:
    """
    Determine if a prune is considered "good" based on sufficiency and attainability thresholds.
    """
    return sufficiency >= suff_threshold and obtainability >= obtain_threshold
