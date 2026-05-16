# Some of the code is originally based on code from the "Thought Anchors" paper by P. Bogdan, U. Macar et al.

from enum import Enum
import re
import warnings
from typing import List, Tuple, Optional
from fractions import Fraction
import torch
from transformers import AutoTokenizer


def wrap_math_boxed_prompt(prompt):
    return f"Solve the following problem. You MUST put your final answer in \\boxed{{}}. Problem: {prompt}\n"


def wrap_general_boxed_prompt(prompt):
    return f"Answer the following question. You MUST put your final answer in \\boxed{{}}. {prompt}\n"


def normalize_ppl(ppl, full_cot_ppl, empty_cot_ppl):
    if full_cot_ppl > empty_cot_ppl:
        print("Full CoT PPL is greater than Empty CoT PPL!")

    if empty_cot_ppl == full_cot_ppl:
        return 0.0
    return (ppl - full_cot_ppl) / (empty_cot_ppl - full_cot_ppl)


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.
    """
    normalized = latex_str.strip().lower()

    # Remove $ signs at start/end (math mode delimiters)
    normalized = normalized.strip("$")

    # GROUP 1: Remove LaTeX spacing commands (\,, \;, \:, \!, \quad, \qquad, etc.)
    normalized = re.sub(r"\\[,;:!]", "", normalized)
    normalized = re.sub(r"\\q?quad", "", normalized)
    normalized = re.sub(r"\\hspace\{[^}]*\}", "", normalized)
    normalized = re.sub(r"\\vspace\{[^}]*\}", "", normalized)

    # Replace different fraction notations (dfrac, tfrac -> frac)
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")

    # Normalize spaces
    normalized = normalized.replace(r"\ ", " ")
    normalized = re.sub(r"\s+", "", normalized)

    # Normalize percentages
    normalized = normalized.replace("\\%", "")

    # Normalize funny commas in LaTeX ({,} for spacing)
    normalized = normalized.replace("{,}", "")

    # GROUP 3: Remove commas used as thousand separators (between digits)
    normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)

    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")

    # Normalize decimal representation
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)

    # GROUP 2: Add leading zero to decimals that start with a dot
    normalized = re.sub(r"^\.(\d)", r"0.\1", normalized)
    normalized = re.sub(r"([^0-9])\.(\d)", r"\g<1>0.\2", normalized)

    # Handle {-} as minus sign before removing braces
    normalized = normalized.replace("{-}", "-")

    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)

    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")

    # Remove LaTeX text commands
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)

    # Normalize date formats (e.g., "October 30" vs "October\\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    # Strip common prefix words (is, equals, approximately) that precede numbers
    normalized = re.sub(
        r"^(is|equals|approximately|about|around)(\d)", r"\2", normalized
    )

    return normalized


def _clean_whitespace(text: str) -> str:
    """
    Clean up whitespace artifacts from sentence removal.
    Preserves paragraph structure while removing redundant spaces.
    """
    # Replace multiple spaces (but not newlines) with single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Replace 3+ consecutive newlines with double newline (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces on each line
    text = re.sub(r" +\n", "\n", text)

    # Remove leading spaces on each line (except indentation might be intentional, so be careful)
    # Only remove if it's clearly an artifact (space after newline followed by more space)
    text = re.sub(r"\n +(?= )", "\n", text)

    return text.strip()


# -----------------------------------------------------------
# CoT parsing and editing functions
# -----------------------------------------------------------


def extract_cot_from_output(solution_text: str, model_name: str):
    """
    Extract the chain of thought from the full model output.
    e.g., for gps-oss, extract text in the <analysis> tags.

    There are multiple heuristics per model, as models often don't follow a single template
    when outputting reasoning chains.

    Args:
        solution_text: The full model output text
        model_name: The name of the model (optional, for future use)
    """
    cot_text = None
    start_cot_idx = 0
    try:
        if "gpt-oss" in model_name.lower():
            # Get whatever is within analysis/commentary <|message|> tags.
            # Note: sometimes the model outputs the CoT in "<|channel|>analysis<...", "<|start|>analysis<...",
            #       or "<|start|>commentary<|message|>...".
            match = re.search(
                r"(?:analysis|commentary)<\|message\|>([\s\S]*?)<\|end\|>",
                solution_text,
            )
            if match:
                cot_text = match.group(1)
                start_cot_idx = match.start(1)
            else:
                # No end tag found, assume everything from analysis/commentary <|message|> onwards is the thinking
                match = re.search(
                    r"(?:analysis|commentary)<\|message\|>([\s\S]*)",
                    solution_text,
                )
                if match:
                    cot_text = match.group(1)
                    start_cot_idx = match.start(1)
                else:
                    # Fallback: CoT in <|start|>...<|end|> block without channel tags,
                    # appearing before <|start|>assistant<|channel|>final
                    match = re.search(
                        r"<\|start\|>((?!system|user|assistant)[\s\S]*?)<\|end\|><\|start\|>assistant<\|channel\|>final",
                        solution_text,
                    )
                    if match:
                        cot_text = match.group(1)
                        start_cot_idx = match.start(1)
                    else:
                        # Fallback: CoT after <|start|> (not system/user/assistant) with no closing tag,
                        # content must not contain <|start|> or <|end|> tags
                        # Use [\s\S] instead of . to match newlines in multi-line CoT
                        match = re.search(
                            r"<\|start\|>((?!system|user|assistant)(?:(?!<\|(?:start|end)\|>)[\s\S])+)$",
                            solution_text,
                        )
                        if match:
                            cot_text = match.group(1)
                            start_cot_idx = match.start(1)
                        else:
                            # Fallback: CoT directly after user message <|end|> without <|start|> tag,
                            # ending at <|end|><|start|>assistant<|channel|>final
                            match = re.search(
                                r"<\|start\|>user<\|message\|>[\s\S]*?<\|end\|>((?!<\|start\|>)[\s\S]+?)<\|end\|><\|start\|>assistant<\|channel\|>final",
                                solution_text,
                            )
                            if match:
                                cot_text = match.group(1)
                                start_cot_idx = match.start(1)
        elif (
            "deepseek" in model_name.lower()
            or "qwen" in model_name.lower()
            or "olmo" in model_name.lower()
        ):
            # Get whatever is within <think> and </think> tags.
            # If the opening <think> tag is missing, assume thinking starts at the beginning
            match = re.search(r"<think>([\s\S]*?)</think>", solution_text)
            if match:
                cot_text = match.group(1)
                start_cot_idx = match.start(
                    1
                )  # Start of the captured group (after <think>)
            else:
                # Didn't find both start and end tags, assume everything up to </think> is the thinking
                match = re.search(r"([\s\S]*)</think>", solution_text)
                if match:
                    cot_text = match.group(1)
                    start_cot_idx = 0
                else:
                    # No end tag found, assume everything from <think> onwards is the thinking
                    match = re.search(r"<think>([\s\S]*)", solution_text)
                    if match:
                        cot_text = match.group(1)
                        start_cot_idx = match.start(1)
    except:
        pass

    if cot_text is None:
        # Default: assume the entire solution is the CoT
        cot_text = solution_text
        start_cot_idx = 0

    return cot_text, start_cot_idx


def cot_sentences_and_token_borders(output, model, tokenizer):
    """
    Split a reasoning chain (taken from a model's output) into reasoning steps (sentences),
    and find the token borders in the original output corresponding to each step.
    """
    cot_output, cot_start_idx = extract_cot_from_output(output, model.model_name)
    sentences, sent_ranges = _split_solution_into_sentences(cot_output)

    # Sanity verification
    assert all(
        ["You are ChatGPT" not in s and "think>" not in s for s in sentences]
    ), "extract_cot_from_output probably failed..."

    sent_ranges = [
        (start + cot_start_idx, end + cot_start_idx) for (start, end) in sent_ranges
    ]  # Adjust sentence ranges to the full output
    token_borders = _get_sentence_token_ranges(output, sent_ranges, tokenizer)
    return sentences, token_borders


def _split_solution_into_sentences(solution_text: str) -> List[str]:
    """
    Split a solution into sentences for rollout generation.

    Args:
        solution_text: The full solution text

    Returns:
        List of sentences
    """
    # Define patterns for sentence boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    # Split the text into sentences
    sentences = []
    current_sent = ""

    # Process the text character by character
    sent_start = i = 0
    while i < len(solution_text):
        current_sent += solution_text[i]

        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True

        # If we found a boundary, add the sentence and reset
        if is_paragraph_end or is_sentence_end:
            sentences.append((current_sent, sent_start, i + 1))
            current_sent = ""
            sent_start = i + 1

        i += 1

    # Add the last sentence if not empty
    if current_sent.strip():
        sentences.append((current_sent, sent_start, len(solution_text)))

    # Merge small sentences (less than 10 characters)
    merged_sentences = []
    for sent, start, end in sentences:
        if merged_sentences and len(sent.strip()) < 10:
            prev_sent, prev_start, prev_end = merged_sentences[-1]
            merged_sentences[-1] = (prev_sent + sent, prev_start, end)
        else:
            merged_sentences.append((sent, start, end))

    return [sent for sent, _, _ in merged_sentences], [
        (start, end) for _, start, end in merged_sentences
    ]


def _get_sentence_token_ranges(
    text: str, sent_ranges: List[Tuple[int, int]], tokenizer: AutoTokenizer
) -> List[Tuple[int, int]]:
    """Convert character positions to token indices"""
    sent_token_ranges = []

    for sent_start, sent_end in sent_ranges:
        sent_start_token = tokenizer.encode(text[:sent_start], add_special_tokens=False)
        sent_start_token_idx = len(sent_start_token)
        sent_end_token = tokenizer.encode(text[:sent_end], add_special_tokens=False)
        sent_end_token_idx = len(sent_end_token)
        sent_token_ranges.append((sent_start_token_idx, sent_end_token_idx))

    return sent_token_ranges


def apply_step_mask_to_cot(
    full_prompt: str,
    sentences: List[str],
    sentence_mask: List[bool],
    model_name: str,
    clean_whitespace: bool = False,
    verbose: bool = False,
) -> str:
    """
    Applies the sentence mask to the full prompt (input + reasoning + summary) or to a CoT reasoning chain,
    removing sentences that are masked out, and removing all text after the CoT.

    Args:
        full_prompt (str): The original full prompt containing the CoT / only the CoT.
        sentences (list of str): List of sentences in the CoT (Not the whole input+output!).
        sentence_mask (list of bool): Boolean mask indicating which sentences to keep.
        model_name (str): Name of the model used for generating the CoT.
        clean_whitespace (bool): Whether to clean up whitespace artifacts after removal.

    Returns:
        str: The modified prompt with masked sentences removed.

    Raises:
        ValueError: If sentences and sentence_mask have different lengths.
        ValueError: If CoT extraction fails.
    """
    # Validate that sentences and mask have same length
    if len(sentences) != len(sentence_mask):
        raise ValueError(
            f"Length mismatch: sentences ({len(sentences)}) != sentence_mask ({len(sentence_mask)})"
        )

    cot, cot_start_idx = extract_cot_from_output(full_prompt, model_name)

    # Handle None CoT from failed extraction
    if cot is None or "<think>" in cot or "</think>" in cot or "You are ChatGPT" in cot:
        raise ValueError(f"Failed to extract CoT from prompt for model {model_name}")

    # Find exact positions of each sentence sequentially
    # This handles duplicates correctly by searching from where we left off
    removal_ranges: List[Tuple[int, int]] = []
    search_start = 0

    for i, sentence in enumerate(sentences):
        # Handle empty sentences
        if not sentence or not sentence.strip():
            continue

        # Find the sentence starting from where we left off
        # This correctly handles duplicate sentences by finding them in order
        pos = cot.find(sentence, search_start)

        if pos == -1:
            # Try normalized matching as fallback
            # Sometimes there are whitespace differences
            normalized_sentence = " ".join(sentence.split())
            normalized_cot_segment = " ".join(cot[search_start:].split())

            # Try to find in normalized form
            norm_pos = normalized_cot_segment.find(normalized_sentence)
            if norm_pos == -1:
                if verbose:
                    warnings.warn(
                        f"Sentence {i} not found in CoT (starting from position {search_start}): "
                        f"'{sentence[:50]}{'...' if len(sentence) > 50 else ''}'".replace(
                            "\n", "\\n"
                        )
                    )
                continue
            else:
                # Found in normalized form - try to map back to original position
                # This is approximate but better than skipping
                pos = (
                    cot.find(sentence.split()[0], search_start)
                    if sentence.split()
                    else -1
                )
                if pos == -1:
                    if verbose:
                        warnings.warn(
                            f"Sentence {i} found in normalized form but couldn't map position: "
                            f"'{sentence[:50]}{'...' if len(sentence) > 50 else ''}'".replace(
                                "\n", "\\n"
                            )
                        )
                    continue

        # Record position for removal if this sentence is masked out
        if not sentence_mask[i]:
            removal_ranges.append((pos, pos + len(sentence)))

        # Move search start past this sentence for the next iteration
        search_start = pos + len(sentence)

    # Bug fix #2: Sort by position descending and remove from end to start
    # This ensures earlier indices remain valid as we remove text
    removal_ranges.sort(key=lambda x: x[0], reverse=True)

    # Remove sentences from end to start
    partial_cot = cot
    for start_pos, end_pos in removal_ranges:
        partial_cot = partial_cot[:start_pos] + partial_cot[end_pos:]

    # Clean up whitespace artifacts
    if clean_whitespace:
        partial_cot = _clean_whitespace(partial_cot)

    new_full_prompt = full_prompt[:cot_start_idx] + partial_cot
    return new_full_prompt


# ----------------------------------------------------------
# Answer searching funtions
# ----------------------------------------------------------
class AnswerError(Enum):
    WRONG_BOXED_ANSWER = -2
    ANSWER_NOT_FOUND = -1


def search_answer(sentences, gt_answer, boxed=True):
    """
    Searches for a specific answer (gt_answer) in a list of reasoning steps,
    either as a boxed answer or as a substring, using various normalization heuristics.
    """
    for i, sent in list(enumerate(sentences))[::-1]:
        if boxed:
            boxed_answers = extract_boxed_answers(sent)
            if len(boxed_answers) > 0:
                boxed_ans = boxed_answers[-1]
                if are_answers_equivalent(boxed_ans, gt_answer):
                    return i, sent
                else:
                    # Found boxed answer, which is probably the final answer, which isn't true
                    return AnswerError.WRONG_BOXED_ANSWER, sent
        else:
            if normalize_latex(gt_answer) in normalize_latex(sent):
                return i, sent
            if re.escape(normalize_latex(gt_answer)) in re.escape(
                normalize_latex(sent)
            ):
                return i, sent
    return AnswerError.ANSWER_NOT_FOUND, ""


def sentence_states_answer(sentence: str, gt_answer: str) -> bool:
    """
    Check if a sentence directly states the answer using heuristic patterns.

    Args:
        sentence: The sentence to check
        gt_answer: The ground truth answer to compare against

    Returns:
        True if the sentence appears to directly state the answer
    """
    # Check #1: Direct answer statement patterns
    escaped_gt_answer = re.escape(gt_answer.replace("$", ""))
    direct_answer_patterns = [
        r"(Thus|Therefore|So|Hence|Consequently|This means).*(answer|result)[^\n]+"
        + escaped_gt_answer,
        r"(The|Our|My)\s+(final\s+)?(answer|result)[^\n]+" + escaped_gt_answer,
        r"(Thus|Therefore|So|Hence|Consequently),? (answer|result)[^\n]+",
    ]
    for pattern in direct_answer_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True

    # Check #2: Boxed answer matching ground truth
    boxed_answers = extract_boxed_answers(sentence)
    for boxed_ans in boxed_answers:
        if are_answers_equivalent(boxed_ans, gt_answer):
            return True

    return False


def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.

    Args:
        text: The text to extract boxed answers from

    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not boxed_starts:
        return []

    answers = []

    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""

        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break

            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char

            idx += 1

        if answer:
            answers.append(answer)

    return answers


# --------------------------------------------------------
# Model-specific token functions
# --------------------------------------------------------
def get_end_thinking_token(model_name):
    """
    Get the token marking the end of the chain-of-thought reasoning for a given model.
    Note - this token isn't neccesarily unique for CoT end (e..g, in gpt-oss).
    """
    if "gpt" in model_name.lower():
        end_thinking_token = "<|end|>"
    elif "deepseek" in model_name.lower() or "olmo" in model_name.lower():
        end_thinking_token = "</think>"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return end_thinking_token


def get_empty_cot_ending(model_name):
    """
    Get the token sequence to append to the prompt when evaluating an empty CoT.
    """
    if "gpt-oss" in model_name.lower():
        return "<|end|><|start|>final<|message|>The answer is \\boxed{"
    elif "deepseek" in model_name.lower() or "olmo" in model_name.lower():
        return "The answer is \\boxed{"
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported for empty CoT ending."
        )


def get_prefill_ending(model_name):
    if "gpt-oss" in model_name.lower():
        return "<|end|><|start|>final<|message|>The answer is \\boxed{"
    elif "deepseek" in model_name.lower() or "olmo" in model_name.lower():
        return " Therefore, the final answer is \\boxed{"
    else:
        raise NotImplementedError(
            f"Model {model_name} not supported for prefill ending."
        )


# --------------------------------------------------------
# Answer comparison heuristic functions + Answer comparison function
# --------------------------------------------------------


def try_parse_fraction(s: str) -> Optional[Fraction]:
    """
    Try to parse a string as a fraction.
    Handles: \frac{a}{b}, a/b, and integer forms.
    """
    s = s.strip()

    # Remove $ and spacing
    s = s.strip("$")
    s = re.sub(r"\\[,;:!]", "", s)
    s = s.strip()

    # Handle \frac{a}{b} format (with or without braces)
    # \frac{num}{denom} or \frac ab (single digits)
    frac_match = re.search(r"\\(?:d|t)?frac\{?(\d+)\}?\{?(\d+)\}?", s)
    if frac_match:
        num, denom = int(frac_match.group(1)), int(frac_match.group(2))
        if denom != 0:
            return Fraction(num, denom)

    # Handle a/b format
    slash_match = re.match(r"^(\d+)/(\d+)$", s)
    if slash_match:
        num, denom = int(slash_match.group(1)), int(slash_match.group(2))
        if denom != 0:
            return Fraction(num, denom)

    # Handle ratio format a:b (GROUP 7)
    ratio_match = re.match(r"^(\d+):(\d+)$", s.replace(" ", ""))
    if ratio_match:
        num, denom = int(ratio_match.group(1)), int(ratio_match.group(2))
        if denom != 0:
            return Fraction(num, denom)

    # Handle integer
    try:
        return Fraction(int(s))
    except ValueError:
        pass

    return None


def try_parse_number(s: str) -> Optional[float]:
    """
    Try to parse a string as a numeric value.
    Handles fractions, decimals, percentages, repeating decimals, mixed numbers, units.
    """
    original = s
    s = s.strip().lower()

    # Remove $ signs
    s = s.strip("$")

    # Remove LaTeX spacing commands
    s = re.sub(r"\\[,;:!]", "", s)
    s = s.strip()

    # Remove commas used as thousand separators
    s = re.sub(r"(\d),(\d)", r"\1\2", s)

    # Handle {-} as minus sign
    s = s.replace("{-}", "-")

    # GROUP 4: Handle unit multipliers
    unit_multipliers = {
        "million": 1_000_000,
        "billion": 1_000_000_000,
        "trillion": 1_000_000_000_000,
        "thousand": 1_000,
        "hundred": 100,
    }

    multiplier = 1
    for unit, mult in unit_multipliers.items():
        if unit in s:
            s = re.sub(r"\\?text\{?\s*" + unit + r"\s*\}?", "", s)
            s = s.replace(unit, "")
            multiplier = mult
            break

    # GROUP 4: Remove non-numeric units (dollars, meters, etc.)
    unit_words = [
        "dollars",
        "dollar",
        "cents",
        "cent",
        "meters",
        "meter",
        "feet",
        "foot",
        "inches",
        "inch",
        "pounds",
        "pound",
        "kilograms",
        "kilogram",
        "kg",
        "km",
        "cm",
        "mm",
        "m",
        "seconds",
        "second",
        "minutes",
        "minute",
        "hours",
        "hour",
        "mph",
        "kph",
        "mi",
        "miles",
        "mile",
    ]
    for unit in unit_words:
        s = re.sub(r"\\?text\{?\s*" + unit + r"\s*\}?", "", s)
        # Only remove standalone unit words, not partial matches
        s = re.sub(r"\b" + unit + r"\b", "", s)

    s = s.strip()

    # Remove remaining \text{} wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # Strip common prefix words (is, equals, approximately, etc.)
    prefix_words = ["is", "equals", "=", "approximately", "about", "around"]
    for prefix in prefix_words:
        s = re.sub(r"^" + re.escape(prefix) + r"\s*", "", s, flags=re.IGNORECASE)

    # Handle time format (4 o'clock -> 4)
    time_match = re.match(r"^(\d{1,2})\s*o['\u2019]?\s*clock", s, re.IGNORECASE)
    if time_match:
        return float(time_match.group(1)) * multiplier

    # GROUP 8: Handle repeating decimals (3.\overline{3})
    overline_match = re.match(r"^(\d*)\.?\\overline\{(\d+)\}", s)
    if overline_match:
        integer_part = overline_match.group(1) or "0"
        repeating_part = overline_match.group(2)
        # Convert repeating decimal to fraction
        # e.g., 3.\overline{3} = 3 + 1/3 = 10/3
        # 0.\overline{142857} = 142857/999999
        repeat_len = len(repeating_part)
        denom = int("9" * repeat_len)
        numer = int(repeating_part)
        frac = Fraction(numer, denom)
        return (int(integer_part) + float(frac)) * multiplier

    # GROUP 9: Handle mixed numbers (83\frac{1}{3})
    mixed_match = re.match(r"^(\d+)\\(?:d|t)?frac\{?(\d+)\}?\{?(\d+)\}?", s)
    if mixed_match:
        whole = int(mixed_match.group(1))
        num = int(mixed_match.group(2))
        denom = int(mixed_match.group(3))
        if denom != 0:
            return (whole + num / denom) * multiplier

    # Handle \frac{a}{b} format (with optional leading minus)
    frac_match = re.search(r"^(-?)\\(?:d|t)?frac\{?(-?\d+)\}?\{?(-?\d+)\}?", s)
    if frac_match:
        sign = -1 if frac_match.group(1) == "-" else 1
        num, denom = int(frac_match.group(2)), int(frac_match.group(3))
        if denom != 0:
            return sign * (num / denom) * multiplier

    # Handle a/b format
    slash_match = re.match(r"^(-?\d+)/(-?\d+)$", s)
    if slash_match:
        num, denom = int(slash_match.group(1)), int(slash_match.group(2))
        if denom != 0:
            return (num / denom) * multiplier

    # Handle ratio format a:b (GROUP 7)
    ratio_match = re.match(r"^(-?\d+):(-?\d+)$", s.replace(" ", ""))
    if ratio_match:
        num, denom = int(ratio_match.group(1)), int(ratio_match.group(2))
        if denom != 0:
            return (num / denom) * multiplier

    # Handle exponentiation (10^4 = 10000)
    exp_match = re.match(r"^(-?\d+(?:\.\d+)?)\^{?(-?\d+)}?$", s)
    if exp_match:
        base = float(exp_match.group(1))
        exponent = int(exp_match.group(2))
        try:
            return (base**exponent) * multiplier
        except (ValueError, OverflowError):
            pass

    # Handle plain numbers (including those with leading zeros issues)
    # Add leading zero if needed
    if s.startswith("."):
        s = "0" + s

    try:
        return float(s) * multiplier
    except ValueError:
        pass

    return None


def normalize_symbolic_addition(s: str) -> Optional[str]:
    """
    Try to normalize expressions with commutative addition.
    Returns sorted terms joined by +, or None if not applicable.
    """
    s = s.strip().lower()
    s = s.strip("$")
    s = re.sub(r"\\[,;:!]", "", s)
    s = s.strip()

    # Only handle simple additions (no nested parentheses)
    if "(" in s or ")" in s:
        return None

    # Check if this looks like a simple addition expression
    if "+" not in s and "-" not in s:
        return None

    # Split by + and - while keeping the signs
    # Replace - with +- for easier splitting
    s = s.replace("-", "+-")
    terms = [t.strip() for t in s.split("+") if t.strip()]

    if len(terms) < 2:
        return None

    # Normalize each term and sort
    normalized_terms = []
    for term in terms:
        # Remove leading + if present
        term = term.lstrip("+")
        # Normalize the term
        term = re.sub(r"\s+", "", term)
        term = re.sub(r"{([^{}]+)}", r"\1", term)
        normalized_terms.append(term)

    # Sort terms (put negative terms at the end)
    normalized_terms.sort(key=lambda x: (x.startswith("-"), x))

    return "+".join(normalized_terms)


def normalize_tuple(s: str) -> Optional[str]:
    """
    Normalize tuple/list format like (a,b,c) or a,b,c to a canonical form.
    Returns None if not a tuple-like expression.
    """
    s = s.strip().strip("$")
    s = re.sub(r"\\[,;:!]", "", s)  # Remove LaTeX spacing
    s = s.strip()

    # Remove outer parentheses if present
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    # Check if this looks like a comma-separated list of values
    # Must have at least one comma and consist of simple values
    if "," not in s:
        return None

    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 2:
        return None

    # Check that each part is a simple value (number, variable, or simple expression)
    for part in parts:
        # Allow numbers, variables, simple expressions
        if not re.match(r"^-?[\d\w\.\+\-\*/\^]+$", part):
            return None

    return "(" + ",".join(parts) + ")"


def are_answers_equivalent(found_answer: str, gt_answer: str) -> bool:
    """
    Check if two answers are equivalent using various normalization and parsing heursitics.
    """
    # Direct normalization comparison
    if normalize_latex(found_answer) == normalize_latex(gt_answer):
        return True

    # Tuple/list comparison (must come before numeric to avoid comma issues)
    found_tuple = normalize_tuple(found_answer)
    gt_tuple = normalize_tuple(gt_answer)
    if found_tuple is not None and gt_tuple is not None:
        if found_tuple == gt_tuple:
            return True

    # Variable assignment extraction (N=2 vs 2, y=5 vs 5)
    def extract_assignment_value(s: str) -> Optional[str]:
        s = s.strip().strip("$")
        # Match patterns like "N=2", "x = 5", "y=-3"
        match = re.match(r"^[a-zA-Z_]\w*\s*=\s*(.+)$", s)
        if match:
            return match.group(1).strip()
        return None

    found_val = extract_assignment_value(found_answer)
    gt_val = extract_assignment_value(gt_answer)

    # If one is an assignment and the other is just the value, compare values
    if found_val is not None and gt_val is None:
        if normalize_latex(found_val) == normalize_latex(gt_answer):
            return True
    if gt_val is not None and found_val is None:
        if normalize_latex(gt_val) == normalize_latex(found_answer):
            return True

    # Inequality to interval conversion (-1 < x < 11 vs (-1,11))
    def inequality_to_interval(s: str) -> Optional[str]:
        s = s.strip().strip("$")
        s = re.sub(r"\s+", "", s)  # Remove spaces
        # Match patterns like "a<x<b" or "a<=x<=b" or "a<x<=b" etc.
        match = re.match(r"^(-?[\d\.]+)[<≤]([a-zA-Z])[<≤](-?[\d\.]+)$", s)
        if match:
            return f"({match.group(1)},{match.group(3)})"
        return None

    def interval_to_tuple(s: str) -> Optional[str]:
        s = s.strip().strip("$")
        s = re.sub(r"\s+", "", s)
        # Match interval format (a,b)
        match = re.match(r"^\((-?[\d\.]+),(-?[\d\.]+)\)$", s)
        if match:
            return f"({match.group(1)},{match.group(2)})"
        return None

    found_interval = inequality_to_interval(found_answer) or interval_to_tuple(
        found_answer
    )
    gt_interval = inequality_to_interval(gt_answer) or interval_to_tuple(gt_answer)
    if found_interval is not None and gt_interval is not None:
        if found_interval == gt_interval:
            return True

    # Numeric equivalence (handles fractions, decimals, units, repeating decimals)
    found_num = try_parse_number(found_answer)
    gt_num = try_parse_number(gt_answer)
    if found_num is not None and gt_num is not None:
        # Use relative tolerance for floating point comparison
        # e.g., 83.33 should match 83 1/3 (=83.333...)
        if abs(gt_num) > 1e-10:
            rel_diff = abs(found_num - gt_num) / abs(gt_num)
            if rel_diff < 5e-4:  # 0.05% tolerance
                return True
        else:
            # For values near zero, use absolute tolerance
            if abs(found_num - gt_num) < 1e-10:
                return True

    # Fraction equivalence
    found_frac = try_parse_fraction(found_answer)
    gt_frac = try_parse_fraction(gt_answer)
    if found_frac is not None and gt_frac is not None:
        if found_frac == gt_frac:
            return True

    # Commutative addition (sqrt{15}+8 == 8+sqrt{15})
    norm_found_add = normalize_symbolic_addition(found_answer)
    norm_gt_add = normalize_symbolic_addition(gt_answer)
    if norm_found_add is not None and norm_gt_add is not None:
        if norm_found_add == norm_gt_add:
            return True

    # Month abbreviation handling (Feb vs February)
    month_abbrevs = {
        "jan": "january",
        "feb": "february",
        "mar": "march",
        "apr": "april",
        "may": "may",
        "jun": "june",
        "jul": "july",
        "aug": "august",
        "sep": "september",
        "sept": "september",
        "oct": "october",
        "nov": "november",
        "dec": "december",
    }

    def normalize_month(s: str) -> str:
        s = s.strip().lower().strip("$")
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).strip()
        return month_abbrevs.get(s, s)

    if normalize_month(found_answer) == normalize_month(gt_answer):
        return True

    # Ratio to fraction (5:19 == \frac{5}{19})
    # Handle ratio in gt_answer matched to fraction in found_answer
    gt_clean = gt_answer.strip().strip("$")
    gt_clean = re.sub(r"\\[,;:!]", "", gt_clean).strip()
    found_clean = found_answer.strip().strip("$")
    found_clean = re.sub(r"\\[,;:!]", "", found_clean).strip()

    ratio_match = re.match(r"^(\d+):(\d+)$", gt_clean.replace(" ", ""))
    if ratio_match:
        ratio_as_frac = f"\\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
        if normalize_latex(found_answer) == normalize_latex(ratio_as_frac):
            return True

    ratio_match = re.match(r"^(\d+):(\d+)$", found_clean.replace(" ", ""))
    if ratio_match:
        ratio_as_frac = f"\\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
        if normalize_latex(gt_answer) == normalize_latex(ratio_as_frac):
            return True

    return False


# --------------------------------------------------------
# CoT pruning function
# --------------------------------------------------------
def prune_cot(
    sent_to_sent_attributions: torch.Tensor,
    threshold: float,
    answer_sentence_idx: int,
    influence_func: callable,
    is_recursive: bool = False,
):
    assert 0.0 <= threshold <= 1.0, "Influence threshold must be between 0.0 and 1.0"

    # Convert to positive index
    if answer_sentence_idx == -1:
        answer_sentence_idx = sent_to_sent_attributions.shape[0] - 1

    sentence_mask = torch.zeros(sent_to_sent_attributions.shape[0], dtype=torch.long)
    sentence_mask[answer_sentence_idx] = 1  # Always keep answer sentence
    ref_idx = answer_sentence_idx
    while ref_idx > 0:
        if sentence_mask[ref_idx] != 0:
            sent_influence = influence_func(sent_to_sent_attributions, ref_idx)

            # Find score threshold that keeps the desired fraction of total influence
            sorted_influences = torch.sort(sent_influence, descending=True).values
            cumulative_score = torch.cumsum(sorted_influences, dim=0) / torch.sum(
                sorted_influences
            )
            threshold_index = torch.searchsorted(cumulative_score, threshold).item()
            threshold_influence_value = sorted_influences[
                min(threshold_index, len(cumulative_score) - 1)
            ]

            sentence_mask |= sent_influence >= threshold_influence_value

        ref_idx -= 1

        # If not recursive, only do one pass to measure effect on answer sentence
        if not is_recursive:
            break

    return sentence_mask
