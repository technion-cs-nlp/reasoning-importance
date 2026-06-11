import csv
from enum import Enum
import gc
import glob
import json
import logging
import re
import sys
import numpy as np
import random
import subprocess
import pickle
import torch
import os
from collections import namedtuple

# import transformer_lens as lens
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

lens = namedtuple("lens", ["HookedTransformer"])  # Hack


class Metric(object):
    def __init__(self):
        self.lst = 0.0
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val, cnt=1):
        self.lst = val
        self.sum += val * cnt
        self.cnt += cnt
        self.avg = self.sum / self.cnt


def save_results(
    outputs: Dict[str, Any], metadata: Dict[str, Any], output_path: str
) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump((outputs, metadata), f, indent=4, sort_keys=True)


def load_model(
    model_id: str,
    dtype: torch.dtype = None,
    device_map: str = None,
    model_kwargs: Dict = {},
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, **model_kwargs
    )
    model.to(dtype)
    model.eval()
    if "deepseek" in model_id.lower() or "olmo" in model_id.lower():
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def to_tokens(s, tokenizer):
    return [
        tokenizer.decode(t)
        for t in tokenizer(s, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
    ]


def len_tokens(s, tokenizer):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].numel()


@torch.no_grad()
def generate_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    new_toks: int = 1000,
    sampling_temp: Optional[float] = None,
    sampling_top_p: Optional[float] = None,
) -> str:
    """
    Generate a long text output from a given prompt using the specified model and tokenizer.

    Args:
        model: The language reasoning model to use for generation.
        tokenizer: The tokenizer corresponding to the model.
        prompt: The input prompt as a string or as a list containing a single dictionary).
        new_toks: The number of new tokens to generate.
        reasoning_effort: The level of reasoning effort to apply (used in ).
    """
    if (
        isinstance(prompt, list)
        and isinstance(prompt[0], dict)
        and "content" in prompt[0]
    ):
        prompt, input_len = get_chat_template(prompt, tokenizer, return_as_str=True)
    else:
        input_len = len(tokenizer(prompt)["input_ids"])

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    output = model.generate(
        **model_inputs,
        max_new_tokens=new_toks,
        num_return_sequences=1,
        do_sample=(sampling_temp is not None and sampling_top_p is not None),
        temperature=sampling_temp,
        top_p=sampling_top_p,
    )
    return tokenizer.decode(output[0, input_len:])


@torch.no_grad()
def perplexity(
    sentence_or_logits: Union[torch.Tensor, str],
    model=None,
    tokenizer: AutoTokenizer = None,
    specific_positions: Optional[List[int]] = None,
    labels: Optional[torch.Tensor] = None,
):
    if isinstance(sentence_or_logits, str):
        # Calculate the logits of the chosen words
        assert (
            model is not None and tokenizer is not None
        ), "Model and tokenizer must be provided when input is a string."
        inputs = tokenizer(sentence_or_logits, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        logits = logits[:, :-1, :].contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
    else:
        raise NotImplementedError
        logits = sentence_or_logits
        if labels is None:
            # Assumes logits are already gathered at relevant labels
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

    if specific_positions is not None:
        assert all(
            [p > 0 for p in specific_positions]
        ), "Isn't possible to calculate perplexity on label at first position."
        shifted_specific_positions = [
            p - 1 for p in specific_positions
        ]  # Shift positions for logits vs labels
        assert all(
            [p < logits.size(1) for p in shifted_specific_positions]
        ), "Specific positions exceed sequence length."
        logits = logits[:, shifted_specific_positions, :]
        labels = labels[:, shifted_specific_positions]

    # Calculate perplexity per token
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
    return torch.exp(nll).item()


def get_chat_template(input_dict, tokenizer: AutoTokenizer, return_as_str: bool = True):
    """
    Given an input dictionary with "role" and "content" keys, generate and return the appropriate chat template
    and it's length in tokens.

    Args:
        input_dict: A dictionary with "role" and "content" keys.
        tokenizer: The tokenizer to use for tokenization.
        return_as_str: If True, return the chat template as a string; otherwise, return as token IDs.
    """
    model_name = tokenizer.name_or_path.lower()
    if "gpt-oss" in model_name:
        full_input_toks = tokenizer.apply_chat_template(
            input_dict, return_tensors="pt"
        )[0]
    elif (
        "qwen3-4b" in model_name
        or "deepseek" in model_name.lower()
        or "olmo" in model_name.lower()
    ):
        full_input_toks = tokenizer.apply_chat_template(
            input_dict, return_tensors="pt", add_generation_prompt=True
        )[0]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    full_input_str = tokenizer.decode(full_input_toks)
    input_token_len = len(full_input_toks)

    if return_as_str:
        return full_input_str, input_token_len
    else:
        return full_input_toks, input_token_len


def translate_to_english(text):
    try:
        if len(text.strip()) == 0:
            return text
        from googletrans import Translator

        translator = Translator()
        translated = translator.translate(text, dest="en")
        return translated.text
    except Exception as e:
        logging.error(f"Error translating text: {text}")
        return "FAILURE"


def topk_2d(tensor, k):
    """
    Gets the H, W indices of the topk values in a 2D PyTorch tensor.

    Args:
      tensor: A 2D PyTorch tensor.
      k: The number of top values to retrieve.

    Returns:
      A tuple of two tensors: (topk_h_indices, topk_w_indices).
      Each tensor contains the indices of the topk values along the height and width dimensions, respectively.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError("Input tensor must be a 2D PyTorch tensor.")
    if k > tensor.numel():
        raise ValueError(
            "k cannot be greater than the number of elements in the tensor."
        )

    topk_values, topk_indices = torch.topk(tensor.flatten(), k)

    h_indices = topk_indices // tensor.shape[1]
    w_indices = topk_indices % tensor.shape[1]

    return (h_indices, w_indices), topk_values


def set_deterministic(seed=1337):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cuda_device(device_idx):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)


def get_gpu_count():
    """
    Runs the nvidia-smi command and parses the output to determine the number of GPUs.

    Returns:
        int: The number of GPUs detected on the system.
    """
    try:
        # Run the nvidia-smi command and capture the output
        output = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], universal_newlines=True
        )

        # Split the output into individual lines
        lines = output.strip().split("\n")

        # Count the number of GPU lines
        gpu_count = len(lines)

        return gpu_count
    except (subprocess.CalledProcessError, ValueError):
        # If there's an error running the command or parsing the output, return 0
        return 0


def get_single_token_tokens(processor, token_list):
    """
    Get a list of tokens from token_list that are tokenized to one token only.
    """
    return [
        t
        for t in token_list
        if processor(text=t, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].numel()
        == 1
    ]


def generate_random_strings(
    model, num_tokens, count=1, batch_size=1, initial_token=None
):
    """
    Generate a random string of tokens from the model.
    """
    result_strings = []
    for idx in range(0, count, batch_size):
        real_bs = min(count - idx, batch_size)
        if initial_token is None:
            initial_token = model.to_tokens("")
        else:
            initial_token = model.to_tokens(initial_token, prepend_bos=False)
        tokens = model.generate(
            initial_token.repeat(real_bs, 1),
            num_tokens - 1,
            prepend_bos=False,
            temperature=1.0,
        )  # -1 because BOS is already included
        result_strings += model.to_string(tokens[:, 1:])  # skip BOS
    return result_strings


def reduce_dimensionality(vectors, target_dim=2, type="tsne"):
    assert target_dim in [2, 3], "Only 2D and 3D reductions are supported."
    if type == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=target_dim, random_state=0)
        tsne_vectors = tsne.fit_transform(vectors.detach().numpy())
        return tsne_vectors.T
        # return tsne_vectors[:, 0], tsne_vectors[:, 1], tsne_vectors[:, 2] if target_dim == 3 else
    elif type == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=target_dim, random_state=0)
        pca_vectors = pca.fit_transform(vectors.detach().numpy())
        return pca_vectors.T
        # return pca_vectors[:, 0], pca_vectors[:, 1], pca_vectors[:, 2] if target_dim == 3 else
    elif type == "umap":
        import umap

        reducer = umap.UMAP()
        umap_vectors = reducer.fit_transform(vectors.detach().numpy())
        return umap_vectors.T
        # return umap_vectors[:, 0], umap_vectors[:, 1], umap_vectors[:, 2] if target_dim == 3 else
    else:
        raise NotImplementedError


def safe_eval(prompt):
    """
    Wrapper for eval function to avoid throwing exceptions where dividing by zero.
    """
    try:
        return int(eval(prompt))
    except ZeroDivisionError as e:
        return torch.nan


def generate_prompt_completion(
    prompt: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
        )
        return tokenizer.decode(output.view(-1)[inputs["input_ids"].shape[1] :])


def flatten(recursive_list):
    """
    Flatten a recursed list (of any recursion depth) in a single list.
    """
    flat_list = []

    def _flatten(lst):
        for item in lst:
            if isinstance(item, list):
                _flatten(item)
            else:
                flat_list.append(item)

    _flatten(recursive_list)
    return flat_list


def get_topk_indices_3d(tensor, k):
    """
    Get the 3-D indices of the top k values in a 3-D tensor.

    Args:
        tensor: Input tensor of shape (A, B, C)
        k: Number of top values to retrieve

    Returns:
        values: Top k values
        indices: Tuple of (dim0_indices, dim1_indices, dim2_indices) for top k values
    """
    # Get tensor shape
    A, B, C = tensor.shape

    # Get top k values and flat indices
    values, flat_indices = torch.topk(tensor.flatten(), k)

    # Calculate indices for each dimension
    dim2_indices = flat_indices % C
    temp = flat_indices // C
    dim1_indices = temp % B
    dim0_indices = temp // B
    return torch.stack([dim0_indices, dim1_indices, dim2_indices]).T


def n_layers(model: Union[lens.HookedTransformer, torch.nn.Module]) -> int:
    """
    Returns the number of layers in the model.
    """
    # if isinstance(model, lens.HookedTransformer):
    # return model.cfg.n_layers
    # else:
    return model.model.config.num_hidden_layers


def n_heads(model: Union[lens.HookedTransformer, torch.nn.Module]) -> int:
    """
    Returns the number of attention heads in the model.
    """
    # if isinstance(model, lens.HookedTransformer):
    # return model.cfg.n_heads
    # else:
    return model.model.config.num_attention_heads


def d_model(model: Union[lens.HookedTransformer, torch.nn.Module]) -> int:
    """
    Returns the model dimension (d_model) of the model.
    """
    # if isinstance(model, lens.HookedTransformer):
    # return model.cfg.d_model
    # else:
    return model.model.config.hidden_size


Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])


def get_topk(
    logits: torch.Tensor, tokenizer, k: int = 5, to_probs: bool = True
) -> List[Tuple[str, float]]:
    vals = logits.squeeze()[-1]
    if to_probs:
        vals = torch.softmax(logits.squeeze()[-1], dim=-1)

    topk = torch.topk(vals, k)
    return [
        (tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)
    ]


@torch.no_grad()
def logit_lens(tensor, model, use_final_ln: bool = True):
    # if isinstance(model, lens.HookedTransformer):
    # if use_final_ln:
    # tensor = model.ln_final(tensor)
    # return model.unembed(tensor)
    # else:
    tensor = tensor.to(device=model.device, dtype=model.lm_head.weight.dtype)
    if use_final_ln:
        tensor = model.model.norm(tensor)
    return model.lm_head(tensor)


#### Memory leak debugging util functions ####
def enable_memory_snapshot():
    # keep a maximum 100,000 alloc/free events from before the snapshot
    torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100_000)


def gpu_memory_snapshot(output_file):
    snapshot = torch.cuda.memory._snapshot()
    with open(output_file, "wb") as f:
        pickle.dump(snapshot, f)


def monitor_out_of_memory():
    """
    Register a monitor to save a GPU memory snapshot right after an Out-Of-Memory error occurs.
    """
    enable_memory_snapshot()

    def oom_observer(device, alloc, device_alloc, device_free):
        gpu_memory_snapshot("oom_snapshot.pkl")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


def monitor_memory(func):
    """
    Decorator wrapper to wrap a function / code block with a memory snapshot before and after it
    """

    def wrapper(*args, **kwargs):
        enable_memory_snapshot()
        func(*args, **kwargs)
        gpu_memory_snapshot("snapshot.pkl")

    return wrapper


def clear_memory():
    """Clear unused memory and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


####
#### External LLM Calls
####
class LLMProvider(Enum):
    """Supported LLM API providers."""

    CLAUDE = "claude"
    GEMINI = "gemini"


# Default models for each provider
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def call_llm_api(
    prompt: str,
    provider: LLMProvider,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Call an LLM API with a prompt and return the response.

    Args:
        prompt: The prompt to send to the LLM.
        provider: The LLM provider to use (CLAUDE or GEMINI).
        model: Optional model name. If not provided, uses provider's default.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0 for deterministic).

    Returns:
        The LLM's response text.

    Raises:
        ValueError: If the provider is not supported.
        ImportError: If the required SDK is not installed.
        Exception: If the API call fails.
    """
    if provider == LLMProvider.CLAUDE:
        return _call_claude_api(prompt, model, max_tokens, temperature)
    elif provider == LLMProvider.GEMINI:
        return _call_gemini_api(prompt, model, max_tokens, temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _call_claude_api(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Call the Claude API (Anthropic) with a prompt.

    Requires ANTHROPIC_API_KEY environment variable to be set.

    Args:
        prompt: The prompt to send.
        model: Model name (default: claude-sonnet-4-20250514).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        The response text from Claude.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    model = model or DEFAULT_CLAUDE_MODEL

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response
    return message.content[0].text


def _call_gemini_api(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Call the Gemini API (Google) with a prompt.

    Requires GOOGLE_API_KEY environment variable to be set.

    Args:
        prompt: The prompt to send.
        model: Model name (default: gemini-2.0-flash).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        The response text from Gemini.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. "
            "Install with: pip install google-generativeai"
        )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    model_name = model or DEFAULT_GEMINI_MODEL

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    model_instance = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    response = model_instance.generate_content(prompt)
    if hasattr(response, 'text'): 
        return response.text
    else:
        return None
