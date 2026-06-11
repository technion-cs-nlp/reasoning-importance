"""
Train a classifier to distinguish between removable and non-removable
sentences (steps) in Chain-of-Thought reasoning chains (prefixes).

Script supporting two modes:
- --no-context: Uses step embeddings (from last model layer) to train a
  linear or MLP classifier.
- --context: Uses full model activations (per-token, per-layer) to train neural
  classifiers (linear, mlp, layerwise_linear, layerwise_mlp).

Pipeline:
1. Loads removability results from eval_prefix_and_removability.py
2. Uses extract_removable_nonremovable_entry_keys to get per-sentence labels
3. Balances classes and splits train/eval
4. Trains and evaluates a binary classifier
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from loading_utils import (
    load_attribution_pruning_results,
    load_generations,
    load_random_pruning_results,
    load_single_step_results,
)
from general_utils import set_deterministic, load_model, n_layers, d_model
from consts import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    SUFFICIENCY_THRESHOLD,
    OBTAINABILITY_THRESHOLD,
    REMOVABILITY_CLASSIFIER_TYPES,
    MODEL_NAME_TO_ID_PATH,
    TEST_SIZE,
    WEIGHT_DECAY,
)
from removability_utils import (
    extract_removable_nonremovable_entry_keys,
    extract_removable_nonremovable_entry_keys_from_random,
)

_CLASSIFIER_TYPES = [
    t
    for t in REMOVABILITY_CLASSIFIER_TYPES
    if t not in ("activation_tensor", "simple_transformer")
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier to distinguish removable vs non-removable CoT sentences"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--context",
        action="store_true",
        help="Use full model activations (per-token, per-layer) for classification",
    )
    mode_group.add_argument(
        "--no-context",
        action="store_true",
        help="Use sentence embeddings (from last model layer) for classification",
    )

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="harp-standard")
    parser.add_argument(
        "--classifier-type",
        type=str,
        required=True,
        choices=_CLASSIFIER_TYPES,
        help="Type of classifier: linear, mlp, layerwise_linear, layerwise_mlp",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-statistics", action="store_true")
    parser.add_argument("--save-output-file", action="store_true")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--sufficiency-threshold", type=float, default=SUFFICIENCY_THRESHOLD
    )
    parser.add_argument(
        "--obtainability-threshold", type=float, default=OBTAINABILITY_THRESHOLD
    )
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--embed-model-id", type=str, default=None)
    parser.add_argument("--embed-model-path", type=str, default=None)
    parser.add_argument("--probe-hidden-dim", type=int, default=4096)
    parser.add_argument("--probe-mlp-layers", type=int, default=2)

    parser.add_argument(
        "--specific-layer",
        type=int,
        default=None,
        help="Use activations from a specific layer (0 = embeddings, 1 = after first block). "
        "Only valid with linear or mlp classifier types.",
    )
    parser.add_argument("--save-embeddings", type=str, default=None)
    parser.add_argument("--load-embeddings", type=str, default=None)

    parser.add_argument(
        "--eval-json",
        type=str,
        default=None,
        help="Path to a JSON file (format: output of removability_pred_llm_as_a_judge.py). "
        "Entries/sentences in this file become the eval set; all remaining pairs are used for training."
        "This arg is used to compare apples-to-apples with the subset of steps evaluated in removability_pred_llm_as_a_judge.py.",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Path to save per-instance eval predictions as JSON.",
    )

    args = parser.parse_args()

    if args.specific_layer is not None:
        if args.classifier_type not in ("linear", "mlp"):
            parser.error(
                "--specific-layer is only valid with linear or mlp classifier types"
            )

    return args


# ---------------------------------------------------------------------------
# Data loading functions for train/eval data
# ---------------------------------------------------------------------------


def load_step_removability_labels(
    model_name: str,
    suff_threshold: float,
    obtn_threshold: float,
    dataset: str = "harp-standard",
) -> Tuple[Dict[str, Dict[int, bool]], Dict]:
    """
    Returns:
        - entry_labels: Dict mapping entry_key -> a dict mapping sentence_idx -> is_removable (bool)
        - step_results: Dict mapping entry_key -> step result dict (from load_single_step_results)
    """
    generations, _ = load_generations(model_name, dataset=dataset)
    attr_pruning_results, _ = load_attribution_pruning_results(
        model_name, dataset=dataset
    )
    step_results, _ = load_single_step_results(model_name, dataset=dataset)
    entry_labels = extract_removable_nonremovable_entry_keys(
        generations,
        attr_pruning_results,
        step_results,
        suff_threshold=suff_threshold,
        obtn_threshold=obtn_threshold,
    )
    n_removable = sum(sum(1 for v in d.values() if v) for d in entry_labels.values())
    n_nonremovable = sum(
        sum(1 for v in d.values() if not v) for d in entry_labels.values()
    )
    logging.info(
        f"  Loaded: {n_removable} removable, {n_nonremovable} non-removable across {len(entry_labels)} entries"
    )
    return entry_labels, step_results


def get_removable_nonremovable_lists(
    entry_labels: Dict[str, Dict[int, bool]],
    step_results: Dict,
    context: bool,
) -> Tuple[list, list]:
    """
    Builds lists of removable and non-removable items for classifier training.

    Returns:
        - removable: list of dicts, each representing a removable step and its relevant context info
        - nonremovable: same but for non-removable steps
    """
    removable = []
    nonremovable = []

    for entry_key, labels in sorted(entry_labels.items(), key=lambda x: x[0]):
        processed_entry = step_results.get(entry_key)
        if processed_entry is None:
            continue

        post_removal = processed_entry.get("post_removal", {})
        sentences = post_removal.get("sentences", [])

        if context:
            token_borders = post_removal.get("token_borders", [])
            full_prompt = post_removal.get("full_prompt", "")
            if not token_borders or not full_prompt:
                continue

        for sent_idx, is_removable in labels.items():
            if sent_idx >= len(sentences):
                continue
            sentence = sentences[sent_idx].strip()
            if not sentence:
                continue

            if context:
                if sent_idx >= len(token_borders):
                    continue
                item = {
                    "sentence": sentence,
                    "token_borders": token_borders[sent_idx],
                    "full_prompt": full_prompt,
                    "entry_key": entry_key,
                    "sentence_idx": sent_idx,
                }
            else:
                item = sentence

            if is_removable:
                removable.append(item)
            else:
                nonremovable.append(item)

    return removable, nonremovable


def split_labels_by_eval_json(
    entry_labels: Dict[str, Dict[int, bool]],
    eval_json_path: str,
) -> Tuple[Dict[str, Dict[int, bool]], Dict[str, Dict[int, bool]]]:
    with open(eval_json_path) as f:
        eval_data, _ = json.load(f)

    eval_pairs = set()
    for entry_key, sentences in eval_data.items():
        for sent_idx_str in sentences:
            eval_pairs.add((entry_key, int(sent_idx_str)))

    train_entry_labels: Dict[str, Dict[int, bool]] = {}
    eval_entry_labels: Dict[str, Dict[int, bool]] = {}

    for entry_key, labels in entry_labels.items():
        for sent_idx, is_removable in labels.items():
            if (entry_key, sent_idx) in eval_pairs:
                eval_entry_labels.setdefault(entry_key, {})[sent_idx] = is_removable
            else:
                train_entry_labels.setdefault(entry_key, {})[sent_idx] = is_removable

    n_eval = sum(len(v) for v in eval_entry_labels.values())
    n_train = sum(len(v) for v in train_entry_labels.values())
    logging.info(
        f"Eval JSON split: {n_eval}/{len(eval_pairs)} eval pairs matched, {n_train} train pairs"
    )
    return train_entry_labels, eval_entry_labels


def balance_and_split(
    removable: list,
    nonremovable: list,
    test_size: float,
    seed: int,
    max_per_class: Optional[int] = None,
) -> Tuple[list, list, np.ndarray, np.ndarray]:
    n_per_class = min(len(removable), len(nonremovable))
    if max_per_class is not None:
        n_per_class = min(n_per_class, max_per_class)

    set_deterministic(seed)
    removable = random.sample(removable, k=n_per_class)
    nonremovable = random.sample(nonremovable, k=n_per_class)
    logging.info(f"Using {n_per_class} samples per class (balanced)")

    all_items = removable + nonremovable
    labels = np.array([1] * len(removable) + [0] * len(nonremovable))

    train_items, eval_items, train_labels, eval_labels = train_test_split(
        all_items,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    logging.info(f"Train set: {len(train_items)} samples")
    logging.info(f"Eval set: {len(eval_items)} samples")
    return train_items, eval_items, train_labels, eval_labels


# ---------------------------------------------------------------------------
# Statistics / sampling / balancing
# ---------------------------------------------------------------------------


def print_statistics(
    model_name: str,
    n_removable: int,
    n_nonremovable: int,
    embed_model_name: Optional[str] = None,
):
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"Model: {model_name}")
    if embed_model_name:
        print(f"Embedding model: {embed_model_name}")
    print(f"Total removable sentences: {n_removable}")
    print(f"Total non-removable sentences: {n_nonremovable}")
    print(
        f"Class ratio (removable:non-removable): {n_removable / max(n_nonremovable, 1):.2f}"
    )
    print("=" * 60 + "\n")


def print_samples(removable: list, nonremovable: list, n_samples: int, context: bool):
    def _fmt(item):
        if context:
            sent = item["sentence"]
            borders = item["token_borders"]
            return f"tokens {borders}: {sent[:150]}{'...' if len(sent) > 150 else ''}"
        return f"{item[:200]}{'...' if len(item) > 200 else ''}"

    print("\n" + "=" * 60)
    print("SAMPLE REMOVABLE SENTENCES:")
    print("=" * 60)
    for i, item in enumerate(removable[:n_samples]):
        print(f"  [{i+1}] {_fmt(item)}")

    print("\n" + "=" * 60)
    print("SAMPLE NON-REMOVABLE SENTENCES:")
    print("=" * 60)
    for i, item in enumerate(nonremovable[:n_samples]):
        print(f"  [{i+1}] {_fmt(item)}")
    print("=" * 60 + "\n")


# ===========================================================================
# Dataset / collation
# ===========================================================================


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: List[torch.Tensor], labels: List[int]):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    embeddings = torch.stack([e.float() for e in embeddings])
    labels = torch.tensor(labels, dtype=torch.float32)
    return embeddings, labels


# ===========================================================================
# Activation extraction (shared by context and no-context paths)
# ===========================================================================


def _extract_from_hidden_states(
    hidden_states: torch.Tensor,
    token_start: int,
    token_end: Optional[int],
    specific_layer: Optional[int],
    average_tokens: bool,
    average_layers: bool,
) -> torch.Tensor:
    """
    Slice and average a (L+1, seq_len, D) hidden-states tensor for a token range.

    Returns shape: (D,) when averaging fully, (L, D) for layerwise, or (L, tokens, D) raw.
    """
    if token_end is None:
        token_end = hidden_states.shape[1]

    if specific_layer is not None:
        return hidden_states[specific_layer, token_start:token_end, :].mean(dim=0)

    hs = hidden_states[
        1:, token_start:token_end, :
    ]  # (L, tokens, D) — skip embedding layer
    if average_tokens:
        hs = hs.mean(dim=1)  # (L, D)
    if average_layers:
        hs = hs.mean(dim=0)  # (D,)
    return hs


@torch.no_grad()
def _cache_prompt_activations(
    data_items: List[Dict],
    model,
    tokenizer,
    batch_size: int = 1,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Cache hidden states for all unique full_prompts in data_items.

    Batches prompts together using right-padding; strips padding before storing
    so each cache entry is a (L+1, unpadded_seq_len, D) tensor.
    """
    unique_prompts = list({item["full_prompt"] for item in data_items})
    logging.info(f"Caching activations for {len(unique_prompts)} unique prompts")
    device = next(model.parameters()).device

    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }
    if max_length is not None:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_length

    cache = {}
    for i in tqdm(
        range(0, len(unique_prompts), batch_size), desc="Caching activations"
    ):
        batch_prompts = unique_prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, **tokenizer_kwargs).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (L+1, B, S, D)
        attention_mask = inputs["attention_mask"]

        for j, prompt in enumerate(batch_prompts):
            seq_len = attention_mask[j].sum().item()
            cache[prompt] = hidden_states[:, j, :seq_len, :].cpu()

        del outputs, inputs, hidden_states
        torch.cuda.empty_cache()

    return cache


def compute_activation_embeddings(
    data_items: List[Dict],
    activation_cache: Dict[str, torch.Tensor],
    average_tokens: bool = True,
    average_layers: bool = False,
    specific_layer: Optional[int] = None,
    desc: str = "Extracting activations",
) -> List[torch.Tensor]:
    """
    Compute activation embeddings by slicing from pre-cached hidden states.

    Returns List of tensors; shape depends on options (see `_extract_from_hidden_states`).
    """
    embeddings = []
    for item in tqdm(data_items, desc=desc):
        token_start, token_end = item["token_borders"]
        try:
            emb = _extract_from_hidden_states(
                activation_cache[item["full_prompt"]],
                token_start,
                token_end,
                specific_layer,
                average_tokens,
                average_layers,
            )
            embeddings.append(emb)
        except Exception as e:
            logging.warning(
                f"Error extracting activations for entry {item.get('entry_key', 'unknown')}: {e}"
            )
    return embeddings


# ===========================================================================
# Embedding orchestration (no-context and context)
# ===========================================================================


def _classifier_aggregation(classifier_type: str) -> Tuple[bool, bool]:
    """How to aggregate cached activations for a given classifier type."""
    average_tokens = True  # all current classifier types expect token-averaged input
    average_layers = classifier_type in ("linear", "mlp")
    return average_tokens, average_layers


def compute_nocontext_embeddings(
    sentences: List[str],
    model,
    tokenizer,
    classifier_type: str,
    specific_layer: Optional[int],
    batch_size: int,
) -> List[torch.Tensor]:
    """
    Compute step embeddings for sentences in isolation (no surrounding context).

    Each sentence is its own prompt. Uses the same cache + extract pipeline as
    the context path.
    """
    items = [{"full_prompt": s, "token_borders": (0, None)} for s in sentences]
    average_tokens, average_layers = _classifier_aggregation(classifier_type)

    activation_cache = _cache_prompt_activations(
        items, model, tokenizer, batch_size=batch_size, max_length=512
    )

    return compute_activation_embeddings(
        items,
        activation_cache,
        average_tokens=average_tokens,
        average_layers=average_layers,
        specific_layer=specific_layer,
        desc="No-context embeddings",
    )


def fix_token_borders(
    orig_tokenizer: AutoTokenizer, new_tokenizer: AutoTokenizer, items: List[Dict]
):
    """
    Remap token borders from orig_tokenizer's token space to new_tokenizer's token space.
    Used when the embedding model is not the original model that generated the analysis of the reasoning chain.
    Uses character-level offset mappings; falls back to sentence-text matching.

    Args:
    - orig_tokenizer: The tokenizer used to originally determine token borders.
    - new_tokenizer: The tokenizer of the model used to calculate step activations.
    - items: A list of dicts, each corresponding to a reasoning step.
    """
    for item in items:
        prompt = item["full_prompt"]
        orig_start, orig_end = item["token_borders"]

        orig_offsets = None
        new_offsets = None

        try:
            orig_offsets = orig_tokenizer(
                prompt, return_offsets_mapping=True, add_special_tokens=False
            ).offset_mapping
        except Exception:
            pass

        try:
            new_offsets = new_tokenizer(
                prompt, return_offsets_mapping=True, add_special_tokens=False
            ).offset_mapping
        except Exception:
            pass

        if new_offsets is None:
            new_ids = new_tokenizer(prompt, add_special_tokens=False).input_ids
            new_offsets = []
            prev_len = 0
            for i in range(len(new_ids)):
                decoded = new_tokenizer.decode(
                    new_ids[: i + 1], skip_special_tokens=False
                )
                cur_len = len(decoded)
                new_offsets.append((prev_len, cur_len))
                prev_len = cur_len

        char_start, char_end = None, None

        if orig_offsets is not None:
            orig_start_c = min(orig_start, len(orig_offsets) - 1)
            orig_end_c = min(orig_end, len(orig_offsets))
            if orig_end_c > orig_start_c:
                char_start = orig_offsets[orig_start_c][0]
                char_end = orig_offsets[orig_end_c - 1][1]

        if char_start is None:
            orig_ids = orig_tokenizer(prompt, add_special_tokens=False).input_ids
            decoded = orig_tokenizer.decode(
                orig_ids[orig_start:orig_end], skip_special_tokens=False
            )
            idx = prompt.find(decoded)
            if idx == -1:
                idx = prompt.find(item["sentence"])
            if idx == -1:
                idx = prompt.find(item["sentence"].strip())
            if idx != -1:
                text = decoded if prompt.find(decoded) != -1 else item["sentence"]
                char_start = idx
                char_end = idx + len(text)

        if char_start is None:
            logging.warning(
                f"Could not determine char range for entry "
                f"{item.get('entry_key', '?')}, sentence_idx={item.get('sentence_idx', '?')}"
            )
            continue

        new_start = None
        new_end = None
        for i, (cs, ce) in enumerate(new_offsets):
            if cs == ce:
                continue
            if ce > char_start and cs < char_end:
                if new_start is None:
                    new_start = i
                new_end = i + 1

        if new_start is not None:
            item["token_borders"] = [new_start, new_end]
        else:
            logging.warning(
                f"Could not remap token borders for entry "
                f"{item.get('entry_key', '?')}, sentence_idx={item.get('sentence_idx', '?')}"
            )

    return items


def compute_context_embeddings(
    train_items: List[Dict],
    eval_items: List[Dict],
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    model,
    tokenizer,
    classifier_type: str,
    specific_layer: Optional[int],
    data_model_name: str,
    embed_model_name: str,
    num_model_layers: int,
    model_hidden_dim: int,
    save_dir: str,
    load_embeddings: Optional[str] = None,
    save_embeddings: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], np.ndarray, np.ndarray, int, int]:
    """
    Compute step embeddings for items in their full surrounding context.

    Handles load/save embeddings cache and cross-model token remapping.

    Returns:
        train_embeddings, eval_embeddings, train_labels, eval_labels,
        num_model_layers, model_hidden_dim
        (labels and dims may be updated vs inputs when loading from .pt cache)
    """
    average_tokens, average_layers = _classifier_aggregation(classifier_type)

    embedding_path = os.path.join(save_dir, load_embeddings or "")

    if load_embeddings and os.path.exists(embedding_path):
        logging.info(f"Loading pre-computed embeddings from {embedding_path}")
        saved_data = torch.load(embedding_path, weights_only=False)
        train_embeddings = saved_data["train_embeddings"]
        eval_embeddings = saved_data["eval_embeddings"]
        train_labels = saved_data["train_labels"]
        eval_labels = saved_data["eval_labels"]
        num_model_layers = saved_data["num_layers"]
        model_hidden_dim = saved_data["hidden_dim"]
    else:
        activation_cache = _cache_prompt_activations(
            train_items + eval_items, model, tokenizer
        )

        if data_model_name != embed_model_name:
            orig_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME_TO_ID_PATH[data_model_name][1]
            )
            train_items = fix_token_borders(orig_tokenizer, tokenizer, train_items)
            eval_items = fix_token_borders(orig_tokenizer, tokenizer, eval_items)
            del orig_tokenizer

        train_embeddings = compute_activation_embeddings(
            train_items,
            activation_cache,
            average_tokens=average_tokens,
            average_layers=average_layers,
            specific_layer=specific_layer,
            desc="Train embeddings",
        )
        eval_embeddings = compute_activation_embeddings(
            eval_items,
            activation_cache,
            average_tokens=average_tokens,
            average_layers=average_layers,
            specific_layer=specific_layer,
            desc="Eval embeddings",
        )

        del activation_cache
        gc.collect()

        if save_embeddings:
            save_path = os.path.join(save_dir, save_embeddings)
            logging.info(f"Saving embeddings to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "train_embeddings": train_embeddings,
                    "eval_embeddings": eval_embeddings,
                    "train_labels": train_labels,
                    "eval_labels": eval_labels,
                    "num_layers": num_model_layers,
                    "hidden_dim": model_hidden_dim,
                    "classifier_type": classifier_type,
                },
                save_path,
            )

    logging.info(
        f"{len(train_embeddings)} train, {len(eval_embeddings)} eval embeddings"
    )
    if not train_embeddings or not eval_embeddings:
        logging.error("No valid embeddings extracted. Check model and data.")
        sys.exit(1)

    logging.info(f"Embedding shape: {train_embeddings[0].shape}")
    return (
        train_embeddings,
        eval_embeddings,
        train_labels,
        eval_labels,
        num_model_layers,
        model_hidden_dim,
    )


# ===========================================================================
# Classifier architectures
# ===========================================================================


class SimpleMLP(nn.Module):
    """MLP classifier for (D,) input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        out_dim = hidden_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = out_dim
            out_dim = out_dim // 2
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LayerwiseLinear(nn.Module):
    """Linear classifier for (L, D) input — flattens to (L*D,)."""

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(num_layers * hidden_dim, 1)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class LayerwiseMLP(nn.Module):
    """MLP classifier for (L, D) input."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        first_hidden: int = 4096,
    ):
        super().__init__()
        input_dim = num_layers * model_dim
        layers = [
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, 1),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


# ===========================================================================
# Classifier training
# ===========================================================================


def create_classifier(
    classifier_type: str,
    model_num_layers: int,
    model_hidden_dim: int,
    probe_hidden_dim: int = 4096,
    probe_mlp_layers: int = 2,
) -> nn.Module:
    if classifier_type == "linear":
        return nn.Linear(model_hidden_dim, 1)
    elif classifier_type == "mlp":
        return SimpleMLP(model_hidden_dim, probe_hidden_dim, probe_mlp_layers)
    elif classifier_type == "layerwise_linear":
        return LayerwiseLinear(model_num_layers, model_hidden_dim)
    elif classifier_type == "layerwise_mlp":
        return LayerwiseMLP(model_num_layers, model_hidden_dim, probe_hidden_dim)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def train_classifier(
    classifier: nn.Module,
    train_embeddings: List[torch.Tensor],
    train_labels: np.ndarray,
    eval_embeddings: List[torch.Tensor],
    eval_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[nn.Module, Dict, np.ndarray]:
    """
    Train and evaluate a binary classifier on List[torch.Tensor] embeddings.

    Uses EmbeddingDataset, CosineAnnealingLR, batched eval, and best-model tracking.

    Returns:
        (trained_classifier, results_dict, final_eval_preds)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Training {classifier.__class__.__name__}...")

    classifier = classifier.to(device)

    train_dataset = EmbeddingDataset(train_embeddings, train_labels.tolist())
    eval_dataset = EmbeddingDataset(eval_embeddings, eval_labels.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_results = []
    best_eval_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(classifier(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        classifier.eval()
        train_preds, train_gt = [], []
        eval_preds, eval_gt = [], []

        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                preds = (
                    (torch.sigmoid(classifier(batch_X.to(device))) > 0.5)
                    .float()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                train_preds.extend(preds)
                train_gt.extend(batch_y.numpy())

            for batch_X, batch_y in eval_loader:
                preds = (
                    (torch.sigmoid(classifier(batch_X.to(device))) > 0.5)
                    .float()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                eval_preds.extend(preds)
                eval_gt.extend(batch_y.numpy())

        train_preds, train_gt = np.array(train_preds), np.array(train_gt)
        eval_preds, eval_gt = np.array(eval_preds), np.array(eval_gt)

        train_acc = accuracy_score(train_gt, train_preds)
        eval_acc = accuracy_score(eval_gt, eval_preds)
        _, _, eval_f1, _ = precision_recall_fscore_support(
            eval_gt, eval_preds, average="binary"
        )
        avg_loss = total_loss / len(train_loader)

        epoch_results.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_accuracy": train_acc,
                "eval_accuracy": eval_acc,
                "eval_f1": eval_f1,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Eval Acc: {eval_acc:.4f}, F1: {eval_f1:.4f}"
        )

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in classifier.state_dict().items()
            }

    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
        classifier = classifier.to(device)

    classifier.eval()
    final_preds, final_gt = [], []
    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            preds = (
                (torch.sigmoid(classifier(batch_X.to(device))) > 0.5)
                .float()
                .cpu()
                .numpy()
                .flatten()
            )
            final_preds.extend(preds)
            final_gt.extend(batch_y.numpy())

    final_preds = np.array(final_preds)
    final_gt = np.array(final_gt)

    results = {
        "epoch_results": epoch_results,
        "best_eval_accuracy": best_eval_acc,
        "final_classification_report": classification_report(
            final_gt, final_preds, target_names=["non-removable", "removable"]
        ),
    }

    return classifier, results, final_preds


# ===========================================================================
# Output formatting, saving, and printing
# ===========================================================================


def build_output_data(
    args,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    train_embeddings: list,
    eval_embeddings: list,
    results: Dict,
    extra_params: Dict,
) -> Dict:
    base_params = {
        "seed": args.seed,
        "classifier_type": args.classifier_type,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "sufficiency_threshold": args.sufficiency_threshold,
        "obtainability_threshold": args.obtainability_threshold,
        "specific_layer": args.specific_layer,
        "eval_json": os.path.basename(args.eval_json) if args.eval_json else None,
    }
    base_params.update(extra_params)

    return {
        "parameters": base_params,
        "data_statistics": {
            "n_removable": int((train_labels == 1).sum() + (eval_labels == 1).sum()),
            "n_nonremovable": int((train_labels == 0).sum() + (eval_labels == 0).sum()),
            "n_train": len(train_embeddings),
            "n_eval": len(eval_embeddings),
        },
        "results": results,
    }


def save_predictions_if_requested(
    args,
    final_preds: np.ndarray,
    eval_labels: np.ndarray,
    eval_items: list,
) -> None:
    if not args.save_predictions:
        return
    predictions_data = []
    for i, (pred, gt) in enumerate(zip(final_preds, eval_labels)):
        item = eval_items[i]
        entry = {"ground_truth": int(gt), "prediction": int(pred)}
        if isinstance(item, dict):
            entry["entry_key"] = item["entry_key"]
            entry["sentence_idx"] = item["sentence_idx"]
            entry["sentence"] = item["sentence"]
        else:
            entry["sentence"] = item
        predictions_data.append(entry)
    os.makedirs(os.path.dirname(args.save_predictions), exist_ok=True)
    with open(args.save_predictions, "w") as f:
        json.dump(predictions_data, f, indent=2)
    logging.info(f"Predictions saved to: {args.save_predictions}")


def print_and_save_results(
    output_data: Dict,
    results: Dict,
    classifier: nn.Module,
    args,
    output_filename: str,
    save_pt: bool,
    save_output_file: bool,
) -> None:
    print("\n" + "=" * 60)
    print("FINAL CLASSIFICATION REPORT")
    print("=" * 60)
    print(results["final_classification_report"])
    print("=" * 60 + "\n")

    if save_output_file:
        output_path = os.path.join(
            "results", args.model_name, "classifiers", output_filename
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"Results saved to: {output_path}")

        if save_pt:
            pt_path = output_path.replace(".json", ".pt")
            torch.save(classifier.state_dict(), pt_path)


# ===========================================================================
# Main functions (for both context and no-context modes)
# ===========================================================================


def run_analysis(
    args,
    train_items: list,
    eval_items: list,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    save_output_file: bool = False,
) -> float:
    """
    Unified orchestrator for both no-context and context classification paths.

    Mode-specific logic is isolated to embedding computation at the top.
    Everything after (classifier creation, training, output) is shared.
    """
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    # Resolve embedding model identity (shared for both modes)
    model_id, model_path = MODEL_NAME_TO_ID_PATH[args.model_name]
    embed_model_id = args.embed_model_id or model_id
    embed_model_path = args.embed_model_path or model_path
    embed_model_name = embed_model_id.split("/")[-1]

    # Load model + tokenizer ONCE
    logging.info(f"Loading embedding model: {embed_model_path}")
    model, tokenizer = load_model(
        embed_model_path,
        dtype=(torch.bfloat16 if "gpt" not in embed_model_id.lower() else None),
        device_map="auto",
    )
    model.model_name = embed_model_name

    if "random" in embed_model_id.lower():
        for _, param in model.named_parameters():
            param.data = torch.randn_like(param.data)
        for _, buff in model.named_buffers():
            buff.data = torch.randn_like(buff.data)

    model_num_layers = n_layers(model)
    model_hidden_dim = d_model(model)

    if args.no_context:
        all_items = train_items + eval_items
        n_train = len(train_items)

        all_embeddings = compute_nocontext_embeddings(
            sentences=all_items,
            model=model,
            tokenizer=tokenizer,
            classifier_type=args.classifier_type,
            specific_layer=args.specific_layer,
            batch_size=args.batch_size,
        )
        train_embeddings = all_embeddings[:n_train]
        eval_embeddings = all_embeddings[n_train:]

        output_filename = f"classifier_noctx_{args.classifier_type}_embedder={embed_model_name}_seed={args.seed}{suffix}.json"
        save_pt = False
        extra_params = {
            "model_name": args.model_name,
            "embed_model_id": embed_model_id,
            "embed_model_path": embed_model_path,
        }

    else:  # context
        save_dir = os.path.join("results", args.model_name, "classifiers")
        (
            train_embeddings,
            eval_embeddings,
            train_labels,
            eval_labels,
            model_num_layers,
            model_hidden_dim,
        ) = compute_context_embeddings(
            train_items=train_items,
            eval_items=eval_items,
            train_labels=train_labels,
            eval_labels=eval_labels,
            model=model,
            tokenizer=tokenizer,
            classifier_type=args.classifier_type,
            specific_layer=args.specific_layer,
            data_model_name=args.model_name,
            embed_model_name=embed_model_name,
            num_model_layers=model_num_layers,
            model_hidden_dim=model_hidden_dim,
            save_dir=save_dir,
            load_embeddings=args.load_embeddings,
            save_embeddings=args.save_embeddings,
        )

        output_filename = f"classifier_ctx_{args.classifier_type}_embedder={embed_model_name}_seed={args.seed}{suffix}.json"
        save_pt = True

        extra_params = {
            "data_model_name": args.model_name,
            "embed_model_name": embed_model_name,
            "embed_model_path": embed_model_path,
            "num_model_layers": model_num_layers,
            "model_hidden_dim": model_hidden_dim,
        }

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    extra_params.update({"probe_hidden_dim": args.probe_hidden_dim})

    # --- Shared from here ---

    classifier = create_classifier(
        args.classifier_type,
        model_num_layers,
        model_hidden_dim,
        args.probe_hidden_dim,
        probe_mlp_layers=args.probe_mlp_layers,
    )
    logging.info(f"Classifier architecture:\n{classifier}")

    classifier, results, final_preds = train_classifier(
        classifier,
        train_embeddings,
        train_labels,
        eval_embeddings,
        eval_labels,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
    )

    save_predictions_if_requested(args, final_preds, eval_labels, eval_items)

    output_data = build_output_data(
        args,
        train_labels,
        eval_labels,
        train_embeddings,
        eval_embeddings,
        results,
        extra_params,
    )

    print_and_save_results(
        output_data,
        results,
        classifier,
        args,
        output_filename,
        save_pt,
        save_output_file,
    )

    return results["best_eval_accuracy"]


def main():
    args = parse_args()
    set_deterministic(args.seed)

    entry_labels, step_results = load_step_removability_labels(
        args.model_name,
        args.sufficiency_threshold,
        args.obtainability_threshold,
        dataset=args.dataset,
    )

    removable, nonremovable = get_removable_nonremovable_lists(
        entry_labels, step_results, context=args.context
    )

    embed_model_name = None
    if args.context:
        embed_model_id = args.embed_model_id or args.model_name
        embed_model_name = embed_model_id.split("/")[-1]

    print_statistics(
        args.model_name, len(removable), len(nonremovable), embed_model_name
    )

    if not removable or not nonremovable:
        logging.error(
            "Not enough data for training. Need both removable and non-removable sentences."
        )
        sys.exit(1)

    print_samples(removable, nonremovable, n_samples=3, context=args.context)

    if args.only_statistics:
        logging.info("Only statistics requested. Exiting.")
        sys.exit(0)

    if args.eval_json:
        # The eval dataset is chosen as the entries in the given json,
        # other entries go to training.
        train_entry_labels, eval_entry_labels = split_labels_by_eval_json(
            entry_labels, args.eval_json
        )
        train_rem, train_nonrem = get_removable_nonremovable_lists(
            train_entry_labels, step_results, context=args.context
        )
        eval_rem, eval_nonrem = get_removable_nonremovable_lists(
            eval_entry_labels, step_results, context=args.context
        )
        if args.max_samples_per_class:
            set_deterministic(args.seed)
            n_per = min(len(train_rem), len(train_nonrem), args.max_samples_per_class)
            train_rem = random.sample(train_rem, k=n_per)
            train_nonrem = random.sample(train_nonrem, k=n_per)
        train_items = train_rem + train_nonrem
        train_labels = np.array([1] * len(train_rem) + [0] * len(train_nonrem))
        eval_items = eval_rem + eval_nonrem
        eval_labels = np.array([1] * len(eval_rem) + [0] * len(eval_nonrem))
        logging.info(
            f"Eval JSON split — train: {len(train_items)} "
            f"({len(train_rem)} rem, {len(train_nonrem)} nonrem), "
            f"eval: {len(eval_items)} "
            f"({len(eval_rem)} rem, {len(eval_nonrem)} nonrem)"
        )
    else:
        # Randomly split the data into train and eval sets, balancing classes and
        # sampling max_samples_per_class if set.
        train_items, eval_items, train_labels, eval_labels = balance_and_split(
            removable,
            nonremovable,
            args.test_size,
            args.seed,
            args.max_samples_per_class,
        )

    best_eval_acc = run_analysis(
        args,
        train_items,
        eval_items,
        train_labels,
        eval_labels,
        args.save_output_file,
    )

    # DON'T REMOVE: This is used to extract the best eval acc in run_classifier_experiments,
    # which calls this script as a subprocess.
    print(f"BEST_EVAL_ACC={best_eval_acc}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
