"""
Removability Probe Analysis Script.

Analyzes what removability classifier probes learn from model activations.
"""

import argparse
import gc
import json
import logging
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from consts import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_NAME_TO_ID_PATH,
    SEEDS,
    SUFFICIENCY_THRESHOLD,
    OBTAINABILITY_THRESHOLD,
    TEST_SIZE,
    WEIGHT_DECAY,
)
from general_utils import set_deterministic, load_model, n_layers, d_model, save_results
from train_removability_classifier import (
    load_step_removability_labels,
    get_removable_nonremovable_lists,
    balance_and_split,
    _cache_prompt_activations,
    LayerwiseMLP,
    train_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Token positions in each reasoning step to extract activations and train a probe on these activations.
# "average" is the standard approach, averaging activations across all tokens in the reasoning step.
TOKEN_POSITIONS = ["average", "first", "middle", "last"]


# Surface-level sentence categories: each maps to a list of marker phrases/patterns.
# These are matched case-insensitively against sentence text.
SENTENCE_CATEGORIES = {
    "filler": [
        "let me think",
        "let's think",
        "lets think",
        "let me consider",
        "let's consider",
        "let me see",
        "let's see",
        "let me ensure",
        "let's ensure",
        "lets ensure",
        "hmm",
        "okay",
        "alright",
        "so basically",
        "in other words",
    ],
    "verification": [
        "wait",
        "double check",
        "double-check",
        "recompute",
        "recalculate",
        "verify",
        "let me check",
        "let's check",
        "checking",
        "is this correct",
        "does this make sense",
        "let me re",
        "actually, let me",
        "hold on",
    ],
    "plan": [
        "we need",
        "first, i",
        "step 1",
        "step 2",
        "my approach",
        "i'll start by",
        "the plan is",
        "i need to",
        "i should",
        "let me start",
        "to solve this",
        "the strategy",
        "i will",
        "my strategy",
    ],
    "computation": [
        "calculating",
        "computing",
        "plugging in",
        "substituting",
        "evaluating",
        "simplifying",
        "expanding",
        "factoring",
        "multiplying",
        "dividing",
        "adding",
        "subtracting",
        "integrating",
        "differentiating",
        "solving for",
    ],
    "fact_retrieval": [
        "we know that",
        "recall that",
        "remember that",
        "by definition",
        "by the formula",
        "using the formula",
        "the formula for",
        "it is known",
        "a known result",
        "from the theorem",
        "by theorem",
        "since we know",
    ],
}

# Regex to count numbers and arithmetic tokens in a sentence
ARITHMETIC_PATTERN = re.compile(r"[+\-*/^=<>≤≥≠±]")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Removability probe analysis: per-layer/position training + surface feature correlation"
    )

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="harp-standard",
        help="Dataset identifier (default: harp-standard)",
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


def compute_preresidual(activations: torch.Tensor) -> torch.Tensor:
    """
    Convert residual-stream activations to pre-residual (per-layer contributions).
    Used to train probes on true per-layer outputs.

    Input: (L, D) or (N, L, D) where layer 0 is the embedding layer output.
    Output: same shape, where layer 0 is unchanged and layer i (i>0) = layer[i] - layer[i-1].
    """
    if activations.dim() == 2:
        # (L, D)
        return torch.cat(
            [activations[:1, :], activations[1:, :] - activations[:-1, :]], dim=0
        )
    elif activations.dim() == 3:
        # (N, L, D)
        return torch.cat(
            [activations[:, :1, :], activations[:, 1:, :] - activations[:, :-1, :]],
            dim=1,
        )
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {activations.dim()}D")


def extract_position_activations(
    full_activations: List[torch.Tensor],
    position: str,
) -> List[torch.Tensor]:
    """
    From per-sentence full activation tensors (each shape (s_i, L, D)),
    extract activations for a specific token position.

    Args:
        full_activations: List of (s_i, L, D) tensors (one per sentence).
        position: One of 'first', 'middle', 'last', 'average'.

    Returns:
        List of (L, D) tensors (one per sentence).
    """
    result = []
    for act in full_activations:
        # act shape: (s_i, L, D) where s_i = number of tokens in sentence
        s = act.shape[0]
        if s == 0:
            # Fallback: zeros
            result.append(torch.zeros(act.shape[1], act.shape[2]))
            continue

        if position == "first":
            result.append(act[0])  # (L, D)
        elif position == "middle":
            result.append(act[s // 2])  # (L, D)
        elif position == "last":
            result.append(act[s - 1])  # (L, D)
        elif position == "average":
            result.append(act.mean(dim=0))  # (L, D)
        else:
            raise ValueError(f"Unknown position: {position}")
    return result


def get_step_activations(
    data_items: List[Dict],
    activation_cache: Dict[str, torch.Tensor],
) -> List[torch.Tensor]:
    """
    Extract raw per-sentence activation tensors (s_i, L, D) from the activation cache.
    Skips the embedding layer (index 0) for consistency with existing code.

    Returns:
        List of (s_i, L, D) tensors where L = num_layers (excluding embedding).
    """
    activations = []
    for item in tqdm(data_items, desc="Extracting raw activations"):
        full_prompt = item["full_prompt"]
        token_start, token_end = item["token_borders"]
        hidden_states = activation_cache[
            full_prompt
        ]  # (L+1, seq_len, D) - also includes embedding layer at hs[0, :, :]
        act = hidden_states[:, token_start:token_end, :].permute(1, 0, 2)  # (s_i, L, D)
        activations.append(act)
    return activations


# ---------------------------------------------------------------------------
# Analysis 1: Per-layer, per-position training
# ---------------------------------------------------------------------------


def run_per_layer_per_position(
    train_activations: List[torch.Tensor],
    eval_activations: List[torch.Tensor],
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    num_layers: int,
    results: dict,
    metadata,
    output_path: str,
    seed: int,
):
    """
    Train probes for each (layer, token_position) combination.
    Updates results dict in-place and saves after each probe.
    """
    results_key = f"per_layer_position_seed{seed}"
    if results_key not in results:
        results[results_key] = {}

    for position in TOKEN_POSITIONS:
        logging.info(f"  Token position: {position}")

        # Extract activations for this position -> list of (L, D) tensors
        train_pos = extract_position_activations(train_activations, position)
        eval_pos = extract_position_activations(eval_activations, position)

        # Stack to (N, L, D)
        X_train_all = torch.stack(train_pos)
        X_eval_all = torch.stack(eval_pos)

        # Compute pre-residual
        X_train_all = compute_preresidual(X_train_all)
        X_eval_all = compute_preresidual(X_eval_all)

        for layer_idx in tqdm(range(num_layers), desc=f"Per-layer probes ({position})"):
            key = (
                f"layer{layer_idx - 1}_{position}"  # -1 to account for embedding layer
            )
            if key in results[results_key]:
                logging.info(f"    Skipping {key} (already computed)")
                continue

            X_train_layer = X_train_all[:, layer_idx, :]  # (N, D)
            X_eval_layer = X_eval_all[:, layer_idx, :]  # (N, D)

            classifier = LayerwiseMLP(
                num_layers=1,
                model_dim=X_train_layer.shape[1],
                first_hidden=128,
            )

            _, probe_results, _ = train_classifier(
                classifier,
                list(X_train_layer.unsqueeze(1)),
                train_labels,
                list(X_eval_layer.unsqueeze(1)),
                eval_labels,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                weight_decay=WEIGHT_DECAY,
            )
            results[results_key][key] = probe_results
            save_results(results, metadata, output_path)

        # Also train on ALL layers (flattened) for this position
        all_layers_key = f"all_layers_{position}"
        if all_layers_key not in results[results_key]:
            logging.info(f"  Training all-layers probe for position={position}")

            classifier = LayerwiseMLP(
                num_layers=X_train_all.shape[1],
                model_dim=X_train_all.shape[2],
                first_hidden=128,
            )
            _, probe_results, _ = train_classifier(
                classifier,
                list(X_train_all),
                train_labels,
                list(X_eval_all),
                eval_labels,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                weight_decay=WEIGHT_DECAY,
            )
            results[results_key][all_layers_key] = probe_results
            save_results(results, metadata, output_path)


# ---------------------------------------------------------------------------
# Analysis 2: Base classifier + surface feature correlation
# ---------------------------------------------------------------------------


def extract_surface_features(
    items: List[Dict],
    processed_results: Dict,
) -> Dict[str, np.ndarray]:
    """
    Extract surface-level features for each sentence item.

    Returns dict mapping feature_name -> np.ndarray of shape (N,).
    """
    positions = []
    frac_positions = []
    token_lengths = []
    char_lengths = []
    n_numbers = []
    n_arithmetic = []
    category_counts = {cat: [] for cat in SENTENCE_CATEGORIES}

    for item in items:
        entry_key = item["entry_key"]
        sent_idx = item["sentence_idx"]
        token_start, token_end = item["token_borders"]
        sentence = item["sentence"]
        sentence_lower = sentence.lower()

        # Position features
        positions.append(sent_idx)
        post_removal = processed_results.get(entry_key, {}).get("post_removal", {})
        n_sents = len(post_removal.get("sentences", []))
        frac_positions.append(sent_idx / max(n_sents - 1, 1))

        # Length features
        token_lengths.append(token_end - token_start)
        char_lengths.append(len(sentence))

        # Number and arithmetic token counts
        n_numbers.append(len(NUMBER_PATTERN.findall(sentence)))
        n_arithmetic.append(len(ARITHMETIC_PATTERN.findall(sentence)))

        # Category features (binary: does sentence contain any marker from category?)
        for cat, markers in SENTENCE_CATEGORIES.items():
            has_cat = any(marker in sentence_lower for marker in markers)
            category_counts[cat].append(int(has_cat))

    features = {
        "position": np.array(positions, dtype=np.float64),
        "frac_position": np.array(frac_positions, dtype=np.float64),
        "token_length": np.array(token_lengths, dtype=np.float64),
        "char_length": np.array(char_lengths, dtype=np.float64),
        "n_numbers": np.array(n_numbers, dtype=np.float64),
        "n_arithmetic": np.array(n_arithmetic, dtype=np.float64),
    }
    for cat, counts in category_counts.items():
        features[f"cat_{cat}"] = np.array(counts, dtype=np.float64)

    return features


def train_base_classifier_and_get_predictions(
    train_activations: List[torch.Tensor],
    eval_activations: List[torch.Tensor],
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    num_model_layers: int,
    model_hidden_dim: int,
    first_hidden: int = 128,
) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """
    Train a LayerwiseMLP base classifier on averaged-token, all-layer, pre-residual activations.

    Returns:
        (best_eval_acc, eval_predictions, eval_logits_np, results_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Average tokens -> (L, D) per sentence
    train_avg = extract_position_activations(train_activations, "average")
    eval_avg = extract_position_activations(eval_activations, "average")

    # Stack and compute pre-residual
    # skipping first layer which is the embedding layer activations
    X_train = compute_preresidual(torch.stack(train_avg))[:, 1:, :]  # (N, L, D)
    X_eval = compute_preresidual(torch.stack(eval_avg))[:, 1:, :]  # (N, L, D)

    # Create and train LayerwiseMLP
    classifier = LayerwiseMLP(
        num_layers=num_model_layers,
        model_dim=model_hidden_dim,
        first_hidden=first_hidden,
    )

    # Use the existing train_classifier
    train_embeddings = [X_train[i] for i in range(X_train.shape[0])]
    eval_embeddings = [X_eval[i] for i in range(X_eval.shape[0])]

    classifier, train_results, _ = train_classifier(
        classifier,
        train_embeddings,
        train_labels,
        eval_embeddings,
        eval_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Get eval predictions and logits
    classifier.eval()
    X_eval_t = X_eval.float().to(device)
    with torch.no_grad():
        eval_logits = classifier(X_eval_t.view(X_eval_t.size(0), -1))
        eval_probs = torch.sigmoid(eval_logits).cpu().numpy().flatten()
        eval_preds = (eval_probs > 0.5).astype(float)

    return train_results["best_eval_accuracy"], eval_preds, eval_probs, train_results


def train_surface_feature_classifier(
    train_items: List[Dict],
    eval_items: List[Dict],
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    processed_results: Dict,
) -> float:
    """
    Train a logistic regression on surface-level features only and return its eval accuracy.
    """
    train_features = extract_surface_features(train_items, processed_results)
    eval_features = extract_surface_features(eval_items, processed_results)

    sorted_keys = sorted(train_features.keys())
    train_matrix = np.column_stack([train_features[k] for k in sorted_keys])
    eval_matrix = np.column_stack([eval_features[k] for k in sorted_keys])

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_matrix)
    eval_matrix = scaler.transform(eval_matrix)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_matrix, train_labels)
    eval_preds = clf.predict(eval_matrix)
    return float(accuracy_score(eval_labels, eval_preds))


def compute_correlations(
    eval_features: Dict[str, np.ndarray],
    eval_labels: np.ndarray,
    eval_predictions: np.ndarray,
) -> dict:
    """
    Compute point-biserial correlations between:
    1. Ground truth labels and surface features
    2. Classifier predictions and surface features
    3. Classifier probabilities and surface features

    Returns a dict of correlation results.
    """
    correlations = {}

    for feat_name, feat_values in eval_features.items():
        feat_corr = {}

        # Correlation with ground truth labels
        try:
            r_label, p_label = pointbiserialr(eval_labels, feat_values)
            feat_corr["label_r"] = float(r_label)
            feat_corr["label_p"] = float(p_label)
        except Exception:
            feat_corr["label_r"] = None
            feat_corr["label_p"] = None

        # Correlation with classifier predictions
        try:
            r_pred, p_pred = pointbiserialr(eval_predictions, feat_values)
            feat_corr["pred_r"] = float(r_pred)
            feat_corr["pred_p"] = float(p_pred)
        except Exception:
            feat_corr["pred_r"] = None
            feat_corr["pred_p"] = None

        # Mean and std values per class (ground truth)
        feat_corr["mean_removable"] = (
            float(feat_values[eval_labels == 1].mean())
            if (eval_labels == 1).any()
            else None
        )
        feat_corr["mean_nonremovable"] = (
            float(feat_values[eval_labels == 0].mean())
            if (eval_labels == 0).any()
            else None
        )
        feat_corr["std_removable"] = (
            float(feat_values[eval_labels == 1].std())
            if (eval_labels == 1).any()
            else None
        )
        feat_corr["std_nonremovable"] = (
            float(feat_values[eval_labels == 0].std())
            if (eval_labels == 0).any()
            else None
        )
        correlations[feat_name] = feat_corr

    return correlations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    model_name = args.model_name
    model_id, model_path = MODEL_NAME_TO_ID_PATH[model_name]

    output_path = os.path.join(
        "results",
        model_name,
        "classifiers",
        f"{args.dataset}_probe_analysis.json",
    )
    metadata = {
        "model_name": model_name,
        "dataset": args.dataset,
        "suff_threshold": args.sufficiency_threshold,
        "obtn_threshold": args.obtainability_threshold,
        "seeds": SEEDS,
    }
    if os.path.exists(output_path):
        with open(output_path) as f:
            results, cached_metadata = json.load(f)
        logging.info(f"Loaded existing results from {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = {}

    # Load data (shared across seeds — labels are the same, only splits differ)
    logging.info("Loading data...")
    entry_labels, step_results = load_step_removability_labels(
        model_name,
        args.sufficiency_threshold,
        args.obtainability_threshold,
        dataset=args.dataset,
    )
    removable, nonremovable = get_removable_nonremovable_lists(
        entry_labels, step_results, context=True
    )
    n_rem = len(removable)
    n_nonrem = len(nonremovable)
    logging.info(f"Total: {n_rem} removable, {n_nonrem} non-removable sentences")
    results["data_statistics"] = {
        "n_removable": n_rem,
        "n_nonremovable": n_nonrem,
    }
    save_results(results, metadata, output_path)

    # Compute raw activations (shared across seeds)
    logging.info(f"Loading model {model_id} from {model_path}")
    model, tokenizer = load_model(
        model_path,
        dtype=(torch.bfloat16 if "gpt" not in model_id.lower() else None),
        device_map="auto",
    )
    model.model_name = model_name
    num_model_layers = n_layers(model)
    model_hidden_dim = d_model(model)

    # Cache activations for all unique prompts
    all_items = removable + nonremovable
    activation_cache = _cache_prompt_activations(all_items, model, tokenizer)
    del model, tokenizer

    # Extract per-step activations (s_i, L, D)
    logging.info("Extracting raw activations for all sentences...")
    all_raw_activations = get_step_activations(all_items, activation_cache)
    del activation_cache
    torch.cuda.empty_cache()
    gc.collect()

    # Split raw activations back to removable / nonremovable
    rem_activations = all_raw_activations[:n_rem]
    nonrem_activations = all_raw_activations[n_rem:]

    # Per-seed loop
    for seed in SEEDS:
        logging.info(f"\n{'='*60}")
        logging.info(f"  Seed: {seed}")
        logging.info(f"{'='*60}")

        set_deterministic(seed)

        # Balance and split
        train_items_acts, eval_items_acts, train_labels, eval_labels = (
            balance_and_split(
                list(zip(removable, rem_activations)),
                list(zip(nonremovable, nonrem_activations)),
                TEST_SIZE,
                seed,
            )
        )
        train_acts = [x[1] for x in train_items_acts]
        eval_acts = [x[1] for x in eval_items_acts]
        train_items = [x[0] for x in train_items_acts]
        eval_items = [x[0] for x in eval_items_acts]
        train_labels = np.array(train_labels)
        eval_labels = np.array(eval_labels)
        logging.info(f"Seed {seed}: {len(train_acts)} train, {len(eval_acts)} eval")

        # -------------------------------------------------------------------
        # Step 1: Per-layer, per-position training
        # -------------------------------------------------------------------
        logging.info(f"Step 1: Per-layer/per-position training (seed={seed})")
        run_per_layer_per_position(
            train_acts,
            eval_acts,
            train_labels,
            eval_labels,
            num_model_layers,
            results,
            metadata,
            output_path,
            seed=seed,
        )

        # -------------------------------------------------------------------
        # Step 2: Base classifier + correlation with surface features
        # -------------------------------------------------------------------
        logging.info(
            f"Step 2: Base classifier + surface feature correlation (seed={seed})"
        )

        old_corr_key = f"correlation_seed{seed}"
        if old_corr_key in results:
            del results[old_corr_key]
        corr_key = f"surface_correlactions_seed{seed}"
        if corr_key not in results:
            # Train base classifier
            best_acc, eval_preds, eval_probs, base_results = (
                train_base_classifier_and_get_predictions(
                    train_acts,
                    eval_acts,
                    train_labels,
                    eval_labels,
                    num_model_layers,
                    model_hidden_dim,
                )
            )

            results[corr_key] = {
                "base_classifier_accuracy": best_acc,
                "base_classifier_results": {
                    "best_eval_accuracy": base_results["best_eval_accuracy"],
                },
            }
            save_results(results, metadata, output_path)

            # Train classifier on surface features only
            surface_acc = train_surface_feature_classifier(
                train_items, eval_items, train_labels, eval_labels, step_results
            )
            results[corr_key]["surface_classifier_accuracy"] = surface_acc
            save_results(results, metadata, output_path)

            # Extract surface features for eval set
            eval_features = extract_surface_features(eval_items, step_results)

            # Compute correlations
            correlations = compute_correlations(
                eval_features,
                eval_labels,
                eval_preds,
            )

            results[corr_key]["correlations"] = correlations
            save_results(results, metadata, output_path)
        else:
            logging.info(f"  Skipping correlation (seed={seed}) — already computed")

    logging.info("Done!")


if __name__ == "__main__":
    main()
