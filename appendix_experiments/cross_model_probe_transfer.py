"""
Cross-model removability probe transfer.

For every ordered pair (A, B) of distinct models in MODEL_NAME_TO_ID_PATH:

  1. Train an activation-based removability probe P_A on model A:
        A's reasoning-step activations (penultimate hidden layer, averaged
        over step tokens) -> binary removability label (1 = removable).
     Labels come from `extract_removable_nonremovable_entry_keys`
     (re-used via `load_step_removability_labels` from train_removability_classifier).

  2. Load the previously-trained B->A linear mapping from
     `train_cross_model_activation_mapping.py`.

  3. Take B's reasoning-step activations (from B, in B's space), pass them
     through the B->A mapping to project into A's activation space, then run
     P_A on the projected activations.

  4. Compare these cross-model predictions to B's ground-truth removability
     labels and report accuracy / F1 / classification report.

Outputs are saved under:
    results/{A}/cross-model-analysis/{DS}_probe_transfer_from={B}_seed={seed}.json
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from consts import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_NAME_TO_ID_PATH,
    SUFFICIENCY_THRESHOLD,
    OBTAINABILITY_THRESHOLD,
    TEST_SIZE,
    WEIGHT_DECAY,
)
from general_utils import set_deterministic, load_model, d_model
from train_removability_classifier import (
    SimpleMLP,
    get_removable_nonremovable_lists,
    load_step_removability_labels,
    train_classifier,
)
from train_cross_model_activation_mapping import compute_penult_step_avg

DEFAULT_PROBE_HIDDEN_DIM = 128
DEFAULT_PROBE_MLP_LAYERS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-model probe transfer: train P_A on A's activations, "
        "then evaluate it on B's activations via the B->A mapping."
    )
    parser.add_argument("--dataset", type=str, default="harp-standard")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument(
        "--probe-hidden-dim", type=int, default=DEFAULT_PROBE_HIDDEN_DIM
    )
    parser.add_argument(
        "--probe-mlp-layers", type=int, default=DEFAULT_PROBE_MLP_LAYERS
    )
    parser.add_argument(
        "--mapping-suffix",
        type=str,
        default="",
        help="Suffix used when saving mappings (must match train_cross_model_activation_mapping.py).",
    )
    return parser.parse_args()


# ===========================================================================
# Labeled step item loading + balancing
# ===========================================================================


def build_labeled_items(
    model_name: str,
    suff_threshold: float,
    obtn_threshold: float,
    dataset: str,
    seed: int,
    max_per_class: Optional[int] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Returns a (items, labels) pair drawn from a model's removability ground
    truth, with classes balanced and (optionally) downsampled.
    """
    entry_labels, step_results = load_step_removability_labels(
        model_name, suff_threshold, obtn_threshold, dataset=dataset
    )
    removable, nonremovable = get_removable_nonremovable_lists(
        entry_labels, step_results, context=True
    )
    n_per = min(len(removable), len(nonremovable))
    if max_per_class is not None:
        n_per = min(n_per, max_per_class)
    if n_per < 10:
        return [], np.array([])

    rng = random.Random(seed)
    removable = rng.sample(removable, k=n_per)
    nonremovable = rng.sample(nonremovable, k=n_per)
    items = removable + nonremovable
    labels = np.array([1] * len(removable) + [0] * len(nonremovable))
    return items, labels


# ===========================================================================
# Helpers
# ===========================================================================


def _load_lm(model_name: str):
    model_id, model_path = MODEL_NAME_TO_ID_PATH[model_name]
    logging.info(f"Loading model {model_name} from {model_path}")
    model, tokenizer = load_model(
        model_path,
        dtype=(torch.bfloat16 if "gpt" not in model_id.lower() else None),
        device_map="auto",
    )
    return model, tokenizer


def _filter_valid(
    items: List[Dict], labels: np.ndarray, acts: List[Optional[torch.Tensor]]
) -> Tuple[List[Dict], np.ndarray, List[torch.Tensor]]:
    valid = [i for i, a in enumerate(acts) if a is not None]
    return (
        [items[i] for i in valid],
        labels[valid] if len(labels) else labels,
        [acts[i] for i in valid],
    )


# ===========================================================================
# Probe evaluation
# ===========================================================================


@torch.no_grad()
def predict_with_probe(
    probe: nn.Module, X: List[torch.Tensor], batch_size: int = 64
) -> np.ndarray:
    device = next(probe.parameters()).device
    probe.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        batch = torch.stack([x.float() for x in X[i : i + batch_size]]).to(device)
        logits = probe(batch)
        preds.extend((torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten())
    return np.array(preds)


# ===========================================================================
# Main
# ===========================================================================


def _compute_model_activations(
    model_name: str, items: List[Dict], max_prompt_tokens: int
) -> Tuple[List[Optional[torch.Tensor]], int]:
    """Load a model, compute averaged penultimate activations for items, then unload."""
    model, tokenizer = _load_lm(model_name)
    d = d_model(model)
    acts = compute_penult_step_avg(
        items,
        model,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        desc=f"{model_name} activations",
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return acts, d


def process_pair(tgt_a: str, src_b: str, args) -> None:
    """
    Run the full pipeline for a single (A, B) pair:
      1. Build A's labeled items + load A + compute A activations + train P_A.
      2. Build B's labeled items + load B + compute B activations.
      3. Load B->A mapping, project B activations, run P_A, save evaluation.
    """
    logging.info(f"\n{'='*60}\nProcessing pair: {src_b} -> {tgt_a}\n{'='*60}")

    out_dir = os.path.join("results", tgt_a, "cross-model-analysis")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(
        out_dir, f"{args.dataset}_probe_transfer_from={src_b}_seed={args.seed}.json"
    )

    if os.path.exists(json_path):
        logging.info(f"Output already exists at {json_path}; skipping pair")
        return

    # --- 1. A side: items, activations, probe training ---
    items_a, labels_a = build_labeled_items(
        tgt_a,
        args.sufficiency_threshold,
        args.obtainability_threshold,
        args.dataset,
        args.seed,
        max_per_class=args.max_samples_per_class,
    )
    logging.info(f"  {tgt_a}: {len(items_a)} labeled step items")
    if len(items_a) < 20:
        logging.warning(f"Too few labeled items for {tgt_a}; skipping pair")
        return

    own_acts_a, d_tgt = _compute_model_activations(
        tgt_a, items_a, args.max_prompt_tokens
    )
    items_a, labels_a, acts_a = _filter_valid(items_a, labels_a, own_acts_a)
    if len(items_a) < 20:
        logging.warning(f"Too few valid {tgt_a} activations; skipping pair")
        return

    idx = list(range(len(items_a)))
    train_idx, eval_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=labels_a
    )
    X_train = [acts_a[i] for i in train_idx]
    y_train = labels_a[train_idx]
    X_eval = [acts_a[i] for i in eval_idx]
    y_eval = labels_a[eval_idx]

    logging.info(
        f"Training probe P_{tgt_a} on {len(X_train)} train / {len(X_eval)} eval (d={d_tgt})"
    )
    probe = SimpleMLP(
        input_dim=d_tgt,
        hidden_dim=args.probe_hidden_dim,
        num_layers=args.probe_mlp_layers,
    )
    probe, probe_results, _ = train_classifier(
        probe,
        X_train,
        y_train,
        X_eval,
        y_eval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    probe_train_summary = {
        "best_eval_accuracy": probe_results["best_eval_accuracy"],
        "classification_report": probe_results["final_classification_report"],
        "n_train": len(X_train),
        "n_eval": len(X_eval),
    }
    del acts_a, own_acts_a, X_train, X_eval
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. B side: items + activations ---
    items_b, labels_b = build_labeled_items(
        src_b,
        args.sufficiency_threshold,
        args.obtainability_threshold,
        args.dataset,
        args.seed,
        max_per_class=args.max_samples_per_class,
    )
    logging.info(f"  {src_b}: {len(items_b)} labeled step items")
    if len(items_b) < 10:
        logging.warning(f"Too few labeled items for {src_b}; skipping pair")
        return

    own_acts_b, d_src = _compute_model_activations(
        src_b, items_b, args.max_prompt_tokens
    )
    items_b, labels_b, acts_b = _filter_valid(items_b, labels_b, own_acts_b)
    if len(items_b) < 10:
        logging.warning(f"Too few valid {src_b} activations; skipping pair")
        return

    # --- 3. Load mapping, project, apply probe ---
    mapping_suffix = f"_{args.mapping_suffix}" if args.mapping_suffix else ""
    mapping_path = os.path.join(
        "results",
        tgt_a,
        "cross-model-analysis",
        f"{src_b}_to_{tgt_a}_seed={args.seed}{mapping_suffix}.pt",
    )
    if not os.path.exists(mapping_path):
        logging.warning(
            f"Missing mapping {mapping_path}; skipping ({src_b} -> {tgt_a})"
        )
        return

    device = next(probe.parameters()).device
    mapping = nn.Linear(d_src, d_tgt).to(device)
    mapping.load_state_dict(torch.load(mapping_path, map_location=device))
    mapping.eval()

    with torch.no_grad():
        src_stack = torch.stack([a.float() for a in acts_b]).to(device)
        projected = mapping(src_stack)
    projected_list = [p.cpu() for p in projected]

    preds = predict_with_probe(probe, projected_list, batch_size=args.batch_size)

    acc = float(accuracy_score(labels_b, preds))
    _, _, f1, _ = precision_recall_fscore_support(
        labels_b, preds, average="binary", zero_division=0
    )
    report = classification_report(
        labels_b, preds, target_names=["non-removable", "removable"], zero_division=0
    )
    logging.info(
        f"Transfer {src_b} -> {tgt_a}: acc={acc:.4f}, F1={f1:.4f} on {len(labels_b)} items"
    )

    output = {
        "parameters": {
            "target_model": tgt_a,
            "source_model": src_b,
            "dataset": args.dataset,
            "seed": args.seed,
            "sufficiency_threshold": args.sufficiency_threshold,
            "obtainability_threshold": args.obtainability_threshold,
            "probe_hidden_dim": args.probe_hidden_dim,
            "probe_mlp_layers": args.probe_mlp_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_samples_per_class": args.max_samples_per_class,
            "d_source": d_src,
            "d_target": d_tgt,
            "layer": "penultimate (hidden_states[-2])",
            "aggregation": "mean over step tokens",
            "mapping_path": mapping_path,
        },
        "probe_in_domain_training": probe_train_summary,
        "transfer_evaluation": {
            "n_eval": int(len(labels_b)),
            "n_removable": int((labels_b == 1).sum()),
            "n_nonremovable": int((labels_b == 0).sum()),
            "accuracy": acc,
            "f1": float(f1),
            "classification_report": report,
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Saved {json_path}")

    del mapping, projected, src_stack, probe, acts_b, own_acts_b
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    set_deterministic(args.seed)

    model_names = list(MODEL_NAME_TO_ID_PATH.keys())
    logging.info(f"Models: {model_names}")

    for tgt_a in model_names:
        for src_b in model_names:
            if tgt_a == src_b:
                continue
            process_pair(tgt_a, src_b, args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
