"""
Token-based removability classification baselines using two approaches:

1. Bag-of-words (TF-IDF features + MLP classifier)
2. Pre-trained sentence transformer (all-MiniLM-L6-v2 embeddings + MLP classifier)

Uses the same data loading and train/eval split logic as train_removability_classifier.py.

Output: results/{model_name}/{dataset}_token_based_removability_seed={seed}.json
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from loading_utils import (
    load_attribution_pruning_results,
    load_generations,
    load_single_step_results,
)
from general_utils import set_deterministic
from consts import (
    BATCH_SIZE,
    EPOCHS,
    SUFFICIENCY_THRESHOLD,
    OBTAINABILITY_THRESHOLD,
    TEST_SIZE,
    WEIGHT_DECAY,
)
from removability_utils import extract_removable_nonremovable_entry_keys
from train_removability_classifier import (
    SimpleMLP,
    balance_and_split,
    get_removable_nonremovable_lists,
)


def load_all_data(
    model_name: str,
    suff_threshold: float,
    obtn_threshold: float,
    dataset: str = "harp-standard",
) -> Tuple[Dict[str, Dict[int, bool]], Dict]:
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
        f"  Loaded: {n_removable} removable, {n_nonremovable} non-removable "
        f"across {len(entry_labels)} entries"
    )
    return entry_labels, step_results


def _train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    input_dim: int,
    seed: int,
    hidden_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a SimpleMLP binary classifier and return (train_preds, eval_preds)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic(seed)

    mlp = SimpleMLP(input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        mlp.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(mlp(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        train_preds = (
            (torch.sigmoid(mlp(X_train_t.to(device)).squeeze(-1)) > 0.5)
            .float()
            .cpu()
            .numpy()
        )
        eval_preds = (
            (torch.sigmoid(mlp(X_eval_t).squeeze(-1)) > 0.5).float().cpu().numpy()
        )
    return train_preds, eval_preds


# ---------------------------------------------------------------------------
# Baseline 1: Bag-of-Words (TF-IDF) + MLP
# ---------------------------------------------------------------------------


def run_bow_classifier(
    train_sentences: List[str],
    train_labels: np.ndarray,
    eval_sentences: List[str],
    eval_labels: np.ndarray,
    seed: int,
    epochs: int = EPOCHS,
) -> Dict:
    logging.info("=== Method 1: Bag-of-Words (TF-IDF + MLP) ===")

    vectorizer = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 5),
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_sentences).toarray()
    X_eval = vectorizer.transform(eval_sentences).toarray()

    input_dim = X_train.shape[1]
    train_preds, eval_preds = _train_mlp_classifier(
        X_train, train_labels, X_eval, input_dim, seed, epochs=epochs
    )

    train_acc = accuracy_score(train_labels, train_preds)
    eval_acc = accuracy_score(eval_labels, eval_preds)
    eval_prec, eval_rec, eval_f1, _ = precision_recall_fscore_support(
        eval_labels, eval_preds, average="binary"
    )

    report = classification_report(
        eval_labels, eval_preds, target_names=["non-removable", "removable"]
    )

    print("\n" + "=" * 60)
    print("BOW (TF-IDF + MLP) RESULTS")
    print("=" * 60)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Eval accuracy:  {eval_acc:.4f}")
    print(f"Eval F1:        {eval_f1:.4f}")
    print(report)
    print("=" * 60 + "\n")

    return {
        "method": "bow_tfidf_mlp",
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "eval_precision": eval_prec,
        "eval_recall": eval_rec,
        "eval_f1": eval_f1,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Baseline 2: Sentence Transformer embeddings + MLP
# ---------------------------------------------------------------------------


def run_sentence_transformer_classifier(
    train_sentences: List[str],
    train_labels: np.ndarray,
    eval_sentences: List[str],
    eval_labels: np.ndarray,
    seed: int,
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    epochs: int = EPOCHS,
) -> Dict:
    logging.info(f"=== Method 2: Sentence Transformer ({st_model_name}) + MLP ===")

    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(st_model_name)

    logging.info(f"Encoding {len(train_sentences)} train sentences...")
    X_train = st_model.encode(
        train_sentences, batch_size=batch_size, show_progress_bar=True
    )
    logging.info(f"Encoding {len(eval_sentences)} eval sentences...")
    X_eval = st_model.encode(
        eval_sentences, batch_size=batch_size, show_progress_bar=True
    )

    del st_model
    torch.cuda.empty_cache()

    input_dim = X_train.shape[1]
    train_preds, eval_preds = _train_mlp_classifier(
        X_train, train_labels, X_eval, input_dim, seed, epochs=epochs
    )

    train_acc = accuracy_score(train_labels, train_preds)
    eval_acc = accuracy_score(eval_labels, eval_preds)
    eval_prec, eval_rec, eval_f1, _ = precision_recall_fscore_support(
        eval_labels, eval_preds, average="binary"
    )

    report = classification_report(
        eval_labels, eval_preds, target_names=["non-removable", "removable"]
    )

    print("\n" + "=" * 60)
    print("SENTENCE TRANSFORMER + MLP RESULTS")
    print("=" * 60)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Eval accuracy:  {eval_acc:.4f}")
    print(f"Eval F1:        {eval_f1:.4f}")
    print(report)
    print("=" * 60 + "\n")

    return {
        "method": f"sentence_transformer_mlp_{st_model_name.split('/')[-1]}",
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "eval_precision": eval_prec,
        "eval_recall": eval_rec,
        "eval_f1": eval_f1,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Token-based removability classification baselines (BoW, SentenceTransformer)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Short model name for results directory (e.g., 'gpt-oss-20b')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="harp-standard",
        help="Dataset identifier (default: harp-standard)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["bow", "sentence_transformer"],
        choices=["bow", "sentence_transformer"],
        help="Which baselines to run (default: both)",
    )
    parser.add_argument(
        "--eval-json",
        type=str,
        default=None,
        help="Path to an eval JSON file (same format as removability_pred_llm_as_a_judge output). "
        "Entries in this file become the eval set, rest become train set.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output filenames",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--sufficiency-threshold", type=float, default=SUFFICIENCY_THRESHOLD
    )
    parser.add_argument(
        "--obtainability-threshold", type=float, default=OBTAINABILITY_THRESHOLD
    )
    parser.add_argument("--max-samples-per-class", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Eval-JSON split
# ---------------------------------------------------------------------------


def split_labels_by_eval_json(
    entry_labels: Dict[str, Dict[int, bool]],
    eval_json_path: str,
) -> Tuple[Dict[str, Dict[int, bool]], Dict[str, Dict[int, bool]]]:
    """Split labels using an eval JSON, taking eval labels from the JSON's
    ``ground_truth_is_removable`` field rather than re-deriving them from
    ``entry_labels``. This preserves the balance of the eval JSON even when
    thresholds or upstream data drift.
    """
    with open(eval_json_path) as f:
        eval_data, _ = json.load(f)

    eval_pairs = set()
    eval_entry_labels: Dict[str, Dict[int, bool]] = {}
    for entry_key, sentences in eval_data.items():
        for sent_idx_str, sent_info in sentences.items():
            sent_idx = int(sent_idx_str)
            eval_pairs.add((entry_key, sent_idx))
            is_removable = sent_info["ground_truth_is_removable"]
            eval_entry_labels.setdefault(entry_key, {})[sent_idx] = is_removable

    train_entry_labels: Dict[str, Dict[int, bool]] = {}
    for entry_key, labels in entry_labels.items():
        for sent_idx, is_removable in labels.items():
            if (entry_key, sent_idx) not in eval_pairs:
                train_entry_labels.setdefault(entry_key, {})[sent_idx] = is_removable

    n_eval = sum(len(v) for v in eval_entry_labels.values())
    n_eval_json = len(eval_pairs)
    n_train = sum(len(v) for v in train_entry_labels.values())
    logging.info(
        f"Eval JSON split: {n_eval}/{n_eval_json} eval pairs matched, "
        f"{n_train} train pairs"
    )

    return train_entry_labels, eval_entry_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    set_deterministic(args.seed)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_filename = (
        f"{args.dataset}_token_based_removability_seed={args.seed}{suffix}.json"
    )
    output_path = os.path.join("results", args.model_name, output_filename)
    if os.path.exists(output_path):
        logging.warning(f"Output file {output_path} already exists")
        return

    entry_labels, step_results = load_all_data(
        args.model_name,
        args.sufficiency_threshold,
        args.obtainability_threshold,
        dataset=args.dataset,
    )

    removable, nonremovable = get_removable_nonremovable_lists(
        entry_labels, step_results, context=False
    )

    logging.info(
        f"Total: {len(removable)} removable, {len(nonremovable)} non-removable"
    )

    if len(removable) == 0 or len(nonremovable) == 0:
        logging.error("Need both removable and non-removable sentences.")
        sys.exit(1)

    if args.eval_json:
        train_entry_labels, eval_entry_labels = split_labels_by_eval_json(
            entry_labels, args.eval_json
        )
        train_rem, train_nonrem = get_removable_nonremovable_lists(
            train_entry_labels, step_results, context=False
        )
        eval_rem, eval_nonrem = get_removable_nonremovable_lists(
            eval_entry_labels, step_results, context=False
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
            f"Eval JSON split -- train: {len(train_items)} "
            f"({len(train_rem)} rem, {len(train_nonrem)} nonrem), "
            f"eval: {len(eval_items)} "
            f"({len(eval_rem)} rem, {len(eval_nonrem)} nonrem)"
        )
    else:
        train_items, eval_items, train_labels, eval_labels = balance_and_split(
            removable,
            nonremovable,
            args.test_size,
            args.seed,
            args.max_samples_per_class,
        )

    all_results = {}

    if "bow" in args.baselines:
        all_results["bow"] = run_bow_classifier(
            train_items,
            train_labels,
            eval_items,
            eval_labels,
            args.seed,
            epochs=args.epochs,
        )

    if "sentence_transformer" in args.baselines:
        all_results["sentence_transformer"] = run_sentence_transformer_classifier(
            train_items,
            train_labels,
            eval_items,
            eval_labels,
            args.seed,
            epochs=args.epochs,
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method_name, result in all_results.items():
        print(
            f"  {method_name:25s}  "
            f"eval_acc={result['eval_accuracy']:.4f}  "
            f"eval_f1={result['eval_f1']:.4f}"
        )
    print("=" * 60 + "\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "parameters": {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "baselines": args.baselines,
            "sufficiency_threshold": args.sufficiency_threshold,
            "obtainability_threshold": args.obtainability_threshold,
        },
        "data_statistics": {
            "n_removable": int(
                (np.concatenate([train_labels, eval_labels]) == 1).sum()
            ),
            "n_nonremovable": int(
                (np.concatenate([train_labels, eval_labels]) == 0).sum()
            ),
            "n_train": len(train_items),
            "n_eval": len(eval_items),
        },
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
            for k, v in all_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
