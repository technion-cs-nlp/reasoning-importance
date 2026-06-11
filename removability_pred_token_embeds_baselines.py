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
from tqdm import tqdm

from loading_utils import (
    load_attribution_pruning_results,
    load_generations,
    load_single_step_results,
)
from general_utils import set_deterministic
from consts import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_NAME_TO_ID_PATH,
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

BASELINE_NAMES = [
    "bow",
    "sentence_transformer",
    "modernbert_step",
    "modernbert_context",
    "confidence",
]


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
# Baselines 3 & 4: ModernBERT fine-tuned (step-only / full-prefix)
# ---------------------------------------------------------------------------


def get_removable_nonremovable_with_prefix(
    entry_labels: Dict[str, Dict[int, bool]],
    step_results: Dict,
) -> Tuple[list, list]:
    """Like get_removable_nonremovable_lists(context=False), but each item is a
    dict {"sentence": str, "prefix_text": str} where ``prefix_text`` contains
    the CoT from sentence 0 up to and including ``sent_idx``."""
    removable, nonremovable = [], []
    for entry_key, labels in sorted(entry_labels.items(), key=lambda x: x[0]):
        processed_entry = step_results.get(entry_key)
        if processed_entry is None:
            continue
        sentences = processed_entry.get("post_removal", {}).get("sentences", [])
        for sent_idx, is_removable in labels.items():
            if sent_idx >= len(sentences):
                continue
            sentence = sentences[sent_idx].strip()
            if not sentence:
                continue
            prefix_text = " ".join(s.strip() for s in sentences[: sent_idx + 1])
            item = {"sentence": sentence, "prefix_text": prefix_text}
            if is_removable:
                removable.append(item)
            else:
                nonremovable.append(item)
    return removable, nonremovable


def _train_modernbert_classifier(
    train_texts: List[str],
    train_labels: np.ndarray,
    eval_texts: List[str],
    eval_labels: np.ndarray,
    seed: int,
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    drop_overflow: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fine-tune ModernBERT for binary classification on ``train_texts``.

    If ``drop_overflow`` is True, examples whose tokenized length exceeds the
    model's max context are dropped (no truncation). Otherwise inputs are
    truncated to the model max length.

    Returns (train_preds, eval_preds, kept_train_labels, kept_eval_labels).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = tokenizer.model_max_length

    if drop_overflow:

        def _keep(texts):
            lens = [
                len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts
            ]
            return [i for i, L in enumerate(lens) if L <= model_max]

        keep_train = _keep(train_texts)
        keep_eval = _keep(eval_texts)
        logging.info(
            f"ModernBERT drop-overflow: kept {len(keep_train)}/{len(train_texts)} "
            f"train, {len(keep_eval)}/{len(eval_texts)} eval (model_max={model_max})"
        )
        train_texts = [train_texts[i] for i in keep_train]
        eval_texts = [eval_texts[i] for i in keep_eval]
        train_labels = np.asarray(train_labels)[keep_train]
        eval_labels = np.asarray(eval_labels)[keep_eval]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    n_pos = int(np.sum(train_labels == 1))
    n_neg = int(np.sum(train_labels == 0))
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def _tokenize(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=not drop_overflow,
            max_length=model_max,
            return_tensors="pt",
        )

    def _predict(texts):
        if not texts:
            return np.array([], dtype=np.int64)
        preds = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = {k: v.to(device) for k, v in _tokenize(batch).items()}
                logits = model(**enc).logits
                preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    def _eval_loss_and_acc(texts, labels):
        if not texts:
            return float("nan"), float("nan")
        was_training = model.training
        model.eval()
        total_loss, total_n, correct = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_y = torch.tensor(
                    [int(labels[i]) for i in range(start, start + len(batch_texts))],
                    dtype=torch.long,
                ).to(device)
                enc = {k: v.to(device) for k, v in _tokenize(batch_texts).items()}
                logits = model(**enc).logits
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_texts)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total_n += len(batch_texts)
        if was_training:
            model.train()
        return total_loss / max(total_n, 1), correct / max(total_n, 1)

    indices = list(range(len(train_texts)))
    for epoch in range(epochs):
        random.shuffle(indices)
        model.train()
        running_loss = 0.0
        for start in tqdm(
            range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            batch_idx = indices[start : start + batch_size]
            batch_texts = [train_texts[i] for i in batch_idx]
            batch_y = torch.tensor(
                [int(train_labels[i]) for i in batch_idx], dtype=torch.long
            ).to(device)
            enc = {k: v.to(device) for k, v in _tokenize(batch_texts).items()}
            optimizer.zero_grad()
            logits = model(**enc).logits
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_idx)
        eval_loss, eval_acc = _eval_loss_and_acc(eval_texts, eval_labels)
        logging.info(
            f"  epoch {epoch + 1}/{epochs} - "
            f"train_loss={running_loss / max(len(indices), 1):.4f}  "
            f"eval_loss={eval_loss:.4f}  eval_acc={eval_acc:.4f}"
        )

    model.eval()

    train_preds = _predict(train_texts)
    eval_preds = _predict(eval_texts)

    del model
    torch.cuda.empty_cache()

    return train_preds, eval_preds, train_labels, eval_labels


def _summarize_modernbert_results(
    method_name: str,
    pretty_name: str,
    train_labels: np.ndarray,
    train_preds: np.ndarray,
    eval_labels: np.ndarray,
    eval_preds: np.ndarray,
    extra: Dict = None,
) -> Dict:
    train_acc = accuracy_score(train_labels, train_preds)
    eval_acc = accuracy_score(eval_labels, eval_preds)
    eval_prec, eval_rec, eval_f1, _ = precision_recall_fscore_support(
        eval_labels, eval_preds, average="binary"
    )
    report = classification_report(
        eval_labels, eval_preds, target_names=["non-removable", "removable"]
    )

    print("\n" + "=" * 60)
    print(f"{pretty_name} RESULTS")
    print("=" * 60)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Eval accuracy:  {eval_acc:.4f}")
    print(f"Eval F1:        {eval_f1:.4f}")
    print(report)
    print("=" * 60 + "\n")

    out = {
        "method": method_name,
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "eval_precision": eval_prec,
        "eval_recall": eval_rec,
        "eval_f1": eval_f1,
        "classification_report": report,
    }
    if extra:
        out.update(extra)
    return out


def run_modernbert_step_classifier(
    train_sentences: List[str],
    train_labels: np.ndarray,
    eval_sentences: List[str],
    eval_labels: np.ndarray,
    seed: int,
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-5,
) -> Dict:
    """Fine-tune ModernBERT on individual reasoning steps."""
    logging.info(f"=== Method 3: ModernBERT step-only ({model_name}) ===")

    train_preds, eval_preds, train_labels, eval_labels = _train_modernbert_classifier(
        train_sentences,
        train_labels,
        eval_sentences,
        eval_labels,
        seed,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        drop_overflow=False,
    )
    return _summarize_modernbert_results(
        method_name=f"modernbert_step_{model_name.split('/')[-1]}",
        pretty_name="MODERNBERT STEP-ONLY",
        train_labels=train_labels,
        train_preds=train_preds,
        eval_labels=eval_labels,
        eval_preds=eval_preds,
    )


def run_modernbert_context_classifier(
    train_items: List[Dict],
    train_labels: np.ndarray,
    eval_items: List[Dict],
    eval_labels: np.ndarray,
    seed: int,
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-5,
) -> Dict:
    """Fine-tune ModernBERT on the CoT prefix up to and including the
    classified step. Examples exceeding the model's max context are dropped."""
    logging.info(f"=== Method 4: ModernBERT full-prefix ({model_name}) ===")

    train_texts = [it["prefix_text"] for it in train_items]
    eval_texts = [it["prefix_text"] for it in eval_items]

    train_preds, eval_preds, train_labels, eval_labels = _train_modernbert_classifier(
        train_texts,
        train_labels,
        eval_texts,
        eval_labels,
        seed,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        drop_overflow=True,
    )
    return _summarize_modernbert_results(
        method_name=f"modernbert_context_{model_name.split('/')[-1]}",
        pretty_name="MODERNBERT FULL-PREFIX",
        train_labels=train_labels,
        train_preds=train_preds,
        eval_labels=eval_labels,
        eval_preds=eval_preds,
        extra={
            "n_train_after_drop": len(train_labels),
            "n_eval_after_drop": len(eval_labels),
        },
    )


# ---------------------------------------------------------------------------
# Baseline 5: Per-step mean token confidence + logistic regression
# ---------------------------------------------------------------------------


def get_removable_nonremovable_with_borders(
    entry_labels: Dict[str, Dict[int, bool]],
    step_results: Dict,
) -> Tuple[list, list]:
    """Returns items carrying entry_key, sentence_idx, token_borders, and
    full_prompt — everything needed to recover per-token model probabilities
    of the generated CoT."""
    removable, nonremovable = [], []
    for entry_key, labels in sorted(entry_labels.items(), key=lambda x: x[0]):
        processed_entry = step_results.get(entry_key)
        if processed_entry is None:
            continue
        post_removal = processed_entry.get("post_removal", {})
        sentences = post_removal.get("sentences", [])
        token_borders = post_removal.get("token_borders", [])
        full_prompt = post_removal.get("full_prompt", "")
        if not token_borders or not full_prompt:
            continue
        for sent_idx, is_removable in labels.items():
            if sent_idx >= len(sentences) or sent_idx >= len(token_borders):
                continue
            sentence = sentences[sent_idx].strip()
            if not sentence:
                continue
            item = {
                "entry_key": entry_key,
                "sentence_idx": sent_idx,
                "sentence": sentence,
                "token_borders": token_borders[sent_idx],
                "full_prompt": full_prompt,
            }
            if is_removable:
                removable.append(item)
            else:
                nonremovable.append(item)
    return removable, nonremovable


@torch.no_grad()
def _compute_entry_token_logprobs(full_prompt: str, model, tokenizer) -> np.ndarray:
    """Tokenize ``full_prompt`` and return per-token log-probabilities of the
    realized tokens under the model. Index ``i`` of the returned array is
    log p(token[i] | token[<i]); index 0 is set to NaN (no context)."""
    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    logits = model(input_ids).logits  # (1, T, V)
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # (T-1,)
    out = np.full(input_ids.shape[1], np.nan, dtype=np.float64)
    out[1:] = token_lp.cpu().numpy()
    return out


def _step_mean_logprob(token_logprobs: np.ndarray, token_borders) -> float:
    start, end = int(token_borders[0]), int(token_borders[1])
    start = max(start, 1)  # skip NaN at position 0
    end = min(end, len(token_logprobs))
    if end <= start:
        return float("nan")
    seg = token_logprobs[start:end]
    seg = seg[~np.isnan(seg)]
    if seg.size == 0:
        return float("nan")
    return float(seg.mean())


def _compute_confidence_features(
    items: List[Dict],
    model,
    tokenizer,
) -> np.ndarray:
    """Compute mean per-token log-probability of each step in ``items``.
    Groups by entry to share one forward pass per ``full_prompt``."""
    by_entry: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        by_entry.setdefault(it["entry_key"], []).append(i)

    feats = np.full(len(items), np.nan, dtype=np.float64)
    for k, (entry_key, idxs) in enumerate(by_entry.items()):
        full_prompt = items[idxs[0]]["full_prompt"]
        try:
            token_lps = _compute_entry_token_logprobs(full_prompt, model, tokenizer)
        except Exception as e:
            logging.warning(f"Forward pass failed for entry {entry_key}: {e}")
            continue
        for i in idxs:
            feats[i] = _step_mean_logprob(token_lps, items[i]["token_borders"])
        if (k + 1) % 25 == 0:
            logging.info(f"  confidence features: {k + 1}/{len(by_entry)} entries")
    return feats


def run_confidence_classifier(
    train_items: List[Dict],
    train_labels: np.ndarray,
    eval_items: List[Dict],
    eval_labels: np.ndarray,
    model_short_name: str,
    seed: int,
) -> Dict:
    """Compute the mean per-token log-probability of each reasoning step under
    the reasoning model, then fit a logistic regression mapping that scalar
    confidence to the removability label."""
    from sklearn.linear_model import LogisticRegression
    from general_utils import load_model

    logging.info(
        f"=== Method 5: Per-step mean token confidence + logistic regression "
        f"({model_short_name}) ==="
    )

    if model_short_name not in MODEL_NAME_TO_ID_PATH:
        raise ValueError(
            f"Unknown model {model_short_name!r}; expected one of "
            f"{list(MODEL_NAME_TO_ID_PATH)}"
        )
    _, local_path = MODEL_NAME_TO_ID_PATH[model_short_name]
    model, tokenizer = load_model(local_path, dtype=torch.bfloat16, device_map="auto")

    try:
        train_feat = _compute_confidence_features(train_items, model, tokenizer)
        eval_feat = _compute_confidence_features(eval_items, model, tokenizer)
    finally:
        del model
        torch.cuda.empty_cache()

    train_mask = ~np.isnan(train_feat)
    eval_mask = ~np.isnan(eval_feat)
    n_train_dropped = int((~train_mask).sum())
    n_eval_dropped = int((~eval_mask).sum())
    if n_train_dropped or n_eval_dropped:
        logging.warning(
            f"Confidence baseline dropped {n_train_dropped} train, "
            f"{n_eval_dropped} eval items with no valid confidence"
        )
    X_train = train_feat[train_mask].reshape(-1, 1)
    y_train = np.asarray(train_labels)[train_mask]
    X_eval = eval_feat[eval_mask].reshape(-1, 1)
    y_eval = np.asarray(eval_labels)[eval_mask]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    clf = LogisticRegression(
        class_weight=class_weight, random_state=seed, max_iter=1000
    )
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    eval_preds = clf.predict(X_eval)

    train_acc = accuracy_score(y_train, train_preds)
    eval_acc = accuracy_score(y_eval, eval_preds)
    eval_prec, eval_rec, eval_f1, _ = precision_recall_fscore_support(
        y_eval, eval_preds, average="binary"
    )
    report = classification_report(
        y_eval, eval_preds, target_names=["non-removable", "removable"]
    )

    coef = float(clf.coef_.ravel()[0])
    intercept = float(clf.intercept_.ravel()[0])

    print("\n" + "=" * 60)
    print("CONFIDENCE (mean per-token logprob) + LOGREG RESULTS")
    print("=" * 60)
    print(
        f"Train mean logprob (removable):     {train_feat[train_mask][y_train == 1].mean():.4f}"
    )
    print(
        f"Train mean logprob (non-removable): {train_feat[train_mask][y_train == 0].mean():.4f}"
    )
    print(f"LR coef={coef:.4f}  intercept={intercept:.4f}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Eval accuracy:  {eval_acc:.4f}")
    print(f"Eval F1:        {eval_f1:.4f}")
    print(report)
    print("=" * 60 + "\n")

    return {
        "method": f"confidence_logreg_{model_short_name}",
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "eval_precision": eval_prec,
        "eval_recall": eval_rec,
        "eval_f1": eval_f1,
        "lr_coef": coef,
        "lr_intercept": intercept,
        "mean_logprob_removable": (
            float(train_feat[train_mask][y_train == 1].mean())
            if (y_train == 1).any()
            else float("nan")
        ),
        "mean_logprob_nonremovable": (
            float(train_feat[train_mask][y_train == 0].mean())
            if (y_train == 0).any()
            else float("nan")
        ),
        "n_train_dropped": n_train_dropped,
        "n_eval_dropped": n_eval_dropped,
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
        default=BASELINE_NAMES,
        choices=BASELINE_NAMES,
        help="Which baselines to run (default: all)",
    )
    parser.add_argument(
        "--use-eval-json",
        action="store_true",
        help="If supplied, uses the eval JSON file outputted by removability_pred_llm_as_a_judge output. "
        "Entries in this file become the eval set, rest become train set.",
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
    parser.add_argument(
        "--modernbert-model-name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="HF model id for ModernBERT baselines",
    )
    parser.add_argument(
        "--modernbert-epochs",
        type=int,
        default=3,
        help="Fine-tuning epochs for ModernBERT baselines (default 3)",
    )
    parser.add_argument(
        "--modernbert-batch-size",
        type=int,
        default=4,
        help="Batch size for ModernBERT step-only baseline",
    )
    parser.add_argument(
        "--modernbert-context-batch-size",
        type=int,
        default=4,
        help="Batch size for ModernBERT full-prefix baseline (smaller — long inputs)",
    )
    parser.add_argument(
        "--modernbert-lr",
        type=float,
        default=2e-5,
        help="Learning rate for ModernBERT fine-tuning",
    )

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

    suffix = f"_eval_json_split" if args.use_eval_json else ""
    output_filename = (
        f"{args.dataset}_token_based_removability_seed={args.seed}{suffix}.json"
    )
    output_path = os.path.join("results", args.model_name, output_filename)

    existing_output: Dict = {}
    existing_results: Dict = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_output = json.load(f)
        existing_results = existing_output.get("results", {}) or {}
        logging.info(
            f"Output file {output_path} already exists; will reuse baselines: "
            f"{sorted(existing_results)}"
        )
        remaining = [b for b in args.baselines if b not in existing_results]
        if not remaining:
            logging.info(
                "All requested baselines already present in results file. Nothing to run."
            )
            return
        logging.info(f"Running missing baselines: {remaining}")
        args.baselines = remaining

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

    if args.use_eval_json:
        eval_json_path = f"./results/{args.model_name}/{args.dataset}_sentence_removability_tags_seed={args.seed}.json"
        train_entry_labels, eval_entry_labels = split_labels_by_eval_json(
            entry_labels, eval_json_path
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
        eval_json_path = None
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

    if "modernbert_step" in args.baselines:
        all_results["modernbert_step"] = run_modernbert_step_classifier(
            train_items,
            train_labels,
            eval_items,
            eval_labels,
            args.seed,
            model_name=args.modernbert_model_name,
            epochs=args.modernbert_epochs,
            batch_size=args.modernbert_batch_size,
            lr=args.modernbert_lr,
        )

    if "modernbert_context" in args.baselines:
        # Build prefix-aware items using the same train/eval split as above.
        if args.use_eval_json:
            train_rem_ctx, train_nonrem_ctx = get_removable_nonremovable_with_prefix(
                train_entry_labels, step_results
            )
            eval_rem_ctx, eval_nonrem_ctx = get_removable_nonremovable_with_prefix(
                eval_entry_labels, step_results
            )
            if args.max_samples_per_class:
                set_deterministic(args.seed)
                n_per = min(
                    len(train_rem_ctx),
                    len(train_nonrem_ctx),
                    args.max_samples_per_class,
                )
                train_rem_ctx = random.sample(train_rem_ctx, k=n_per)
                train_nonrem_ctx = random.sample(train_nonrem_ctx, k=n_per)
            train_items_ctx = train_rem_ctx + train_nonrem_ctx
            train_labels_ctx = np.array(
                [1] * len(train_rem_ctx) + [0] * len(train_nonrem_ctx)
            )
            eval_items_ctx = eval_rem_ctx + eval_nonrem_ctx
            eval_labels_ctx = np.array(
                [1] * len(eval_rem_ctx) + [0] * len(eval_nonrem_ctx)
            )
        else:
            removable_ctx, nonremovable_ctx = get_removable_nonremovable_with_prefix(
                entry_labels, step_results
            )
            (
                train_items_ctx,
                eval_items_ctx,
                train_labels_ctx,
                eval_labels_ctx,
            ) = balance_and_split(
                removable_ctx,
                nonremovable_ctx,
                args.test_size,
                args.seed,
                args.max_samples_per_class,
            )

        all_results["modernbert_context"] = run_modernbert_context_classifier(
            train_items_ctx,
            train_labels_ctx,
            eval_items_ctx,
            eval_labels_ctx,
            args.seed,
            model_name=args.modernbert_model_name,
            epochs=args.modernbert_epochs,
            batch_size=args.modernbert_context_batch_size,
            lr=args.modernbert_lr,
        )

    if "confidence" in args.baselines:
        # Build items carrying full_prompt + token_borders so we can recover
        # per-token model probabilities for each reasoning step.
        if args.use_eval_json:
            train_rem_b, train_nonrem_b = get_removable_nonremovable_with_borders(
                train_entry_labels, step_results
            )
            eval_rem_b, eval_nonrem_b = get_removable_nonremovable_with_borders(
                eval_entry_labels, step_results
            )
            if args.max_samples_per_class:
                set_deterministic(args.seed)
                n_per = min(
                    len(train_rem_b),
                    len(train_nonrem_b),
                    args.max_samples_per_class,
                )
                train_rem_b = random.sample(train_rem_b, k=n_per)
                train_nonrem_b = random.sample(train_nonrem_b, k=n_per)
            train_items_b = train_rem_b + train_nonrem_b
            train_labels_b = np.array(
                [1] * len(train_rem_b) + [0] * len(train_nonrem_b)
            )
            eval_items_b = eval_rem_b + eval_nonrem_b
            eval_labels_b = np.array([1] * len(eval_rem_b) + [0] * len(eval_nonrem_b))
        else:
            removable_b, nonremovable_b = get_removable_nonremovable_with_borders(
                entry_labels, step_results
            )
            (
                train_items_b,
                eval_items_b,
                train_labels_b,
                eval_labels_b,
            ) = balance_and_split(
                removable_b,
                nonremovable_b,
                args.test_size,
                args.seed,
                args.max_samples_per_class,
            )

        all_results["confidence"] = run_confidence_classifier(
            train_items_b,
            train_labels_b,
            eval_items_b,
            eval_labels_b,
            model_short_name=args.model_name,
            seed=args.seed,
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

    new_results = {
        k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
        for k, v in all_results.items()
    }
    merged_results = {**existing_results, **new_results}

    parameters = existing_output.get("parameters", {}) if existing_output else {}
    parameters.update(
        {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sufficiency_threshold": args.sufficiency_threshold,
            "obtainability_threshold": args.obtainability_threshold,
        }
    )
    parameters["baselines"] = sorted(merged_results)

    data_statistics = (
        existing_output.get("data_statistics", {}) if existing_output else {}
    )
    data_statistics.update(
        {
            "n_removable": int(
                (np.concatenate([train_labels, eval_labels]) == 1).sum()
            ),
            "n_nonremovable": int(
                (np.concatenate([train_labels, eval_labels]) == 0).sum()
            ),
            "n_train": len(train_items),
            "n_eval": len(eval_items),
        }
    )

    output_data = {
        "parameters": parameters,
        "data_statistics": data_statistics,
        "results": merged_results,
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
