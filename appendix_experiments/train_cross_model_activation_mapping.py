"""
Train linear mappings between models' penultimate-layer activation spaces.

For every ordered pair (A, B) of distinct models in MODEL_NAME_TO_ID_PATH we
train an `nn.Linear(d_B, d_A)` mapping that maps B's penultimate hidden-layer
activations into A's penultimate hidden-layer activation space.

Per-reasoning-step training items:
  - One item per reasoning step from B's `processed_sentences` (post_removal)
  - The "position" for each step is the full token range of the step (i.e. the
    tokens that make up that step, ending at its last token); the activation
    used for training is the AVERAGE of the model's penultimate hidden states
    over those tokens (same aggregation as train_removability_classifier).
  - Source-side (B) activation: mean over the step's tokens of model B's
    hidden_states[-2] on the step's `full_prompt`.
  - Target-side (A) activation: mean over the step's tokens of model A's
    hidden_states[-2] on the same `full_prompt`, with the step's token
    boundaries remapped from B's tokenizer to A's tokenizer.

Outputs (per pair) are saved under:
    results/{A}/cross-model-analysis/{B}_to_{A}_seed={seed}.{json,pt}
"""

import argparse
import copy
import gc
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from loading_utils import load_single_step_results
from general_utils import set_deterministic, load_model, d_model
from consts import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_NAME_TO_ID_PATH,
    TEST_SIZE,
    WEIGHT_DECAY,
)
from train_removability_classifier import fix_token_borders

DEFAULT_MAX_ENTRIES_PER_MODEL = 200


# ===========================================================================
# Argument parsing
# ===========================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train linear B->A mappings between models' penultimate activations, "
        "for every (A, B) pair in MODEL_NAME_TO_ID_PATH."
    )
    parser.add_argument("--dataset", type=str, default="harp-standard")
    parser.add_argument(
        "--max-entries-per-model",
        type=int,
        default=DEFAULT_MAX_ENTRIES_PER_MODEL,
        help="Cap on the number of generation entries used per source model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix on output filenames.",
    )
    return parser.parse_args()


# ===========================================================================
# Step item collection (from processed_sentences post_removal)
# ===========================================================================


def collect_step_items(
    model_name: str,
    dataset: str,
    max_entries: Optional[int],
    seed: int,
) -> List[Dict]:
    """
    Build per-reasoning-step items from a model's processed_sentences results.
    Each item carries the step's text, the step's token range (in this model's
    tokenizer), and the entry's full_prompt.
    """
    step_results, _ = load_single_step_results(model_name, dataset=dataset)
    entries = sorted(step_results.items(), key=lambda kv: kv[0])
    if max_entries is not None and len(entries) > max_entries:
        rng = random.Random(seed)
        entries = rng.sample(entries, k=max_entries)

    items: List[Dict] = []
    for entry_key, entry in entries:
        post = entry.get("post_removal", {})
        sentences = post.get("sentences", [])
        borders = post.get("token_borders", [])
        full_prompt = post.get("full_prompt", "")
        if not sentences or not borders or not full_prompt:
            continue
        for sent_idx, (sent, bord) in enumerate(zip(sentences, borders)):
            if not sent or not sent.strip():
                continue
            items.append(
                {
                    "entry_key": entry_key,
                    "sentence_idx": sent_idx,
                    "sentence": sent,
                    "token_borders": list(bord),
                    "full_prompt": full_prompt,
                }
            )
    return items


# ===========================================================================
# Activation extraction: penultimate hidden layer, averaged over step tokens
# ===========================================================================


@torch.no_grad()
def compute_penult_step_avg(
    items: List[Dict],
    model,
    tokenizer,
    max_prompt_tokens: int,
    desc: str,
) -> List[Optional[torch.Tensor]]:
    """
    For each item, run the model on its full_prompt and return the mean of
    hidden_states[-2] across the item's [token_start, token_end) range.
    """
    device = next(model.parameters()).device

    # Group by unique prompt so each forward runs once.
    prompts_to_idx: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        prompts_to_idx.setdefault(it["full_prompt"], []).append(i)

    out: List[Optional[torch.Tensor]] = [None] * len(items)

    for prompt, idxs in tqdm(prompts_to_idx.items(), desc=desc):
        try:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_tokens,
            ).to(device)
            outputs = model(**enc, output_hidden_states=True)
        except Exception as e:
            logging.warning(f"Forward pass failed for one prompt: {e}")
            continue

        penult = outputs.hidden_states[-2][0].to(torch.float32).cpu()  # (S, D)
        seq_len = penult.shape[0]

        for i in idxs:
            start, end = items[i]["token_borders"]
            start = max(0, min(start, seq_len))
            end = max(start, min(end, seq_len))
            if end <= start:
                continue
            out[i] = penult[start:end].mean(dim=0).clone()

        del outputs, penult
        torch.cuda.empty_cache()

    return out


# ===========================================================================
# Mapping training
# ===========================================================================


class PairedActivationDataset(Dataset):
    def __init__(self, src: List[torch.Tensor], tgt: List[torch.Tensor]):
        assert len(src) == len(tgt)
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def _collate(batch):
    src, tgt = zip(*batch)
    return torch.stack(src).float(), torch.stack(tgt).float()


def train_linear_mapping(
    src_train: List[torch.Tensor],
    tgt_train: List[torch.Tensor],
    src_eval: List[torch.Tensor],
    tgt_eval: List[torch.Tensor],
    d_src: int,
    d_tgt: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[nn.Linear, Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = nn.Linear(d_src, d_tgt).to(device)

    train_loader = DataLoader(
        PairedActivationDataset(src_train, tgt_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )
    eval_loader = DataLoader(
        PairedActivationDataset(src_eval, tgt_eval),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        mapping.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_results = []
    best_eval_mse = float("inf")
    best_state = None

    for epoch in range(epochs):
        mapping.train()
        total_loss = 0.0
        n_batches = 0
        for sx, tx in train_loader:
            sx = sx.to(device)
            tx = tx.to(device)
            optimizer.zero_grad()
            pred = mapping(sx)
            loss = criterion(pred, tx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        mapping.eval()
        with torch.no_grad():
            eval_sse = 0.0
            eval_n = 0
            cos_sum = 0.0
            cos_n = 0
            for sx, tx in eval_loader:
                sx = sx.to(device)
                tx = tx.to(device)
                pred = mapping(sx)
                eval_sse += ((pred - tx) ** 2).sum().item()
                eval_n += tx.numel()
                cos = nn.functional.cosine_similarity(pred, tx, dim=-1)
                cos_sum += cos.sum().item()
                cos_n += cos.numel()
            eval_mse = eval_sse / max(eval_n, 1)
            eval_cos = cos_sum / max(cos_n, 1)

        train_loss = total_loss / max(n_batches, 1)
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "train_mse": train_loss,
                "eval_mse": eval_mse,
                "eval_cos": eval_cos,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        logging.info(
            f"  Epoch {epoch+1}/{epochs} - train MSE: {train_loss:.4f}, "
            f"eval MSE: {eval_mse:.4f}, eval cos: {eval_cos:.4f}"
        )

        if eval_mse < best_eval_mse:
            best_eval_mse = eval_mse
            best_state = {k: v.cpu().clone() for k, v in mapping.state_dict().items()}

    if best_state is not None:
        mapping.load_state_dict(best_state)
        mapping = mapping.to(device)

    with torch.no_grad():
        tgt_eval_stack = torch.stack([t.float() for t in tgt_eval])
        mean_tgt = tgt_eval_stack.mean(dim=0, keepdim=True)
        baseline_mse_mean = ((tgt_eval_stack - mean_tgt) ** 2).mean().item()
        baseline_mse_zero = (tgt_eval_stack**2).mean().item()

    return mapping, {
        "epoch_results": epoch_results,
        "best_eval_mse": best_eval_mse,
        "baseline_eval_mse_predict_mean": baseline_mse_mean,
        "baseline_eval_mse_predict_zero": baseline_mse_zero,
    }


# ===========================================================================
# Main
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


def process_pair(tgt_a: str, src_b: str, args) -> None:
    """
    Run the full pipeline (collect items -> compute B & A activations ->
    train B->A mapping -> save) for a single (A, B) model pair.
    """
    logging.info(f"\n{'='*60}\nProcessing pair: {src_b} -> {tgt_a}\n{'='*60}")

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_dir = os.path.join("results", tgt_a, "cross-model-analysis")
    os.makedirs(out_dir, exist_ok=True)
    base = f"{src_b}_to_{tgt_a}_seed={args.seed}{suffix}"
    json_path = os.path.join(out_dir, base + ".json")
    pt_path = os.path.join(out_dir, base + ".pt")
    if os.path.exists(json_path) and os.path.exists(pt_path):
        logging.info(f"Output for {src_b} -> {tgt_a} already exists; skipping")
        return

    # --- 1. Collect B's step items (with B's token borders) ---
    items_b = collect_step_items(
        src_b, args.dataset, args.max_entries_per_model, args.seed
    )
    logging.info(f"  {src_b}: {len(items_b)} step items collected")
    if not items_b:
        logging.warning(f"No items for {src_b}; skipping pair")
        return

    # --- 2. Compute B's source-side activations (load + unload B) ---
    b_model, b_tokenizer = _load_lm(src_b)
    d_src = d_model(b_model)
    src_acts = compute_penult_step_avg(
        items_b,
        b_model,
        b_tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        desc=f"{src_b} source-side activations",
    )
    del b_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 3. Build A-remapped items using both tokenizers, then drop B's tokenizer ---
    a_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_ID_PATH[tgt_a][1])
    items_a = copy.deepcopy(items_b)
    fix_token_borders(b_tokenizer, a_tokenizer, items_a)
    del b_tokenizer
    gc.collect()

    # --- 4. Compute A's target-side activations on the remapped items ---
    a_model, a_tokenizer = _load_lm(tgt_a)
    d_tgt = d_model(a_model)
    tgt_acts = compute_penult_step_avg(
        items_a,
        a_model,
        a_tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        desc=f"{tgt_a} target-side activations on {src_b}'s items",
    )
    del a_model, a_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- 5. Pair up usable activations ---
    paired_src, paired_tgt = [], []
    for s, t in zip(src_acts, tgt_acts):
        if s is not None and t is not None:
            paired_src.append(s)
            paired_tgt.append(t)
    n_paired = len(paired_src)
    logging.info(
        f"{src_b} -> {tgt_a}: {n_paired}/{len(src_acts)} usable paired step activations"
    )
    if n_paired < 10:
        logging.warning(f"Too few pairs for {src_b} -> {tgt_a}; skipping")
        return

    idx = list(range(n_paired))
    train_idx, eval_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed
    )
    src_train = [paired_src[i] for i in train_idx]
    tgt_train = [paired_tgt[i] for i in train_idx]
    src_eval = [paired_src[i] for i in eval_idx]
    tgt_eval = [paired_tgt[i] for i in eval_idx]

    # --- 6. Train mapping ---
    logging.info(
        f"Training {src_b} (d={d_src}) -> {tgt_a} (d={d_tgt}) "
        f"on {len(src_train)} train / {len(src_eval)} eval"
    )
    mapping, results = train_linear_mapping(
        src_train=src_train,
        tgt_train=tgt_train,
        src_eval=src_eval,
        tgt_eval=tgt_eval,
        d_src=d_src,
        d_tgt=d_tgt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --- 7. Save ---
    output = {
        "parameters": {
            "target_model": tgt_a,
            "source_model": src_b,
            "dataset": args.dataset,
            "seed": args.seed,
            "max_entries_per_model": args.max_entries_per_model,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_prompt_tokens": args.max_prompt_tokens,
            "d_source": d_src,
            "d_target": d_tgt,
            "layer": "penultimate (hidden_states[-2])",
            "aggregation": "mean over step tokens",
        },
        "data_statistics": {
            "n_step_items_source": len(src_acts),
            "n_paired": n_paired,
            "n_train": len(src_train),
            "n_eval": len(src_eval),
        },
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    torch.save(mapping.state_dict(), pt_path)
    logging.info(f"Saved {base}.json/.pt under {out_dir}")

    del mapping, src_acts, tgt_acts, paired_src, paired_tgt
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
