"""
Runs train_removability_classifier.py with multiple configuration sets.
"""

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime

from consts import LR, MODEL_NAME_TO_ID_PATH, REMOVABILITY_CLASSIFIER_TYPES, SEEDS


def generate_embedding_configs():
    embedding_changing_args = {
        "seed": SEEDS,
        "context_flag": ["--context", "--no-context"],
        "classifier-type": [
            "linear",
            "mlp",
            "layerwise_linear",
            "layerwise_mlp",
        ],
    }
    embedding_changing_combs = [
        dict(zip(embedding_changing_args.keys(), v))
        for v in itertools.product(*embedding_changing_args.values())
    ]

    return embedding_changing_combs


def generate_classifier_configs(classifier_type):
    """
    Generates settings that affect the classifier architecture and training (but not the input embeddings).
    """
    model_spec = {"classifier-type": [classifier_type]}
    if classifier_type == "linear":
        pass
    elif classifier_type == "layerwise_linear":
        pass
    elif classifier_type == "mlp":
        model_spec.update({"probe-mlp-layers": [2, 5], "probe-hidden-dim": [128, 256]})
    elif classifier_type == "layerwise_mlp":
        model_spec.update({"probe-hidden-dim": [16, 128, 1024, 4096]})
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Global settings that apply to everything
    lrs = sorted([LR, 3e-3, 3e-5])

    all_combinations = []
    for values in itertools.product(*model_spec.values()):
        config = dict(zip(model_spec.keys(), values))

        for lr in lrs:
            combo = config.copy()
            combo["learning-rate"] = lr
            all_combinations.append(combo)

    return all_combinations


def build_command(
    model_name: str,
    model_id: str,
    model_path: str,
    experiment: dict,
    suffix: str,
    dataset: str = "harp-standard",
    save_predictions: str = None,
) -> list[str]:
    """Convert an experiment dict into a CLI arg list."""
    cmd = [
        sys.executable,
        "train_removability_classifier.py",
        experiment.pop("context_flag"),
        "--model-name",
        model_name,
        "--dataset",
        dataset,
        "--output-suffix",
        suffix,
    ]
    cmd.extend(["--embed-model-id", model_id, "--embed-model-path", model_path])
    cmd.extend(["--sufficiency-threshold", "0.8", "--obtainability-threshold", "0.8"])
    cmd.extend(["--epochs", "30"])

    for key, value in experiment.items():
        cmd.extend([f"--{key}", str(value)])

    if save_predictions:
        cmd.extend(["--save-predictions", save_predictions])

    cmd = [part for part in cmd if part]  # Remove empty strings
    return cmd


def experiment_label(experiment: dict) -> str:
    """Create a unique, shortened, human-readable label from an experiment config.

    Abbreviations:
        s42        = seed 42
        ctx/noctx  = context / no-context
        lin/mlp/lwmlp/lwlin = classifier types
        phd256     = probe-hidden-dim 256
        pml2       = probe-mlp-layers 2
        lr3e-4     = learning rate 3e-4
    """
    parts = []

    # Embedding-changing params
    parts.append(f"s{experiment.get('seed', '?')}")
    parts.append("ctx" if experiment.get("context_flag") == "--context" else "noctx")

    # Classifier type (abbreviated)
    clf_abbrev = {
        "linear": "lin",
        "mlp": "mlp",
        "layerwise_linear": "lwlin",
        "layerwise_mlp": "lwmlp",
    }
    clf = experiment.get("classifier-type", "?")
    parts.append(clf_abbrev.get(clf, clf))

    # Classifier-specific params (only include when they vary for that type)
    if clf == "mlp":
        parts.append(f"phd{experiment.get('probe-hidden-dim', '?')}")
        parts.append(f"pml{experiment.get('probe-mlp-layers', '?')}")
    elif clf == "layerwise_mlp":
        parts.append(f"phd{experiment.get('probe-hidden-dim', '?')}")

    # Learning rate
    lr = experiment.get("learning-rate")
    if lr is not None:
        parts.append(f"lr{lr:g}")

    return "_".join(parts)


def summarize_across_seeds(scores_dict: dict):
    """
    Summarize results for a single setting across multiple seeds (which control the train/eval split).
    """
    config_summary = {}
    config_scores = {}
    for label, acc in scores_dict.items():
        if acc is None:
            continue
        # Strip the seed part (third underscore-delimited token, e.g. "s42")
        parts = label.split("_")
        config_key = "_".join(parts[1:])  # drop seed part
        config_scores.setdefault(config_key, []).append(acc)

    for config_key, accs in config_scores.items():
        config_summary[config_key] = {
            "mean": sum(accs) / len(accs),
            "std": (sum((a - sum(accs) / len(accs)) ** 2 for a in accs) / len(accs))
            ** 0.5,
            "n_seeds": len(accs),
        }
    return config_summary


def print_summary(summary):
    best_config = max(summary, key=lambda k: summary[k]["mean"])
    best = summary[best_config]
    print(f"\n{'=' * 60}")
    print("  RESULTS AVERAGED ACROSS SEEDS")
    print(f"{'=' * 60}")
    for cfg in sorted(summary, key=lambda k: summary[k]["mean"], reverse=True):
        s = summary[cfg]
        print(f"  {cfg}: {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
    print(f"\n  Best config: {best_config}")
    print(f"  Mean acc: {best['mean']:.4f} +/- {best['std']:.4f}")
    print(f"{'=' * 60}\n")


def parse_best_eval_acc(stdout: str) -> float | None:
    """Extract BEST_EVAL_ACC=<value> from subprocess stdout."""
    match = re.search(r"BEST_EVAL_ACC=([\d.]+)", stdout)
    if match:
        return float(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run train_removability_classifier with multiple configurations."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Short model name for results directory (e.g. 'gpt-oss-20b')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="harp-standard",
        help="Dataset identifier (default: harp-standard)",
    )
    args = parser.parse_args()
    model_id, model_path = MODEL_NAME_TO_ID_PATH[args.model_name]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scores_path = f"results/{args.model_name}/classifiers/{args.dataset}_classifier_experiment_scores.json"
    if os.path.exists(scores_path):
        loaded = json.load(open(scores_path))
        if len(loaded) >= 5:
            (
                scores,
                config_summary,
                cross_model_scores,
                cross_model_summary,
                extra_scores,
            ) = loaded
        else:
            scores, config_summary, cross_model_scores, cross_model_summary = loaded
            extra_scores = {}
    else:
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        (
            scores,
            config_summary,
            cross_model_scores,
            cross_model_summary,
            extra_scores,
        ) = ({}, {}, {}, {}, {})

    # Activation-effecting arguments
    embedding_changing_combs = generate_embedding_configs()

    # Cartesian product of all experiments
    experiments = []
    for comb in embedding_changing_combs:
        embeds_path = f"seed{comb['seed']}_{comb['classifier-type']}_{comb['context_flag']}_embeddings.pt".replace(
            "-", ""
        )

        # Classifier-arch / training arguments (can used the same cached embeddings across these)
        classifier_args = generate_classifier_configs(comb["classifier-type"])

        for cls_args in classifier_args:
            exp = {**comb, **cls_args}
            exp.update({"save-embeddings": embeds_path, "load-embeddings": embeds_path})
            experiments.append(exp)
    total = len(experiments)

    print(f"Generated {total=} experiments to run.")

    # Run experiments
    failed = []
    for i, experiment in enumerate(experiments, 1):
        exp = dict(experiment)  # Copy so we don't mutate the original
        label = experiment_label(exp)

        if label in scores and scores[label]:
            print(
                f"\n*** Skipping experiment {i}/{total} (label '{label}') "
                f"because it already has a recorded score. ***\n"
            )
            continue

        suffix = f"{timestamp}_{label}"
        cmd = build_command(
            args.model_name, model_id, model_path, exp, suffix, dataset=args.dataset
        )

        print(f"\n{'=' * 60}")
        print(f"  Experiment {i}/{total}: {label}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, capture_output=True, text=True)
        # Stream the captured output
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if result.returncode != 0:
            print(f"\n*** Experiment {i} FAILED (exit code {result.returncode}) ***\n")
            failed.append(i)
            scores[label] = None
        else:
            best_acc = parse_best_eval_acc(result.stdout)
            scores[label] = best_acc
            print(
                f"\n*** Experiment {i}/{total} completed successfully "
                f"(best eval acc: {best_acc}) ***\n"
            )

        with open(scores_path, "w") as f:
            json.dump(
                (
                    scores,
                    config_summary,
                    cross_model_scores,
                    cross_model_summary,
                    extra_scores,
                ),
                f,
                indent=2,
            )

    # Average across seeds and find the best performing setting
    config_summary = summarize_across_seeds(scores)
    with open(scores_path, "w") as f:
        json.dump(
            (
                scores,
                config_summary,
                cross_model_scores,
                cross_model_summary,
                extra_scores,
            ),
            f,
            indent=2,
        )
    print_summary(config_summary)

    # Cross-model experiments
    print("Running cross-model experiments...")
    predictions_dir = f"results/{args.model_name}/classifiers/preds"
    os.makedirs(predictions_dir, exist_ok=True)
    for seed in SEEDS:
        # Run self-model experiment with predictions (same config as cross-model for fair comparison)
        self_pred_path = os.path.join(
            predictions_dir, f"{args.dataset}_predictions_self_seed{seed}.json"
        )
        if not os.path.exists(self_pred_path):
            print(f"\n*** Running self-model prediction run for seed {seed} ***\n")
            self_exp = {
                "seed": seed,
                "context_flag": "--context",
                "classifier-type": "layerwise_mlp",
                "probe-hidden-dim": 1024,
                "learning-rate": LR,
            }
            suffix = f"{timestamp}_self_predictions_seed{seed}"
            cmd = build_command(
                args.model_name,
                model_id,
                model_path,
                self_exp,
                suffix,
                dataset=args.dataset,
                save_predictions=self_pred_path,
            )
            print(f"  Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            if result.returncode != 0:
                print(f"\n*** Self-model prediction run FAILED for seed {seed} ***\n")
            else:
                print(
                    f"\n*** Self-model prediction run completed for seed {seed} ***\n"
                )
        else:
            print(
                f"\n*** Skipping self-model prediction run for seed {seed} (already exists) ***\n"
            )

        other_model_ids_paths = list(MODEL_NAME_TO_ID_PATH.values()) + [
            ("random/random", model_path)
        ]
        for other_model_id, other_model_path in other_model_ids_paths:
            if other_model_id == model_id:
                # Not cross-model, allready evaluated above
                continue
            other_model_name = other_model_id.split("/")[-1]
            key = f"seed{seed}_{other_model_name}"

            cross_pred_path = os.path.join(
                predictions_dir,
                f"{args.dataset}_predictions_crossmodel_{other_model_name}_seed{seed}.json",
            )

            if key in cross_model_scores and cross_model_scores[key]:
                print(
                    f"\n*** Skipping cross-model experiment for seed {seed} and model {other_model_id} "
                    f"because it already has a recorded score. ***\n"
                )
                continue

            cross_model_exp = {
                "seed": seed,
                "context_flag": "--context",
                "classifier-type": "layerwise_mlp",
                "probe-hidden-dim": 1024,
                "learning-rate": LR,
            }

            suffix = f"{timestamp}_crossmodel_{other_model_name}_seed{seed}"
            cmd = build_command(
                args.model_name,
                other_model_id,
                other_model_path,
                cross_model_exp,
                suffix,
                dataset=args.dataset,
                save_predictions=cross_pred_path,
            )
            print(f"\n{'=' * 60}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"{'=' * 60}\n")

            result = subprocess.run(cmd, capture_output=True, text=True)
            # Stream the captured output
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)

            if result.returncode != 0:
                print(f"\n*** Experiment FAILED***\n")
                cross_model_scores[key] = None
            else:
                best_acc = parse_best_eval_acc(result.stdout)
                cross_model_scores[key] = best_acc
                print(
                    f"\n*** Experiment completed successfully: (best eval acc: {best_acc}) ***\n"
                )
            with open(scores_path, "w") as f:
                json.dump(
                    (
                        scores,
                        config_summary,
                        cross_model_scores,
                        cross_model_summary,
                        extra_scores,
                    ),
                    f,
                    indent=2,
                )

        with open(scores_path, "w") as f:
            json.dump(
                (
                    scores,
                    config_summary,
                    cross_model_scores,
                    cross_model_summary,
                    extra_scores,
                ),
                f,
                indent=2,
            )

    # Average across seeds and find the best performing setting
    cross_model_summary = summarize_across_seeds(cross_model_scores)
    with open(scores_path, "w") as f:
        json.dump(
            (
                scores,
                config_summary,
                cross_model_scores,
                cross_model_summary,
                extra_scores,
            ),
            f,
            indent=2,
        )
    print_summary(cross_model_summary)

    # Compute cross-model confusion matrices
    print("\nComputing cross-model confusion matrices...")
    confusion_results = {}
    for seed in SEEDS:
        self_pred_path = os.path.join(
            predictions_dir, f"{args.dataset}_predictions_self_seed{seed}.json"
        )
        if not os.path.exists(self_pred_path):
            print(f"  Skipping seed {seed}: self-model predictions not found")
            continue
        with open(self_pred_path) as f:
            self_preds = json.load(f)
        # Index self predictions by (entry_key, sentence_idx)
        self_by_key = {
            (p["entry_key"], p["sentence_idx"]): p
            for p in self_preds
            if "entry_key" in p
        }

        other_model_ids_paths = list(MODEL_NAME_TO_ID_PATH.values()) + [
            ("random/random", model_path)
        ]
        for other_model_id, _ in other_model_ids_paths:
            if other_model_id == model_id:
                continue
            other_model_name = other_model_id.split("/")[-1]
            cross_pred_path = os.path.join(
                predictions_dir,
                f"{args.dataset}_predictions_crossmodel_{other_model_name}_seed{seed}.json",
            )
            if not os.path.exists(cross_pred_path):
                print(
                    f"  Skipping seed {seed}, {other_model_name}: cross-model predictions not found"
                )
                continue
            with open(cross_pred_path) as f:
                cross_preds = json.load(f)
            cross_by_key = {
                (p["entry_key"], p["sentence_idx"]): p
                for p in cross_preds
                if "entry_key" in p
            }

            # Align by intersection of keys
            common_keys = set(self_by_key.keys()) & set(cross_by_key.keys())
            if not common_keys:
                print(f"  No aligned samples for seed {seed}, {other_model_name}")
                continue

            # Compute confusion matrix
            cells = {
                "correct_self_correct_cross": [],
                "correct_self_incorrect_cross": [],
                "incorrect_self_correct_cross": [],
                "incorrect_self_incorrect_cross": [],
            }
            for key in common_keys:
                sp = self_by_key[key]
                cp = cross_by_key[key]
                gt = sp["ground_truth"]
                self_correct = sp["prediction"] == gt
                cross_correct = cp["prediction"] == gt

                example = {
                    "entry_key": key[0],
                    "sentence_idx": key[1],
                    "sentence": sp.get("sentence", ""),
                    "ground_truth": gt,
                    "pred_self": sp["prediction"],
                    "pred_cross": cp["prediction"],
                }

                if self_correct and cross_correct:
                    cells["correct_self_correct_cross"].append(example)
                elif self_correct and not cross_correct:
                    cells["correct_self_incorrect_cross"].append(example)
                elif not self_correct and cross_correct:
                    cells["incorrect_self_correct_cross"].append(example)
                else:
                    cells["incorrect_self_incorrect_cross"].append(example)

            confusion_key = f"seed{seed}_{other_model_name}"
            conf_matrix = {
                cell_name: len(examples) for cell_name, examples in cells.items()
            }
            agreement_ratio = (
                conf_matrix["correct_self_correct_cross"]
                + conf_matrix["incorrect_self_incorrect_cross"]
            ) / len(common_keys)
            confusion_results[confusion_key] = {
                "self_model": args.model_name,
                "other_model": other_model_name,
                "seed": seed,
                "n_aligned_samples": len(common_keys),
                "agreement_ratio": agreement_ratio,
                "confusion_matrix": conf_matrix,
                "examples": {
                    cell_name: random.sample(examples, min(1000, len(examples)))
                    for cell_name, examples in cells.items()
                },
            }

            print(
                f"  seed{seed}_{other_model_name}: "
                f"{len(common_keys)} aligned samples, "
                f"confusion={confusion_results[confusion_key]['confusion_matrix']}"
            )

    if confusion_results:
        confusion_path = os.path.join(
            predictions_dir, f"{args.dataset}_cross_model_confusion_matrices.json"
        )
        with open(confusion_path, "w") as f:
            json.dump(confusion_results, f, indent=2)
        print(f"\nConfusion matrices saved to: {confusion_path}")

    # Extra experiments (eval-json based, seed 42 only)
    if args.dataset == "harp-standard":
        eval_json_path = f"./results/{args.model_name}/sentence_removability_tags_difficulty=all_seed=42.json"
    else:
        eval_json_path = f"./results/{args.model_name}/{args.dataset}_sentence_removability_tags_seed=42.json"

    extra_experiments = {
        "eval_json_context_lwmlp": {
            "seed": 42,
            "context_flag": "--context",
            "classifier-type": "layerwise_mlp",
            "probe-hidden-dim": 1024,
            "learning-rate": LR,
            "eval-json": eval_json_path,
        },
        "eval_json_no_context_lwmlp": {
            "seed": 42,
            "context_flag": "--no-context",
            "classifier-type": "layerwise_mlp",
            "probe-hidden-dim": 1024,
            "learning-rate": LR,
            "eval-json": eval_json_path,
        },
    }

    print(f"\nRunning {len(extra_experiments)} extra experiments...")
    for label, experiment in extra_experiments.items():
        if experiment["context_flag"] == "--context":
            eval_json_pred_path = os.path.join(
                predictions_dir, f"{dataset_str}predictions_eval_json.json"
            )
        else:
            eval_json_pred_path = None
        if label in extra_scores and extra_scores[label]:
            print(
                f"\n*** Skipping extra experiment '{label}' "
                f"because it already has a recorded score. ***\n"
            )
            continue

        exp = dict(experiment)
        suffix = f"{timestamp}_extra_{label}"
        cmd = build_command(
            args.model_name,
            model_id,
            model_path,
            exp,
            suffix,
            dataset=args.dataset,
            save_predictions=eval_json_pred_path,
        )

        print(f"\n{'=' * 60}")
        print(f"  Extra experiment: {label}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if result.returncode != 0:
            print(f"\n*** Extra experiment '{label}' FAILED ***\n")
            extra_scores[label] = None
        else:
            best_acc = parse_best_eval_acc(result.stdout)
            extra_scores[label] = best_acc
            print(
                f"\n*** Extra experiment '{label}' completed successfully "
                f"(best eval acc: {best_acc}) ***\n"
            )

        with open(scores_path, "w") as f:
            json.dump(
                (
                    scores,
                    config_summary,
                    cross_model_scores,
                    cross_model_summary,
                    extra_scores,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
