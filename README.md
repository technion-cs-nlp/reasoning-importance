# Chain-of-Thought Importance Concept Detection 

This codebase accompanies the paper ["Reasoning Models Know What's Important, and Encode It in Their Activations"](https://arxiv.org/abs/2604.18307).

## Key findings

- Many reasoning steps in CoT are *removable*: they can be removed while maintaining the correct answer.
- Probes trained on reasoning steps activations succesfully distinguish important from removable steps, even **prior** to generation of subsequent steps (that determine the importance of steps).
- Token-level features (e.g., LLM labels, surface-level features, semantic embeddings, lexcial statistics) fail at this classification task.


## File hierarchy
```
.
├── data/                             # Datasets (HARP variants)
├── results/                          # Pipeline outputs (per model)
├── appendix_experiments/             # Additional ablations that mostly appear in the paper's appendix
│
├── generate_reasoning_chains.py      # Stage 1: CoT generation
├── identify_shortcut_steps.py        # Stage 2: shortcut detection
├── eval_attribution_pruning.py       # Stage 3.1: attribution-based pruning
├── eval_llm_pruning.py               # Stage 3.2: LLM-based pruning
├── eval_random_pruning.py            # Stage 3.3: random pruning baseline
├── train_removability_classifier.py  # Stage 4: activation-based importance (through removability) probes
├── run_classifier_experiments.py     # Stage 4 aggregated

├── run_token_based_experiments.py    # Stage 5: token-based classification
├── train_removability_classifier.py  # Stage 6: probe training
│
├── cot_utils.py                      # CoT parsing (sentences, answers, LaTeX)
├── general_utils.py                  # Model loading, generation, perplexity
├── gradient_attribution.py           # Gradient-based sentence influence scores
├── loading_utils.py                  # Load result files across pipeline stages
├── removability_utils.py             # Removability labeling helpers
├── consts.py                         # Shared constants
└── tests/                            # Unit tests
```

## Pipeline

To recreate the results, you can run the stages in order:

```bash
# 1. Generate chain-of-thought outputs
python generate_reasoning_chains.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard

# 2. Find shortcut sentences sentences
python identify_shortcut_steps.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard

# 3. Find the core reasoning subsequence, with three variants of a greedy algorithm.
# 3.1. Activation-based (gradient attribution) variant:
python eval_attribution_pruning.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard
# 3.2. LLM-based pruning variant:
python eval_llm_pruning.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard --external-model gemini-2.5-pro
# 3.3. Random pruning baseline:
python eval_random_pruning.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard

# 4. Train and evaluate reasoning-step-embedding-based importance probes (removability probes)
python train_removability_classifier.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard --context --classifier-type layerwise_mlp --probe-hidden-dim 128
# OR (for all classifier settings combined):
python run_classifier_experiments.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard

# 5. Baselines for importance classification
# 5.1. External LLM-as-a-judge baseline
python removability_pred_llm_as_a_judge.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard --setups external local 
# 5.2. Other (TF-IDF, Semantic Embeddings) baselines 
python removability_pred_token_embeds_baselines.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard --baselines bow sentence_transformer

# 6. Analyze importance probes
python analyze_removability_probes.py --model-name DeepSeek-R1-Distill-Qwen-7B --dataset harp-standard
```

Results are saved under `results/{model-name}/`.

You may also download the pre-computed results (reported in the paper) and run a subset of the scripts on them.


## Supported model ids

- `openai/gpt-oss-20b`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `allenai/Olmo-3-7B-Think`

## Dataset

Two mathematical reasoning dataset: **HARP** (Download from https://github.com/aadityasingh/HARP into the `data` folder) and **MATH-500**