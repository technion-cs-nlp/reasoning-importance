DATASET_REGISTRY = {
    "harp-standard": {
        "text_column": "problem",
        "answer_column": "answer",
        "has_difficulty": True,
        "prompt_wrapper": "math",
    },
    "math-500": {
        "text_column": "problem",
        "answer_column": "answer",
        "has_difficulty": False,
        "prompt_wrapper": "math",
    },
}

SUFFICIENCY_THRESHOLD = 0.8
OBTAINABILITY_THRESHOLD = 0.8
NECCESITY_THRESHOLD = 0.8
N_SUFFICINECY_EVAL_RESAMPLES = 5
MAX_EXTERNAL_COMPLETION_TOKENS = 24576
VALID_PREFIX_SENT_LIMITS = (10, 200)
SEEDS = [42, 43, 44]
PREFIX_ITERATION_KEY = "incremental_prefix_results"
THRESHOLDS_KEY = "threshold_results"
GREEDY_KEY = "greedy_results"
PRUNING_THRESHOLDS = sorted(
    [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
)


# Classifier default values
TEST_SIZE = 0.2
EPOCHS = 30
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
LR = 3e-4

REMOVABILITY_CLASSIFIER_TYPES = [
    "linear",
    "mlp",
    "layerwise_linear",
    "layerwise_mlp",
    "activation_tensor",
    "simple_transformer",
]

# TODO DELETE PATHS PRIOR TO PUBLISHING OR REPLACE DIR
MODEL_NAME_TO_ID_PATH = {
    "gpt-oss-20b": (
        "openai/gpt-oss-20b",
        "/mnt/nlp/models/models--openai--gpt-oss-20b/snapshots/ef533c924d202f1bffdd43e1f0593ad7d42484d6/",
    ),
    "DeepSeek-R1-Distill-Qwen-1.5B": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "/mnt/nlp/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
    ),
    "DeepSeek-R1-Distill-Qwen-7B": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "/mnt/nlp/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
    ),
    "DeepSeek-R1-Distill-Qwen-14B": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "/mnt/nlp/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761",
    ),
    "Olmo-3-7B-Think": (
        "allenai/Olmo-3-7B-Think",
        "/mnt/nlp/models/models--allenai--Olmo-3-7B-Think/snapshots/7c991fde2671813ab41745054310f60a610b0fac",
    ),
}

LONG_TO_SHORT_MODEL_NAMES = {
    "gpt-oss-20b": "GPT-OSS",
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-1.5",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-7",
    "DeepSeek-R1-Distill-Qwen-14B": "DS-14",
    "Olmo-3-7B-Think": "Olmo3",
}
