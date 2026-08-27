import os

# Project root paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Default settings
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "collision_1m.yaml")
