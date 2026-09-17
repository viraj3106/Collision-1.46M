import os

# Project root paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments")
TOKENIZER_DIR = os.environ.get("COLLISION_TOKENIZER_PATH", os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Model checkpoints
PRODUCTION_MODEL_PATH = os.environ.get("COLLISION_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt"))
RESEARCH_V9_MODEL_PATH = os.environ.get("COLLISION_RESEARCH_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt"))

# Server configuration
COLLISION_HOST = os.environ.get("COLLISION_HOST", "0.0.0.0" if os.environ.get("COLLISION_ENVIRONMENT") == "production" else "127.0.0.1")
COLLISION_PORT = int(os.environ.get("COLLISION_PORT", "8000"))
COLLISION_LOG_LEVEL = os.environ.get("COLLISION_LOG_LEVEL", "INFO").upper()
COLLISION_ENVIRONMENT = os.environ.get("COLLISION_ENVIRONMENT", "production").lower()
COLLISION_CORS_ORIGINS = os.environ.get("COLLISION_CORS_ORIGINS", "")

# Grounding & safety configuration
COLLISION_WEB_ENABLED = os.environ.get("COLLISION_WEB_ENABLED", "true").lower() in ("1", "true", "yes", "on")
COLLISION_LOCAL_RAG_ENABLED = os.environ.get("COLLISION_LOCAL_RAG_ENABLED", "true").lower() in ("1", "true", "yes", "on")
COLLISION_MAX_INPUT_LENGTH = int(os.environ.get("COLLISION_MAX_INPUT_LENGTH", "2000"))
COLLISION_MAX_CONTEXT_LENGTH = int(os.environ.get("COLLISION_MAX_CONTEXT_LENGTH", "256"))
COLLISION_DEFAULT_TOP_K = int(os.environ.get("COLLISION_DEFAULT_TOP_K", "3"))
COLLISION_DEFAULT_TEMPERATURE = float(os.environ.get("COLLISION_DEFAULT_TEMPERATURE", "0.2"))
COLLISION_DEFAULT_REPETITION_PENALTY = float(os.environ.get("COLLISION_DEFAULT_REPETITION_PENALTY", "1.15"))

# Reliability & Timeout limits
COLLISION_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("COLLISION_REQUEST_TIMEOUT_SECONDS", "30.0"))
COLLISION_WEB_TIMEOUT_SECONDS = float(os.environ.get("COLLISION_WEB_TIMEOUT_SECONDS", "4.0"))
COLLISION_RATE_LIMIT_PER_MINUTE = int(os.environ.get("COLLISION_RATE_LIMIT", os.environ.get("COLLISION_RATE_LIMIT_PER_MINUTE", "60")))
COLLISION_RATE_LIMIT_ENABLED = os.environ.get("COLLISION_RATE_LIMIT_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Protected Checkpoint SHA256 Hashes
PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

def validate_checkpoints() -> bool:
    """Validates that protected checkpoints exist and match expected SHA-256 hashes."""
    import hashlib
    pairs = [
        (PRODUCTION_MODEL_PATH, PROTECTED_COLLISION_10M_SHA256),
        (RESEARCH_V9_MODEL_PATH, PROTECTED_PHASE91_V9_SHA256)
    ]
    for path, expected_hash in pairs:
        if not os.path.exists(path):
            return False
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192 * 1024):
                sha.update(chunk)
        if sha.hexdigest().lower() != expected_hash.lower():
            return False
    return True

# Default settings
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "collision_1m.yaml")

