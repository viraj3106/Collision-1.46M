import os
import sys
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.inference.engine import CollisionInferenceEngine

_engine_instance = None

def get_model_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64kb chunks
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_inference_engine() -> CollisionInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        model_dir = os.environ.get(
            "MODEL_PATH", 
            os.path.join(PROJECT_ROOT, "models", "collision-10m")
        )
        model_pt_path = os.path.join(model_dir, "model.pt")
        
        # 1. Verify file exists
        if not os.path.exists(model_pt_path):
            raise FileNotFoundError(f"Production model.pt checkpoint not found at: {model_pt_path}")
            
        # 2. Check SHA256 checksum
        expected_sha = os.environ.get(
            "MODEL_SHA256", 
            "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
        )
        print(f"Verifying model checksum for {model_pt_path}...")
        actual_sha = get_model_sha256(model_pt_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Model integrity validation failed! SHA256 checksum mismatch.\n"
                f"Expected: {expected_sha}\n"
                f"Actual:   {actual_sha}"
            )
        print("Model checksum verified successfully.")

        # 3. Load engine and verify parameter count
        print("Initializing CollisionInferenceEngine dependency...")
        engine = CollisionInferenceEngine(model_dir=model_dir)
        
        expected_params = int(os.environ.get("MODEL_PARAM_COUNT", "10282304"))
        actual_params = sum(p.numel() for p in engine.model.parameters())
        if actual_params != expected_params:
            raise ValueError(
                f"Model parameter count mismatch!\n"
                f"Expected: {expected_params}\n"
                f"Actual:   {actual_params}"
            )
        print(f"Model parameter count verified: {actual_params:,} parameters.")
        
        _engine_instance = engine
        
    return _engine_instance
