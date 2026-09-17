import os
import sys
import json
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer

config_path = os.path.join(PROJECT_ROOT, "configs/collision_10m.yaml")
with open(config_path, "r") as f:
    cfg_dict = yaml.safe_load(f)["model"]

cfg = ModelConfig.from_yaml(config_path)
model = CollisionTransformer(cfg)

num_params = sum(p.numel() for p in model.parameters())

arch_manifest = {
    "task": "TASK 5 - MODEL ARCHITECTURE CONTROL",
    "architecture_name": "CollisionTransformer",
    "target_parameters": 10282304,
    "actual_parameters": num_params,
    "architecture_match": num_params == 10282304,
    "configuration": cfg_dict
}

with open(os.path.join(PROJECT_ROOT, "experiments/phase91/phase91_architecture_manifest.json"), "w") as f:
    json.dump(arch_manifest, f, indent=2)

print(f"Architecture manifest generated! Parameter count: {num_params:,}")
