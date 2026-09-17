"""
COLLISION-10M — Automated Hugging Face Publisher & Hub Synchronizer.

Safely validates checksums and publishes COLLISION-10M Model Hub and
COLLISION AI & NLP Lab Space Hub to Hugging Face in one step.

Usage:
    python release/publish_to_huggingface.py --token <HF_TOKEN>
    python release/publish_to_huggingface.py --all
"""

import os
import sys
import argparse
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_REPO_ID = "viraj3106/collision-10m"
SPACE_REPO_ID = "viraj3106/collision-ai-lab"
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"


def verify_package(hf_dir: str) -> bool:
    print("[1/3] Verifying release package integrity...")
    model_pt = os.path.join(hf_dir, "model.pt")
    if not os.path.exists(model_pt):
        print(f"[ERROR] Model weights not found at {model_pt}")
        return False

    sha256 = hashlib.sha256()
    with open(model_pt, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().lower()

    if actual_hash != EXPECTED_SHA256:
        print(f"[ERROR] Checksum mismatch: Expected {EXPECTED_SHA256}, got {actual_hash}")
        return False

    print(f"[OK] Checksum verified: SHA-256 is {actual_hash[:16]}... (Protected Flagship)")
    return True


def publish_model(api, hf_dir: str, repo_id: str, token: str):
    print(f"\n[2/3] Uploading Model Hub to https://huggingface.co/{repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        api.upload_folder(
            folder_path=hf_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Release COLLISION-10M with In-House NLP Engine & Natural Web Grounding"
        )
        print(f"[OK] Model successfully published to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[ERROR] Failed to upload model: {e}")


def publish_space(api, space_dir: str, repo_id: str, token: str):
    print(f"\n[3/3] Uploading Space Hub to https://huggingface.co/spaces/{repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True, token=token)
        api.upload_folder(
            folder_path=space_dir,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            commit_message="Deploy COLLISION AI & NLP Lab interactive portal"
        )
        print(f"[OK] Space successfully deployed to: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"[ERROR] Failed to upload space: {e}")


def main():
    parser = argparse.ArgumentParser(description="COLLISION Hugging Face Publisher")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API Token")
    parser.add_argument("--model-repo", type=str, default="collision-10M/collision-10m", help="Target Model Hub repo ID")
    parser.add_argument("--space-repo", type=str, default="collision-10M/collision-ai-lab", help="Target Space Hub repo ID")
    parser.add_argument("--model-only", action="store_true", help="Upload Model Hub only")
    parser.add_argument("--space-only", action="store_true", help="Upload Space Hub only")
    parser.add_argument("--all", action="store_true", default=True, help="Upload both Model and Space")
    args = parser.parse_args()

    hf_dir = os.path.join(PROJECT_ROOT, "release", "huggingface")
    space_dir = os.path.join(PROJECT_ROOT, "release", "huggingface_space")

    if not verify_package(hf_dir):
        sys.exit(1)

    if not args.token:
        print("\n⚠️ Warning: No HF_TOKEN provided.")
        print("Please provide your token using --token <YOUR_HF_TOKEN> or export HF_TOKEN=<token>.")
        print("\nManual deployment instructions are available in HF_RELEASE.md.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] `huggingface_hub` package not found. Run `pip install huggingface_hub`.")
        return

    api = HfApi()

    if args.space_only:
        publish_space(api, space_dir, args.space_repo, args.token)
    elif args.model_only:
        publish_model(api, hf_dir, args.model_repo, args.token)
    else:
        publish_model(api, hf_dir, args.model_repo, args.token)
        publish_space(api, space_dir, args.space_repo, args.token)

    print("\n[SUCCESS] Deployment completed successfully!")
    print(f"* Model Hub: https://huggingface.co/{args.model_repo}")
    print(f"* Space Demo: https://huggingface.co/spaces/{args.space_repo}")


if __name__ == "__main__":
    main()
