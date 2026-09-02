import os
import json
import re
import argparse
from typing import Dict, List, Tuple

RAW_DIR = os.path.join("data", "real_world", "raw")
CLEANED_DIR = os.path.join("data", "real_world", "cleaned")
REJECTED_DIR = os.path.join("data", "real_world", "rejected")

REQUIRED_FIELDS = ["prompt", "response", "rating"]

def validate_and_clean_records(raw_records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Validates records, removes duplicates, excludes rejected/unconsented records,
    and returns (cleaned_records, rejected_records).
    """
    cleaned = []
    rejected = []
    seen_prompts_responses = set()

    for idx, rec in enumerate(raw_records):
        rejection_reasons = []

        # 1. Consent check
        # Consent must be explicitly True if provided (defaults to True if consent key is omitted for backward compat or explicit consent field)
        if rec.get("consent") is False:
            rejection_reasons.append("Consent explicitly declined (consent=False)")

        # 2. Check required fields
        for field in REQUIRED_FIELDS:
            val = rec.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                rejection_reasons.append(f"Missing or empty required field: {field}")

        # 3. Rating check (only thumbs up/down allowed for positive dataset inclusion)
        rating = str(rec.get("rating", "")).lower().strip()
        if rating not in ["thumbs_up", "thumbs_down", "up", "down", "+1", "-1", "positive", "negative"]:
            rejection_reasons.append(f"Invalid rating value: {rec.get('rating')}")
        elif rating in ["thumbs_down", "down", "-1", "negative"]:
            rejection_reasons.append("Thumbs down / negative rating")

        # 4. Sensitive data check (passwords, tokens, api keys, emails, phone numbers)
        prompt_str = str(rec.get("prompt", ""))
        resp_str = str(rec.get("response", ""))
        combined_text = (prompt_str + " " + resp_str).lower()

        # Regex patterns for emails, phone numbers, API keys
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        
        if re.search(email_pattern, combined_text):
            rejection_reasons.append("Potential sensitive data detected: email address")
        if re.search(phone_pattern, combined_text):
            rejection_reasons.append("Potential sensitive data detected: phone number")

        for sensitive_keyword in ["api_key", "bearer ", "password=", "passwd=", "secret_key", "col_", "sk-", "token=", "auth_key"]:
            if sensitive_keyword in combined_text:
                rejection_reasons.append(f"Potential sensitive credential detected: {sensitive_keyword}")

        # 5. Duplicate check
        norm_prompt = " ".join(prompt_str.split())
        norm_response = " ".join(resp_str.split())
        pair_key = (norm_prompt, norm_response)
        if pair_key in seen_prompts_responses:
            rejection_reasons.append("Duplicate prompt-response example")
        
        # 6. Excessive length check (> 4000 characters)
        if len(prompt_str) > 4000 or len(resp_str) > 4000:
            rejection_reasons.append("Excessively long record (exceeds 4000 characters)")

        if rejection_reasons:
            rec_copy = dict(rec)
            rec_copy["rejection_reasons"] = rejection_reasons
            rejected.append(rec_copy)
        else:
            seen_prompts_responses.add(pair_key)
            cleaned.append({
                "user_id": rec.get("user_id", "anonymous"),
                "prompt": norm_prompt,
                "model": rec.get("model", "collision-10m"),
                "response": norm_response,
                "rating": rating,
                "feedback": rec.get("feedback", ""),
                "category": rec.get("category", "general"),
                "timestamp": rec.get("timestamp", ""),
                "consent": rec.get("consent", True)
            })


    return cleaned, rejected

def process_data_pipeline(raw_file_path: str = None) -> Dict[str, int]:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    raw_records = []
    
    # Locate raw data files
    if raw_file_path and os.path.exists(raw_file_path):
        files_to_read = [raw_file_path]
    else:
        files_to_read = [
            os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) 
            if f.endswith(".json") or f.endswith(".jsonl")
        ]

    for fpath in files_to_read:
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        try:
                            raw_records.append(json.loads(line))
                        except Exception:
                            pass
            else:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        raw_records.extend(data)
                    else:
                        raw_records.append(data)
                except Exception:
                    pass

    cleaned, rejected = validate_and_clean_records(raw_records)

    # Save cleaned dataset as JSONL without overwriting raw files
    cleaned_out_path = os.path.join(CLEANED_DIR, "real_world_cleaned.jsonl")
    versioned_out_path = os.path.join(CLEANED_DIR, "collision_real_world_v1.jsonl")
    
    with open(cleaned_out_path, "w", encoding="utf-8") as f1, \
         open(versioned_out_path, "w", encoding="utf-8") as f2:
        for item in cleaned:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            f1.write(line)
            f2.write(line)

    # Save rejected dataset
    rejected_out_path = os.path.join(REJECTED_DIR, "real_world_rejected.jsonl")
    with open(rejected_out_path, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "raw_count": len(raw_records),
        "cleaned_count": len(cleaned),
        "rejected_count": len(rejected),
        "cleaned_file": cleaned_out_path,
        "versioned_file": versioned_out_path,
        "rejected_file": rejected_out_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and clean real-world COLLISION feedback records.")
    parser.add_argument("--raw-file", type=str, default=None, help="Optional direct path to a raw data file.")
    args = parser.parse_args()
    
    stats = process_data_pipeline(args.raw_file)
    print(f"Data Pipeline Execution Summary:")
    print(f"  Raw Records Evaluated: {stats['raw_count']}")
    print(f"  Cleaned Training Examples: {stats['cleaned_count']} -> {stats['cleaned_file']}")
    print(f"  Rejected Examples: {stats['rejected_count']} -> {stats['rejected_file']}")
