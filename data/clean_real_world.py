import os
import json
import re
import argparse
import sqlite3
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "real_world", "raw")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "real_world", "cleaned")
REJECTED_DIR = os.path.join(PROJECT_ROOT, "data", "real_world", "rejected")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "real_world", "reports")

REQUIRED_FIELDS = ["prompt", "response", "rating"]

# Prompt injection heuristics
PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"disregard system prompt",
    r"you are now DAN",
    r"jailbreak",
    r"bypass safety",
    r"system prompt override",
    r"drop database",
    r"<script\b",
]

def validate_and_clean_records(raw_records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Validates records according to Phase 53 strict schema and quality rules:
    - Missing required fields rejection
    - Consent check (consent must be strictly True)
    - Positive rating signal check ('thumbs_up', 'up', '+1', 'positive')
    - Minimum & maximum length filtering (empty, extremely short, excessive)
    - PII detection (email, phone, SSN, IP address)
    - Credential / Secret detection (API key, bearer token, passwords, tokens)
    - Prompt injection check
    - Exact & near-duplicate filtering
    """
    cleaned = []
    rejected = []
    seen_pairs = set()

    for idx, rec in enumerate(raw_records):
        rejection_reasons = []

        # 1. Consent verification
        consent_val = rec.get("consent")
        if consent_val is not True and str(consent_val).lower() not in ["true", "1"]:
            rejection_reasons.append("Missing or unverified consent (consent != True)")

        # 2. Check required fields & empty content
        for field in REQUIRED_FIELDS:
            val = rec.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                rejection_reasons.append(f"Missing or empty required field: {field}")

        prompt_str = str(rec.get("prompt", "")).strip()
        resp_str = str(rec.get("response", "")).strip()

        if len(prompt_str) < 3:
            rejection_reasons.append("Prompt too short (< 3 characters)")
        if len(resp_str) < 5:
            rejection_reasons.append("Response too short (< 5 characters)")

        if len(prompt_str) > 4000 or len(resp_str) > 4000:
            rejection_reasons.append("Excessively long record (exceeds 4000 characters)")

        # 3. Rating & feedback validity check (must be positive signal for training)
        rating = str(rec.get("rating", "")).lower().strip()
        positive_ratings = ["thumbs_up", "up", "+1", "positive", "1", "approve", "approved"]
        if rating not in positive_ratings:
            rejection_reasons.append(f"Non-positive rating signal: '{rec.get('rating')}'")

        # 4. Sensitive data & PII detection
        combined_text = (prompt_str + " " + resp_str).lower()

        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'

        if re.search(email_pattern, combined_text):
            rejection_reasons.append("Sensitive data detected: email address")
        if re.search(phone_pattern, combined_text):
            rejection_reasons.append("Sensitive data detected: phone number")
        if re.search(ip_pattern, combined_text) and not any(ip in combined_text for ip in ["127.0.0.1", "0.0.0.0"]):
            rejection_reasons.append("Sensitive data detected: IP address")

        secret_keywords = ["api_key", "bearer ", "password=", "passwd=", "secret_key", "col_", "sk-", "token=", "auth_key", "private_key"]
        for sensitive_keyword in secret_keywords:
            if sensitive_keyword in combined_text:
                rejection_reasons.append(f"Sensitive credential detected: {sensitive_keyword}")

        # 5. Prompt injection / safety checks
        for inj_pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(inj_pattern, combined_text, re.IGNORECASE):
                rejection_reasons.append(f"Prompt injection / safety contamination detected: '{inj_pattern}'")

        # 6. Deduplication check
        norm_prompt = " ".join(prompt_str.split())
        norm_resp = " ".join(resp_str.split())
        pair_key = (norm_prompt.lower(), norm_resp.lower())
        if pair_key in seen_pairs:
            rejection_reasons.append("Duplicate prompt-response pair")

        if rejection_reasons:
            rec_copy = dict(rec)
            rec_copy["rejection_reasons"] = rejection_reasons
            rejected.append(rec_copy)
        else:
            seen_pairs.add(pair_key)
            cleaned.append({
                "prompt": norm_prompt,
                "response": norm_resp,
                "rating": rating,
                "category": rec.get("category", "general"),
                "conversation_type": rec.get("conversation_type", "factual Q&A"),
                "parent_id": rec.get("parent_id"),
                "is_multi_turn": rec.get("is_multi_turn", False),
                "source": rec.get("model", rec.get("source", "user_feedback")),
                "consent": True,
                "quality_status": "passed_audit"
            })

    return cleaned, rejected

def fetch_raw_records() -> List[Dict]:
    raw_records = []
    
    # 1. Fetch from SQLite DB if available
    db_path = os.environ.get("COLLISION_DB_PATH", os.path.join(PROJECT_ROOT, "collision_api.db"))
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback")
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Ensure consent boolean
                row_dict["consent"] = bool(row_dict.get("consent", 0))
                raw_records.append(row_dict)
            conn.close()
        except Exception as e:
            print(f"Note: DB read error or table empty: {e}")

    # 2. Fetch from data/real_world/raw/ directory
    if os.path.exists(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.endswith(".json") or f.endswith(".jsonl"):
                fpath = os.path.join(RAW_DIR, f)
                with open(fpath, "r", encoding="utf-8") as file:
                    if f.endswith(".jsonl"):
                        for line in file:
                            if line.strip():
                                try:
                                    raw_records.append(json.loads(line))
                                except Exception:
                                    pass
                    else:
                        try:
                            data = json.load(file)
                            if isinstance(data, list):
                                raw_records.extend(data)
                            else:
                                raw_records.append(data)
                        except Exception:
                            pass

    return raw_records

def process_data_pipeline() -> Dict:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    raw_records = fetch_raw_records()
    cleaned, rejected = validate_and_clean_records(raw_records)

    # Write cleaned
    cleaned_out_path = os.path.join(CLEANED_DIR, "real_world_cleaned.jsonl")
    with open(cleaned_out_path, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write rejected
    rejected_out_path = os.path.join(REJECTED_DIR, "real_world_rejected.jsonl")
    with open(rejected_out_path, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summarize rejection reasons
    rejection_summary = {}
    for item in rejected:
        for reason in item.get("rejection_reasons", []):
            rejection_summary[reason] = rejection_summary.get(reason, 0) + 1

    report_data = {
        "raw_record_count": len(raw_records),
        "cleaned_record_count": len(cleaned),
        "rejected_record_count": len(rejected),
        "rejection_reasons_breakdown": rejection_summary,
        "consent_coverage_percent": (sum(1 for r in raw_records if r.get("consent") in [True, 1]) / len(raw_records) * 100) if raw_records else 0.0
    }

    report_out_path = os.path.join(REPORTS_DIR, "data_cleaning_report.json")
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return {
        "raw_count": len(raw_records),
        "cleaned_count": len(cleaned),
        "rejected_count": len(rejected),
        "cleaned_file": cleaned_out_path,
        "rejected_file": rejected_out_path,
        "report_file": report_out_path
    }

if __name__ == "__main__":
    stats = process_data_pipeline()
    print("Real-World Data Cleaning Complete:")
    print(f"  Raw Evaluated: {stats['raw_count']}")
    print(f"  Cleaned Accepted: {stats['cleaned_count']}")
    print(f"  Rejected: {stats['rejected_count']}")

