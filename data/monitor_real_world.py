import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

PHASE59_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase59")
PHASE60_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase60")
PHASE61_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase61")
PHASE62_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase62")
PHASE63_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase63")
PHASE64_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase64")
PHASE65_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase65")
PHASE67_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase67")
PHASE68_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase68")
PHASE69_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase69")
PHASE70_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase70")

def print_real_world_data_status(phase_dir=PHASE70_DIR):
    status = generate_data_status_reports(phase_dir=phase_dir)
    gate = check_training_readiness_gate(phase_dir=phase_dir)

    clean = status["clean_records"]
    target = status["target_clean_records"]
    remaining = status["remaining_records_required"]
    diversity = status.get("data_diversity_status", "HIGHLY_CONCENTRATED")
    
    domain_dist = status.get("domain_distribution", {})
    top_domain = max(domain_dist, key=domain_dist.get) if domain_dist else "General Knowledge"
    zero_domains_count = len(status.get("zero_record_domains", []))
    
    training_status = "READY" if gate["readiness_verdict"] == "REAL_WORLD_DATA_READY_FOR_SFT" else "BLOCKED"
    production_status = "FROZEN"

    report_lines = [
        "COLLISION PUBLIC BETA",
        "--------------------",
        "",
        f"Clean records: {clean} / {target}",
        f"Remaining: {remaining}",
        "",
        "Diversity:",
        f"{diversity}",
        "",
        "Top domain:",
        f"{top_domain}",
        "",
        "Zero-record domains:",
        f"{zero_domains_count}",
        "",
        "Training:",
        f"{training_status}",
        "",
        "Production:",
        f"{production_status}"
    ]

    output_text = "\n".join(report_lines)
    print(output_text)
    return status, gate, output_text

if __name__ == "__main__":
    print_real_world_data_status()
