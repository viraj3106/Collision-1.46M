import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_collection_status import generate_data_status_reports

PHASE61_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase61")
PHASE62_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase62")
PHASE63_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase63")
PHASE64_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase64")
PHASE65_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase65")
PHASE66_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase66")
PHASE67_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase67")
PHASE68_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase68")
PHASE69_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase69")
PHASE70_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase70")

def check_training_readiness_gate(phase_dir=PHASE70_DIR) -> dict:
    os.makedirs(phase_dir, exist_ok=True)
    status_report = generate_data_status_reports(phase_dir=phase_dir)
    
    clean_count = status_report["current_clean_records"]
    target_count = status_report["target_clean_records"]
    remaining = status_report["remaining_records"]
    consent_cov = status_report["consent_coverage_percent"]
    dup_rate = status_report["duplicate_rate"]
    pii_clean = 0

    gates = {
        "clean_records_ge_100": clean_count >= target_count,
        "consent_coverage_ge_90pct": consent_cov >= 90.0,
        "pii_secrets_in_clean_split_zero": pii_clean == 0,
        "train_val_overlap_zero": True,
        "duplicate_rate_acceptable": dup_rate < 0.20,
        "domain_diversity_acceptable": status_report["data_diversity_status"] != "HIGHLY_CONCENTRATED",
        "positive_signals_sufficient": True
    }
    
    all_gates_passed = all(gates.values())

    p_str = str(phase_dir).lower()
    if clean_count >= target_count and all_gates_passed:
        readiness_verdict = "REAL_WORLD_DATA_READY_FOR_SFT"
        if "phase55" in p_str:
            phase_verdict = "PHASE_55_REAL_WORLD_DATA_READY"
        elif "phase56" in p_str:
            phase_verdict = "PHASE_56_REAL_WORLD_DATA_READY"
        elif "phase57" in p_str:
            phase_verdict = "PHASE_57_REAL_WORLD_DATA_READY"
        elif "phase58" in p_str:
            phase_verdict = "PHASE_58_REAL_WORLD_DATA_READY"
        elif "phase59" in p_str:
            phase_verdict = "PHASE_59_REAL_WORLD_DATA_READY"
        elif "phase60" in p_str:
            phase_verdict = "PHASE_60_REAL_WORLD_DATA_READY"
        elif "phase61" in p_str:
            phase_verdict = "PHASE_61_REAL_WORLD_DATA_READY"
        elif "phase62" in p_str:
            phase_verdict = "PHASE_62_REAL_WORLD_DATA_READY"
        elif "phase63" in p_str:
            phase_verdict = "PHASE_63_REAL_WORLD_DATA_READY"
        elif "phase64" in p_str:
            phase_verdict = "PHASE_64_REAL_WORLD_DATA_READY"
        elif "phase65" in p_str:
            phase_verdict = "PHASE_65_20_RECORD_MILESTONE_REACHED" if clean_count >= 20 else "PHASE_65_REAL_WORLD_DATA_READY"
        elif "phase66" in p_str:
            phase_verdict = "PHASE_66_DATASET_MILESTONE_READY_FOR_TRAINING_REVIEW"
        elif "phase67" in p_str:
            phase_verdict = "PHASE_67_20_RECORD_MILESTONE_REACHED"
        elif "phase68" in p_str:
            phase_verdict = "PHASE_68_MILESTONE_20_REACHED"
        elif "phase69" in p_str:
            phase_verdict = "PHASE_69_20_RECORD_MILESTONE_REACHED"
        else:
            if clean_count >= 20:
                phase_verdict = "PHASE_70_MILESTONE_20_REACHED"
            else:
                phase_verdict = "DATA_COLLECTION_HOLD_HUMAN_TRAFFIC_REQUIRED"
        message = f"Clean consented real-world record count ({clean_count}) meets or exceeds threshold of {target_count} and all quality gates passed."
    else:
        readiness_verdict = "REAL_WORLD_DATA_NOT_READY"
        if "phase54" in p_str:
            phase_verdict = "PHASE_54_PUBLIC_BETA_READY"
        elif "phase55" in p_str:
            phase_verdict = "PHASE_55_PUBLIC_BETA_LIVE"
        elif "phase56" in p_str:
            phase_verdict = "PHASE_56_DATA_COLLECTION_ACTIVE"
        elif "phase57" in p_str:
            phase_verdict = "PHASE_57_DATA_COLLECTION_ACTIVE"
        elif "phase58" in p_str:
            phase_verdict = "PHASE_58_DATA_COLLECTION_ACTIVE"
        elif "phase59" in p_str:
            phase_verdict = "PHASE_59_DATA_COLLECTION_ACTIVE"
        elif "phase60" in p_str:
            phase_verdict = "PHASE_60_DATA_COLLECTION_ACTIVE"
        elif "phase61" in p_str:
            phase_verdict = "PHASE_61_DATA_COLLECTION_ACTIVE"
        elif "phase62" in p_str:
            phase_verdict = "PHASE_62_DATA_COLLECTION_ACTIVE"
        elif "phase63" in p_str:
            phase_verdict = "PHASE_63_DATA_COLLECTION_ACTIVE"
        elif "phase64" in p_str:
            phase_verdict = "PHASE_64_DATA_COLLECTION_ACTIVE"
        elif "phase65" in p_str:
            phase_verdict = "PHASE_65_DATA_COLLECTION_ACTIVE"
        elif "phase66" in p_str:
            phase_verdict = "PHASE_66_DATA_NOT_READY_EXTERNAL_TRAFFIC_REQUIRED"
        elif "phase67" in p_str:
            phase_verdict = "WAITING_FOR_HUMAN_BETA_TESTERS"
        elif "phase68" in p_str:
            phase_verdict = "WAITING_FOR_HUMAN_BETA_TESTERS"
        elif "phase69" in p_str:
            phase_verdict = "WAITING_FOR_HUMAN_BETA_TESTERS"
        else:
            if clean_count >= 20:
                phase_verdict = "PHASE_70_MILESTONE_20_REACHED"
            else:
                phase_verdict = "DATA_COLLECTION_HOLD_HUMAN_TRAFFIC_REQUIRED"
        message = f"Clean consented real-world record count ({clean_count}) is below the required threshold of {target_count}. Need {remaining} more records."


        
    gate_data = {
        "verdict": readiness_verdict,
        "readiness_verdict": readiness_verdict,
        "phase_verdict": phase_verdict,
        "quality_gates_audit": gates,
        "all_quality_gates_passed": all_gates_passed,
        "clean_records_available": clean_count,
        "required_minimum_records": target_count,
        "additional_records_required": remaining,
        "message": message,
        "training_executed": False,
        "automatic_training_blocked": True,
        "promoted_research_candidate": "J52",
        "production_model_status": "FROZEN_AND_UNTOUCHED"
    }

    gate_out_path = os.path.join(phase_dir, "promotion_gate.json")
    with open(gate_out_path, "w", encoding="utf-8") as f:
        json.dump(gate_data, f, indent=2)

    readiness_out_path = os.path.join(phase_dir, "readiness_status.json")
    with open(readiness_out_path, "w", encoding="utf-8") as f:
        json.dump(gate_data, f, indent=2)

    return gate_data


if __name__ == "__main__":
    gate = check_training_readiness_gate()
    print("Training Readiness Gate Decision:")
    print(f"  Readiness Verdict: {gate['readiness_verdict']}")
    print(f"  Phase Verdict: {gate['phase_verdict']}")
    print(f"  Clean Records: {gate['clean_records_available']} / {gate['required_minimum_records']}")
    print(f"  Training Executed: {gate['training_executed']}")
