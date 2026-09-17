import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_SPEC_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase74", "dataset_spec.json")

DATASET_SPECIFICATION = {
    "dataset_name": "collision_conversation_v1",
    "version": "1.0.0",
    "target_model_budget": "10.28M parameters",
    "total_target_conversations": 1400,
    "train_val_split": {"train": 1200, "validation": 200},
    "serialization_format": {
        "multi_turn_delimiter": "\n",
        "user_prefix": "User: ",
        "assistant_prefix": "Assistant: ",
        "token_masking_rule": "Loss weight = 0 for User prompt & prefixes; Loss weight = 1 for Assistant response turns."
    },
    "category_distribution": {
        "casual_conversation": 0.08,
        "general_question_answering": 0.08,
        "explanations": 0.08,
        "technical_conversations": 0.08,
        "follow_up_questions": 0.08,
        "multi_turn_conversations": 0.10,
        "clarification_requests": 0.05,
        "ambiguous_questions": 0.05,
        "topic_changes": 0.05,
        "corrections": 0.04,
        "short_answers": 0.05,
        "detailed_answers": 0.05,
        "reasoning_conversations": 0.05,
        "friendly_acknowledgements": 0.04,
        "i_dont_know_scenarios": 0.04,
        "context_dependent_questions": 0.08
    },
    "turn_depth_distribution": {
        "1_turn": 0.40,
        "2_turns": 0.35,
        "3_turns": 0.25
    }
}

def create_dataset_spec():
    os.makedirs(os.path.dirname(DATASET_SPEC_PATH), exist_ok=True)
    with open(DATASET_SPEC_PATH, "w", encoding="utf-8") as f:
        json.dump(DATASET_SPECIFICATION, f, indent=2)
    print(f"Dataset specification saved to {DATASET_SPEC_PATH}")

if __name__ == "__main__":
    create_dataset_spec()
