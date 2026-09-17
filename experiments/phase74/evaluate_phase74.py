import os
import sys
import json
import torch
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase72.phase72_diagnostic import classify_response, generate_text
from data.audit_generation_quality import calculate_repetition_metrics
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
J71_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase71", "checkpoints", "collision_10m_capability_j71.pt")
J73C_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase73", "checkpoints", "collision_10m_sft_j73c.pt")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def get_phase74_benchmark_prompts() -> Dict[str, List[str]]:
    return {
        "casual_conversation": [
            "Hello! How are you doing today?",
            "Good morning! Are you ready to chat?",
            "Hey! What's up?"
        ],
        "context_retention": [
            "User: My name is Alex.\nAssistant: Nice to meet you Alex!\nUser: What is my name?",
            "User: I like space science.\nAssistant: Space science is fascinating.\nUser: What is my favorite subject?"
        ],
        "follow_up_understanding": [
            "User: What is Python?\nAssistant: Python is a programming language.\nUser: Why is it popular?",
            "User: Tell me about Paris.\nAssistant: Paris is the capital of France.\nUser: What landmarks are there?"
        ],
        "explanation": [
            "Explain how computer memory RAM works in simple terms.",
            "Explain the concept of gravity to a high school student."
        ],
        "instruction_following": [
            "List 3 benefits of drinking water daily.",
            "Summarize the goal of neural networks in one sentence."
        ],
        "technical_conversation": [
            "What is the difference between TCP and UDP?",
            "Explain what a database primary key is."
        ],
        "topic_switching": [
            "User: Let's talk about cars.\nAssistant: Sure! Cars use engines to travel.\nUser: Actually, let's switch to cooking. How do you bake bread?"
        ],
        "clarification": [
            "Can you explain what you meant by algorithms?"
        ],
        "ambiguity": [
            "Can you tell me more about that thing we discussed?"
        ],
        "short_response_control": [
            "Reply with only one word: What is the capital of Japan?"
        ],
        "long_response_control": [
            "Provide a detailed 3-paragraph summary of how photosyntesis works."
        ],
        "consistency": [
            "Is the earth round or flat?"
        ],
        "repetition_resistance": [
            "Say hello five times without repeating yourself unnecessarily."
        ],
        "conversational_naturalness": [
            "Thanks for your help! Have a great day."
        ]
    }
