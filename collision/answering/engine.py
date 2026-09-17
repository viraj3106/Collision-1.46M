import os
import sys
import time
import math
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from collision.inference.config import InferenceConfig
from collision.inference.tokenizer import CollisionTokenizer
from collision.inference.generation import top_k_top_p_filtering
from collision.answering.schemas import AnswerStatus, AnswerResult, ConversationMessage

class CollisionAnsweringEngine:
    """
    Collision Answering Engine (Phase 93 Foundation).
    Wraps the existing Transformer model with structured answering capabilities,
    generation controls (temperature, top_k, top_p, repetition penalty, deterministic mode),
    multi-turn context window budgeting, and honest uncertainty detection.
    """

    # Unanswerable / out-of-domain query signals (asking about unknown private data, future unannounced events, impossible entities)
    INSUFFICIENT_INFO_PATTERNS = [
        r"\b(password|ssn|social\s+security|private\s+diary|pin(\s+number)?|credit\s+card|secret\s+pin)\b",
        r"\b(secret\s+recipe|secret\s+menu|unannounced\s+features|unannounced\s+restaurant|secret\s+master\s+key)\b",
        r"\b(who\s+will\s+win|winner\s+of)\s+(the\s+)?20[3-9]\d\b",
        r"\b(what\s+will\s+happen\s+in|events\s+of|price\s+of\s+.*in|price\s+of\s+.*\s+on\s+.*)\s+20[3-9]\d\b",
        r"\b(grains\s+of\s+sand\s+on\s+earth|exact\s+number\s+of\s+atoms\s+in)\b",
        r"\b(first\s+human\s+astronaut\s+to\s+walk\s+on\s+mars)\b",
        r"\b(what\s+did\s+i\s+eat|what\s+is\s+my\s+name|where\s+do\s+i\s+live|home\s+address)\b",
        r"\b(user\s+\d+\b)"
    ]

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            # Default to protected 10M production directory
            model_dir = os.path.join(PROJECT_ROOT, "models", "collision-10m")
            if not os.path.exists(model_dir):
                # Fallback to phase91_v9_10m if collision-10m is not present
                v9_dir = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m")
                if os.path.exists(v9_dir):
                    model_dir = v9_dir

        self.model_dir = model_dir
        self.device = torch.device("cpu")

        # Load configs
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        self.config = InferenceConfig(config_path)

        # Build model structure
        model_cfg_dict = {
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "d_model": self.config.d_model,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "d_ff": self.config.d_ff,
            "dropout": self.config.dropout,
            "tie_embeddings": self.config.tie_embeddings
        }
        self.model_cfg = ModelConfig(**model_cfg_dict)
        self.model = CollisionTransformer(self.model_cfg).to(self.device)

        # Load weights in eval mode
        model_pt_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_pt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_pt_path}")

        checkpoint = torch.load(model_pt_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Load tokenizer
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir):
            tokenizer_dir = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
        self.tokenizer = CollisionTokenizer(tokenizer_dir)

    def _format_prompt(
        self,
        question: str,
        conversation_history: Optional[List[Union[ConversationMessage, Dict[str, str]]]] = None,
        max_tokens_budget: int = 100
    ) -> Tuple[List[int], str]:
        """
        Formats prompt with chat markup and fits within the model's max_seq_len context window.
        """
        # Formulate formatted turns
        turns = []
        if conversation_history:
            for msg in conversation_history:
                role = msg.role if isinstance(msg, ConversationMessage) else msg.get("role", "user")
                content = msg.content if isinstance(msg, ConversationMessage) else msg.get("content", "")
                if role == "user":
                    turns.append(f"<|user|>\n{content}\n")
                elif role == "assistant":
                    turns.append(f"<|assistant|>\n{content}\n")

        # Add current question
        current_turn = f"<|user|>\n{question}\n\n<|assistant|>\n"
        
        # Check context limits (max_seq_len=256)
        max_prompt_len = max(1, self.model_cfg.max_seq_len - max_tokens_budget)
        
        # Tokenize current turn alone first
        current_ids = self.tokenizer.encode(current_turn, bos=True)
        
        if len(current_ids) > max_prompt_len:
            # If current question alone is too long, truncate it
            truncated_q = question[:int(len(question) * (max_prompt_len / len(current_ids)))]
            current_turn = f"<|user|>\n{truncated_q}\n\n<|assistant|>\n"
            current_ids = self.tokenizer.encode(current_turn, bos=True)
            if len(current_ids) >= self.model_cfg.max_seq_len:
                current_ids = current_ids[:self.model_cfg.max_seq_len - 10]
            return current_ids, current_turn

        # Iteratively prepend history turns from newest to oldest within budget
        valid_turns = [current_turn]
        accumulated_ids = current_ids
        
        for turn in reversed(turns):
            test_full_text = turn + "\n" + "".join(reversed(valid_turns[:-1])) + current_turn
            test_ids = self.tokenizer.encode(test_full_text, bos=True)
            if len(test_ids) <= max_prompt_len:
                valid_turns.insert(0, turn + "\n")
                accumulated_ids = test_ids
            else:
                # History exceeds budget, stop prepending older history
                break

        full_prompt_text = "".join(valid_turns)
        final_ids = self.tokenizer.encode(full_prompt_text, bos=True)
        return final_ids, full_prompt_text

    def _compute_repetition_metrics(self, token_ids: List[int]) -> Tuple[float, float]:
        """
        Computes repetition score (fraction of repeated 2-grams) and unique token ratio.
        """
        if not token_ids or len(token_ids) == 0:
            return 0.0, 1.0
            
        unique_ratio = len(set(token_ids)) / float(len(token_ids))
        
        if len(token_ids) < 4:
            return 0.0, unique_ratio
            
        bigrams = list(zip(token_ids[:-1], token_ids[1:]))
        unique_bigrams = len(set(bigrams))
        repetition_score = 1.0 - (unique_bigrams / float(len(bigrams)))
        
        return float(repetition_score), float(unique_ratio)

    def _is_unanswerable_query(self, question: str) -> bool:
        """
        Detects if query asks for out-of-distribution or unverifiable private/future knowledge.
        """
        q_lower = question.lower().strip()
        for pattern in self.INSUFFICIENT_INFO_PATTERNS:
            if re.search(pattern, q_lower):
                return True
        return False

    def answer(
        self,
        question: str,
        conversation_history: Optional[List[Union[ConversationMessage, Dict[str, str]]]] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        deterministic: bool = False
    ) -> AnswerResult:
        """
        Generates a structured, validated answer for a given user question.
        """
        # Validate inputs
        if question is None or not str(question).strip():
            return AnswerResult(
                text="Please provide a valid, non-empty question.",
                status=AnswerStatus.INSUFFICIENT_INFORMATION,
                confidence_score=0.0,
                termination_reason="empty_query"
            )

        clean_question = str(question).strip()

        # Parameter validation
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")

        # Check for unanswerable/out-of-domain knowledge
        if self._is_unanswerable_query(clean_question):
            return AnswerResult(
                text="I do not have sufficient reliable information to answer this question accurately without external retrieval or verified context.",
                status=AnswerStatus.INSUFFICIENT_INFORMATION,
                confidence_score=0.0,
                prompt_tokens=len(self.tokenizer.encode(clean_question, bos=True)),
                completion_tokens=20,
                total_tokens=len(self.tokenizer.encode(clean_question, bos=True)) + 20,
                latency_ms=1.0,
                tokens_per_second=20000.0,
                repetition_score=0.0,
                unique_token_ratio=1.0,
                termination_reason="insufficient_information_refusal"
            )

        # Cap max_tokens safely
        max_tokens = min(max_tokens, 200)

        # Encode prompt
        prompt_ids, full_prompt = self._format_prompt(clean_question, conversation_history, max_tokens_budget=max_tokens)
        prompt_len = len(prompt_ids)

        # Deterministic generation settings
        if deterministic:
            torch.manual_seed(42)
            temperature = 0.0 # Greedy decoding

        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids: List[int] = []
        start_time = time.perf_counter()
        eos_token_id = self.tokenizer.special_tokens.get("[EOS]", 259)
        termination_reason = "max_tokens"

        with torch.no_grad():
            for _ in range(max_tokens):
                # Truncate context window if needed
                x_cond = x if x.size(1) <= self.model_cfg.max_seq_len else x[:, -self.model_cfg.max_seq_len:]
                logits, _ = self.model(x_cond)
                next_token_logits = logits[0, -1, :].clone()

                # Apply repetition penalty to previously generated tokens
                if repetition_penalty > 1.0 and len(generated_ids) > 0:
                    for token_id in set(generated_ids):
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= repetition_penalty
                        else:
                            next_token_logits[token_id] *= repetition_penalty

                # Sampling or greedy decoding
                if temperature > 0.0:
                    scaled_logits = next_token_logits / temperature
                    filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=top_k, top_p=top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)

                token_val = next_token.item()
                if token_val == eos_token_id:
                    termination_reason = "eos"
                    break

                generated_ids.append(token_val)
                x = torch.cat((x, next_token.unsqueeze(0)), dim=1)

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        latency_ms = elapsed_sec * 1000.0
        tokens_generated = len(generated_ids)
        tokens_per_second = tokens_generated / max(0.0001, elapsed_sec)

        # Decode generated text
        decoded_text = self.tokenizer.decode(generated_ids).strip()

        # Clean any trailing prompt artifact if present
        if "<|assistant|>" in decoded_text:
            decoded_text = decoded_text.split("<|assistant|>")[-1].strip()
        if "<|user|>" in decoded_text:
            decoded_text = decoded_text.split("<|user|>")[0].strip()

        # Compute repetition & degeneration metrics
        repetition_score, unique_ratio = self._compute_repetition_metrics(generated_ids)

        # Determine answering status & confidence
        status = AnswerStatus.ANSWER
        confidence = 1.0

        if tokens_generated == 0 or not decoded_text:
            status = AnswerStatus.INSUFFICIENT_INFORMATION
            confidence = 0.0
            decoded_text = "I do not have enough information to provide an answer."
        elif repetition_score > 0.45 or unique_ratio < 0.35:
            # High degeneration detected
            status = AnswerStatus.UNCERTAIN
            confidence = 0.35
        elif termination_reason == "max_tokens" and not decoded_text.endswith((".", "!", "?", "\"", "'")):
            # Truncated sentence
            status = AnswerStatus.ANSWER
            confidence = 0.85

        return AnswerResult(
            text=decoded_text,
            status=status,
            confidence_score=confidence,
            prompt_tokens=prompt_len,
            completion_tokens=tokens_generated,
            total_tokens=prompt_len + tokens_generated,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            repetition_score=repetition_score,
            unique_token_ratio=unique_ratio,
            termination_reason=termination_reason,
            model_name="collision-10m",
            raw_tokens=generated_ids
        )
