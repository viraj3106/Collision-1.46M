from typing import Dict, Any

class ScoringRubric:
    """Standardized 0-4 qualitative scoring rubric for conversational evaluations."""
    
    RUBRIC_SPEC = {
        "coherence": "0 = Incoherent babble; 4 = Perfectly natural, coherent english phrasing",
        "relevance": "0 = Unrelated response; 4 = Directly addresses prompt topic and intent",
        "instruction_following": "0 = Completely ignores prompt constraints; 4 = Fully satisfies all constraints",
        "context_retention": "0 = Forgets previous turns/entities; 4 = Retains and correctly applies multi-turn state",
        "repetition": "0 = Infinite loop / severe repetitive n-grams; 4 = Clean, non-repetitive response",
        "response_quality": "0 = Completely unusable; 4 = High quality, fluent, helpful output"
    }

    @staticmethod
    def score_response(
        prompt: str,
        generated_text: str,
        expected_behavior: str,
        category: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        text_strip = generated_text.strip()
        words = text_strip.split()
        num_words = len(words)
        
        # 1. Repetition Score (0 to 4)
        rep_ratio = metrics.get("repetition_ratio", 0.0)
        if rep_ratio > 0.6:
            rep_score = 0.0
        elif rep_ratio > 0.4:
            rep_score = 1.0
        elif rep_ratio > 0.2:
            rep_score = 2.5
        elif rep_ratio > 0.1:
            rep_score = 3.2
        else:
            rep_score = 4.0
            
        # 2. Coherence Score (0 to 4)
        if num_words < 2:
            coh_score = 0.5
        elif rep_ratio > 0.5:
            coh_score = 1.0
        elif any(c in text_strip for c in ["\n\n\n\n", "???", "!!!"]):
            coh_score = 2.0
        else:
            coh_score = 3.5 if num_words >= 5 else 3.0

        # 3. Relevance Score (0 to 4)
        rel_score = 3.5
        if num_words < 3:
            rel_score = 1.0
        elif rep_ratio > 0.5:
            rel_score = 1.5

        # 4. Instruction Following Score (0 to 4)
        inst_score = 3.5
        if "one word" in prompt.lower() and num_words > 5:
            inst_score = 1.0
        elif "list 3" in prompt.lower() and not any(c.isdigit() for c in text_strip):
            inst_score = 1.5

        # 5. Context Retention Score (0 to 4)
        ctx_score = 3.5
        if ("context_retention" in category or "follow_up" in category):
            if not metrics.get("context_retained", True):
                ctx_score = 0.5
            else:
                ctx_score = 4.0

        # 6. Overall Response Quality (0 to 4)
        quality = (coh_score + rel_score + inst_score + ctx_score + rep_score) / 5.0

        return {
            "coherence": round(coh_score, 2),
            "relevance": round(rel_score, 2),
            "instruction_following": round(inst_score, 2),
            "context_retention": round(ctx_score, 2),
            "repetition": round(rep_score, 2),
            "response_quality": round(quality, 2)
        }
