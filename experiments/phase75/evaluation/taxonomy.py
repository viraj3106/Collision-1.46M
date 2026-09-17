import re
from typing import List, Dict, Any

FAILURE_CATEGORIES = [
    "CONTEXT_LOSS",
    "INSTRUCTION_FAILURE",
    "REPETITION",
    "FRAGMENTATION",
    "HALLUCINATION",
    "IRRELEVANCE",
    "PREMATURE_TERMINATION",
    "OVER_GENERATION",
    "TONE_DRIFT",
    "CONTRADICTION"
]

class FailureTaxonomy:
    """Structured failure classification engine for conversational evaluation."""
    
    @staticmethod
    def classify_response(
        prompt: str,
        generated_text: str,
        expected_behavior: str,
        category: str,
        metrics: Dict[str, Any]
    ) -> List[str]:
        failures = []
        text_strip = generated_text.strip()
        words = text_strip.split()
        
        # 1. FRAGMENTATION: very short or broken token endings
        if len(words) < 2 or text_strip.endswith(("...", "the", "a", "and", "or", "of", "in", "to")):
            if len(words) < 3:
                failures.append("FRAGMENTATION")
        
        # 2. PREMATURE_TERMINATION
        if len(words) < 4 and "short" not in category.lower() and "one word" not in prompt.lower():
            if "PREMATURE_TERMINATION" not in failures:
                failures.append("PREMATURE_TERMINATION")
                
        # 3. REPETITION
        rep_ratio = metrics.get("repetition_ratio", 0.0)
        unigram_rep = metrics.get("unigram_repetition", 0.0)
        if rep_ratio > 0.35 or unigram_rep > 0.45:
            failures.append("REPETITION")
            
        # 4. OVER_GENERATION
        if len(words) > 100 and "short" in prompt.lower():
            failures.append("OVER_GENERATION")
            
        # 5. INSTRUCTION_FAILURE
        if "one word" in prompt.lower() and len(words) > 5:
            failures.append("INSTRUCTION_FAILURE")
        if "list 3" in prompt.lower() and not any(char.isdigit() for char in text_strip):
            failures.append("INSTRUCTION_FAILURE")
            
        # 6. CONTEXT_LOSS
        if "context_retention" in category or "follow_up" in category:
            expected_lower = expected_behavior.lower()
            # check key expected entity in prompt if present
            if "name is" in prompt.lower() or "favorite" in prompt.lower():
                context_match = metrics.get("context_retained", True)
                if not context_match:
                    failures.append("CONTEXT_LOSS")
                    
        # 7. IRRELEVANCE
        relevance_score = metrics.get("relevance_score", 1.0)
        if relevance_score < 0.2:
            failures.append("IRRELEVANCE")
            
        # 8. CONTRADICTION
        if "contradiction" in category.lower() and metrics.get("contradiction_detected", False):
            failures.append("CONTRADICTION")
            
        # 9. TONE_DRIFT
        if "tone_consistency" in category.lower() and ("!!!" in text_strip or text_strip.isupper()):
            failures.append("TONE_DRIFT")
            
        # 10. HALLUCINATION
        if "knowledge" in category.lower() and metrics.get("hallucination_detected", False):
            failures.append("HALLUCINATION")

        return sorted(list(set(failures)))

    @staticmethod
    def summarize_failures(results: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {cat: 0 for cat in FAILURE_CATEGORIES}
        for item in results:
            for failure in item.get("failures", []):
                if failure in counts:
                    counts[failure] += 1
        return counts
