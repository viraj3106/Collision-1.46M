import math
import re
from typing import Dict, Any, List, Optional

def calculate_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.strip().lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ngrams = set(ngrams)
    repeat_count = len(ngrams) - len(unique_ngrams)
    return repeat_count / float(len(ngrams))

def calculate_unigram_repetition(text: str) -> float:
    words = text.strip().lower().split()
    if not words:
        return 0.0
    unique_words = set(words)
    return 1.0 - (len(unique_words) / float(len(words)))

def check_context_retention(prompt: str, response: str, expected_behavior: str) -> bool:
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    
    # Check key entity extractions from prompt (e.g., "name is Alex" -> "alex")
    if "name is" in prompt_lower:
        match = re.search(r"name is ([a-zA-Z]+)", prompt_lower)
        if match:
            target_name = match.group(1).strip()
            if target_name in response_lower:
                return True
            else:
                return False
                
    if "favorite" in prompt_lower and "is" in prompt_lower:
        match = re.search(r"favorite ([a-zA-Z\s]+) is ([a-zA-Z]+)", prompt_lower)
        if match:
            target_fact = match.group(2).strip()
            if target_fact in response_lower:
                return True
            else:
                return False
                
    # Default check based on expected behavior keywords
    expected_words = [w.lower() for w in expected_behavior.split() if len(w) > 4]
    if expected_words:
        matches = sum(1 for w in expected_words if w in response_lower)
        return matches >= max(1, len(expected_words) // 2)
        
    return True

def calculate_metrics_for_generation(
    prompt: str,
    response: str,
    expected_behavior: str,
    category: str,
    loss: Optional[float] = None
) -> Dict[str, Any]:
    words = response.strip().split()
    token_count = len(words)
    
    tri_rep = calculate_repetition_ratio(response, n=3)
    uni_rep = calculate_unigram_repetition(response)
    ctx_retained = check_context_retention(prompt, response, expected_behavior)
    
    perplexity = math.exp(loss) if (loss is not None and loss < 20) else None
    
    return {
        "generation_length_words": token_count,
        "repetition_ratio": round(tri_rep, 4),
        "unigram_repetition": round(uni_rep, 4),
        "context_retained": ctx_retained,
        "loss": round(loss, 4) if loss is not None else None,
        "perplexity": round(perplexity, 4) if perplexity is not None else None,
        "relevance_score": 1.0 if token_count >= 3 and tri_rep < 0.4 else 0.1
    }
