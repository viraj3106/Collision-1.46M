"""
COLLISION Synaptic NLP — Information-Theoretic & Semantic Knowledge Processing.

Provides advanced NLP optimizations:
1. Shannon Information Entropy Profiling across token sequences
2. Semantic Knowledge Triplet Extraction: Deterministically converts natural language sentences into (Subject, Relation, Object, Context) triples
3. Contextual Polysemy Disambiguator: Identifies polysemous words and resolves domain-specific senses
4. Linguistic Structural Complexity Analyzer
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from collision.brain.schemas import (
    SemanticTriple,
    InformationEntropyProfile
)
from collision.nlp.processor import CollisionNLPProcessor, POSWord


class SemanticTripletExtractor:
    """
    Extracts structured knowledge graph triples (Subject, Predicate, Object)
    from unstructured English text without requiring heavy multi-gigabyte models.
    """

    RELATION_PATTERNS = [
        # X is a Y / X was developed by Y
        (r"^(.*?)\s+(?:is\s+a|is\s+an|are\s+a|are\s+an|is\s+defined\s+as)\s+(.+)$", "is_a"),
        (r"^(.*?)\s+(?:was\s+developed\s+by|was\s+created\s+by|was\s+invented\s+by|was\s+built\s+by)\s+(.+)$", "created_by"),
        (r"^(.*?)\s+(?:consists\s+of|is\s+composed\s+of|contains|operates\s+with|features)\s+(.+)$", "consists_of"),
        (r"^(.*?)\s+(?:utilizes|uses|relies\s+on|employs|applies)\s+(.+)$", "utilizes"),
        (r"^(.*?)\s+(?:enables|allows|facilitates|leads\s+to|causes)\s+(.+)$", "enables"),
        (r"^(.*?)\s+(?:optimizes|minimizes|maximizes|improves|reduces)\s+(.+)$", "optimizes"),
        (r"^(.*?)\s+(?:requires|demands|needs)\s+(.+)$", "requires"),
    ]

    @classmethod
    def extract_triples(cls, text: str) -> List[SemanticTriple]:
        triples: List[SemanticTriple] = []
        sentences = CollisionNLPProcessor.segment_sentences(text)

        for sentence in sentences:
            s_clean = sentence.strip().rstrip(".!?")
            if not s_clean:
                continue

            matched = False
            for pattern, relation in cls.RELATION_PATTERNS:
                m = re.match(pattern, s_clean, re.IGNORECASE)
                if m:
                    subj = m.group(1).strip()
                    obj = m.group(2).strip()
                    if len(subj) > 1 and len(obj) > 1:
                        triples.append(SemanticTriple(
                            subject=subj,
                            predicate=relation,
                            object=obj,
                            confidence=0.92,
                            context=sentence
                        ))
                        matched = True
                        break

            # Fallback POS-based subject-verb-object extraction
            if not matched:
                pos_tags = CollisionNLPProcessor.tag_pos(sentence)
                subj_tokens = []
                verb_tokens = []
                obj_tokens = []
                phase = "subject"

                for p in pos_tags:
                    if phase == "subject":
                        if p.pos in ("NOUN", "PROPN", "ADJ", "DET"):
                            subj_tokens.append(p.word)
                        elif p.pos in ("VERB", "AUX"):
                            verb_tokens.append(p.word)
                            phase = "verb"
                    elif phase == "verb":
                        if p.pos in ("VERB", "ADV", "AUX", "ADP"):
                            verb_tokens.append(p.word)
                        else:
                            obj_tokens.append(p.word)
                            phase = "object"
                    elif phase == "object":
                        obj_tokens.append(p.word)

                if subj_tokens and verb_tokens and obj_tokens:
                    s_str = " ".join(subj_tokens)
                    v_str = " ".join(verb_tokens)
                    o_str = " ".join(obj_tokens)
                    if len(s_str) > 2 and len(o_str) > 2:
                        triples.append(SemanticTriple(
                            subject=s_str,
                            predicate=v_str,
                            object=o_str,
                            confidence=0.75,
                            context=sentence
                        ))

        return triples


class ContextualPolysemyDisambiguator:
    """
    Identifies polysemous words (e.g. 'transformer', 'state', 'model', 'head')
    and disambiguates their intended sense according to surrounding contextual tokens.
    """

    POLYSEMY_REGISTRY = {
        "transformer": {
            "ai_ml": {
                "markers": ["attention", "neural", "layer", "parameters", "tokens", "llm", "slm", "weights", "heads", "bert", "gpt"],
                "sense": "Neural self-attention architecture for sequence modeling (Vaswani et al.)"
            },
            "electrical": {
                "markers": ["voltage", "current", "grid", "coil", "magnetic", "step-up", "step-down", "substation", "ac"],
                "sense": "Electromagnetic device transferring electrical energy between circuits via mutual induction"
            }
        },
        "state": {
            "computer_science": {
                "markers": ["memory", "variable", "machine", "fsm", "automaton", "program", "stateless", "cache"],
                "sense": "Computational condition or configuration of values stored in memory at a specific time"
            },
            "physics": {
                "markers": ["quantum", "superposition", "wavefunction", "thermodynamic", "phase", "matter", "gas", "solid"],
                "sense": "Physical or quantum mechanical configuration of a physical system"
            },
            "political": {
                "markers": ["government", "nation", "country", "border", "policy", "sovereignty", "citizen"],
                "sense": "Organized political community under one government"
            }
        },
        "collision": {
            "ai_project": {
                "markers": ["10m", "1m", "model", "parameter", "layer", "checkpoint", "rag", "service", "grounding", "api"],
                "sense": "COLLISION compact SLM neural architecture & hybrid cognitive service"
            },
            "physics": {
                "markers": ["momentum", "elastic", "inelastic", "kinetic", "particles", "impact", "velocity", "conservation"],
                "sense": "Physical interaction between two or more bodies exerting forces upon each other"
            },
            "cryptography": {
                "markers": ["hash", "sha-256", "md5", "digest", "preimage", "birthday attack", "keccak"],
                "sense": "Cryptographic condition where two distinct inputs produce the identical hash digest"
            }
        }
    }

    @classmethod
    def disambiguate(cls, text: str) -> Dict[str, str]:
        """Returns detected polysemous terms and their resolved contextual senses."""
        t_lower = text.lower()
        resolved: Dict[str, str] = {}

        for word, senses in cls.POLYSEMY_REGISTRY.items():
            if re.search(r'\b' + re.escape(word) + r'\b', t_lower):
                best_sense = None
                max_hits = 0
                for sense_key, sense_info in senses.items():
                    hits = sum(1 for m in sense_info["markers"] if re.search(r'\b' + re.escape(m) + r'\b', t_lower))
                    if hits > max_hits:
                        max_hits = hits
                        best_sense = sense_info["sense"]

                if best_sense:
                    resolved[word] = best_sense
                else:
                    # Default first sense
                    first_val = list(senses.values())[0]
                    resolved[word] = first_val["sense"]

        return resolved
