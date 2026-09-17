"""
COLLISION NLP Comprehension — Reading Comprehension (Context QA) and Text Transformation Engines.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field

from collision.nlp.processor import CollisionNLPProcessor


class ContextQAResult(BaseModel):
    answer: str
    sentence: str
    confidence: float
    start_char: int
    end_char: int
    context_found: bool


class TextTransformationResult(BaseModel):
    transformation_type: str
    original_text: str
    transformed_text: str


class ReadingComprehensionEngine:
    """
    Extractive context-grounded question answering (SQuAD-style span & sentence extraction).
    """
    @classmethod
    def answer_question(cls, context: str, question: str) -> ContextQAResult:
        c_clean = CollisionNLPProcessor.clean_text(context)
        q_clean = question.strip()

        if not c_clean or not q_clean:
            return ContextQAResult(
                answer="Insufficient context or question provided.",
                sentence="",
                confidence=0.0,
                start_char=0,
                end_char=0,
                context_found=False
            )

        sentences = CollisionNLPProcessor.segment_sentences(c_clean)
        if not sentences:
            sentences = [c_clean]

        q_tokens = set(CollisionNLPProcessor.tokenize_words(q_clean, remove_stopwords=True))
        q_lower = q_clean.lower()

        # Identify question intent target
        is_who = bool(re.search(r'\b(who|whose|whom)\b', q_lower))
        is_when = bool(re.search(r'\b(when|what year|what date|what time)\b', q_lower))
        is_where = bool(re.search(r'\b(where|what location|which place|which country|which city)\b', q_lower))
        is_quantity = bool(re.search(r'\b(how many|how much|what percentage|how long|how far)\b', q_lower))
        is_why_how = bool(re.search(r'\b(why|how does|how do|how is|how can)\b', q_lower))

        scored_sentences: List[Tuple[float, int, str]] = []

        for idx, s in enumerate(sentences):
            s_words = set(CollisionNLPProcessor.tokenize_words(s, remove_stopwords=True))
            overlap = len(q_tokens.intersection(s_words))
            score = overlap * 3.0

            # Target type weighting
            if is_who and re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', s):
                score += 2.5
            if is_when and re.search(r'\b(1[89]\d{2}|20\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b', s, re.I):
                score += 3.0
            if is_where and any(kw in s.lower() for kw in ['in', 'at', 'city', 'country', 'region', 'state', 'university', 'located']):
                score += 2.0
            if is_quantity and re.search(r'\b(\$\d+|\d+[\d,\.]*|\d+%\b|\d+\s*(?:million|billion|km|miles|meters|kg|years))\b', s, re.I):
                score += 3.0
            if is_why_how and any(kw in s.lower() for kw in ['because', 'by', 'through', 'due to', 'mechanism', 'process', 'causes']):
                score += 2.5

            if idx == 0:
                score += 0.5

            scored_sentences.append((score, idx, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_sent = scored_sentences[0]

        if best_score <= 0.5 and not q_tokens.intersection(set(CollisionNLPProcessor.tokenize_words(best_sent))):
            return ContextQAResult(
                answer="The answer could not be determined conclusively from the provided context.",
                sentence="",
                confidence=0.20,
                start_char=0,
                end_char=0,
                context_found=False
            )

        start_pos = c_clean.find(best_sent)
        end_pos = start_pos + len(best_sent) if start_pos != -1 else 0

        # Refine span if exact clause is obvious
        extracted_answer = best_sent

        conf = min(0.99, max(0.60, (best_score / (len(q_tokens) * 3.0 + 1.0)) * 0.40 + 0.55))

        return ContextQAResult(
            answer=extracted_answer,
            sentence=best_sent,
            confidence=round(conf, 2),
            start_char=max(0, start_pos),
            end_char=max(0, end_pos),
            context_found=True
        )


class TextTransformer:
    """
    Style, formality, simplification, and structure text transformations.
    """
    INFORMAL_TO_FORMAL = {
        r'\bkinda\b': 'somewhat',
        r'\bsorta\b': 'rather',
        r'\bgonna\b': 'going to',
        r'\bwanna\b': 'wish to',
        r'\bgotta\b': 'must',
        r'\byep\b': 'yes',
        r'\bnope\b': 'no',
        r'\blol\b': '',
        r'\bomg\b': '',
        r'\bhlo\b': 'hello',
        r'\bhey\b': 'hello',
        r'\bbro\b': 'colleague',
        r'\bbuddy\b': 'associate',
        r'\bdude\b': 'individual',
        r'\bawesome\b': 'exemplary',
        r'\bcool\b': 'favorable',
        r'\bbad\b': 'suboptimal',
        r'\bget\b': 'obtain',
        r'\bbig\b': 'substantial',
        r'\bfix\b': 'rectify',
        r'\bshow\b': 'demonstrate'
    }

    COMPLEX_TO_SIMPLE = {
        r'\butilize\b': 'use',
        r'\bcommence\b': 'start',
        r'\bterminate\b': 'end',
        r'\bfacilitate\b': 'help',
        r'\bimplement\b': 'apply',
        r'\bsubsequently\b': 'later',
        r'\bconsequently\b': 'so',
        r'\bnevertheless\b': 'still',
        r'\baforementioned\b': 'previous',
        r'\bsubstantial\b': 'large',
        r'\bdemonstrates\b': 'shows',
        r'\bexhibits\b': 'shows',
        r'\belucidate\b': 'explain',
        r'\bparadigm\b': 'model',
        r'\bmethodology\b': 'method'
    }

    @classmethod
    def formalize(cls, text: str) -> TextTransformationResult:
        """Transforms informal or colloquial text into formal prose."""
        transformed = text
        for pat, repl in cls.INFORMAL_TO_FORMAL.items():
            transformed = re.sub(pat, repl, transformed, flags=re.I)

        transformed = re.sub(r'\s+', ' ', transformed).strip()
        if transformed and transformed[-1] not in '.!?':
            transformed += '.'
        if transformed and transformed[0].islower():
            transformed = transformed[0].upper() + transformed[1:]

        return TextTransformationResult(
            transformation_type="Formalize",
            original_text=text,
            transformed_text=transformed
        )

    @classmethod
    def simplify(cls, text: str) -> TextTransformationResult:
        """Simplifies complex or academic text into plain English."""
        transformed = text
        for pat, repl in cls.COMPLEX_TO_SIMPLE.items():
            transformed = re.sub(pat, repl, transformed, flags=re.I)

        transformed = re.sub(r'\s+', ' ', transformed).strip()
        return TextTransformationResult(
            transformation_type="Simplify",
            original_text=text,
            transformed_text=transformed
        )

    @classmethod
    def bulletize(cls, text: str) -> TextTransformationResult:
        """Converts paragraph text into concise structured bullet points."""
        sentences = CollisionNLPProcessor.segment_sentences(text)
        if not sentences:
            sentences = [text.strip()]

        bullets = [f"• {s.strip()}" for s in sentences if len(s.strip()) > 5]
        return TextTransformationResult(
            transformation_type="Bulletize",
            original_text=text,
            transformed_text="\n".join(bullets)
        )
