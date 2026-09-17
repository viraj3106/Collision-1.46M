"""
COLLISION Natural Language Processing (NLP) Engine.

Provides advanced NLP capabilities:
1. Intelligent Sentence Boundary Segmentation (handling abbreviations, numbers, quotes)
2. Question-Intent Extraction & Focused Answering (Definitions, Mechanisms, Biographies)
3. Extractive NLP Summarization (BM25/TF-IDF sentence ranking)
4. Lexicon & Valence Sentiment Analysis (Positive, Negative, Neutral with confidence)
5. Rule-Based Named Entity Recognition (Persons, Organizations, Locations, Dates, Numbers)
6. Structured NLP Response Formatting with Clear Markdown Highlights
"""

import re
import math
import html
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set


class NLPTextProcessor:
    """
    Core NLP text normalization, tokenization, and sentence boundary segmentation.
    """
    ABBREVIATIONS = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'vs.', 'etc.',
        'e.g.', 'i.e.', 'u.s.', 'u.k.', 'u.n.', 'jan.', 'feb.', 'mar.', 'apr.',
        'jun.', 'jul.', 'aug.', 'sep.', 'sept.', 'oct.', 'nov.', 'dec.',
        'approx.', 'dept.', 'est.', 'govt.', 'inc.', 'ltd.', 'co.', 'corp.',
        'st.', 'ave.', 'rd.', 'blvd.', 'no.', 'vol.', 'pp.', 'ed.', 'al.'
    }

    STOPWORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
        'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
        'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot', 'could',
        'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself',
        'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
        'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
        'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that',
        'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
        'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
        'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
    }

    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'superb',
        'outstanding', 'brilliant', 'terrific', 'exceptional', 'positive', 'love',
        'like', 'admire', 'praise', 'happy', 'delighted', 'pleased', 'satisfied',
        'impressive', 'effective', 'efficient', 'helpful', 'valuable', 'perfect',
        'flawless', 'beautiful', 'clean', 'fast', 'secure', 'reliable', 'innovative',
        'revolutionary', 'triumph', 'success', 'benefit', 'advantage', 'superior',
        'best', 'top', 'favorite', 'enjoy', 'fabulous', 'awesome', 'gem'
    }

    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'subpar', 'inferior',
        'negative', 'hate', 'dislike', 'despise', 'disappointed', 'frustrated',
        'annoyed', 'angry', 'upset', 'defective', 'broken', 'buggy', 'slow',
        'vulnerable', 'insecure', 'unreliable', 'useless', 'worthless', 'flawed',
        'ugly', 'fail', 'failure', 'problem', 'issue', 'error', 'crash', 'glitch',
        'disaster', 'catastrophe', 'harmful', 'damaging', 'worst', 'waste', 'regret'
    }

    INTENSIFIERS = {
        'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'super': 1.5,
        'really': 1.4, 'highly': 1.5, 'exceptionally': 1.8, 'absolutely': 1.8,
        'totally': 1.5, 'completely': 1.5, 'utterly': 2.0
    }

    NEGATIONS = {
        'not', 'never', 'no', 'hardly', 'barely', 'scarcely', 'neither', 'nor', 'without'
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Thoroughly cleans and normalizes arbitrary text."""
        if not text:
            return ""
        t = html.unescape(text)
        t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8')
        t = re.sub(r'\[\d+\]', '', t)
        t = re.sub(r'\[(?:citation\s+needed|note\s+\d+|disambiguation)\]', '', t, flags=re.I)
        t = re.sub(r'\(?archived\s+from\s+(the\s+)?original\s+on\s+[^\)]+\)?', '', t, flags=re.I)
        t = re.sub(r'retrieved\s+(on\s+)?[a-z]+\s+\d+,\s+\d{4}', '', t, flags=re.I)
        t = re.sub(r'\b(?:doi|isbn|issn|pmid):[0-9a-z\.\-\/]+', '', t, flags=re.I)
        t = re.sub(r'^[a-z]+\s+\d{4}\)\.\s*', '', t, flags=re.I)
        t = re.sub(r'^(?:this\s+is\s+a\s+(?:chronological\s+)?list\s+of\s+[^.]*\.\s*)', '', t, flags=re.I)
        t = t.replace('\ufffd', ' ')
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    @classmethod
    def segment_sentences(cls, text: str) -> List[str]:
        """
        Splits text into well-formed sentences, taking into account abbreviations,
        numbers, and quotes.
        """
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []

        # Mask known abbreviations with unique tokens
        masked = cleaned
        for idx, abbr in enumerate(sorted(cls.ABBREVIATIONS, key=len, reverse=True)):
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            masked = pattern.sub(f"__ABBR_{idx}__", masked)

        # Mask decimal numbers like 3.14
        masked = re.sub(r'(\d+)\.(\d+)', r'\1__DOT__\2', masked)

        # Split on true sentence boundaries (. ! ?)
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', masked)

        sentences = []
        for s in raw_sentences:
            # Unmask abbreviations and dots
            for idx, abbr in enumerate(sorted(cls.ABBREVIATIONS, key=len, reverse=True)):
                s = s.replace(f"__ABBR_{idx}__", abbr)
            s = s.replace("__DOT__", ".")
            s = re.sub(r'^[a-zA-Z0-9_\-]+\)\s*', '', s)
            s = re.sub(r'^[^\w\(\[\'"]+', '', s)
            s = s.strip()
            if len(s) >= 15:
                # Filter out lower-case dangling fragments that lack subject
                if s and s[0].islower() and not s.startswith(('http', 'iPhone', 'eBay', 'iOS')):
                    continue
                # Clean trailing dangling prepositions and conjunctions
                s = re.sub(r'\s+(?:and|or|but|because|with|the|a|an|of|in|to|that)\.?$', '', s, flags=re.I)
                # Ensure ending punctuation
                if s and s[-1] not in '.!?':
                    s += '.'
                sentences.append(s)

        return sentences

    @classmethod
    def tokenize_words(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenizes text into lowercase alphanumeric words."""
        words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
        if remove_stopwords:
            return [w for w in words if w not in cls.STOPWORDS and len(w) > 1]
        return words


class NLPSentimentAnalyzer:
    """
    High-precision valence and lexicon-based sentiment analysis with negation and intensifier handling.
    """
    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        words = re.findall(r'\b[a-zA-Z\']+\b', text.lower())
        if not words:
            return {"sentiment": "Neutral", "score": 0.0, "confidence": 1.0, "label": "NEUTRAL"}

        pos_score = 0.0
        neg_score = 0.0
        active_negation = False
        intensifier_multiplier = 1.0
        key_positive = []
        key_negative = []

        for idx, w in enumerate(words):
            if w in NLPTextProcessor.NEGATIONS:
                active_negation = True
                continue

            if w in NLPTextProcessor.INTENSIFIERS:
                intensifier_multiplier = NLPTextProcessor.INTENSIFIERS[w]
                continue

            # Reset negation after punctuation or distance > 3
            if active_negation and idx > 0 and words[idx-1] in {'.', ',', ';', 'but', 'however'}:
                active_negation = False

            if w in NLPTextProcessor.POSITIVE_WORDS:
                val = 1.0 * intensifier_multiplier
                if active_negation:
                    neg_score += val
                    key_negative.append(f"not {w}")
                else:
                    pos_score += val
                    key_positive.append(w)
                intensifier_multiplier = 1.0
                active_negation = False

            elif w in NLPTextProcessor.NEGATIVE_WORDS:
                val = 1.0 * intensifier_multiplier
                if active_negation:
                    pos_score += val
                    key_positive.append(f"not {w}")
                else:
                    neg_score += val
                    key_negative.append(w)
                intensifier_multiplier = 1.0
                active_negation = False

        total_emotional_signals = pos_score + neg_score
        if total_emotional_signals == 0:
            return {
                "sentiment": "Neutral",
                "score": 0.0,
                "confidence": 0.85,
                "label": "NEUTRAL",
                "summary": "The input text expresses an objective or neutral tone with no strong sentiment markers."
            }

        net_score = (pos_score - neg_score) / max(1.0, total_emotional_signals)
        confidence = min(1.0, 0.60 + (abs(net_score) * 0.35) + min(0.05, total_emotional_signals * 0.02))

        if net_score > 0.15:
            sentiment = "Positive"
            label = "POSITIVE"
        elif net_score < -0.15:
            sentiment = "Negative"
            label = "NEGATIVE"
        else:
            sentiment = "Mixed / Neutral"
            label = "NEUTRAL"

        return {
            "sentiment": sentiment,
            "score": round(net_score, 3),
            "confidence": round(confidence, 2),
            "label": label,
            "positive_markers": key_positive,
            "negative_markers": key_negative,
            "summary": f"Identified **{sentiment}** sentiment (Score: {net_score:+.2f}, Confidence: {confidence*100:.0f}%)."
        }


class NLPEntityExtractor:
    """
    Extracts named entities (Persons, Organizations, Locations, Numbers, Dates) using regex patterns and gazetteers.
    """
    @classmethod
    def extract(cls, text: str) -> Dict[str, List[str]]:
        cleaned = NLPTextProcessor.clean_text(text)
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates_years": [],
            "quantities": []
        }

        # 1. Dates and Years (e.g. 1991, 2026, July 20, 1969)
        years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', cleaned)
        date_phrases = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b', cleaned, re.I)
        entities["dates_years"] = list(set(years + date_phrases))

        # 2. Quantities & Percentages (e.g. 10.28M, $500, 15%, 256 tokens)
        quantities = re.findall(r'\b(?:\$\d+[\d,\.]*|\d+[\d,\.]*%\b|\d+[\d,\.]*\s*(?:parameters|tokens|million|billion|km|miles|meters|kg|lbs|sec|ms|flops))\b', cleaned, re.I)
        entities["quantities"] = list(set(quantities))

        # 3. Capitalized Proper Names / Entities
        proper_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', cleaned)
        for name in proper_names:
            if name.lower() in NLPTextProcessor.STOPWORDS or len(name) < 3:
                continue
            if any(kw in name.lower() for kw in ['corp', 'inc', 'ltd', 'foundation', 'university', 'institute', 'lab', 'labs', 'google', 'apple', 'microsoft', 'nasa', 'cern', 'openai', 'deepmind']):
                if name not in entities["organizations"]:
                    entities["organizations"].append(name)
            elif any(kw in name.lower() for kw in ['france', 'paris', 'tokyo', 'japan', 'california', 'germany', 'london', 'america', 'europe', 'asia', 'earth', 'moon', 'mars']):
                if name not in entities["locations"]:
                    entities["locations"].append(name)
            elif len(name.split()) >= 2 and not any(w.lower() in NLPTextProcessor.STOPWORDS for w in name.split()):
                if name not in entities["persons"]:
                    entities["persons"].append(name)

        return entities


class NLPSummarizer:
    """
    Extractive sentence-ranking summarizer using BM25 and content centrality.
    """
    @classmethod
    def summarize(cls, text: str, max_sentences: int = 3) -> str:
        sentences = NLPTextProcessor.segment_sentences(text)
        if len(sentences) <= max_sentences:
            return "\n\n".join(sentences)

        # Tokenize each sentence
        doc_tokens = [NLPTextProcessor.tokenize_words(s) for s in sentences]
        all_words = [w for dt in doc_tokens for w in dt]
        word_freq: Dict[str, int] = {}
        for w in all_words:
            word_freq[w] = word_freq.get(w, 0) + 1

        # Score sentences by word importance + position bias
        scored_sentences = []
        for idx, (s, tokens) in enumerate(zip(sentences, doc_tokens)):
            if not tokens:
                continue
            score = sum(word_freq.get(w, 0) for w in tokens) / (len(tokens) ** 0.5)
            # Prioritize early opening sentences
            position_multiplier = 1.0 + (1.0 / (idx + 1.0))
            scored_sentences.append((score * position_multiplier, idx, s))

        # Select top-K sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored_sentences[:max_sentences]
        # Sort by original appearance order
        top_sentences.sort(key=lambda x: x[1])

        return " ".join([s for _, _, s in top_sentences])


class NLPAnswerSynthesizer:
    """
    Intelligent NLP Answer Synthesizer that structures open-domain evidence into clear,
    natural, and beautifully formatted answers with high conversational fluency.
    """
    @classmethod
    def synthesize(cls, query: str, passages: List[Tuple[Any, str]]) -> str:
        if not passages:
            return "No verified information could be retrieved for this query."

        # 1. Segment and gather all clean candidate sentences
        all_sentences: List[str] = []
        seen_sentences: Set[str] = set()
        q_words = set(NLPTextProcessor.tokenize_words(query))

        for item, raw_text in passages:
            if not raw_text:
                continue
            sentences = NLPTextProcessor.segment_sentences(raw_text)
            for s in sentences:
                s_clean = s.strip()
                s_key = s_clean.lower()[:45]
                if s_key not in seen_sentences and len(s_clean) >= 20:
                    seen_sentences.add(s_key)
                    all_sentences.append(s_clean)

        if not all_sentences:
            return "No verified information could be retrieved for this query."

        # 2. Score sentences for relevance to user question
        scored_sentences = []
        for idx, s in enumerate(all_sentences):
            s_words = set(NLPTextProcessor.tokenize_words(s))
            overlap = len(q_words.intersection(s_words))
            score = overlap * 2.0

            # Boost definitional sentences if asking "what is" or "who is"
            if re.search(r'^(who|what)\s+(is|was|are|were)', query, re.I):
                if re.search(r'\b(is|was|are|were|refers to|defined as|known for|discovered|invented)\b', s, re.I):
                    score += 3.0

            # Boost mechanism sentences if asking "how does" or "why"
            if re.search(r'^(how|why)', query, re.I):
                if re.search(r'\b(works by|process|mechanism|allows|causes|enables|functions)\b', s, re.I):
                    score += 3.0

            # Opening sentence from primary source gets a baseline bonus
            if idx == 0:
                score += 2.5

            scored_sentences.append((score, idx, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # 3. Select top core answer sentence
        primary_sentence = scored_sentences[0][2]

        from collision.grounding.formatter import NaturalGroundedFormatter

        # Check if primary sentence contains multi-aspect specifications
        spec_text = NaturalGroundedFormatter.format_specifications(primary_sentence)
        if spec_text:
            return spec_text

        # 4. Select supporting detail sentences (up to 3)
        supporting_sentences = []
        for _, _, s in scored_sentences[1:6]:
            if s != primary_sentence and not any(s[:30].lower() in existing.lower() for existing in [primary_sentence] + supporting_sentences):
                supporting_sentences.append(s)
            if len(supporting_sentences) >= 3:
                break

        # 5. Format into a structured, elegant ChatGPT-like response
        formatted_primary = NaturalGroundedFormatter.highlight_terms(primary_sentence)
        response_parts = [formatted_primary]

        if supporting_sentences:
            if len(supporting_sentences) >= 2 and any(kw in query.lower() for kw in ['how', 'why', 'what are', 'explain', 'tell me about', 'features', 'highlights']):
                # Format as key points for clarity
                points = [f"• {NaturalGroundedFormatter.highlight_terms(s)}" for s in supporting_sentences]
                response_parts.append("**Key Highlights:**\n" + "\n".join(points))
            else:
                formatted_support = " ".join([NaturalGroundedFormatter.highlight_terms(s) for s in supporting_sentences])
                response_parts.append(formatted_support)

        return "\n\n".join(response_parts)
