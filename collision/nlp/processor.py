"""
COLLISION NLP Processor — Industrial-Strength Natural Language Processing Algorithms.

Provides high-performance text normalization, sentence boundary detection, tokenization,
morphological lemmatization, part-of-speech (POS) tagging, readability metrics,
phonetic encoding, edit distances, and language detection.
"""

import re
import math
import html
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field


class POSWord(BaseModel):
    word: str
    lemma: str
    pos: str
    tag: str


class ReadabilityMetrics(BaseModel):
    flesch_reading_ease: float
    reading_ease_description: str
    flesch_kincaid_grade: float
    gunning_fog_index: float
    lexical_diversity_ttr: float
    word_count: int
    sentence_count: int
    syllable_count: int
    avg_words_per_sentence: float
    avg_syllables_per_word: float


class LanguageDetectionResult(BaseModel):
    language: str
    language_code: str
    confidence: float
    top_candidates: List[Tuple[str, float]] = Field(default_factory=list)


class CollisionNLPProcessor:
    """
    Core algorithmic processor for text segmentation, tokenization, morphological lemmatization,
    POS tagging, readability scoring, and language detection.
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
        'best', 'top', 'favorite', 'enjoy', 'fabulous', 'awesome', 'gem', 'buddy', 'friend'
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

    # Irregular Lemmatization mapping
    IRREGULAR_LEMMAS = {
        'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
        'has': 'have', 'had': 'have', 'having': 'have',
        'does': 'do', 'did': 'do', 'doing': 'do', 'done': 'do',
        'goes': 'go', 'went': 'go', 'going': 'go', 'gone': 'go',
        'says': 'say', 'said': 'say', 'saying': 'say',
        'makes': 'make', 'made': 'make', 'making': 'make',
        'knows': 'know', 'knew': 'know', 'known': 'know', 'knowing': 'know',
        'thinks': 'think', 'thought': 'think', 'thinking': 'think',
        'takes': 'take', 'took': 'take', 'taken': 'take', 'taking': 'take',
        'sees': 'see', 'saw': 'see', 'seen': 'see', 'seeing': 'see',
        'comes': 'come', 'came': 'come', 'coming': 'come',
        'finds': 'find', 'found': 'find', 'finding': 'find',
        'gives': 'give', 'gave': 'give', 'given': 'give', 'giving': 'give',
        'tells': 'tell', 'told': 'tell', 'telling': 'tell',
        'becomes': 'become', 'became': 'become', 'becoming': 'become',
        'leaves': 'leave', 'left': 'leave', 'leaving': 'leave',
        'feels': 'feel', 'felt': 'feel', 'feeling': 'feel',
        'children': 'child', 'people': 'person', 'men': 'man', 'women': 'woman',
        'mice': 'mouse', 'teeth': 'tooth', 'feet': 'foot', 'geese': 'goose',
        'criteria': 'criterion', 'phenomena': 'phenomenon', 'data': 'datum',
        'better': 'good', 'best': 'good', 'worse': 'bad', 'worst': 'bad',
        'faster': 'fast', 'fastest': 'fast', 'easier': 'easy', 'easiest': 'easy'
    }

    # Lexicon mappings for POS tagging
    PRONOUNS = {
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
        'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what'
    }

    PREPOSITIONS = {
        'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'under', 'over', 'of', 'off', 'around', 'near', 'since', 'without', 'within'
    }

    CONJUNCTIONS = {
        'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'although', 'because', 'since',
        'unless', 'while', 'whereas', 'if', 'even', 'though', 'whether', 'as'
    }

    DETERMINERS = {
        'the', 'a', 'an', 'every', 'each', 'either', 'neither', 'some', 'any', 'no',
        'all', 'both', 'half', 'many', 'much', 'several', 'few', 'another', 'such'
    }

    COMMON_VERBS = {
        'be', 'have', 'do', 'say', 'go', 'get', 'make', 'know', 'think', 'take',
        'see', 'come', 'want', 'look', 'use', 'find', 'give', 'tell', 'work',
        'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean',
        'keep', 'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start', 'show',
        'hear', 'play', 'run', 'move', 'like', 'live', 'believe', 'hold', 'bring',
        'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
        'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow',
        'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open',
        'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy',
        'wait', 'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut',
        'reach', 'kill', 'remain', 'solve', 'compute', 'train', 'process', 'generate'
    }

    COMMON_ADJECTIVES = {
        'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
        'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
        'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able',
        'fast', 'slow', 'quick', 'strong', 'weak', 'hard', 'soft', 'clean', 'dirty',
        'easy', 'difficult', 'simple', 'complex', 'smart', 'intelligent', 'bright',
        'dark', 'hot', 'cold', 'deep', 'shallow', 'wide', 'narrow', 'rich', 'poor',
        'powerful', 'accurate', 'precise', 'efficient', 'robust', 'stable', 'modern'
    }

    # Language trigrams / frequent words profiles
    LANG_PROFILES = {
        'en': {'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'was', 'for', 'it', 'with', 'as', 'on', 'be', 'at'},
        'es': {'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'una'},
        'fr': {'de', 'la', 'le', 'et', 'les', 'des', 'en', 'un', 'du', 'une', 'que', 'est', 'pour', 'qui', 'dans', 'a'},
        'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem'},
        'it': {'di', 'e', 'il', 'la', 'che', 'in', 'del', 'per', 'un', 'i', 'della', 'si', 'le', 'non', 'da', 'dei'},
        'pt': {'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'nao', 'uma', 'os', 'no', 'se'},
        'hi': {'hai', 'mein', 'ko', 'aur', 'se', 'ki', 'ka', 'ke', 'yeh', 'kya', 'bhi', 'tha', 'thi', 'kar', 'ho', 'nahin'},
        'la': {'et', 'in', 'ad', 'non', 'est', 'cum', 'ut', 'de', 'qui', 'per', 'sed', 'ab', 'ex', 'te', 'quam', 'se'}
    }

    LANG_NAMES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'hi': 'Hindi / Hinglish',
        'la': 'Latin'
    }

    PAST_TENSE_VERBS = {
        'wrote', 'went', 'saw', 'ate', 'built', 'ran', 'held', 'spoke', 'drove', 'flew',
        'grew', 'knew', 'rose', 'sang', 'took', 'wore', 'won', 'bought', 'brought',
        'caught', 'felt', 'found', 'heard', 'kept', 'left', 'lost', 'made', 'paid',
        'read', 'said', 'sent', 'sold', 'spent', 'stood', 'told', 'thought', 'understood',
        'began', 'became', 'came', 'gave', 'chose', 'fell', 'got', 'hit', 'led', 'met', 'put', 'set'
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalizes and decodes text, preserving punctuation and sentence boundaries."""
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
        """Splits text into coherent sentences with exact abbreviation preservation."""
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []

        abbr_map: Dict[str, str] = {}
        masked = cleaned
        for idx, abbr in enumerate(sorted(cls.ABBREVIATIONS, key=len, reverse=True)):
            def _repl(m, i=idx):
                tok = f"__ABBR_{i}__"
                abbr_map[tok] = m.group(0)
                return tok
            masked = re.sub(re.escape(abbr), _repl, masked, flags=re.IGNORECASE)

        masked = re.sub(r'(\d+)\.(\d+)', r'\1__DOT__\2', masked)
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', masked)

        sentences = []
        for s in raw_sentences:
            for tok, orig in abbr_map.items():
                s = s.replace(tok, orig)
            s = s.replace("__DOT__", ".")
            s = s.strip()
            if len(s) >= 5:
                if s and s[-1] not in '.!?':
                    s += '.'
                sentences.append(s)

        return sentences

    @classmethod
    def tokenize_words(cls, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenizes text into lowercase words."""
        words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
        if remove_stopwords:
            return [w for w in words if w not in cls.STOPWORDS and len(w) > 1]
        return words

    @classmethod
    def lemmatize_word(cls, word: str) -> str:
        """
        Morphological rule-based lemmatizer mapping inflections to base lemma.
        """
        w = word.lower().strip()
        if not w:
            return ""

        # Check irregular lookup
        if w in cls.IRREGULAR_LEMMAS:
            return cls.IRREGULAR_LEMMAS[w]

        # Suffix stripping rules
        if len(w) > 4:
            if w.endswith('ies') and len(w) > 4:
                return w[:-3] + 'y'
            if w.endswith('ying'):
                return w[:-4] + 'ie'
            if w.endswith('ing'):
                base = w[:-3]
                if len(base) > 2 and base[-1] == base[-2] and base[-1] not in 'lsz':
                    return base[:-1]
                if base.endswith(('at', 'iz', 'is', 'id', 'iv', 'ut', 'or', 'ac', 'us')):
                    return base + 'e'
                if len(base) >= 3 and not base.endswith('e'):
                    return base
                return base + 'e' if len(base) >= 3 else w
            if w.endswith('ed'):
                base = w[:-2]
                if len(base) > 2 and base[-1] == base[-2] and base[-1] not in 'lsz':
                    return base[:-1]
                if base.endswith('i'):
                    return base[:-1] + 'y'
                if base.endswith(('at', 'iz', 'is', 'id', 'iv', 'ut', 'or', 'ac', 'us')):
                    return base + 'e'
                return base if base.endswith('e') or len(base) >= 4 else w
            if w.endswith('sses'):
                return w[:-2]
            if w.endswith('ches') or w.endswith('shes') or w.endswith('xes'):
                return w[:-2]
            if w.endswith('s') and not w.endswith('ss') and not w.endswith('us') and not w.endswith('is'):
                return w[:-1]
            if w.endswith('ly') and len(w) > 4:
                return w[:-2]
            if w.endswith('ment') and len(w) > 6:
                return w[:-4]
            if w.endswith('tion') and len(w) > 6:
                return w[:-4] + 'te'
            if w.endswith('able') and len(w) > 6:
                return w[:-4]

        return w

    @classmethod
    def tag_pos(cls, text: str) -> List[POSWord]:
        """
        Fast rule-based + affix POS tagger.
        """
        tokens = re.findall(r'\b[a-zA-Z0-9_\-\']+\b', text)
        tagged: List[POSWord] = []

        for idx, token in enumerate(tokens):
            w_lower = token.lower()
            lemma = cls.lemmatize_word(w_lower)

            # Number
            if re.match(r'^\d+(\.\d+)?%?$', token):
                pos, tag = 'NUM', 'CD'
            # Pronoun
            elif w_lower in cls.PRONOUNS:
                pos, tag = 'PRON', 'PRP'
            # Determiner
            elif w_lower in cls.DETERMINERS:
                pos, tag = 'DET', 'DT'
            # Preposition
            elif w_lower in cls.PREPOSITIONS:
                pos, tag = 'PREP', 'IN'
            # Conjunction
            elif w_lower in cls.CONJUNCTIONS:
                pos, tag = 'CONJ', 'CC'
            # Adverb
            elif w_lower.endswith('ly') or w_lower in {'now', 'then', 'here', 'there', 'always', 'never', 'very', 'quite', 'too', 'also', 'quickly', 'slowly', 'well'}:
                pos, tag = 'ADV', 'RB'
            # Adjective
            elif (
                w_lower in cls.COMMON_ADJECTIVES or
                w_lower.endswith(('able', 'ible', 'al', 'ful', 'ic', 'ish', 'ive', 'less', 'ous', 'est', 'er', 'ary'))
            ):
                pos, tag = 'ADJ', 'JJ'
            # Verb
            elif (
                w_lower in cls.COMMON_VERBS or
                w_lower in cls.PAST_TENSE_VERBS or
                lemma in cls.COMMON_VERBS or
                w_lower.endswith(('ing', 'ed', 'ize', 'ise', 'ate', 'ify')) or
                (idx > 0 and tagged[-1].pos in ('PRON', 'NOUN', 'ADV') and not w_lower.endswith(('tion', 'ment', 'ity', 'ness', 'ware')))
            ):
                pos, tag = 'VERB', 'VB'
            # Proper Noun
            elif token[0].isupper() and idx > 0:
                pos, tag = 'NOUN', 'NNP'
            # General Noun
            else:
                pos, tag = 'NOUN', 'NN'

            tagged.append(POSWord(word=token, lemma=lemma, pos=pos, tag=tag))

        return tagged

    @classmethod
    def count_syllables(cls, word: str) -> int:
        """Estimates syllable count in an English word."""
        w = word.lower().strip()
        if len(w) <= 3:
            return 1
        w = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', w)
        w = re.sub(r'^y', '', w)
        syllables = len(re.findall(r'[aeiouy]{1,2}', w))
        return max(1, syllables)

    @classmethod
    def compute_readability(cls, text: str) -> ReadabilityMetrics:
        """
        Calculates standard NLP readability statistics:
        - Flesch Reading Ease (0-100)
        - Flesch-Kincaid Grade Level
        - Gunning Fog Index
        - Lexical Diversity / Type-Token Ratio (TTR)
        """
        cleaned = cls.clean_text(text)
        sentences = cls.segment_sentences(cleaned) or [cleaned]
        words = cls.tokenize_words(cleaned, remove_stopwords=False)

        num_sentences = max(1, len(sentences))
        num_words = max(1, len(words))
        num_syllables = sum(cls.count_syllables(w) for w in words)
        unique_words = len(set(w.lower() for w in words))

        avg_words_per_sentence = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words

        # Flesch Reading Ease Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0.0, min(100.0, flesch_score))

        # Flesch-Kincaid Grade Level Formula: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
        fk_grade = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
        fk_grade = max(0.0, fk_grade)

        # Complex words (3+ syllables)
        complex_words = sum(1 for w in words if cls.count_syllables(w) >= 3 and not w.endswith(('ed', 'es', 'ing')))
        pct_complex = (complex_words / num_words) * 100.0
        # Gunning Fog Formula: 0.4 * ( (words/sentences) + 100*(complex_words/words) )
        fog_index = 0.4 * (avg_words_per_sentence + pct_complex)

        # Lexical diversity (Type-Token Ratio)
        ttr = unique_words / num_words

        # Qualitative assessment
        if flesch_score >= 90:
            desc = "Very Easy (5th grade level)"
        elif flesch_score >= 80:
            desc = "Easy (6th grade level)"
        elif flesch_score >= 70:
            desc = "Fairly Easy (7th grade level)"
        elif flesch_score >= 60:
            desc = "Standard / Plain English (8th-9th grade level)"
        elif flesch_score >= 50:
            desc = "Fairly Difficult (10th-12th grade level)"
        elif flesch_score >= 30:
            desc = "Difficult (College level)"
        else:
            desc = "Very Difficult / Academic (Graduate level)"

        return ReadabilityMetrics(
            flesch_reading_ease=round(flesch_score, 2),
            reading_ease_description=desc,
            flesch_kincaid_grade=round(fk_grade, 2),
            gunning_fog_index=round(fog_index, 2),
            lexical_diversity_ttr=round(ttr, 3),
            word_count=num_words,
            sentence_count=num_sentences,
            syllable_count=num_syllables,
            avg_words_per_sentence=round(avg_words_per_sentence, 2),
            avg_syllables_per_word=round(avg_syllables_per_word, 2)
        )

    @classmethod
    def detect_language(cls, text: str) -> LanguageDetectionResult:
        """
        N-Gram & frequency distribution language detector across major languages.
        """
        tokens = set(cls.tokenize_words(text, remove_stopwords=False))
        if not tokens:
            return LanguageDetectionResult(language="English", language_code="en", confidence=0.95)

        scores: Dict[str, float] = {}
        for code, profile in cls.LANG_PROFILES.items():
            matches = tokens.intersection(profile)
            scores[code] = len(matches) / max(1, len(profile))

        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_code, top_score = sorted_langs[0]

        if top_score == 0:
            top_code = 'en'
            conf = 0.85
        else:
            conf = min(0.99, max(0.75, 0.70 + (top_score * 0.30)))

        top_candidates = [(cls.LANG_NAMES.get(c, c), round(s, 3)) for c, s in sorted_langs[:3]]

        return LanguageDetectionResult(
            language=cls.LANG_NAMES.get(top_code, "English"),
            language_code=top_code,
            confidence=round(conf, 2),
            top_candidates=top_candidates
        )

    @classmethod
    def levenshtein_distance(cls, s1: str, s2: str) -> int:
        """Computes minimum character edits (insertions, deletions, substitutions) between s1 and s2."""
        if s1 == s2: return 0
        if len(s1) == 0: return len(s2)
        if len(s2) == 0: return len(s1)

        v0 = list(range(len(s2) + 1))
        v1 = [0] * (len(s2) + 1)

        for i in range(len(s1)):
            v1[0] = i + 1
            for j in range(len(s2)):
                cost = 0 if s1[i] == s2[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0, v1 = v1, v0

        return v0[len(s2)]

    @classmethod
    def soundex(cls, name: str) -> str:
        """Generates Soundex phonetic encoding for words/names."""
        name = name.upper().strip()
        if not name: return ""

        mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }

        first_letter = name[0]
        tail = name[1:]

        encoded = []
        last_code = mapping.get(first_letter, '0')

        for char in tail:
            code = mapping.get(char, '0')
            if code != '0':
                if code != last_code:
                    encoded.append(code)
                last_code = code
            elif char in 'AEIOUYHW':
                last_code = '0'

        soundex_code = (first_letter + "".join(encoded) + "000")[:4]
        return soundex_code
