"""
COLLISION NLP Analytics — Advanced Text Analytics, Keyphrase Extraction, Topic Classification,
Tone Analysis, Semantic Similarity, and Grammar Proofreading.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field

from collision.nlp.processor import CollisionNLPProcessor


class KeyphraseResult(BaseModel):
    keyphrases: List[str] = Field(default_factory=list)
    top_keywords: List[Tuple[str, float]] = Field(default_factory=list)


class TopicClassificationResult(BaseModel):
    primary_topic: str
    confidence: float
    topic_distribution: List[Tuple[str, float]] = Field(default_factory=list)
    keywords_matched: List[str] = Field(default_factory=list)


class ToneAnalysisResult(BaseModel):
    primary_tone: str
    formality_score: float
    formality_label: str
    is_objective: bool
    subjectivity_score: float
    tone_markers: List[str] = Field(default_factory=list)
    summary: str


class SemanticSimilarityResult(BaseModel):
    overall_similarity: float
    jaccard_similarity: float
    cosine_similarity: float
    char_ngram_similarity: float
    levenshtein_ratio: float
    shared_keywords: List[str] = Field(default_factory=list)
    unique_to_first: List[str] = Field(default_factory=list)
    unique_to_second: List[str] = Field(default_factory=list)


class GrammarIssue(BaseModel):
    issue_type: str
    original: str
    replacement: str
    position: int
    explanation: str


class ProofreadResult(BaseModel):
    original_text: str
    corrected_text: str
    issues_found: int
    issues: List[GrammarIssue] = Field(default_factory=list)
    is_clean: bool


class TextRankKeyphraseExtractor:
    """
    Graph-centrality TextRank algorithm for extracting prominent keyphrases and keywords.
    """
    @classmethod
    def extract(cls, text: str, top_n: int = 5) -> KeyphraseResult:
        tokens = CollisionNLPProcessor.tokenize_words(text, remove_stopwords=True)
        if not tokens:
            return KeyphraseResult()

        lemmas = [CollisionNLPProcessor.lemmatize_word(t) for t in tokens if len(t) > 2]
        if not lemmas:
            return KeyphraseResult()

        # Build co-occurrence graph within window size = 3
        window = 3
        graph: Dict[str, Set[str]] = {}
        for i, w in enumerate(lemmas):
            if w not in graph:
                graph[w] = set()
            for j in range(max(0, i - window), min(len(lemmas), i + window + 1)):
                if i != j and lemmas[j] != w:
                    graph[w].add(lemmas[j])

        # PageRank / Degree centrality power iteration
        scores = {w: 1.0 for w in graph}
        damping = 0.85
        for _ in range(15):
            new_scores = {}
            for node, neighbors in graph.items():
                rank_sum = sum(scores[nbr] / max(1, len(graph[nbr])) for nbr in neighbors)
                new_scores[node] = (1.0 - damping) + (damping * rank_sum)
            scores = new_scores

        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_kw = [(w, round(s, 2)) for w, s in sorted_words[:top_n * 2]]

        # Extract multi-word candidate phrases (N-grams) matching noun phrases
        words_raw = re.findall(r'\b[A-Za-z0-9_\-]+\b', text)
        phrases: Set[str] = set()
        for i in range(len(words_raw) - 1):
            w1, w2 = words_raw[i].lower(), words_raw[i + 1].lower()
            if w1 not in CollisionNLPProcessor.STOPWORDS and w2 not in CollisionNLPProcessor.STOPWORDS:
                if len(w1) > 2 and len(w2) > 2:
                    phrases.add(f"{words_raw[i]} {words_raw[i+1]}")

        for i in range(len(words_raw) - 2):
            w1, w2, w3 = words_raw[i].lower(), words_raw[i + 1].lower(), words_raw[i + 2].lower()
            if w1 not in CollisionNLPProcessor.STOPWORDS and w3 not in CollisionNLPProcessor.STOPWORDS:
                phrases.add(f"{words_raw[i]} {words_raw[i+1]} {words_raw[i+2]}")

        # Score phrases by sum of constituent word scores
        phrase_scores = []
        for p in phrases:
            p_words = CollisionNLPProcessor.tokenize_words(p, remove_stopwords=True)
            score = sum(scores.get(CollisionNLPProcessor.lemmatize_word(w), 0.5) for w in p_words)
            phrase_scores.append((p, score))

        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        top_phrases = [p for p, _ in phrase_scores[:top_n]]

        # Fallback to single keywords if no multi-word phrases found
        if not top_phrases:
            top_phrases = [w for w, _ in top_kw[:top_n]]

        return KeyphraseResult(
            keyphrases=top_phrases,
            top_keywords=[(w, s) for w, s in top_kw[:top_n]]
        )


class TopicClassifier:
    """
    Multi-domain topic classifier across 10 academic and practical disciplines.
    """
    TOPIC_LEXICONS = {
        "Computer Science & AI": {
            'neural', 'algorithm', 'model', 'python', 'inference', 'software', 'programming',
            'database', 'api', 'server', 'code', 'hardware', 'transformer', 'gpu', 'tensor',
            'compiler', 'network', 'cybersecurity', 'kernel', 'quantum', 'computing', 'data'
        },
        "Physics & Astronomy": {
            'quantum', 'gravity', 'electron', 'photon', 'atom', 'relativity', 'energy',
            'mass', 'speed', 'light', 'force', 'particle', 'galaxy', 'planet', 'orbit',
            'telescope', 'spacetime', 'thermodynamics', 'velocity', 'acceleration'
        },
        "Chemistry & Biology": {
            'photosynthesis', 'cell', 'dna', 'rna', 'protein', 'enzyme', 'organism',
            'chemical', 'molecule', 'reaction', 'acid', 'compound', 'bacteria', 'virus',
            'evolution', 'gene', 'membrane', 'metabolism', 'chlorophyll', 'cellular'
        },
        "Mathematics & Statistics": {
            'equation', 'matrix', 'vector', 'calculus', 'algebra', 'probability', 'statistics',
            'integral', 'derivative', 'theorem', 'geometry', 'variance', 'mean', 'median',
            'standard deviation', 'prime', 'logarithm', 'polynomial', 'dimension'
        },
        "Finance & Business": {
            'revenue', 'profit', 'market', 'stock', 'investment', 'capital', 'funding',
            'economy', 'gdp', 'inflation', 'currency', 'dollar', 'bank', 'interest',
            'valuation', 'equity', 'shares', 'quarter', 'fiscal', 'sales', 'growth'
        },
        "Medicine & Health": {
            'doctor', 'hospital', 'patient', 'disease', 'treatment', 'drug', 'therapy',
            'clinical', 'diagnosis', 'symptom', 'vaccine', 'surgery', 'health', 'medical',
            'physician', 'cardiology', 'neurology', 'immune', 'infection', 'cancer',
            'trial', 'trials', 'antibody', 'antibodies', 'immunity', 'treatments'
        },
        "History & Politics": {
            'war', 'treaty', 'empire', 'king', 'president', 'government', 'century',
            'revolution', 'constitution', 'nation', 'democracy', 'parliament', 'ancient',
            'dynasty', 'republic', 'election', 'minister', 'monarchy', 'diplomacy'
        },
        "Law & Legal": {
            'court', 'judge', 'plaintiff', 'defendant', 'lawsuit', 'statute', 'jurisdiction',
            'contract', 'liability', 'attorney', 'legal', 'verdict', 'clause', 'legislation',
            'patent', 'copyright', 'intellectual property', 'regulation', 'compliance'
        },
        "Philosophy & Ethics": {
            'morality', 'ethics', 'epistemology', 'existentialism', 'logic', 'ontology',
            'consciousness', 'rationalism', 'virtue', 'utilitarianism', 'philosophy',
            'reason', 'truth', 'metaphysics', 'dualism', 'free will', 'knowledge'
        },
        "Everyday & Social": {
            'friend', 'chat', 'weather', 'food', 'travel', 'movie', 'game', 'music',
            'hobby', 'family', 'home', 'vacation', 'holiday', 'greeting', 'conversation',
            'relax', 'weekend', 'morning', 'night', 'dinner', 'lunch', 'party'
        }
    }

    @classmethod
    def classify(cls, text: str) -> TopicClassificationResult:
        tokens = set(CollisionNLPProcessor.tokenize_words(text, remove_stopwords=True))
        lemmas = set(CollisionNLPProcessor.lemmatize_word(t) for t in tokens)
        combined = tokens.union(lemmas)

        if not combined:
            return TopicClassificationResult(
                primary_topic="General Knowledge",
                confidence=0.50,
                topic_distribution=[("General Knowledge", 0.50)]
            )

        scores: Dict[str, float] = {}
        matches_dict: Dict[str, List[str]] = {}

        for topic, lexicon in cls.TOPIC_LEXICONS.items():
            matches = combined.intersection(lexicon)
            scores[topic] = len(matches) * 1.5
            matches_dict[topic] = list(matches)

        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_topic, top_score = sorted_topics[0]

        total_score = sum(scores.values())
        if total_score == 0:
            return TopicClassificationResult(
                primary_topic="General Knowledge",
                confidence=0.70,
                topic_distribution=[("General Knowledge", 0.70)]
            )

        distribution = [(t, round(s / total_score, 3)) for t, s in sorted_topics if s > 0][:4]
        conf = min(0.99, max(0.55, top_score / (total_score + 0.1) * 0.85 + 0.15))

        return TopicClassificationResult(
            primary_topic=top_topic,
            confidence=round(conf, 2),
            topic_distribution=distribution,
            keywords_matched=matches_dict.get(top_topic, [])
        )


class ToneAnalyzer:
    """
    Measures formality, subjectivity, urgency, and dominant tone.
    """
    FORMAL_INDICATORS = {
        'furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless',
        'subsequently', 'demonstrates', 'indicates', 'establish', 'utilize',
        'implementation', 'regarding', 'aforementioned', 'substantial', 'comprehensive'
    }

    CASUAL_INDICATORS = {
        'hey', 'hlo', 'cool', 'awesome', 'bro', 'buddy', 'gonna', 'wanna', 'gotta',
        'yep', 'nope', 'lol', 'omg', 'super', 'kinda', 'sorta', 'dude', 'mate'
    }

    URGENT_INDICATORS = {
        'immediately', 'urgent', 'asap', 'critical', 'emergency', 'now', 'instantly',
        'warning', 'danger', 'alert', 'crucial', 'fatal', 'severe'
    }

    ACADEMIC_INDICATORS = {
        'hypothesis', 'empirical', 'methodology', 'framework', 'synthesize', 'paradigm',
        'correlation', 'quantitative', 'qualitative', 'theoretical', 'phenomenon'
    }

    SUBJECTIVE_WORDS = {
        'feel', 'believe', 'think', 'opinion', 'guess', 'suppose', 'love', 'hate',
        'beautiful', 'ugly', 'best', 'worst', 'awful', 'amazing', 'terrible'
    }

    @classmethod
    def analyze(cls, text: str) -> ToneAnalysisResult:
        tokens = CollisionNLPProcessor.tokenize_words(text, remove_stopwords=False)
        total_words = max(1, len(tokens))
        t_set = set(t.lower() for t in tokens)

        formal_count = len(t_set.intersection(cls.FORMAL_INDICATORS))
        casual_count = len(t_set.intersection(cls.CASUAL_INDICATORS))
        urgent_count = len(t_set.intersection(cls.URGENT_INDICATORS))
        academic_count = len(t_set.intersection(cls.ACADEMIC_INDICATORS))
        subj_count = len(t_set.intersection(cls.SUBJECTIVE_WORDS))

        avg_word_len = sum(len(t) for t in tokens) / total_words

        # Formality score (0 to 100)
        formality = 50.0 + (formal_count * 15.0) + (academic_count * 12.0) - (casual_count * 20.0)
        if avg_word_len > 5.5: formality += 10.0
        if avg_word_len < 4.0: formality -= 10.0
        formality = max(0.0, min(100.0, formality))

        # Subjectivity score (0.0 = purely objective, 1.0 = purely subjective)
        subjectivity = min(1.0, (subj_count * 0.20) + (0.1 if any(t in t_set for t in ['i', 'my', 'me']) else 0.0))

        # Primary tone selection
        markers = []
        if urgent_count > 0:
            primary_tone = "Urgent / Action-Oriented"
            markers.extend(list(t_set.intersection(cls.URGENT_INDICATORS)))
        elif academic_count >= 2 or (formal_count >= 2 and avg_word_len > 6.0):
            primary_tone = "Academic & Analytical"
            markers.extend(list(t_set.intersection(cls.ACADEMIC_INDICATORS | cls.FORMAL_INDICATORS)))
        elif casual_count > 0 or formality < 40:
            primary_tone = "Casual & Conversational"
            markers.extend(list(t_set.intersection(cls.CASUAL_INDICATORS)))
        elif formality >= 65:
            primary_tone = "Formal & Professional"
            markers.extend(list(t_set.intersection(cls.FORMAL_INDICATORS)))
        else:
            primary_tone = "Neutral & Informative"

        if formality >= 75:
            form_label = "Highly Formal"
        elif formality >= 55:
            form_label = "Formal"
        elif formality >= 40:
            form_label = "Moderate / Balanced"
        else:
            form_label = "Casual / Informal"

        summary = (
            f"The text exhibits a **{primary_tone}** tone ({form_label}, Formality: {formality:.0f}%, "
            f"{'Objective' if subjectivity < 0.35 else 'Subjective'})."
        )

        return ToneAnalysisResult(
            primary_tone=primary_tone,
            formality_score=round(formality, 1),
            formality_label=form_label,
            is_objective=bool(subjectivity < 0.35),
            subjectivity_score=round(subjectivity, 2),
            tone_markers=markers,
            summary=summary
        )


class SemanticSimilarityCalculator:
    """
    Computes multi-metric lexical and semantic similarity between two texts.
    """
    @classmethod
    def compare(cls, text1: str, text2: str) -> SemanticSimilarityResult:
        t1_words = CollisionNLPProcessor.tokenize_words(text1, remove_stopwords=True)
        t2_words = CollisionNLPProcessor.tokenize_words(text2, remove_stopwords=True)

        s1 = set(CollisionNLPProcessor.lemmatize_word(w) for w in t1_words)
        s2 = set(CollisionNLPProcessor.lemmatize_word(w) for w in t2_words)

        # 1. Jaccard token overlap
        intersection = s1.intersection(s2)
        union = s1.union(s2)
        jaccard = len(intersection) / max(1, len(union)) if union else 1.0

        # 2. Term Frequency Cosine Similarity
        all_terms = list(union)
        if all_terms:
            vec1 = [t1_words.count(t) for t in all_terms]
            vec2 = [t2_words.count(t) for t in all_terms]
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            cosine = dot_product / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
        else:
            cosine = 1.0

        # 3. Character 3-gram overlap
        def char_ngrams(s: str, n: int = 3) -> Set[str]:
            s_clean = re.sub(r'\s+', '', s.lower())
            return {s_clean[i:i+n] for i in range(len(s_clean) - n + 1)} if len(s_clean) >= n else {s_clean}

        ng1, ng2 = char_ngrams(text1), char_ngrams(text2)
        ngram_sim = len(ng1.intersection(ng2)) / max(1, len(ng1.union(ng2))) if (ng1 or ng2) else 1.0

        # 4. Levenshtein ratio
        dist = CollisionNLPProcessor.levenshtein_distance(text1.lower(), text2.lower())
        max_len = max(len(text1), len(text2))
        lev_ratio = 1.0 - (dist / max(1, max_len)) if max_len > 0 else 1.0
        lev_ratio = max(0.0, lev_ratio)

        # Blended overall similarity
        overall = (0.50 * cosine) + (0.25 * jaccard) + (0.15 * ngram_sim) + (0.10 * lev_ratio)

        return SemanticSimilarityResult(
            overall_similarity=round(overall, 3),
            jaccard_similarity=round(jaccard, 3),
            cosine_similarity=round(cosine, 3),
            char_ngram_similarity=round(ngram_sim, 3),
            levenshtein_ratio=round(lev_ratio, 3),
            shared_keywords=sorted(list(intersection)),
            unique_to_first=sorted(list(s1 - s2)),
            unique_to_second=sorted(list(s2 - s1))
        )


class GrammarProofreader:
    """
    Automated rule-based grammar, punctuation, and typographical proofreader.
    """
    @classmethod
    def proofread(cls, text: str) -> ProofreadResult:
        if not text:
            return ProofreadResult(original_text="", corrected_text="", issues_found=0, is_clean=True)

        corrected = text
        issues: List[GrammarIssue] = []

        # 1. Repeated words check (e.g. "the the", "is is")
        repeated_pattern = re.compile(r'\b([a-zA-Z]+)\s+\1\b', re.IGNORECASE)
        for m in repeated_pattern.finditer(text):
            word = m.group(1)
            issues.append(GrammarIssue(
                issue_type="Repeated Word",
                original=m.group(0),
                replacement=word,
                position=m.start(),
                explanation=f"Duplicate word '{word}' detected."
            ))
        corrected = repeated_pattern.sub(r'\1', corrected)

        # 2. Article misuse: "a" vs "an"
        # "a [vowel sound]" -> "an [vowel sound]"
        a_vowel_pattern = re.compile(r'\b(a)\s+([aeiouAEIOU][a-z]+)\b')
        for m in a_vowel_pattern.finditer(corrected):
            next_w = m.group(2)
            # Avoid exceptions like 'a university', 'a european'
            if not next_w.lower().startswith(('univ', 'europ', 'one', 'user')):
                issues.append(GrammarIssue(
                    issue_type="Article Agreement",
                    original=f"a {next_w}",
                    replacement=f"an {next_w}",
                    position=m.start(),
                    explanation=f"Use 'an' before vowel sounds ('an {next_w}')."
                ))
        corrected = re.sub(r'\ba\s+([aeiouAEIOU](?!niv|urop|one|ser)[a-z]+)\b', r'an \1', corrected)

        # "an [consonant]" -> "a [consonant]"
        an_cons_pattern = re.compile(r'\b(an)\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ][a-z]+)\b')
        for m in an_cons_pattern.finditer(corrected):
            next_w = m.group(2)
            # Avoid exceptions like 'an hour', 'an honest'
            if not next_w.lower().startswith(('hour', 'honest', 'honor')):
                issues.append(GrammarIssue(
                    issue_type="Article Agreement",
                    original=f"an {next_w}",
                    replacement=f"a {next_w}",
                    position=m.start(),
                    explanation=f"Use 'a' before consonant sounds ('a {next_w}')."
                ))
        corrected = re.sub(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ](?!our|onest|onor)[a-z]+)\b', r'a \1', corrected)

        # 3. Space before punctuation fix (e.g. "hello ," -> "hello,")
        space_punct_pattern = re.compile(r'\s+([,;.!?])')
        for m in space_punct_pattern.finditer(corrected):
            issues.append(GrammarIssue(
                issue_type="Punctuation Spacing",
                original=m.group(0),
                replacement=m.group(1),
                position=m.start(),
                explanation="Removed superfluous space before punctuation."
            ))
        corrected = space_punct_pattern.sub(r'\1', corrected)

        # 4. Standalone lowercase 'i' -> 'I'
        i_pattern = re.compile(r'(?<=\s)i(?=[\s\',.!?])|^i(?=[\s\',.!?])')
        if i_pattern.search(corrected):
            issues.append(GrammarIssue(
                issue_type="Capitalization",
                original="i",
                replacement="I",
                position=0,
                explanation="Capitalized pronoun 'I'."
            ))
            corrected = i_pattern.sub('I', corrected)

        # 5. Sentence start capitalization
        sentences = re.split(r'([.!?]\s+)', corrected)
        reconstructed = []
        for s in sentences:
            if s and re.match(r'^[a-z]', s):
                cap = s[0].upper() + s[1:]
                issues.append(GrammarIssue(
                    issue_type="Sentence Capitalization",
                    original=s[:10],
                    replacement=cap[:10],
                    position=0,
                    explanation="Capitalized first letter of sentence."
                ))
                reconstructed.append(cap)
            else:
                reconstructed.append(s)
        corrected = "".join(reconstructed)

        return ProofreadResult(
            original_text=text,
            corrected_text=corrected,
            issues_found=len(issues),
            issues=issues,
            is_clean=bool(len(issues) == 0)
        )
