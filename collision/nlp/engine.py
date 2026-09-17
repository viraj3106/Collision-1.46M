"""
COLLISION NLP Engine — Official dedicated Natural Language Processing subsystem for COLLISION.

Unifies conversational dialogue management, deterministic mathematical calculation,
extractive summarization, sentiment analysis, named entity recognition, keyphrase extraction,
topic classification, tone analysis, grammar proofreading, reading comprehension,
semantic text comparison, and structured evidence-grounded answer synthesis.
"""

import re
import ast
import math
import html
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field

from collision.nlp.processor import (
    CollisionNLPProcessor,
    POSWord,
    ReadabilityMetrics,
    LanguageDetectionResult
)
from collision.nlp.analytics import (
    TextRankKeyphraseExtractor,
    TopicClassifier,
    ToneAnalyzer,
    SemanticSimilarityCalculator,
    GrammarProofreader,
    KeyphraseResult,
    TopicClassificationResult,
    ToneAnalysisResult,
    SemanticSimilarityResult,
    ProofreadResult
)
from collision.nlp.comprehension import (
    ReadingComprehensionEngine,
    TextTransformer,
    ContextQAResult,
    TextTransformationResult
)


class NLPIntentType(str, Enum):
    CONVERSATION = "CONVERSATION"
    MATH = "MATH"
    SENTIMENT = "SENTIMENT"
    SUMMARIZATION = "SUMMARIZATION"
    NER = "NER"
    KEYPHRASES = "KEYPHRASES"
    TOPIC = "TOPIC"
    TONE = "TONE"
    PROOFREAD = "PROOFREAD"
    READABILITY = "READABILITY"
    LANGUAGE_ID = "LANGUAGE_ID"
    SIMILARITY = "SIMILARITY"
    CONTEXT_QA = "CONTEXT_QA"
    TRANSFORMATION = "TRANSFORMATION"
    KNOWLEDGE_QA = "KNOWLEDGE_QA"
    CREATIVE = "CREATIVE"


class NLPSentimentResult(BaseModel):
    sentiment: str
    score: float
    confidence: float
    label: str
    positive_markers: List[str] = Field(default_factory=list)
    negative_markers: List[str] = Field(default_factory=list)
    summary: str


class NLPEntityResult(BaseModel):
    persons: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    dates_years: List[str] = Field(default_factory=list)
    quantities: List[str] = Field(default_factory=list)


class NLPSummaryResult(BaseModel):
    summary: str
    sentence_count: int
    compression_ratio: float


class NLPResult(BaseModel):
    intent: NLPIntentType
    answer: str
    confidence: float = 1.0
    sentiment_data: Optional[NLPSentimentResult] = None
    entity_data: Optional[NLPEntityResult] = None
    summary_data: Optional[NLPSummaryResult] = None
    keyphrase_data: Optional[KeyphraseResult] = None
    topic_data: Optional[TopicClassificationResult] = None
    tone_data: Optional[ToneAnalysisResult] = None
    proofread_data: Optional[ProofreadResult] = None
    readability_data: Optional[ReadabilityMetrics] = None
    similarity_data: Optional[SemanticSimilarityResult] = None
    context_qa_data: Optional[ContextQAResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollisionNLPEngine:
    """
    Official Dedicated NLP Engine for COLLISION.
    """

    GREETING_PATTERNS = [
        r'^(h[eai]+l+o+|h+i+|h+e+y+|h+a+i+|h+l+o+|h+o+l+a+|howdy|namaste|greetings|yo+|s+u+p+|wassup|whats?\s*up|wsup)(\s+(buddy|friend|bro|mate|there|collision|all))?(\b|[!?. ])',
        r'^(hi\s+there|hello\s+there|hey\s+there|hey\s+collision|hello\s+collision|hi\s+collision|hey\s+buddy|hello\s+buddy|hi\s+buddy)(\b|[!?. ])',
        r'^(good\s+(morning|afternoon|evening|day|night)|gm|gn|morning)(\b|[!?. ])',
        r'^(hello\s+world)(\b|[!?. ])'
    ]

    IDENTITY_PATTERNS = [
        r'who\s+(are|r|made|created|built|designed)\s+(you|u)',
        r'what\s+(is|are|r)\s+(you|collision|u)',
        r'tell\s+me\s+about\s+yourself',
        r'who\s+is\s+your\s+(creator|maker|developer)',
        r'what\s+model\s+(are|r)\s+(you|u)',
        r'are\s+(you|u)\s+(an?\s+)?(ai|robot|bot|human|llm)'
    ]

    COURTESY_PATTERNS = [
        r'\bhow\s+(are|r)\s+(you|u)\b',
        r'\bhow\s+is\s+it\s+going\b',
        r'\bhows\s+it\s+going\b',
        r'\bhow\s+do\s+you\s+do\b',
        r'\bhow\s+(are|r)\s+(you|u)\s+doing\b',
        r'\b(hru|wbu)\b',
        r'\bwhat\s+about\s+(you|u)\b',
        r'\bwhat(\'s|\s+is)?\s*up\b',
        r'\b(thank(s|\s+you|\s+u)?|thx|ty|much\s+appreciated|appreciate\s+it)\b',
        r'\b(good\s+job|great\s+work|well\s+done|nice\s+work|awesome\s+job)\b',
        r'\b(bye|goodbye|cya|see\s+(you|ya)|talk\s+to\s+you\s+later|ttyl|have\s+a\s+nice\s+day|take\s+care)\b'
    ]

    ACKNOWLEDGEMENT_PATTERNS = [
        r'^(ok|okay|okk|k|kk|cool|nice|sure|fine|alright|all\s+right|got\s+it|understood|awesome|great|wow|perfect|sounds\s+good|yep|yeah|yea|yes|no|nope|nah|lol|lmao|haha|hahaha|hehe)[.!]?$'
    ]

    SYSTEM_CHECK_PATTERNS = [
        r'^(test|testing|ping|hello\s*world|status)[.!]?$'
    ]

    HELP_PATTERNS = [
        r'^(help|what\s+can\s+(you|u)\s+do|how\s+do\s+(you|u)\s+work|how\s+does\s+this\s+work|features|what\s+do\s+(you|u)\s+do)\??$'
    ]

    @classmethod
    def handle_conversation(cls, query: str) -> Optional[str]:
        """Evaluates conversational, greetings, and small-talk queries with natural responses."""
        q = query.lower().strip()
        q_norm = re.sub(r'^[^\w]+|[^\w]+$', '', q)

        # 1. Identity & capabilities
        for p in cls.IDENTITY_PATTERNS:
            if re.search(p, q):
                return (
                    "I am **COLLISION**, an ultra-efficient neural AI system designed for intelligent reasoning, "
                    "conversational assistance, mathematical calculations, and grounded knowledge retrieval across open domains."
                )

        # 2. Greetings (hlo, hlo buddy, hi, heyy, sup, etc.)
        for p in cls.GREETING_PATTERNS:
            if re.search(p, q) or re.search(p, q_norm):
                if any(w in q for w in ["buddy", "friend", "mate", "bro"]):
                    return "Hello there! Great to chat with you. How can I help you today?"
                return "Hello! I am **COLLISION**, your AI assistant. How can I help you today?"

        # 3. Courtesies & Gratitude
        for p in cls.COURTESY_PATTERNS:
            if re.search(p, q) or re.search(p, q_norm):
                if any(w in q for w in ["thank", "thx", "ty", "great work", "good job", "well done", "nice work", "appreciated"]):
                    return "You're very welcome! If there is anything else you need, feel free to ask."
                if any(w in q for w in ["bye", "see you", "see ya", "goodbye", "cya", "take care", "ttyl"]):
                    return "Goodbye! Have a fantastic day ahead!"
                return "I'm doing great and running smoothly at peak performance! How can I assist you today?"

        # 4. Acknowledgements / Small-talk
        for p in cls.ACKNOWLEDGEMENT_PATTERNS:
            if re.search(p, q) or re.search(p, q_norm):
                return "Understood! Let me know whenever you're ready for your next question or task."

        # 5. System checks
        for p in cls.SYSTEM_CHECK_PATTERNS:
            if re.search(p, q) or re.search(p, q_norm):
                return "Pong! 🏓 COLLISION is online, healthy, and ready to answer any questions."

        # 6. Help
        for p in cls.HELP_PATTERNS:
            if re.search(p, q) or re.search(p, q_norm):
                return (
                    "**How I can help you:**\n\n"
                    "• **Conversations**: Chat, brainstorm, and answer everyday questions.\n"
                    "• **Math & Problem Solving**: Solve arithmetic, geometry, statistics, percentages, and unit conversions.\n"
                    "• **NLP Analytics**: Sentiment analysis, text summarization, NER, keyphrase extraction, topic classification, and tone scoring.\n"
                    "• **Proofreading & Language**: Fix grammar, compute readability (Flesch-Kincaid), and detect languages.\n"
                    "• **Reading Comprehension**: Answer questions from arbitrary context passages.\n"
                    "• **Live Web & Wikipedia Knowledge**: Real-time factual information with verified citations.\n\n"
                    "Just type your query and I will provide an answer!"
                )

        return None

    @classmethod
    def solve_math(cls, query: str) -> Optional[str]:
        """Safely and deterministically evaluates math, geometry, statistics, and unit conversions."""
        q = query.lower().strip()
        q = re.sub(r'^(?:calculate|compute|solve|what\s+is|evaluate)\s*[:\s]*', '', q)
        q = q.rstrip('?!=. ')

        # 1. Statistics (mean, average, median, std dev, variance)
        stat_match = re.match(r'^(?:(mean|average|median|std\s*dev|standard\s+deviation|variance)\s+(?:of|for))\s*[:\s]*([0-9\.,\s\-]+)$', q)
        if stat_match:
            op, nums_str = stat_match.groups()
            nums = [float(x.strip()) for x in re.findall(r'-?\d+(?:\.\d+)?', nums_str)]
            if nums:
                n = len(nums)
                mean_val = sum(nums) / n
                if op in ('mean', 'average'):
                    return f"Mean of [{', '.join(f'{x:g}' for x in nums)}] = **{mean_val:g}**"
                elif op == 'median':
                    s_nums = sorted(nums)
                    med_val = (s_nums[n//2] if n % 2 != 0 else (s_nums[n//2 - 1] + s_nums[n//2]) / 2.0)
                    return f"Median of [{', '.join(f'{x:g}' for x in nums)}] = **{med_val:g}**"
                elif op in ('std dev', 'standard deviation'):
                    variance = sum((x - mean_val) ** 2 for x in nums) / max(1, n - 1 if n > 1 else 1)
                    std_val = math.sqrt(variance)
                    return f"Standard Deviation of [{', '.join(f'{x:g}' for x in nums)}] = **{std_val:.4g}**"
                elif op == 'variance':
                    variance = sum((x - mean_val) ** 2 for x in nums) / max(1, n - 1 if n > 1 else 1)
                    return f"Variance of [{', '.join(f'{x:g}' for x in nums)}] = **{variance:.4g}**"

        # 2. Geometry calculations
        # Circle area
        circle_area = re.match(r'^(?:area\s+of\s+(?:a\s+)?circle\s+(?:with\s+)?radius)\s*([\d\.]+)$', q)
        if circle_area:
            r = float(circle_area.group(1))
            area = math.pi * (r ** 2)
            return f"Area of circle (radius = {r:g}) = pi * r^2 = **{area:.4g}**"

        # Circle circumference
        circle_circ = re.match(r'^(?:circumference\s+of\s+(?:a\s+)?circle\s+(?:with\s+)?radius)\s*([\d\.]+)$', q)
        if circle_circ:
            r = float(circle_circ.group(1))
            circ = 2.0 * math.pi * r
            return f"Circumference of circle (radius = {r:g}) = 2 * pi * r = **{circ:.4g}**"

        # Rectangle area
        rect_area = re.match(r'^(?:area\s+of\s+(?:a\s+)?rectangle\s+(?:with\s+)?(?:length\s+)?)([\d\.]+)\s*(?:by|and|x|\*|,|\s+width\s+)\s*([\d\.]+)$', q)
        if rect_area:
            l, w = float(rect_area.group(1)), float(rect_area.group(2))
            return f"Area of rectangle ({l:g} x {w:g}) = **{l * w:g}**"

        # Triangle area
        tri_area = re.match(r'^(?:area\s+of\s+(?:a\s+)?triangle\s+(?:with\s+)?(?:base\s+)?)([\d\.]+)\s*(?:and|by|height|\s+height\s+)\s*([\d\.]+)$', q)
        if tri_area:
            b, h = float(tri_area.group(1)), float(tri_area.group(2))
            return f"Area of triangle (base = {b:g}, height = {h:g}) = 0.5 * b * h = **{0.5 * b * h:g}**"

        # Sphere volume
        sphere_vol = re.match(r'^(?:volume\s+of\s+(?:a\s+)?sphere\s+(?:with\s+)?radius)\s*([\d\.]+)$', q)
        if sphere_vol:
            r = float(sphere_vol.group(1))
            vol = (4.0 / 3.0) * math.pi * (r ** 3)
            return f"Volume of sphere (radius = {r:g}) = 4/3 * pi * r^3 = **{vol:.4g}**"

        # Cylinder volume
        cyl_vol = re.match(r'^(?:volume\s+of\s+(?:a\s+)?cylinder\s+(?:with\s+)?radius)\s*([\d\.]+)\s*(?:and|height|\s+height\s+)\s*([\d\.]+)$', q)
        if cyl_vol:
            r, h = float(cyl_vol.group(1)), float(cyl_vol.group(2))
            vol = math.pi * (r ** 2) * h
            return f"Volume of cylinder (radius = {r:g}, height = {h:g}) = pi * r^2 * h = **{vol:.4g}**"

        # 3. Temperature Conversions
        c_to_f = re.match(r'^(-?[\d\.]+)\s*(?:c|celsius|degrees\s+celsius)\s*(?:in|to|into)\s*(?:f|fahrenheit|degrees\s+fahrenheit)$', q)
        if c_to_f:
            c = float(c_to_f.group(1))
            f = (c * 9.0 / 5.0) + 32.0
            return f"{c:g} °C = **{f:g} °F**"

        f_to_c = re.match(r'^(-?[\d\.]+)\s*(?:f|fahrenheit|degrees\s+fahrenheit)\s*(?:in|to|into)\s*(?:c|celsius|degrees\s+celsius)$', q)
        if f_to_c:
            f = float(f_to_c.group(1))
            c = (f - 32.0) * 5.0 / 9.0
            return f"{f:g} °F = **{c:.4g} °C**"

        # 4. Digital Storage Conversions
        gb_to_mb = re.match(r'^([\d\.]+)\s*(?:gb|gigabytes?)\s*(?:in|to|into)\s*(?:mb|megabytes?)$', q)
        if gb_to_mb:
            v = float(gb_to_mb.group(1))
            return f"{v:g} GB = **{v * 1024:g} MB**"

        mb_to_gb = re.match(r'^([\d\.]+)\s*(?:mb|megabytes?)\s*(?:in|to|into)\s*(?:gb|gigabytes?)$', q)
        if mb_to_gb:
            v = float(mb_to_gb.group(1))
            return f"{v:g} MB = **{v / 1024.0:.4g} GB**"

        tb_to_gb = re.match(r'^([\d\.]+)\s*(?:tb|terabytes?)\s*(?:in|to|into)\s*(?:gb|gigabytes?)$', q)
        if tb_to_gb:
            v = float(tb_to_gb.group(1))
            return f"{v:g} TB = **{v * 1024:g} GB**"

        # 5. Speed Conversions
        kmh_to_mph = re.match(r'^([\d\.]+)\s*(?:km\/h|kmh|kph)\s*(?:in|to|into)\s*(?:mph|miles\s+per\s+hour)$', q)
        if kmh_to_mph:
            v = float(kmh_to_mph.group(1))
            return f"{v:g} km/h = **{v * 0.621371:.4g} mph**"

        mph_to_kmh = re.match(r'^([\d\.]+)\s*(?:mph|miles\s+per\s+hour)\s*(?:in|to|into)\s*(?:km\/h|kmh|kph)$', q)
        if mph_to_kmh:
            v = float(mph_to_kmh.group(1))
            return f"{v:g} mph = **{v * 1.60934:.4g} km/h**"

        # 6. Distance Conversions
        km_to_miles = re.match(r'^([\d\.]+)\s*(?:km|kilometers?)\s*(?:in|to|into)\s*(?:miles?|mi)$', q)
        if km_to_miles:
            v = float(km_to_miles.group(1))
            return f"{v:g} km = **{v * 0.621371:g} miles**"

        miles_to_km = re.match(r'^([\d\.]+)\s*(?:miles?|mi)\s*(?:in|to|into)\s*(?:km|kilometers?)$', q)
        if miles_to_km:
            v = float(miles_to_km.group(1))
            return f"{v:g} miles = **{v * 1.60934:g} km**"

        m_to_ft = re.match(r'^([\d\.]+)\s*(?:meters?|m)\s*(?:in|to|into)\s*(?:feet|ft)$', q)
        if m_to_ft:
            v = float(m_to_ft.group(1))
            return f"{v:g} meters = **{v * 3.28084:g} feet**"

        ft_to_m = re.match(r'^([\d\.]+)\s*(?:feet|ft)\s*(?:in|to|into)\s*(?:meters?|m)$', q)
        if ft_to_m:
            v = float(ft_to_m.group(1))
            return f"{v:g} feet = **{v * 0.3048:g} meters**"

        # 7. Weight Conversions
        kg_to_lbs = re.match(r'^([\d\.]+)\s*(?:kg|kilograms?)\s*(?:in|to|into)\s*(?:lbs?|pounds?)$', q)
        if kg_to_lbs:
            v = float(kg_to_lbs.group(1))
            return f"{v:g} kg = **{v * 2.20462:g} lbs**"

        lbs_to_kg = re.match(r'^([\d\.]+)\s*(?:lbs?|pounds?)\s*(?:in|to|into)\s*(?:kg|kilograms?)$', q)
        if lbs_to_kg:
            v = float(lbs_to_kg.group(1))
            return f"{v:g} lbs = **{v * 0.453592:g} kg**"

        # 8. Percentage pattern (e.g. "15% of 850")
        pct_match = re.match(r'^([\d\.]+)\s*%\s*(?:of)?\s*([\d\.]+)$', q)
        if pct_match:
            pct = float(pct_match.group(1))
            val = float(pct_match.group(2))
            res = (pct / 100.0) * val
            return f"{pct:g}% of {val:g} = **{res:g}**"

        # 9. Square root pattern (e.g. "square root of 144", "sqrt 144")
        sqrt_match = re.match(r'^(?:square\s+root\s+of|sqrt\s*\(?)\s*([\d\.]+)\)?$', q)
        if sqrt_match:
            val = float(sqrt_match.group(1))
            if val < 0:
                return f"Square root of {val:g} is not a real number."
            res = math.sqrt(val)
            return f"sqrt({val:g}) = **{res:g}**"

        # 10. Simple linear equations (e.g. "solve 2x + 10 = 30", "3x - 6 = 18")
        eq_match = re.match(r'^(?:solve\s+)?(\d*)\s*x\s*([\+\-])\s*(\d+)\s*=\s*(\d+)$', q)
        if eq_match:
            a_str, sign, b_str, c_str = eq_match.groups()
            a = float(a_str) if a_str else 1.0
            b = float(b_str)
            c = float(c_str)
            if sign == '+':
                x_val = (c - b) / a if a != 0 else 0
            else:
                x_val = (c + b) / a if a != 0 else 0
            return f"Equation: {q} => **x = {x_val:g}**"

        # 11. Natural language arithmetic replacement
        norm_expr = q
        norm_expr = re.sub(r'\bplus\b', '+', norm_expr)
        norm_expr = re.sub(r'\bminus\b', '-', norm_expr)
        norm_expr = re.sub(r'\b(?:multiplied\s+by|times)\b', '*', norm_expr)
        norm_expr = re.sub(r'\b(?:divided\s+by|over)\b', '/', norm_expr)
        norm_expr = re.sub(r'\bto\s+the\s+power\s+(?:of)?\b', '**', norm_expr)
        norm_expr = re.sub(r'\bsquared\b', '**2', norm_expr)
        norm_expr = re.sub(r'\bcubed\b', '**3', norm_expr)
        norm_expr = re.sub(r'\bpi\b', str(math.pi), norm_expr)
        norm_expr = re.sub(r'\be\b', str(math.e), norm_expr)

        # 12. AST evaluation
        clean_expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\s\^%]', '', norm_expr).strip()
        if not clean_expr or not any(op in clean_expr for op in '+-*/^%'):
            return None

        if not re.search(r'\d', clean_expr):
            return None

        try:
            expr_py = clean_expr.replace('^', '**')
            node = ast.parse(expr_py, mode='eval')

            def _eval_node(n):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    return n.value
                elif isinstance(n, ast.BinOp):
                    left, right = _eval_node(n.left), _eval_node(n.right)
                    if isinstance(n.op, ast.Add): return left + right
                    elif isinstance(n.op, ast.Sub): return left - right
                    elif isinstance(n.op, ast.Mult): return left * right
                    elif isinstance(n.op, ast.Div): return left / right if right != 0 else float('inf')
                    elif isinstance(n.op, ast.Pow): return left ** right
                    elif isinstance(n.op, ast.Mod): return left % right
                elif isinstance(n, ast.UnaryOp):
                    operand = _eval_node(n.operand)
                    if isinstance(n.op, ast.USub): return -operand
                    elif isinstance(n.op, ast.UAdd): return +operand
                raise ValueError("Unsupported AST node")

            ans = _eval_node(node.body)
            if isinstance(ans, float) and ans.is_integer():
                ans = int(ans)
            elif isinstance(ans, float):
                ans = round(ans, 6)

            return f"{query.strip()} = **{ans}**"
        except Exception:
            return None

    @classmethod
    def analyze_sentiment(cls, text: str) -> NLPSentimentResult:
        """Computes polarity, confidence, valence, and emotional markers."""
        words = CollisionNLPProcessor.tokenize_words(text, remove_stopwords=False)
        pos_score, neg_score = 0.0, 0.0
        key_pos, key_neg = [], []

        intensifier = 1.0
        active_negation = False

        for i, w in enumerate(words):
            if w in CollisionNLPProcessor.NEGATIONS:
                active_negation = not active_negation
                continue
            if w in CollisionNLPProcessor.INTENSIFIERS:
                intensifier = CollisionNLPProcessor.INTENSIFIERS[w]
                continue

            if w in CollisionNLPProcessor.POSITIVE_WORDS:
                v = 1.0 * intensifier
                if active_negation:
                    neg_score += v
                    key_neg.append(f"not {w}")
                else:
                    pos_score += v
                    key_pos.append(w)
                intensifier = 1.0
                active_negation = False
            elif w in CollisionNLPProcessor.NEGATIVE_WORDS:
                v = 1.0 * intensifier
                if active_negation:
                    pos_score += v
                    key_pos.append(f"not {w}")
                else:
                    neg_score += v
                    key_neg.append(w)
                intensifier = 1.0
                active_negation = False

        total = pos_score + neg_score
        if total == 0:
            return NLPSentimentResult(sentiment="Neutral", score=0.0, confidence=0.85, label="NEUTRAL", summary="The input text expresses an objective or neutral tone.")

        net = (pos_score - neg_score) / max(1.0, total)
        conf = min(1.0, 0.60 + (abs(net) * 0.35) + min(0.05, total * 0.02))
        sent = "Positive" if net > 0.15 else ("Negative" if net < -0.15 else "Neutral")

        return NLPSentimentResult(
            sentiment=sent,
            score=round(net, 3),
            confidence=round(conf, 2),
            label=sent.upper(),
            positive_markers=key_pos,
            negative_markers=key_neg,
            summary=f"Identified **{sent}** sentiment (Score: {net:+.2f}, Confidence: {conf*100:.0f}%)."
        )

    @classmethod
    def summarize(cls, text: str, max_sentences: int = 3) -> NLPSummaryResult:
        """Executes extractive sentence ranking summarization."""
        sentences = CollisionNLPProcessor.segment_sentences(text)
        if len(sentences) <= max_sentences:
            return NLPSummaryResult(summary="\n\n".join(sentences), sentence_count=len(sentences), compression_ratio=1.0)

        doc_tokens = [CollisionNLPProcessor.tokenize_words(s) for s in sentences]
        all_words = [w for dt in doc_tokens for w in dt]
        word_freq: Dict[str, int] = {}
        for w in all_words:
            word_freq[w] = word_freq.get(w, 0) + 1

        scored = []
        for idx, (s, tokens) in enumerate(zip(sentences, doc_tokens)):
            if not tokens:
                continue
            score = sum(word_freq.get(w, 0) for w in tokens) / (len(tokens) ** 0.5)
            pos_mult = 1.0 + (1.0 / (idx + 1.0))
            scored.append((score * pos_mult, idx, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_sentences]
        top.sort(key=lambda x: x[1])

        summary_text = " ".join([s for _, _, s in top])
        return NLPSummaryResult(
            summary=summary_text,
            sentence_count=len(top),
            compression_ratio=round(len(summary_text) / max(1, len(text)), 2)
        )

    @classmethod
    def extract_entities(cls, text: str) -> NLPEntityResult:
        """Extracts named entities from text."""
        cleaned = CollisionNLPProcessor.clean_text(text)
        years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', cleaned)
        date_phrases = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b', cleaned, re.I)
        quantities = re.findall(r'\b(?:\$\d+[\d,\.]*|\d+[\d,\.]*%\b|\d+[\d,\.]*\s*(?:parameters|tokens|million|billion|km|miles|meters|kg|lbs|sec|ms|flops))\b', cleaned, re.I)

        persons, orgs, locs = [], [], []
        proper_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', cleaned)
        for name in proper_names:
            if name.lower() in CollisionNLPProcessor.STOPWORDS or len(name) < 3:
                continue
            if any(kw in name.lower() for kw in ['corp', 'inc', 'ltd', 'foundation', 'university', 'institute', 'lab', 'labs', 'google', 'apple', 'microsoft', 'nasa', 'cern', 'openai', 'deepmind']):
                if name not in orgs: orgs.append(name)
            elif any(kw in name.lower() for kw in ['france', 'paris', 'tokyo', 'japan', 'california', 'germany', 'london', 'america', 'europe', 'asia', 'earth', 'moon', 'mars']):
                if name not in locs: locs.append(name)
            elif len(name.split()) >= 2 and not any(w.lower() in CollisionNLPProcessor.STOPWORDS for w in name.split()):
                if name not in persons: persons.append(name)

        return NLPEntityResult(
            persons=persons,
            organizations=orgs,
            locations=locs,
            dates_years=list(set(years + date_phrases)),
            quantities=list(set(quantities))
        )

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 5) -> KeyphraseResult:
        return TextRankKeyphraseExtractor.extract(text, top_n=top_n)

    @classmethod
    def classify_topic(cls, text: str) -> TopicClassificationResult:
        return TopicClassifier.classify(text)

    @classmethod
    def analyze_tone(cls, text: str) -> ToneAnalysisResult:
        return ToneAnalyzer.analyze(text)

    @classmethod
    def compare_texts(cls, text1: str, text2: str) -> SemanticSimilarityResult:
        return SemanticSimilarityCalculator.compare(text1, text2)

    @classmethod
    def proofread(cls, text: str) -> ProofreadResult:
        return GrammarProofreader.proofread(text)

    @classmethod
    def compute_readability(cls, text: str) -> ReadabilityMetrics:
        return CollisionNLPProcessor.compute_readability(text)

    @classmethod
    def detect_language(cls, text: str) -> LanguageDetectionResult:
        return CollisionNLPProcessor.detect_language(text)

    @classmethod
    def answer_from_context(cls, context: str, question: str) -> ContextQAResult:
        return ReadingComprehensionEngine.answer_question(context, question)

    @classmethod
    def transform_text(cls, text: str, style: str = "formal") -> TextTransformationResult:
        style_lower = style.lower().strip()
        if "simple" in style_lower or "simplify" in style_lower:
            return TextTransformer.simplify(text)
        elif "bullet" in style_lower:
            return TextTransformer.bulletize(text)
        else:
            return TextTransformer.formalize(text)

    @classmethod
    def synthesize_grounded_answer(cls, query: str, passages: List[Tuple[Any, str]]) -> str:
        """Synthesizes open-domain passages into structured NLP answers."""
        if not passages:
            return "No verified information could be retrieved for this query."

        all_sentences: List[str] = []
        seen: Set[str] = set()
        q_words = set(CollisionNLPProcessor.tokenize_words(query))

        for item, raw_text in passages:
            if not raw_text: continue
            sentences = CollisionNLPProcessor.segment_sentences(raw_text)
            for s in sentences:
                s_clean = s.strip()
                s_key = s_clean.lower()[:45]
                if s_key not in seen and len(s_clean) >= 20:
                    seen.add(s_key)
                    all_sentences.append(s_clean)

        if not all_sentences:
            return "No verified information could be retrieved for this query."

        scored = []
        for idx, s in enumerate(all_sentences):
            s_words = set(CollisionNLPProcessor.tokenize_words(s))
            score = len(q_words.intersection(s_words)) * 2.0
            if re.search(r'^(who|what)\s+(is|was|are|were)', query, re.I):
                if re.search(r'\b(is|was|are|were|refers to|defined as|known for|discovered|invented)\b', s, re.I):
                    score += 3.0
            if re.search(r'^(how|why)', query, re.I):
                if re.search(r'\b(works by|process|mechanism|allows|causes|enables|functions)\b', s, re.I):
                    score += 3.0
            if idx == 0: score += 2.5
            scored.append((score, idx, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        primary = scored[0][2]
        supporting = []
        for _, _, s in scored[1:6]:
            if s != primary and not any(s[:30].lower() in existing.lower() for existing in [primary] + supporting):
                supporting.append(s)
            if len(supporting) >= 3:
                break

        parts = [primary]
        if supporting:
            if len(supporting) >= 2 and any(kw in query.lower() for kw in ['how', 'why', 'what are', 'explain', 'tell me about']):
                parts.append("\n".join([f"• {s}" for s in supporting]))
            else:
                parts.append(" ".join(supporting))

        return "\n\n".join(parts)

    @classmethod
    def handle_nlp_task(cls, query: str) -> Optional[str]:
        """Detects and executes dedicated NLP tasks across the expanded NLP subsystem."""
        q = query.strip()

        # 1. Reading Comprehension (Context: ... Question: ...)
        ctx_qa_match = re.match(r'^(?:context|passage|text)\s*:\s*(.+?)\s*(?:question|q)\s*:\s*(.+)$', q, re.I | re.DOTALL)
        if ctx_qa_match:
            ctx_text, q_text = ctx_qa_match.group(1).strip(), ctx_qa_match.group(2).strip()
            res = cls.answer_from_context(ctx_text, q_text)
            out = [
                "### Reading Comprehension Answer",
                f"**Answer**: {res.answer}",
                f"• **Confidence**: `{res.confidence*100:.0f}%`",
                f"• **Source Sentence**: *\"{res.sentence}\"*"
            ]
            return "\n\n".join(out)

        # 2. Text Comparison & Semantic Similarity (Compare texts: ... [vs] ...)
        sim_match = re.match(r'^(?:compare\s+(?:the\s+)?(?:texts?|documents?)|similarity\s+between)\s*[:\s]+(.+?)\s+(?:vs|and|with)\s+(.+)$', q, re.I | re.DOTALL)
        if sim_match:
            t1, t2 = sim_match.group(1).strip(" \"'"), sim_match.group(2).strip(" \"'")
            sim = cls.compare_texts(t1, t2)
            out = [
                "### Semantic Text Similarity",
                f"• **Overall Similarity Score**: `{sim.overall_similarity*100:.1f}%`",
                f"• **Cosine TF-IDF**: `{sim.cosine_similarity*100:.1f}%` | **Jaccard Overlap**: `{sim.jaccard_similarity*100:.1f}%` | **Char N-Gram**: `{sim.char_ngram_similarity*100:.1f}%`",
                f"• **Shared Key Concepts**: {', '.join(sim.shared_keywords) if sim.shared_keywords else 'None'}"
            ]
            if sim.unique_to_first:
                out.append(f"• **Unique to First**: {', '.join(sim.unique_to_first[:5])}")
            if sim.unique_to_second:
                out.append(f"• **Unique to Second**: {', '.join(sim.unique_to_second[:5])}")
            return "\n".join(out)

        # 3. Keyphrase & Keyword Extraction
        kw_match = re.match(r'^(?:extract\s+(?:the\s+)?(?:keywords?|keyphrases?|topics?)\s+(?:from|in)|keywords?\s*[:\s]|keyphrases?\s*[:\s]|main\s+topics?\s+of)\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if kw_match:
            target = kw_match.group(1).strip(" \"'")
            kp = cls.extract_keywords(target)
            out = [
                "### TextRank Keyphrase Extraction",
                f"• **Keyphrases**: {', '.join(f'**{p}**' for p in kp.keyphrases)}",
                f"• **Salient Terms**: {', '.join(f'`{w}` ({s})' for w, s in kp.top_keywords[:6])}"
            ]
            return "\n".join(out)

        # 4. Topic Classification
        topic_match = re.match(r'^(?:classify\s+(?:the\s+)?(?:topic|domain|category)(?:\s+(?:of|for|in))?|what\s+topic\s+is|category\s+of)\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if topic_match:
            target = topic_match.group(1).strip(" \"'")
            top = cls.classify_topic(target)
            dist_str = ", ".join(f"{t} ({s*100:.0f}%)" for t, s in top.topic_distribution)
            out = [
                "### Topic Classification",
                f"• **Primary Topic**: **{top.primary_topic}** (Confidence: `{top.confidence*100:.0f}%`)",
                f"• **Domain Distribution**: {dist_str}",
                f"• **Key Indicators**: {', '.join(top.keywords_matched) if top.keywords_matched else 'Contextual patterns'}"
            ]
            return "\n".join(out)

        # 5. Tone & Formality Analysis
        tone_match = re.match(r'^(?:analyze\s+(?:the\s+)?tone(?:\s+(?:of|for|in))?|tone\s+(?:analysis|of)\s*[:\s]|what\s+is\s+the\s+tone\s+of|formality\s+(?:score|of))\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if tone_match:
            target = tone_match.group(1).strip(" \"'")
            tone = cls.analyze_tone(target)
            out = [
                "### Tone & Formality Analysis",
                f"• **Primary Tone**: **{tone.primary_tone}**",
                f"• **Formality Score**: `{tone.formality_score:.0f}/100` ({tone.formality_label})",
                f"• **Objectivity**: `{'Objective / Factual' if tone.is_objective else 'Subjective / Opinionated'}` (Subjectivity: `{tone.subjectivity_score*100:.0f}%`)",
                f"• **Assessment**: {tone.summary}"
            ]
            if tone.tone_markers:
                out.append(f"• **Tone Markers**: {', '.join(tone.tone_markers)}")
            return "\n".join(out)

        # 6. Grammar & Spell Proofreading
        proof_match = re.match(r'^(?:proofread|check\s+grammar|fix\s+grammar|correct\s+spelling|fix\s+text)\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if proof_match:
            target = proof_match.group(1).strip(" \"'")
            proof = cls.proofread(target)
            out = ["### Grammar & Proofreading Report"]
            if proof.is_clean:
                out.append("✅ **No grammatical or typographical issues detected.** The text is clean.")
            else:
                out.append(f"**Corrected Text**:\n> {proof.corrected_text}\n")
                out.append(f"**Issues Identified ({proof.issues_found})**:")
                for issue in proof.issues:
                    out.append(f"• **{issue.issue_type}**: Changed `\"{issue.original}\"` -> `\"{issue.replacement}\"` (*{issue.explanation}*)")
            return "\n".join(out)

        # 7. Readability & Complexity Score
        read_match = re.match(r'^(?:compute\s+readability|readability\s+(?:score|of)|complexity\s+of)\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if read_match:
            target = read_match.group(1).strip(" \"'")
            metrics = cls.compute_readability(target)
            out = [
                "### Text Readability & Complexity Metrics",
                f"• **Flesch Reading Ease**: `{metrics.flesch_reading_ease}/100` — **{metrics.reading_ease_description}**",
                f"• **Flesch-Kincaid Grade Level**: `Grade {metrics.flesch_kincaid_grade:.1f}`",
                f"• **Gunning Fog Index**: `{metrics.gunning_fog_index:.1f}`",
                f"• **Lexical Richness (TTR)**: `{metrics.lexical_diversity_ttr*100:.1f}%` ({metrics.word_count} words across {metrics.sentence_count} sentences)",
                f"• **Averages**: `{metrics.avg_words_per_sentence:.1f}` words/sentence, `{metrics.avg_syllables_per_word:.2f}` syllables/word"
            ]
            return "\n".join(out)

        # 8. Language Detection
        lang_match = re.match(r'^(?:detect\s+language|what\s+language\s+is)\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if lang_match:
            target = lang_match.group(1).strip(" \"'")
            lang = cls.detect_language(target)
            cands = ", ".join(f"{l} ({c*100:.0f}%)" for l, c in lang.top_candidates)
            out = [
                "### Language Identification",
                f"• **Detected Language**: **{lang.language}** (`{lang.language_code}`)",
                f"• **Confidence**: `{lang.confidence*100:.0f}%`",
                f"• **Top Candidates**: {cands}"
            ]
            return "\n".join(out)

        # 9. Text Transformations (Formalize, Simplify, Bulletize)
        trans_match = re.match(r'^(?:(formalize|simplify|bulletize|convert\s+to\s+bullets?|rewrite\s+formally))\s*[:\n\s]+(.+)$', q, re.I | re.DOTALL)
        if trans_match:
            op, target = trans_match.group(1).lower(), trans_match.group(2).strip(" \"'")
            res = cls.transform_text(target, style=op)
            return f"### Transformed Text ({res.transformation_type})\n\n{res.transformed_text}"

        # 10. Sentiment Analysis task
        sent_match = re.match(
            r'^(?:analyze\s+(?:the\s+)?sentiment(?:\s+(?:of|for|in))?|what\s+is\s+the\s+sentiment\s+(?:of|for|in)|sentiment\s+analysis\s*[:\n\s]|sentiment\s*[:\n\s]|is\s+this\s+(?:text\s+)?positive\s+or\s+negative)\s*[:\n\s]+(.+)$',
            q,
            re.I | re.DOTALL
        )
        if sent_match:
            target = sent_match.group(1).strip(" \"'")
            res = cls.analyze_sentiment(target)
            out = [
                "### NLP Sentiment Analysis",
                f"• **Sentiment**: **{res.sentiment}**",
                f"• **Polarity Score**: `{res.score:+.2f}` (Scale: -1.0 to +1.0)",
                f"• **Confidence**: `{res.confidence*100:.0f}%`",
                f"• **Analysis**: {res.summary}"
            ]
            if res.positive_markers:
                out.append(f"• **Positive Markers**: {', '.join(res.positive_markers)}")
            if res.negative_markers:
                out.append(f"• **Negative Markers**: {', '.join(res.negative_markers)}")
            return "\n".join(out)

        # 11. Extractive Summarization task
        sum_match = re.match(
            r'^(?:summarize(?:\s+(?:the\s+)?(?:following|given|text|article)(?:\s+text)?)?|summary\s+of|give\s+me\s+a\s+summary\s+(?:of|for)|tldr:?|summarise)\s*[:\n\s]+(.+)$',
            q,
            re.I | re.DOTALL
        )
        if sum_match:
            target = sum_match.group(1).strip(" \"'")
            if len(target) > 40:
                summary_res = cls.summarize(target, max_sentences=3)
                return f"### Summary\n\n{summary_res.summary}"

        # 12. Named Entity Recognition (NER) task
        ner_match = re.match(
            r'^(?:extract\s+(?:the\s+)?entities(?:\s+(?:from|in))?|ner\s*[:\n\s]|find\s+(?:names|places|dates|entities)\s+(?:from|in)|extract\s+names\s+and\s+places\s+from)\s*[:\n\s]+(.+)$',
            q,
            re.I | re.DOTALL
        )
        if ner_match:
            target = ner_match.group(1).strip(" \"'")
            ent = cls.extract_entities(target)
            out = ["### Named Entity Recognition (NER)"]
            if ent.persons:
                out.append(f"• **Persons**: {', '.join(ent.persons)}")
            if ent.organizations:
                out.append(f"• **Organizations**: {', '.join(ent.organizations)}")
            if ent.locations:
                out.append(f"• **Locations**: {', '.join(ent.locations)}")
            if ent.dates_years:
                out.append(f"• **Dates & Years**: {', '.join(ent.dates_years)}")
            if ent.quantities:
                out.append(f"• **Quantities & Metrics**: {', '.join(ent.quantities)}")
            if len(out) == 1:
                out.append("No distinct named entities detected in the provided input.")
            return "\n".join(out)

        return None

    @classmethod
    def process_query(cls, query: str) -> Optional[NLPResult]:
        """Unified dispatch method evaluating conversation, math, and all NLP tasks."""
        conv = cls.handle_conversation(query)
        if conv:
            return NLPResult(intent=NLPIntentType.CONVERSATION, answer=conv, confidence=1.0)

        math_ans = cls.solve_math(query)
        if math_ans:
            return NLPResult(intent=NLPIntentType.MATH, answer=math_ans, confidence=1.0)

        nlp_ans = cls.handle_nlp_task(query)
        if nlp_ans:
            return NLPResult(intent=NLPIntentType.KNOWLEDGE_QA, answer=nlp_ans, confidence=1.0)

        return None
