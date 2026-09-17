"""
COLLISION NLP Module — Industrial-Strength NLP subsystem for COLLISION.
"""

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
    ProofreadResult,
    GrammarIssue
)
from collision.nlp.comprehension import (
    ReadingComprehensionEngine,
    TextTransformer,
    ContextQAResult,
    TextTransformationResult
)
from collision.nlp.engine import (
    CollisionNLPEngine,
    NLPIntentType,
    NLPResult,
    NLPSentimentResult,
    NLPEntityResult,
    NLPSummaryResult
)

__all__ = [
    "CollisionNLPProcessor",
    "POSWord",
    "ReadabilityMetrics",
    "LanguageDetectionResult",
    "TextRankKeyphraseExtractor",
    "TopicClassifier",
    "ToneAnalyzer",
    "SemanticSimilarityCalculator",
    "GrammarProofreader",
    "KeyphraseResult",
    "TopicClassificationResult",
    "ToneAnalysisResult",
    "SemanticSimilarityResult",
    "ProofreadResult",
    "GrammarIssue",
    "ReadingComprehensionEngine",
    "TextTransformer",
    "ContextQAResult",
    "TextTransformationResult",
    "CollisionNLPEngine",
    "NLPIntentType",
    "NLPResult",
    "NLPSentimentResult",
    "NLPEntityResult",
    "NLPSummaryResult"
]
