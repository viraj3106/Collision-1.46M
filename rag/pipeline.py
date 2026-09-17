import re
import time
import math
import ast
from typing import List, Dict, Any, Optional, Tuple
from rag.schemas import SearchItem, SearchResult, SourceItem, RAGRequest, RAGResponse, RAGTokenBudget
from rag.search import (
    BaseSearchProvider, 
    HybridSearchProvider, 
    WikipediaSearchProvider, 
    DuckDuckGoSearchProvider, 
    APIWebSearchProvider, 
    MockSearchProvider
)
from rag.fetch import safe_fetch_webpage
from rag.clean import extract_text_from_html
from rag.rank import LexicalRanker
from rag.context import ContextManager


class ConversationalIntentHandler:
    """
    Handles conversational interactions (greetings, courtesies, identity, acknowledgements, and help)
    instantly with zero latency and natural fluency, preventing web search overthinking on small queries.
    """
    GREETING_PATTERNS = [
        r'^(h[eai]+l+o+|h+i+|h+e+y+|h+a+i+|h+l+o+|h+o+l+a+|howdy|namaste|greetings|yo+|s+u+p+|wassup|whats?\s*up|wsup)(\b|[!?. ])',
        r'^(hi\s+there|hello\s+there|hey\s+there|hey\s+collision|hello\s+collision|hi\s+collision)(\b|[!?. ])',
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
    def match(cls, query: str) -> Optional[str]:
        q = query.lower().strip()
        q_norm = re.sub(r'^[^\w]+|[^\w]+$', '', q)
        
        # 1. Identity & capabilities
        for p in cls.IDENTITY_PATTERNS:
            if re.search(p, q):
                return (
                    "I am **COLLISION**, an ultra-efficient neural AI system designed for intelligent reasoning, "
                    "conversational assistance, mathematical calculations, and grounded knowledge retrieval across open domains."
                )

        # 2. Greetings (hlo, hello, hi, heyy, sup, etc.)
        for p in cls.GREETING_PATTERNS:
            if re.search(p, q) or re.search(p, q_norm):
                return "Hello! I am **COLLISION**, your AI assistant. How can I help you today?"

        # 3. Courtesies, Greetings & Gratitude
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
                    "• **Math & Calculations**: Solve arithmetic expressions, percentages, and formulas.\n"
                    "• **Live Web & Wikipedia Knowledge**: Real-time factual information with verified citations.\n"
                    "• **Science, History & Geography**: Clear explanations of concepts and historical facts.\n\n"
                    "Just type your query and I will provide an answer!"
                )

        return None


class MathEvaluator:
    """
    Safely parses and computes mathematical, unit conversion, geometry, statistics, and arithmetic expressions
    with 100% precision and zero execution vulnerabilities.
    """
    @classmethod
    def evaluate(cls, query: str) -> Optional[str]:
        from collision.nlp import CollisionNLPEngine
        return CollisionNLPEngine.solve_math(query)


from rag.nlp import (
    NLPTextProcessor,
    NLPSentimentAnalyzer,
    NLPEntityExtractor,
    NLPSummarizer,
    NLPAnswerSynthesizer
)


class NLPTaskHandler:
    """
    Dedicated NLP task handler supporting summarization, sentiment analysis,
    NER, keyphrase extraction, topic classification, tone analysis, grammar proofreading,
    semantic comparison, and reading comprehension.
    """
    @classmethod
    def handle(cls, query: str) -> Optional[str]:
        from collision.nlp import CollisionNLPEngine
        return CollisionNLPEngine.handle_nlp_task(query)


class UniversalAnswerSynthesizer:
    """
    Synthesizes retrieved passages into clear, factual, and well-structured responses.
    """
    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        return NLPTextProcessor.clean_text(raw_text)

    @classmethod
    def synthesize(cls, query: str, passages: List[Tuple[SearchItem, str]]) -> str:
        return NLPAnswerSynthesizer.synthesize(query, passages)


class QueryRouter:
    """
    Determines routing mode for universal query answering.
    """
    CREATIVE_KEYWORDS = {
        "write a poem", "write a story", "write python code", "write a function",
        "generate a essay", "continue the story", "compose a song"
    }

    def should_search(self, query: str) -> bool:
        q_lower = query.lower().strip()
        
        # Conversational, Math, or NLP tasks do not need web search
        if ConversationalIntentHandler.match(q_lower) is not None:
            return False
            
        if MathEvaluator.evaluate(q_lower) is not None:
            return False

        if NLPTaskHandler.handle(query) is not None:
            return False
            
        # Explicit creative generation can use direct LM
        if any(kw in q_lower for kw in self.CREATIVE_KEYWORDS):
            return False
            
        # All open-domain factual, question-asking, definitional, or temporal queries use knowledge search
        return True


class RAGPipeline:
    """
    Universal Hybrid Intelligence Pipeline:
    Routes queries across Conversational Intent, Math Evaluation, Multi-Source Knowledge Retrieval,
    and Local Neural Generation.
    """
    def __init__(self, search_provider: Optional[BaseSearchProvider] = None, tokenizer=None, inference_engine=None):
        self.search_provider = search_provider or HybridSearchProvider()
        self.router = QueryRouter()
        self.ranker = LexicalRanker()
        self.context_manager = ContextManager(tokenizer=tokenizer)
        self.inference_engine = inference_engine

    def process_universal(self, query: str, mode: str = "auto", max_tokens: int = 150, engine=None) -> RAGResponse:
        """
        Universal entry point to answer ANY question from greetings ('hi') to complex knowledge queries.
        """
        req = RAGRequest(query=query, mode=mode, max_tokens=max_tokens)
        return self.process(req, engine=engine)

    def process(self, request: RAGRequest, engine=None) -> RAGResponse:
        t0 = time.perf_counter()
        active_engine = engine or self.inference_engine
        mode = request.mode.lower().strip()
        if mode not in ("auto", "on", "off"):
            mode = "auto"

        query_clean = request.query.strip()

        # 1. Check Conversational Intent (Zero Latency)
        if mode != "off":
            conv_response = ConversationalIntentHandler.match(query_clean)
            if conv_response:
                total_latency = (time.perf_counter() - t0) * 1000.0
                return RAGResponse(
                    query=request.query,
                    mode_used=mode,
                    web_search_used=False,
                    context_text="",
                    prompt_formatted=request.query,
                    generated_answer=conv_response,
                    sources=[],
                    total_rag_latency_ms=total_latency
                )

        # 2. Check Math Evaluation
        if mode != "off":
            math_response = MathEvaluator.evaluate(query_clean)
            if math_response:
                total_latency = (time.perf_counter() - t0) * 1000.0
                return RAGResponse(
                    query=request.query,
                    mode_used=mode,
                    web_search_used=False,
                    context_text="",
                    prompt_formatted=request.query,
                    generated_answer=math_response,
                    sources=[],
                    total_rag_latency_ms=total_latency
                )

        # 3. Check Dedicated NLP Tasks (Summarization, Sentiment Analysis, NER)
        if mode != "off":
            nlp_response = NLPTaskHandler.handle(query_clean)
            if nlp_response:
                total_latency = (time.perf_counter() - t0) * 1000.0
                return RAGResponse(
                    query=request.query,
                    mode_used=mode,
                    web_search_used=False,
                    context_text="",
                    prompt_formatted=request.query,
                    generated_answer=nlp_response,
                    sources=[],
                    total_rag_latency_ms=total_latency
                )

        # 4. Determine if knowledge retrieval is needed
        web_search_used = False
        if mode == "off":
            web_search_used = False
        elif mode == "on":
            web_search_used = True
        elif mode == "auto":
            web_search_used = self.router.should_search(query_clean)

        if not web_search_used:
            # Mode OFF or Creative query decided search not required
            total_latency = (time.perf_counter() - t0) * 1000.0
            gen_ans = None
            if active_engine:
                res = active_engine.generate(prompt=request.query, max_tokens=request.max_tokens)
                gen_ans = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
            return RAGResponse(
                query=request.query,
                mode_used=mode,
                web_search_used=False,
                context_text="",
                prompt_formatted=request.query,
                generated_answer=gen_ans or "Query processed successfully.",
                sources=[],
                total_rag_latency_ms=total_latency
            )

        # 4. Multi-Source Knowledge Retrieval
        t_search_start = time.perf_counter()
        search_res = self.search_provider.search(request.query, top_k=request.top_k)
        search_latency = (time.perf_counter() - t_search_start) * 1000.0

        if search_res.error or not search_res.results:
            gen_ans = None
            if active_engine:
                res = active_engine.generate(prompt=request.query, max_tokens=request.max_tokens)
                gen_ans = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
                
            total_latency = (time.perf_counter() - t0) * 1000.0
            fallback_msg = gen_ans or f"No external search results found: {search_res.error or 'Topic not found'}"
            return RAGResponse(
                query=request.query,
                mode_used=mode,
                web_search_used=False,
                context_text="",
                prompt_formatted=request.query,
                generated_answer=fallback_msg,
                sources=[],
                search_latency_ms=search_latency,
                total_rag_latency_ms=total_latency,
                error=search_res.error
            )

        # 5. Extract and Rank passages
        t_fetch_start = time.perf_counter()
        candidate_passages = []
        for item in search_res.results[:request.top_k]:
            if item.snippet:
                candidate_passages.append((item, item.snippet))
                
            # Fetch full page if snippet is short and not mock
            if not getattr(self.search_provider, "is_mock", False) and item.source not in ("mock_web", "wikipedia"):
                html, err = safe_fetch_webpage(item.url, timeout_sec=2.0)
                if html:
                    extracted = extract_text_from_html(html)
                    if extracted:
                        candidate_passages.append((item, extracted))

        fetch_latency = (time.perf_counter() - t_fetch_start) * 1000.0

        t_rank_start = time.perf_counter()
        passages_text = [p[1] for p in candidate_passages]
        ranked_scored = self.ranker.score_passages(request.query, passages_text)
        
        ranked_items = []
        seen_texts = set()
        for score, text in ranked_scored:
            if text in seen_texts:
                continue
            seen_texts.add(text)
            for item, orig_text in candidate_passages:
                if orig_text == text:
                    ranked_items.append((item, text))
                    break

        rank_latency = (time.perf_counter() - t_rank_start) * 1000.0

        # 6. Build Context & Sources
        context_str, formatted_prompt, sources, budget_info = self.context_manager.build_rag_context(
            query=request.query,
            retrieved_passages=ranked_items,
            max_completion_tokens=request.max_tokens
        )

        # 7. Synthesize Verified Grounded Answer
        synthesized_answer = UniversalAnswerSynthesizer.synthesize(request.query, ranked_items)

        total_latency = (time.perf_counter() - t0) * 1000.0

        return RAGResponse(
            query=request.query,
            mode_used=mode,
            web_search_used=True,
            context_text=context_str,
            prompt_formatted=formatted_prompt,
            generated_answer=synthesized_answer,
            sources=sources,
            token_budget=budget_info,
            search_latency_ms=search_latency,
            fetch_latency_ms=fetch_latency,
            rank_latency_ms=rank_latency,
            total_rag_latency_ms=total_latency
        )
