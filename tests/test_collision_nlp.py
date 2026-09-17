import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.nlp import (
    CollisionNLPProcessor,
    CollisionNLPEngine,
    TextRankKeyphraseExtractor,
    TopicClassifier,
    ToneAnalyzer,
    SemanticSimilarityCalculator,
    GrammarProofreader,
    ReadingComprehensionEngine,
    TextTransformer,
    NLPIntentType
)


class TestNLPProcessor:
    def test_sentence_segmentation(self):
        text = "Dr. Smith arrived in Washington D.C. at 5 p.m. He met with prof. Alan Turing."
        sentences = CollisionNLPProcessor.segment_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "Alan Turing" in sentences[1]

    def test_tokenization_and_stopwords(self):
        text = "The rapid neural inference engine processes 10,000 tokens per second."
        tokens = CollisionNLPProcessor.tokenize_words(text, remove_stopwords=True)
        assert "the" not in tokens
        assert "rapid" in tokens
        assert "neural" in tokens

    def test_lemmatization(self):
        assert CollisionNLPProcessor.lemmatize_word("running") == "run"
        assert CollisionNLPProcessor.lemmatize_word("faster") == "fast"
        assert CollisionNLPProcessor.lemmatize_word("processes") == "process"
        assert CollisionNLPProcessor.lemmatize_word("children") == "child"
        assert CollisionNLPProcessor.lemmatize_word("calculated") == "calculate"
        assert CollisionNLPProcessor.lemmatize_word("better") == "good"

    def test_pos_tagging(self):
        text = "Linus Torvalds quickly wrote the revolutionary software code."
        pos_tags = CollisionNLPProcessor.tag_pos(text)
        assert len(pos_tags) >= 7
        tag_dict = {p.word: p.pos for p in pos_tags}
        assert tag_dict.get("quickly") == "ADV"
        assert tag_dict.get("wrote") == "VERB"
        assert tag_dict.get("the") == "DET"
        assert tag_dict.get("software") in ("NOUN", "ADJ")

    def test_readability_metrics(self):
        text = "Artificial intelligence and deep neural networks have fundamentally transformed computational linguistics."
        metrics = CollisionNLPProcessor.compute_readability(text)
        assert metrics.word_count > 5
        assert metrics.sentence_count >= 1
        assert 0.0 <= metrics.flesch_reading_ease <= 100.0
        assert metrics.flesch_kincaid_grade > 0
        assert 0.0 < metrics.lexical_diversity_ttr <= 1.0

    def test_language_detection(self):
        en_res = CollisionNLPProcessor.detect_language("The quick brown fox jumps over the lazy dog in the field.")
        assert en_res.language == "English"
        assert en_res.language_code == "en"
        assert en_res.confidence > 0.70

        es_res = CollisionNLPProcessor.detect_language("La casa de mi amigo es muy grande y hermosa en la ciudad.")
        assert es_res.language == "Spanish"
        assert es_res.language_code == "es"

    def test_levenshtein_and_soundex(self):
        dist = CollisionNLPProcessor.levenshtein_distance("kitten", "sitting")
        assert dist == 3
        s1 = CollisionNLPProcessor.soundex("Robert")
        s2 = CollisionNLPProcessor.soundex("Rupert")
        assert s1 == s2 == "R163"


class TestNLPAnalytics:
    def test_textrank_keyphrases(self):
        text = (
            "Quantum computing utilizes quantum states and superposition to perform complex mathematical calculations. "
            "Quantum algorithms run exponentially faster on quantum hardware compared to classical computing systems."
        )
        kp = TextRankKeyphraseExtractor.extract(text, top_n=3)
        assert len(kp.keyphrases) > 0
        assert any("quantum" in p.lower() for p in kp.keyphrases)
        assert len(kp.top_keywords) > 0

    def test_topic_classification(self):
        cs_text = "The deep learning transformer architecture optimizes neural network weights using backpropagation on GPUs."
        res_cs = TopicClassifier.classify(cs_text)
        assert res_cs.primary_topic == "Computer Science & AI"
        assert res_cs.confidence > 0.60

        bio_text = "Photosynthesis enables plants to convert sunlight into chemical energy via chlorophyll in cellular chloroplasts."
        res_bio = TopicClassifier.classify(bio_text)
        assert res_bio.primary_topic == "Chemistry & Biology"

        fin_text = "The company reported record quarterly revenue, profit margins, and increased shareholder dividends on stock markets."
        res_fin = TopicClassifier.classify(fin_text)
        assert res_fin.primary_topic == "Finance & Business"

    def test_tone_and_formality(self):
        formal_text = "Furthermore, empirical evidence demonstrates substantial performance advantages regarding the proposed methodology."
        formal_res = ToneAnalyzer.analyze(formal_text)
        assert formal_res.formality_score > 65
        assert formal_res.is_objective is True

        casual_text = "Hey bro, that was super cool and awesome, gotta check it out asap!"
        casual_res = ToneAnalyzer.analyze(casual_text)
        assert casual_res.formality_score < 50
        assert "Casual" in casual_res.primary_tone or "Urgent" in casual_res.primary_tone

    def test_semantic_similarity(self):
        t1 = "Quantum computing relies on qubits, superposition, and entanglement."
        t2 = "Quantum computers use qubits and quantum states like superposition to process data."
        sim = SemanticSimilarityCalculator.compare(t1, t2)
        assert sim.overall_similarity > 0.50
        assert "qubits" in sim.shared_keywords or "qubit" in sim.shared_keywords or "quantum" in sim.shared_keywords

    def test_grammar_proofreader(self):
        text = "This is a apple on the the table ."
        proof = GrammarProofreader.proofread(text)
        assert proof.issues_found >= 2
        assert "an apple" in proof.corrected_text
        assert "the table" in proof.corrected_text
        assert "table." in proof.corrected_text


class TestNLPComprehension:
    def test_reading_comprehension_qa(self):
        context = (
            "The Apollo 11 mission was launched on July 16, 1969. "
            "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on the Moon on July 20, 1969. "
            "Armstrong became the first human to walk on the lunar surface."
        )
        q1 = "Who became the first human to walk on the lunar surface?"
        ans1 = ReadingComprehensionEngine.answer_question(context, q1)
        assert ans1.context_found is True
        assert "Armstrong" in ans1.answer
        assert ans1.confidence > 0.70

        q2 = "When did they land on the Moon?"
        ans2 = ReadingComprehensionEngine.answer_question(context, q2)
        assert "July 20, 1969" in ans2.answer or "1969" in ans2.answer

    def test_text_transformations(self):
        casual = "hlo bro this is gonna be super cool"
        form = TextTransformer.formalize(casual)
        assert "going to" in form.transformed_text

        complex_t = "We utilize this methodology to elucidate the paradigm."
        simple = TextTransformer.simplify(complex_t)
        assert "use" in simple.transformed_text
        assert "explain" in simple.transformed_text

        bullets = TextTransformer.bulletize("First point. Second point. Third point.")
        assert bullets.transformed_text.count("•") >= 2


class TestNLPEngineMathAndTasks:
    def test_expanded_math_statistics(self):
        assert "30" in CollisionNLPEngine.solve_math("mean of 10, 20, 30, 40, 50")
        assert "25" in CollisionNLPEngine.solve_math("median of 10, 20, 30, 40")

    def test_expanded_math_geometry(self):
        circle = CollisionNLPEngine.solve_math("area of circle with radius 10")
        assert circle is not None
        assert "314" in circle

        rect = CollisionNLPEngine.solve_math("area of rectangle 5 by 20")
        assert rect is not None
        assert "100" in rect

    def test_expanded_math_conversions(self):
        temp = CollisionNLPEngine.solve_math("100 celsius to fahrenheit")
        assert temp is not None
        assert "212" in temp

        storage = CollisionNLPEngine.solve_math("2 tb in gb")
        assert storage is not None
        assert "2048" in storage

        speed = CollisionNLPEngine.solve_math("100 km/h in mph")
        assert speed is not None
        assert "62" in speed

    def test_handle_nlp_task_dispatching(self):
        # Keyphrases
        kp_res = CollisionNLPEngine.handle_nlp_task("Extract keywords from: Neural network backpropagation on GPU clusters.")
        assert "Keyphrase Extraction" in kp_res

        # Topic
        topic_res = CollisionNLPEngine.handle_nlp_task("Classify topic: Clinical trials for vaccine immunity and antibody treatments.")
        assert "Medicine & Health" in topic_res

        # Tone
        tone_res = CollisionNLPEngine.handle_nlp_task("Analyze tone of: Furthermore, we establish the quantitative correlation.")
        assert "Tone & Formality" in tone_res

        # Proofread
        proof_res = CollisionNLPEngine.handle_nlp_task("Proofread: I ate a apple in the the kitchen .")
        assert "Grammar & Proofreading Report" in proof_res
        assert "an apple" in proof_res

        # Readability
        read_res = CollisionNLPEngine.handle_nlp_task("Compute readability: The quick brown fox jumps over the lazy dog.")
        assert "Flesch Reading Ease" in read_res

        # Language ID
        lang_res = CollisionNLPEngine.handle_nlp_task("Detect language: La maison blanche est magnifique avec les fleurs.")
        assert "French" in lang_res or "Language Identification" in lang_res

        # Similarity
        sim_res = CollisionNLPEngine.handle_nlp_task("Compare texts: Machine learning models require training data vs Deep learning systems need datasets")
        assert "Semantic Text Similarity" in sim_res

        # Context QA
        ctx_res = CollisionNLPEngine.handle_nlp_task("Context: Python was created by Guido van Rossum in 1991. Question: Who created Python?")
        assert "Guido van Rossum" in ctx_res
