"""Tests for markovonnx.opm — OVOS plugin integrations."""

from markovonnx.opm import (
    MarkovChatEngine,
    MarkovG2P,
    MarkovIntentEngine,
    MarkovKeywordExtractor,
    MarkovLangDetector,
    MarkovPosTagger,
    MarkovSegmenter,
    _normalize,
    _ppx_to_confidence,
)

# re-use OVOS types
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


# ── Helpers ─────────────────────────────────────────────────────────────

class TestHelpers:
    def test_normalize(self) -> None:
        assert _normalize("  Hello  World! ") == "hello world"

    def test_ppx_to_confidence(self) -> None:
        assert _ppx_to_confidence(1.0) == 1.0
        assert 0 < _ppx_to_confidence(100) < 0.3


# ── Intent Engine ───────────────────────────────────────────────────────

class TestMarkovIntentEngine:
    def _engine(self) -> MarkovIntentEngine:
        e = MarkovIntentEngine(order=1, kneser_ney=False, backoff=False)
        e.add_intent("weather:get", [
            "what is the weather", "how is the weather",
            "tell me the weather", "what is the forecast",
        ])
        e.add_intent("timer:set", [
            "set a timer", "start a timer",
            "set a countdown", "timer for five minutes",
        ])
        e.train()
        return e

    def test_classify(self) -> None:
        e = self._engine()
        scores = e.calc_intents("what is the weather")
        assert scores[0][0] == "weather:get"

    def test_empty(self) -> None:
        e = MarkovIntentEngine()
        assert e.calc_intents("hello") == []


# ── Language Detection ──────────────────────────────────────────────────

class TestMarkovLangDetector:
    def test_detect(self) -> None:
        det = MarkovLangDetector()
        det.train_from_samples("en", ["the quick brown fox"] * 5)
        det.train_from_samples("fr", ["le petit prince est"] * 5)
        assert det.detect("the brown fox jumps") == "en"

    def test_available_languages(self) -> None:
        det = MarkovLangDetector()
        det.train_from_samples("en", ["hello world"] * 3)
        assert "en" in det.available_languages


# ── POS Tagging ─────────────────────────────────────────────────────────

class TestMarkovPosTagger:
    def test_postag(self) -> None:
        tagger = MarkovPosTagger()
        tagger.train("en",
                      [["the", "cat", "sat"]] * 5,
                      [["DT", "NN", "VBD"]] * 5)
        tags = tagger.postag([(0, 3, "the"), (4, 7, "cat"), (8, 11, "sat")], "en")
        assert len(tags) == 3
        assert all(len(t) == 4 for t in tags)

    def test_fallback(self) -> None:
        tagger = MarkovPosTagger()
        tags = tagger.postag([(0, 5, "hello")], "xx")
        assert tags[0][3] == "NN"


# ── Keyword Extraction ──────────────────────────────────────────────────

class TestMarkovKeywordExtractor:
    def test_extract(self) -> None:
        ext = MarkovKeywordExtractor({"top_k": 3})
        ext.train("en", ["the cat sat on the mat"] * 10)
        kw = ext.extract("the quantum physicist", lang="en")
        assert isinstance(kw, dict)
        assert "quantum" in kw or "physicist" in kw

    def test_empty(self) -> None:
        ext = MarkovKeywordExtractor()
        assert ext.extract("") == {}


# ── Segmentation ────────────────────────────────────────────────────────

class TestMarkovSegmenter:
    def test_fallback(self) -> None:
        seg = MarkovSegmenter()
        result = seg.segment("Hello world. How are you?", lang="xx")
        assert len(result) == 2

    def test_empty(self) -> None:
        seg = MarkovSegmenter()
        assert seg.segment("") == []


# ── G2P ─────────────────────────────────────────────────────────────────

class TestMarkovG2P:
    def test_get_ipa(self) -> None:
        g2p = MarkovG2P()
        g2p.train("en",
                   [list("cat"), list("bat"), list("hat")] * 5,
                   [["K", "AE", "T"], ["B", "AE", "T"], ["HH", "AE", "T"]] * 5)
        result = g2p.get_ipa("cat", "en")
        assert len(result) == 3

    def test_oov(self) -> None:
        g2p = MarkovG2P()
        assert g2p.get_ipa("hello", "xx", ignore_oov=True) == []


# ── Chat Engine ─────────────────────────────────────────────────────────

class TestMarkovChatEngine:
    def test_continue_chat(self) -> None:
        engine = MarkovChatEngine({"mode": "word", "order": 1, "response_length": 10})
        engine.train_from_samples([
            "darkness falls across the land",
            "creatures crawl in search of blood",
        ] * 5)
        msg = [AgentMessage(role=MessageRole.USER, content="tell me about darkness")]
        response = engine.continue_chat(msg)
        assert response.role == MessageRole.ASSISTANT
        assert len(response.content) > 0

    def test_no_model(self) -> None:
        engine = MarkovChatEngine({})
        response = engine.continue_chat([
            AgentMessage(role=MessageRole.USER, content="hi")
        ])
        assert response.content == "..."

    def test_stream(self) -> None:
        engine = MarkovChatEngine({"mode": "word", "order": 1, "response_length": 5})
        engine.train_from_samples(["hello world test"] * 5)
        tokens = list(engine.stream_tokens([
            AgentMessage(role=MessageRole.USER, content="hello")
        ]))
        assert len(tokens) > 0
