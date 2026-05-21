"""Tests for markovonnx.opm — OVOS plugin integrations."""

from markovonnx.opm import (
    MarkovChatEngine,
    MarkovG2P,
    MarkovUtteranceTransformer,
    MarkovKeywordExtractor,
    MarkovLangDetector,
    MarkovPosTagger,
    MarkovSegmenter,
)

# re-use OVOS types
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


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


# ── Utterance Transformer ──────────────────────────────────────────────

class TestMarkovUtteranceTransformer:
    def _make_transformer(self) -> MarkovUtteranceTransformer:
        t = MarkovUtteranceTransformer(config={"order": 1, "alpha": 0.9})
        # Manually inject domain data (simulating bus registration)
        from markovonnx import word_tokenize
        samples = [
            "set a timer for five minutes",
            "set a timer for ten minutes",
            "set a timer for three minutes",
            "set a timer for two hours",
            "start a timer for five minutes",
            "activate sleep mode",
            "what is the weather today",
            "play some music",
        ] * 5  # repeat to strengthen learned patterns
        for s in samples:
            tokens = word_tokenize(s.lower())
            t._all_samples.append(tokens)
            t._domain_words.update(tokens)
        t._rebuild_model()
        return t

    def test_nbest_rescoring(self) -> None:
        """Approach A: N-best list reordered by domain LM."""
        t = self._make_transformer()
        utterances = [
            "set a timer for hive minutes",  # STT error: "hive" instead of "five"
            "set a timer for five minutes",   # correct but lower acoustic rank
            "set a time or five minutes",     # worse
        ]
        result, ctx = t.transform(utterances)
        # The domain LM should rank "five minutes" higher
        assert result[0] == "set a timer for five minutes"
        assert ctx.get("markov_rescored") is True

    def test_single_utterance_correction(self) -> None:
        """Approach B: word-level correction on single hypothesis."""
        t = self._make_transformer()
        # "sleek" is edit-distance 2 from "sleep"
        utterances = ["activate sleek mode"]
        result, ctx = t.transform(utterances)
        # Should either correct to "sleep" or pass through unchanged
        assert len(result) >= 1
        if ctx.get("markov_corrected"):
            assert "sleep" in result[0]

    def test_no_model_passthrough(self) -> None:
        """Without a trained model, utterances pass through unchanged."""
        t = MarkovUtteranceTransformer()
        utterances = ["hello world"]
        result, ctx = t.transform(utterances)
        assert result == ["hello world"]
        assert ctx == {}

    def test_empty_utterances(self) -> None:
        t = self._make_transformer()
        result, ctx = t.transform([])
        assert result == []

    def test_domain_word_not_corrected(self) -> None:
        """Words already in domain vocab should not be touched."""
        t = self._make_transformer()
        utterances = ["set a timer for five minutes"]
        result, ctx = t.transform(utterances)
        assert result[0] == "set a timer for five minutes"

    def test_edit_distance(self) -> None:
        assert MarkovUtteranceTransformer._edit_distance("cat", "bat") == 1
        assert MarkovUtteranceTransformer._edit_distance("sleep", "sleek") == 1
        assert MarkovUtteranceTransformer._edit_distance("hello", "hello") == 0
        assert MarkovUtteranceTransformer._edit_distance("", "abc") == 3
