"""OVOS Plugin Manager (OPM) integrations for markovonnx.

This module provides native OVOS plugin support. All OVOS-specific imports
are confined to this file — the rest of markovonnx has zero OVOS dependencies.

The intent pipeline plugin lives in the standalone ``ovos-markov-pipeline-plugin``
package.

Plugins:
    - ``MarkovLangDetector`` — Language detection (opm.lang.detect)
    - ``MarkovPosTagger`` — POS tagging via HMM Viterbi (opm.postag)
    - ``MarkovKeywordExtractor`` — Keyword extraction (opm.keywords)
    - ``MarkovSegmenter`` — Sentence segmentation (opm.segmentation)
    - ``MarkovG2P`` — Grapheme-to-phoneme (opm.g2p)
    - ``MarkovChatEngine`` — Persona chat agent (opm.agents.chat)
    - ``MarkovUtteranceTransformer`` — Domain LM rescoring/correction (opm.transformer.text)
"""

import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# OVOS imports — all confined to this file
from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole
from ovos_plugin_manager.templates.g2p import Grapheme2PhonemePlugin
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_plugin_manager.templates.language import LanguageDetector
from ovos_plugin_manager.templates.postag import PosTagger
from ovos_plugin_manager.templates.segmentation import Segmenter
from ovos_plugin_manager.templates.transformers import UtteranceTransformer
from ovos_utils.log import LOG

# markovonnx imports — no OVOS deps
from markovonnx import (
    HiddenMarkovModel,
    MarkovChain,
    Vocabulary,
    char_tokenize,
    word_tokenize,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Language Detection (opm.lang.detect)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovLangDetector(LanguageDetector):
    """Language detector using character-level Markov chain perplexity."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.order = self.config.get("order", 3)
        self._models: Dict[str, MarkovChain] = {}
        # Load pretrained JSON models: {"en": "/path/to/en_lm.json", ...}
        pretrained = self.config.get("pretrained", {})
        for lang, model_path in pretrained.items():
            if Path(model_path).is_file():
                try:
                    self._models[lang] = MarkovChain.load(model_path)
                    LOG.info(f"Loaded pretrained lang model: {lang}")
                except Exception as e:
                    LOG.error(f"Failed to load pretrained {lang}: {e}")
        # Train from raw text files (skip langs already loaded)
        training_data = self.config.get("training_data", {})
        for lang, path in training_data.items():
            if lang not in self._models and Path(path).is_file():
                self.train_language(lang, path)

    def train_language(self, lang: str, corpus_path: str) -> None:
        """Train a language model from a text file."""
        with open(corpus_path, encoding="utf-8", errors="ignore") as f:
            corpus = [char_tokenize(line) for line in f if line.strip()]
        if not corpus:
            return
        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        mc = MarkovChain(order=self.order, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)
        self._models[lang] = mc

    def train_from_samples(self, lang: str, samples: List[str]) -> None:
        """Train a language model from in-memory samples."""
        corpus = [char_tokenize(s) for s in samples if s.strip()]
        if not corpus:
            return
        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        mc = MarkovChain(order=self.order, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)
        self._models[lang] = mc

    @property
    def available_languages(self) -> Set[str]:
        return set(self._models.keys())

    def detect(self, text: str) -> str:
        probs = self.detect_probs(text)
        if not probs:
            return self.config.get("lang", "en")
        return max(probs, key=probs.get)

    def detect_probs(self, text: str) -> Dict[str, float]:
        if not self._models:
            return {}
        tokens = char_tokenize(text)
        if len(tokens) < self.order:
            return {}
        scores: Dict[str, float] = {}
        for lang, mc in self._models.items():
            scores[lang] = 1.0 / max(mc.perplexity([tokens]), 1e-10)
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores


# ═══════════════════════════════════════════════════════════════════════════
#  POS Tagging (opm.postag)
# ═══════════════════════════════════════════════════════════════════════════

Tag = Tuple[int, int, str, str]


class MarkovPosTagger(PosTagger):
    """POS tagger using HMM Viterbi decoding."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.smoothing = (self.config or {}).get("smoothing", 1e-5)
        self._models: Dict[str, HiddenMarkovModel] = {}
        # Load pretrained HMM models: {"en": "/path/to/en_pos.json"}
        pretrained = (self.config or {}).get("pretrained", {})
        for lang, model_path in pretrained.items():
            if Path(model_path).is_file():
                try:
                    self._models[lang] = HiddenMarkovModel.load(model_path)
                    LOG.info(f"Loaded pretrained POS model: {lang}")
                except Exception as e:
                    LOG.error(f"Failed to load pretrained POS {lang}: {e}")

    def train(self, lang: str, word_sequences: List[List[str]],
              tag_sequences: List[List[str]]) -> None:
        """Train a POS model for a language."""
        vocab = Vocabulary()
        vocab.build_from_sequences(word_sequences)
        all_tags: Set[str] = set()
        for tags in tag_sequences:
            all_tags.update(tags)
        hmm = HiddenMarkovModel(n_states=len(all_tags) + 1,
                                obs_vocab=vocab, smoothing=self.smoothing)
        hmm.fit_supervised(word_sequences, tag_sequences)
        self._models[lang] = hmm

    def postag(self, spans: List[Tuple[int, int, str]],
               lang: Optional[str] = None) -> List[Tag]:
        lang = lang or self.lang
        if lang not in self._models:
            return [(s, e, w, "NN") for s, e, w in spans]
        words = [w for _, _, w in spans]
        tags = self._models[lang].viterbi(words)
        return [(s, e, w, t) for (s, e, w), t in zip(spans, tags)]


# ═══════════════════════════════════════════════════════════════════════════
#  Keyword Extraction (opm.keywords)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovKeywordExtractor(KeywordExtractor):
    """Keyword extractor using inverse frequency surprise scoring."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.order = (self.config or {}).get("order", 1)
        self.top_k = (self.config or {}).get("top_k", 5)
        self._vocabs: Dict[str, Vocabulary] = {}

    def train(self, lang: str, corpus: List[str]) -> None:
        """Train a background language model."""
        sequences = [word_tokenize(s.lower()) for s in corpus if s.strip()]
        if not sequences:
            return
        vocab = Vocabulary()
        vocab.build_from_sequences(sequences)
        self._vocabs[lang] = vocab

    def extract(self, text: str, lang: Optional[str] = None) -> Dict[str, float]:
        lang = lang or self.lang
        tokens = word_tokenize(text.lower())
        if not tokens:
            return {}
        if lang not in self._vocabs:
            return self._fallback(tokens)
        vocab = self._vocabs[lang]
        total_count = sum(vocab._counts.values())
        if total_count == 0:
            return self._fallback(tokens)
        scores: Dict[str, float] = {}
        seen: set = set()
        for tok in tokens:
            if tok in seen or len(tok) <= 2:
                continue
            seen.add(tok)
            count = vocab._counts.get(tok, 0)
            scores[tok] = 1.0 if count == 0 else 1.0 / (1.0 + math.log1p(count))
        if not scores:
            return {}
        mx = max(scores.values())
        if mx > 0:
            scores = {k: v / mx for k, v in scores.items()}
        return dict(sorted(scores.items(), key=lambda x: -x[1])[:self.top_k])

    def _fallback(self, tokens: List[str]) -> Dict[str, float]:
        seen: set = set()
        scores: Dict[str, float] = {}
        for tok in tokens:
            if tok in seen or len(tok) <= 2:
                continue
            seen.add(tok)
            scores[tok] = len(tok) / 20.0
        return dict(sorted(scores.items(), key=lambda x: -x[1])[:self.top_k])


# ═══════════════════════════════════════════════════════════════════════════
#  Sentence Segmentation (opm.segmentation)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovSegmenter(Segmenter):
    """Sentence segmenter using character-level Markov boundary scoring."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.order = (self.config or {}).get("order", 4)
        self.threshold = (self.config or {}).get("threshold", 0.3)
        self._models: Dict[str, MarkovChain] = {}
        self._boundary_ids: Dict[str, List[int]] = {}

    def train(self, lang: str, sentences: List[str]) -> None:
        """Train boundary model from sentence-separated text."""
        corpus = [char_tokenize(s.strip() + "\n") for s in sentences if s.strip()]
        if not corpus:
            return
        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        mc = MarkovChain(order=self.order, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)
        self._models[lang] = mc
        self._boundary_ids[lang] = [vocab.tok2id[c] for c in [".", "!", "?", "\n"]
                                     if c in vocab.tok2id]

    def segment(self, text: str, lang: Optional[str] = None) -> List[str]:
        lang = lang or self.lang
        if lang not in self._models:
            return self._fallback(text)
        mc = self._models[lang]
        bids = self._boundary_ids.get(lang, [])
        if not bids:
            return self._fallback(text)
        tokens = char_tokenize(text)
        if len(tokens) < self.order:
            return [text.strip()] if text.strip() else []
        splits: List[int] = []
        for i in range(self.order, len(tokens)):
            probs = mc._get_probs(tokens[i - self.order:i])
            bp = sum(float(probs[b]) for b in bids if b < len(probs))
            if bp > self.threshold:
                splits.append(i)
        segments: List[str] = []
        prev = 0
        for sp in splits:
            seg = "".join(tokens[prev:sp]).strip()
            if seg:
                segments.append(seg)
            prev = sp
        last = "".join(tokens[prev:]).strip()
        if last:
            segments.append(last)
        return segments or [text.strip()]

    def _fallback(self, text: str) -> List[str]:
        segments = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in segments if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════
#  G2P (opm.g2p)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovG2P(Grapheme2PhonemePlugin):
    """Grapheme-to-phoneme plugin using HMM Viterbi decoding."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.smoothing = (self.config or {}).get("smoothing", 1e-5)
        self._models: Dict[str, HiddenMarkovModel] = {}
        self._trained_langs: Set[str] = set()
        # Load pretrained HMM models: {"en": "/path/to/en_g2p.json"}
        pretrained = (self.config or {}).get("pretrained", {})
        for lang, model_path in pretrained.items():
            if Path(model_path).is_file():
                try:
                    self._models[lang] = HiddenMarkovModel.load(model_path)
                    self._trained_langs.add(lang)
                    LOG.info(f"Loaded pretrained G2P model: {lang}")
                except Exception as e:
                    LOG.error(f"Failed to load pretrained G2P {lang}: {e}")

    @property
    def available_languages(self) -> Set[str]:
        return self._trained_langs

    def train(self, lang: str, grapheme_sequences: List[List[str]],
              phoneme_sequences: List[List[str]]) -> None:
        """Train from aligned character-phoneme pairs."""
        vocab = Vocabulary()
        vocab.build_from_sequences(grapheme_sequences)
        all_phonemes: Set[str] = set()
        for seq in phoneme_sequences:
            all_phonemes.update(seq)
        hmm = HiddenMarkovModel(n_states=len(all_phonemes) + 1,
                                obs_vocab=vocab, smoothing=self.smoothing)
        hmm.fit_supervised(grapheme_sequences, phoneme_sequences)
        self._models[lang] = hmm
        self._trained_langs.add(lang)

    def get_ipa(self, word: str, lang: str, ignore_oov: bool = False) -> List[str]:
        if lang not in self._models:
            if ignore_oov:
                return []
            from ovos_plugin_manager.templates.g2p import OutOfVocabulary
            raise OutOfVocabulary(f"No G2P model for language: {lang}")
        chars = list(word.lower())
        if not chars:
            return []
        return self._models[lang].viterbi(chars)


# ═══════════════════════════════════════════════════════════════════════════
#  Chat Agent (opm.agents.chat)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovChatEngine(ChatEngine):
    """Chat engine that generates responses by sampling a Markov chain.

    Train on a persona corpus (death metal, Shakespeare, pirate, etc.)
    and generate stylistically consistent responses.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.mode: str = self.config.get("mode", "word")
        self.order: int = self.config.get("order", 2)
        self.temperature: float = self.config.get("temperature", 0.7)
        self.response_length: int = self.config.get("response_length", 20)
        self.kneser_ney: bool = self.config.get("kneser_ney", True)
        self.seed_from_input: bool = self.config.get("seed_from_input", True)
        self._model: Optional[MarkovChain] = None
        self._vocab: Optional[Vocabulary] = None
        self._corpus_lines: List[str] = []
        corpus_path = self.config.get("corpus", "")
        if corpus_path and Path(corpus_path).is_file():
            self.train_from_file(corpus_path)

    def train_from_file(self, path: str) -> None:
        """Train from a text file."""
        with open(path, encoding="utf-8", errors="ignore") as f:
            self._corpus_lines = [line.strip() for line in f if line.strip()]
        if not self._corpus_lines:
            return
        self._train_model()

    def train_from_samples(self, samples: List[str]) -> None:
        """Train from in-memory samples."""
        self._corpus_lines = [s.strip() for s in samples if s.strip()]
        if not self._corpus_lines:
            return
        self._train_model()

    def _train_model(self) -> None:
        tokenize = char_tokenize if self.mode == "char" else word_tokenize
        sequences = [tokenize(line) for line in self._corpus_lines]
        sequences = [s for s in sequences if len(s) > self.order]
        self._vocab = Vocabulary()
        self._vocab.build_from_sequences(sequences)
        self._model = MarkovChain(order=self.order, vocab=self._vocab,
                                  smoothing=1e-5, kneser_ney=self.kneser_ney,
                                  backoff=True)
        self._model.fit(sequences)

    def _pick_seed(self, user_text: str) -> str:
        if self.seed_from_input and user_text:
            if self.mode == "char":
                return user_text[-self.order:] if len(user_text) >= self.order else user_text
            words = user_text.lower().split()
            for w in reversed(words):
                if self._vocab and w in self._vocab.tok2id:
                    return w
            return words[-1] if words else ""
        if self._corpus_lines:
            line = random.choice(self._corpus_lines)
            if self.mode == "char":
                return line[:self.order]
            words = line.lower().split()
            return " ".join(words[:self.order]) if words else ""
        return ""

    def _generate(self, seed: str) -> str:
        if self._model is None or self._vocab is None:
            return ""
        tokenize = char_tokenize if self.mode == "char" else word_tokenize
        context = tokenize(seed.lower())
        while len(context) < self.order:
            if self._vocab.id2tok:
                context.insert(0, random.choice(self._vocab.id2tok[1:]))
            else:
                break
        result = list(context)
        for _ in range(self.response_length):
            nxt = self._model.sample(context[-self.order:], self.temperature)
            result.append(nxt)
            context = (context + [nxt])[-self.order:]
        return "".join(result) if self.mode == "char" else " ".join(result)

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        user_text = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                user_text = msg.content
                break
        response = self._generate(self._pick_seed(user_text))
        if not response:
            response = random.choice(self._corpus_lines) if self._corpus_lines else "..."
        return AgentMessage(role=MessageRole.ASSISTANT, content=response)

    def stream_tokens(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> Iterable[str]:
        user_text = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                user_text = msg.content
                break
        seed = self._pick_seed(user_text)
        if self._model is None or self._vocab is None:
            yield "..."
            return
        tokenize = char_tokenize if self.mode == "char" else word_tokenize
        context = tokenize(seed.lower())
        while len(context) < self.order:
            if self._vocab.id2tok:
                context.insert(0, random.choice(self._vocab.id2tok[1:]))
            else:
                break
        yield "".join(context) if self.mode == "char" else " ".join(context)
        for _ in range(self.response_length):
            nxt = self._model.sample(context[-self.order:], self.temperature)
            context = (context + [nxt])[-self.order:]
            yield nxt if self.mode == "char" else " " + nxt


# ═══════════════════════════════════════════════════════════════════════════
#  Utterance Transformer — Domain LM rescoring (opm.transformer.text)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovUtteranceTransformer(UtteranceTransformer):
    """Rescore/correct STT utterances using a domain language model.

    Builds a word-level Markov chain from all registered intent samples.
    On each utterance:

    **Approach A (N-best rescoring)**: If the utterance list contains
    multiple STT hypotheses, reorders them by combined acoustic + LM score.

    **Approach B (word correction)**: For each word in the top utterance,
    checks if replacing it with a phonetically similar word from the
    domain vocab reduces perplexity. Only substitutes on significant drop.

    Configuration:

    .. code-block:: json

        {
            "utterance_transformers": {
                "ovos-markov-utterance-transformer": {
                    "active": true,
                    "alpha": 0.4,
                    "correction_threshold": 0.5,
                    "order": 2,
                    "kneser_ney": true
                }
            }
        }

    ``alpha`` controls the LM weight in N-best rescoring:
    ``final = (1-alpha) * position_score + alpha / ppx``
    """

    def __init__(self, name: str = "ovos-markov-utterance-transformer",
                 priority: int = 5, config: Optional[Dict] = None):
        super().__init__(name, priority, config)
        self.alpha: float = self.config.get("alpha", 0.4)
        self.correction_threshold: float = self.config.get("correction_threshold", 0.5)
        self.order: int = self.config.get("order", 2)
        self.kneser_ney: bool = self.config.get("kneser_ney", True)
        self._model: Optional[MarkovChain] = None
        self._vocab: Optional[Vocabulary] = None
        self._domain_words: Set[str] = set()
        self._all_samples: List[List[str]] = []

    def initialize(self) -> None:
        """Register bus handlers to collect intent samples for the domain LM."""
        if self.bus is None:
            return
        self.bus.on("padatious:register_intent", self._handle_register)
        self.bus.on("register_vocab", self._handle_vocab)
        self.bus.on("mycroft.skills.trained", self._rebuild_model)

    def _handle_register(self, message) -> None:
        """Collect intent samples for domain LM training."""
        samples = message.data.get("samples")
        file_name = message.data.get("file_name")
        if not samples and file_name:
            try:
                with open(file_name) as f:
                    samples = [line.strip() for line in f.readlines()]
            except (OSError, IOError):
                return
        if samples:
            for s in samples:
                tokens = word_tokenize(s.lower().strip())
                if tokens:
                    self._all_samples.append(tokens)
                    self._domain_words.update(tokens)

    def _handle_vocab(self, message) -> None:
        """Collect vocabulary words from Adapt registrations."""
        entity = message.data.get("entity_value", "")
        if entity:
            words = word_tokenize(entity.lower())
            self._domain_words.update(words)

    def _rebuild_model(self, message=None) -> None:
        """Rebuild domain LM from accumulated samples."""
        if not self._all_samples:
            return
        self._vocab = Vocabulary()
        self._vocab.build_from_sequences(self._all_samples)
        self._model = MarkovChain(
            order=self.order, vocab=self._vocab,
            smoothing=1e-5, kneser_ney=self.kneser_ney, backoff=True,
        )
        self._model.fit(self._all_samples)
        LOG.info(
            f"MarkovUtteranceTransformer rebuilt: {len(self._all_samples)} samples, "
            f"vocab={self._vocab.size}, domain_words={len(self._domain_words)}"
        )

    def transform(self, utterances: List[str],
                  context: Optional[Dict] = None) -> Tuple[List[str], Dict]:
        """Rescore and optionally correct utterances using domain LM.

        Args:
            utterances: List of STT hypotheses (first = best acoustic).
            context: Message context.

        Returns:
            Tuple of (reordered/corrected utterances, additional context).
        """
        context = context or {}
        if not self._model or not self._vocab or not utterances:
            return utterances, {}

        # === Approach A: N-best rescoring ===
        if len(utterances) > 1:
            scored = []
            for i, utt in enumerate(utterances):
                tokens = word_tokenize(utt.lower())
                if len(tokens) < self.order:
                    scored.append((utt, 0.0))
                    continue
                ppx = self._model.perplexity([tokens])
                # Position bonus: first hypothesis gets slight boost
                position_score = 1.0 / (1 + i)
                lm_score = 1.0 / max(ppx, 1e-10)
                combined = (1 - self.alpha) * position_score + self.alpha * lm_score
                scored.append((utt, combined))

            scored.sort(key=lambda x: -x[1])
            reordered = [utt for utt, _ in scored]

            if reordered[0] != utterances[0]:
                LOG.debug(
                    f"MarkovUTT rescore: '{utterances[0]}' → '{reordered[0]}'"
                )

            return reordered, {"markov_rescored": True}

        # === Approach B: Word-level correction (single hypothesis) ===
        utt = utterances[0]
        tokens = word_tokenize(utt.lower())
        if len(tokens) < self.order:
            return utterances, {}

        original_ppx = self._model.perplexity([tokens])
        corrected_tokens = list(tokens)
        made_correction = False

        for i, tok in enumerate(tokens):
            if tok in self._domain_words:
                continue  # already a domain word, skip

            # Find phonetically similar domain words (edit distance <= 2)
            candidates = self._find_similar(tok)
            if not candidates:
                continue

            best_candidate = None
            best_ppx = original_ppx

            for candidate in candidates:
                trial = list(corrected_tokens)
                trial[i] = candidate
                trial_ppx = self._model.perplexity([trial])
                if trial_ppx < best_ppx * self.correction_threshold:
                    best_ppx = trial_ppx
                    best_candidate = candidate

            if best_candidate is not None:
                LOG.debug(
                    f"MarkovUTT correct: '{corrected_tokens[i]}' → '{best_candidate}' "
                    f"(ppx {original_ppx:.1f} → {best_ppx:.1f})"
                )
                corrected_tokens[i] = best_candidate
                made_correction = True

        if made_correction:
            corrected = " ".join(corrected_tokens)
            return [corrected] + utterances, {"markov_corrected": True}

        return utterances, {}

    def _find_similar(self, word: str, max_dist: int = 2) -> List[str]:
        """Find domain words within edit distance *max_dist* of *word*."""
        results = []
        for dw in self._domain_words:
            if abs(len(dw) - len(word)) > max_dist:
                continue
            if self._edit_distance(word, dw) <= max_dist:
                results.append(dw)
        return results

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Levenshtein edit distance."""
        if len(a) < len(b):
            return MarkovUtteranceTransformer._edit_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if ca == cb else 1),
                ))
            prev = curr
        return prev[len(b)]

    def default_shutdown(self) -> None:
        """Remove bus handlers."""
        if self.bus:
            self.bus.remove("padatious:register_intent", self._handle_register)
            self.bus.remove("register_vocab", self._handle_vocab)
            self.bus.remove("mycroft.skills.trained", self._rebuild_model)
