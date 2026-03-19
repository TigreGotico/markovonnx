"""OVOS Plugin Manager (OPM) integrations for markovonnx.

This module provides native OVOS plugin support. All OVOS-specific imports
are confined to this file — the rest of markovonnx has zero OVOS dependencies.

Plugins:
    - ``MarkovPipeline`` — Intent matching via perplexity ensemble (opm.pipeline)
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
import string
from collections import defaultdict
from pathlib import Path
from threading import Event, RLock
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# OVOS imports — all confined to this file
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole
from ovos_plugin_manager.templates.g2p import Grapheme2PhonemePlugin
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_plugin_manager.templates.language import LanguageDetector
from ovos_plugin_manager.templates.pipeline import (
    ConfidenceMatcherPipeline,
    IntentHandlerMatch,
)
from ovos_plugin_manager.templates.postag import PosTagger
from ovos_plugin_manager.templates.segmentation import Segmenter
from ovos_plugin_manager.templates.transformers import UtteranceTransformer
from ovos_utils.fakebus import FakeBus
from ovos_utils.lang import standardize_lang_tag
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
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

class _Stemmer:
    """Snowball stemmer wrapper."""

    _LANGS = {
        "ar": "arabic", "eu": "basque", "ca": "catalan", "da": "danish",
        "nl": "dutch", "en": "english", "fi": "finnish", "fr": "french",
        "de": "german", "el": "greek", "hi": "hindi", "hu": "hungarian",
        "id": "indonesian", "ga": "irish", "it": "italian", "lt": "lithuanian",
        "ne": "nepali", "no": "norwegian", "pt": "portuguese", "ro": "romanian",
        "ru": "russian", "sr": "serbian", "es": "spanish", "sv": "swedish",
        "ta": "tamil", "tr": "turkish",
    }

    def __init__(self, lang: str):
        import snowballstemmer
        lang2 = lang.split("-")[0].lower()
        if lang2 not in self._LANGS:
            raise ValueError(f"Unsupported stemmer language: {lang}")
        self._stemmer = snowballstemmer.stemmer(self._LANGS[lang2])

    @classmethod
    def supports(cls, lang: str) -> bool:
        """Check if stemming is available for *lang*."""
        return lang.split("-")[0].lower() in cls._LANGS

    def stem(self, sentence: str) -> str:
        """Stem all words in a sentence."""
        return " ".join(self._stemmer.stemWords(sentence.split()))


def _normalize(text: str, stemmer: Optional[_Stemmer] = None) -> str:
    """Lowercase, collapse whitespace, strip punctuation, optionally stem."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(string.punctuation)
    if stemmer is not None:
        text = stemmer.stem(text)
    return text


def _ppx_to_confidence(ppx: float) -> float:
    """Convert perplexity to a 0-1 confidence score."""
    if ppx <= 1.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.log(ppx))))


# ═══════════════════════════════════════════════════════════════════════════
#  Intent Engine (used by MarkovPipeline)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovIntentEngine:
    """Per-language intent matching engine using Markov chain perplexity."""

    def __init__(self, order: int = 2, smoothing: float = 1e-5,
                 kneser_ney: bool = True, backoff: bool = True,
                 stemmer: Optional[_Stemmer] = None,
                 char_fallback: bool = False,
                 char_fallback_threshold: float = 0.05):
        self.order = order
        self.smoothing = smoothing
        self.kneser_ney = kneser_ney
        self.backoff = backoff
        self.stemmer = stemmer
        self.char_fallback = char_fallback
        self.char_fallback_threshold = char_fallback_threshold
        self._intent_samples: Dict[str, List[str]] = {}
        self._models: Dict[str, MarkovChain] = {}
        self._char_models: Dict[str, MarkovChain] = {}
        self._vocab: Optional[Vocabulary] = None
        self._char_vocab: Optional[Vocabulary] = None
        self._trained = False

    @property
    def must_train(self) -> bool:
        return not self._trained and len(self._intent_samples) > 0

    def add_intent(self, name: str, samples: List[str]) -> None:
        self._intent_samples[name] = [s.strip() for s in samples if s.strip()]
        self._trained = False

    def remove_intent(self, name: str) -> None:
        self._intent_samples.pop(name, None)
        self._models.pop(name, None)
        self._char_models.pop(name, None)
        self._trained = False

    def train(self) -> None:
        if not self._intent_samples:
            self._trained = True
            return
        all_seqs: List[List[str]] = []
        intent_seqs: Dict[str, List[List[str]]] = {}
        for name, raw in self._intent_samples.items():
            seqs = [word_tokenize(_normalize(s, self.stemmer)) for s in raw
                    if _normalize(s, self.stemmer)]
            intent_seqs[name] = seqs
            all_seqs.extend(seqs)
        self._vocab = Vocabulary()
        self._vocab.build_from_sequences(all_seqs)
        self._models = {}
        for name, seqs in intent_seqs.items():
            if not seqs:
                continue
            mc = MarkovChain(order=self.order, vocab=self._vocab,
                             smoothing=self.smoothing, backoff=self.backoff,
                             kneser_ney=self.kneser_ney)
            mc.fit(seqs)
            self._models[name] = mc
        if self.char_fallback:
            all_char: List[List[str]] = []
            intent_char: Dict[str, List[List[str]]] = {}
            for name, raw in self._intent_samples.items():
                seqs = [char_tokenize(_normalize(s, self.stemmer)) for s in raw
                        if _normalize(s, self.stemmer)]
                intent_char[name] = seqs
                all_char.extend(seqs)
            self._char_vocab = Vocabulary()
            self._char_vocab.build_from_sequences(all_char)
            self._char_models = {}
            for name, seqs in intent_char.items():
                if not seqs:
                    continue
                mc = MarkovChain(order=3, vocab=self._char_vocab,
                                 smoothing=self.smoothing, backoff=True,
                                 kneser_ney=self.kneser_ney)
                mc.fit(seqs)
                self._char_models[name] = mc
        self._trained = True

    def calc_intents(self, utterance: str,
                     blacklisted_intents: Optional[set] = None,
                     blacklisted_skills: Optional[set] = None) -> List[Tuple[str, float]]:
        if not self._models or self._vocab is None:
            return []
        blacklisted_intents = blacklisted_intents or set()
        blacklisted_skills = blacklisted_skills or set()
        norm = _normalize(utterance, self.stemmer)
        tokens = word_tokenize(norm)
        if len(tokens) < self.order:
            return []
        scores: List[Tuple[str, float]] = []
        for name, mc in self._models.items():
            if name in blacklisted_intents:
                continue
            skill_id = name.split(":")[0] if ":" in name else name
            if skill_id in blacklisted_skills:
                continue
            scores.append((name, _ppx_to_confidence(mc.perplexity([tokens]))))
        scores.sort(key=lambda x: -x[1])
        if (self.char_fallback and self._char_models and len(scores) >= 2
                and scores[0][1] - scores[1][1] < self.char_fallback_threshold):
            char_tokens = char_tokenize(norm)
            if len(char_tokens) >= 3:
                cs: Dict[str, float] = {}
                for name, mc in self._char_models.items():
                    if name in blacklisted_intents:
                        continue
                    sid = name.split(":")[0] if ":" in name else name
                    if sid in blacklisted_skills:
                        continue
                    cs[name] = _ppx_to_confidence(mc.perplexity([char_tokens]))
                blended = [(n, 0.6 * wc + 0.4 * cs[n]) if n in cs else (n, wc)
                           for n, wc in scores]
                blended.sort(key=lambda x: -x[1])
                return blended
        return scores

    def update_online(self, intent_name: str, utterance: str) -> None:
        if intent_name not in self._intent_samples:
            return
        self._intent_samples[intent_name].append(utterance.strip())
        if self._vocab is None:
            return
        self._trained = False
        self.train()


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Plugin (opm.pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class MarkovPipeline(ConfidenceMatcherPipeline):
    """OVOS pipeline plugin for Markov chain perplexity-based intent matching."""

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        intent_config = Configuration().get("intents", {})
        config = config or intent_config.get("ovos-markov-pipeline-plugin") or {}
        super().__init__(bus, config)
        self.lock = RLock()
        core_config = Configuration()
        self.lang = standardize_lang_tag(core_config.get("lang", "en-US"))
        langs = core_config.get("secondary_langs") or []
        langs = [standardize_lang_tag(l) for l in langs]
        if self.lang not in langs:
            langs.append(self.lang)
        self.conf_high = self.config.get("conf_high", 0.75)
        self.conf_med = self.config.get("conf_med", 0.55)
        self.conf_low = self.config.get("conf_low", 0.30)
        if not (self.conf_low <= self.conf_med <= self.conf_high):
            LOG.warning(f"Thresholds not ordered: {self.conf_low}/{self.conf_med}/{self.conf_high}")
        order = max(1, int(self.config.get("order", 2)))
        kneser_ney = self.config.get("kneser_ney", True)
        backoff = self.config.get("backoff", True)
        smoothing = self.config.get("smoothing", 1e-5)
        use_stemmer = self.config.get("stem", False)
        char_fallback = self.config.get("char_fallback", False)
        char_fb_thresh = self.config.get("char_fallback_threshold", 0.05)
        self.online_learning = self.config.get("online_learning", False)
        self.stemmers: Dict[str, _Stemmer] = {}
        if use_stemmer:
            for lang in langs:
                if _Stemmer.supports(lang):
                    try:
                        self.stemmers[lang] = _Stemmer(lang)
                    except Exception:
                        pass
        self.engines: Dict[str, MarkovIntentEngine] = {
            lang: MarkovIntentEngine(order=order, smoothing=smoothing,
                                     kneser_ney=kneser_ney, backoff=backoff,
                                     stemmer=self.stemmers.get(lang),
                                     char_fallback=char_fallback,
                                     char_fallback_threshold=char_fb_thresh)
            for lang in langs
        }
        self.first_train = Event()
        self.finished_training_event = Event()
        self.finished_training_event.set()
        self.registered_intents: List[str] = []
        self._skill2intent: Dict[str, List[str]] = defaultdict(list)
        self.max_words = self.config.get("max_words", 50)
        self.bus.on("padatious:register_intent", self.register_intent)
        self.bus.on("detach_intent", self.handle_detach_intent)
        self.bus.on("detach_skill", self.handle_detach_skill)
        self.bus.on("mycroft.skills.train", self.train)
        self.bus.on("intent.service.markov.manifest.get", self.handle_manifest)

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        lang = standardize_lang_tag(lang)
        if lang in self.engines:
            return lang
        prefix = lang.split("-")[0]
        for reg in self.engines:
            if reg.startswith(prefix):
                return reg
        return None

    def register_intent(self, message: Message) -> None:
        skill_id = message.data.get("skill_id") or message.context.get("skill_id") or "anonymous_skill"
        name = message.data.get("name")
        if not name:
            return
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        samples = message.data.get("samples")
        if samples is not None and not isinstance(samples, list):
            samples = list(samples) if hasattr(samples, "__iter__") else None
        file_name = message.data.get("file_name")
        if not samples and file_name:
            try:
                with open(file_name) as f:
                    samples = [line.strip() for line in f.readlines()]
            except (OSError, IOError):
                pass
        if not samples:
            return
        self._skill2intent[skill_id].append(name)
        self.registered_intents.append(name)
        closest = self._get_closest_lang(lang)
        if closest and closest in self.engines:
            self.engines[closest].add_intent(name, samples)
        if self.config.get("instant_train", False) or self.first_train.is_set():
            self.train(message)

    def handle_detach_intent(self, message: Message) -> None:
        intent_name = message.data.get("intent_name")
        if intent_name and intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            for engine in self.engines.values():
                engine.remove_intent(intent_name)

    def handle_detach_skill(self, message: Message) -> None:
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            return
        for intent_name in self._skill2intent.pop(skill_id, []):
            if intent_name in self.registered_intents:
                self.registered_intents.remove(intent_name)
            for engine in self.engines.values():
                engine.remove_intent(intent_name)

    def train(self, message: Optional[Message] = None) -> None:
        if not self.finished_training_event.is_set():
            self.finished_training_event.wait()
        with self.lock:
            if not any(e.must_train for e in self.engines.values()):
                self.bus.emit(Message("mycroft.skills.trained"))
                self.finished_training_event.set()
                return
            self.finished_training_event.clear()
            for lang, engine in self.engines.items():
                if engine.must_train:
                    engine.train()
            self.bus.emit(Message("mycroft.skills.trained"))
            self.finished_training_event.set()
        if not self.first_train.is_set():
            self.first_train.set()

    def _match_level(self, utterances: List[str], limit: float,
                     lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        lang = self._get_closest_lang(lang or self.lang)
        if lang is None or lang not in self.engines:
            return None
        engine = self.engines[lang]
        if not engine._trained:
            return None
        sess = SessionManager.get(message)
        utterances = [u for u in utterances if len(u.split()) <= self.max_words]
        if not utterances:
            return None
        best_intent: Optional[str] = None
        best_conf: float = 0.0
        for utt in utterances:
            scores = engine.calc_intents(utt,
                                         blacklisted_intents=sess.blacklisted_intents,
                                         blacklisted_skills=sess.blacklisted_skills)
            if scores and scores[0][1] > best_conf:
                best_intent, best_conf = scores[0]
        if best_intent is not None and best_conf > limit:
            skill_id = best_intent.split(":")[0] if ":" in best_intent else best_intent
            if self.online_learning and best_conf > self.conf_high:
                with self.lock:
                    engine.update_online(best_intent, utterances[0])
            return IntentHandlerMatch(
                match_type=best_intent,
                match_data={"utterance": utterances[0], "confidence": best_conf},
                skill_id=skill_id, utterance=utterances[0])
        return None

    def match_high(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        return self._match_level(utterances, self.conf_high, lang, message)

    def match_medium(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        return self._match_level(utterances, self.conf_med, lang, message)

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        return self._match_level(utterances, self.conf_low, lang, message)

    def handle_manifest(self, message: Message) -> None:
        self.bus.emit(message.reply("intent.service.markov.manifest",
                                    {"intents": self.registered_intents}))

    def shutdown(self) -> None:
        self.bus.remove("padatious:register_intent", self.register_intent)
        self.bus.remove("detach_intent", self.handle_detach_intent)
        self.bus.remove("detach_skill", self.handle_detach_skill)
        self.bus.remove("mycroft.skills.train", self.train)
        self.bus.remove("intent.service.markov.manifest.get", self.handle_manifest)


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
