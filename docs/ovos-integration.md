# OVOS Integration Guide

markovonnx provides 7 native OVOS plugins via `markovonnx.opm`. All OVOS imports are confined to this single file. The intent pipeline plugin ships separately as [`ovos-markov-pipeline-plugin`](https://github.com/TigreGotico/ovos-markov-pipeline-plugin).

```bash
pip install markovonnx[ovos]
```

## Plugins Overview

| Plugin | OPM Entry Point | Config Section |
|--------|-----------------|----------------|
| [MarkovUtteranceTransformer](#stt-rescoring) | `opm.transformer.text` | `utterance_transformers` |
| [MarkovLangDetector](#language-detection) | `opm.lang.detect` | `lang_detect` |
| [MarkovPosTagger](#pos-tagging) | `opm.postag` | `postag` |
| [MarkovKeywordExtractor](#keyword-extraction) | `opm.keywords` | `keyword_extract` |
| [MarkovSegmenter](#sentence-segmentation) | `opm.segmentation` | `segmentation` |
| [MarkovG2P](#grapheme-to-phoneme) | `opm.g2p` | `g2p` |
| [MarkovChatEngine](#persona-chat) | `opm.agents.chat` | per-persona |

---

## Intent Matching

The OVOS intent pipeline plugin is a separate package,
[`ovos-markov-pipeline-plugin`](https://github.com/TigreGotico/ovos-markov-pipeline-plugin)
(`opm.pipeline` entry point). Install and configure it from that repository.

It builds on the same perplexity-classifier pattern markovonnx provides — see
[guide.md](guide.md) for using `MarkovChain` as a classifier directly.

---

## STT Rescoring

**Entry point**: `opm.transformer.text` → `ovos-markov-utterance-transformer`

Improves STT accuracy by rescoring utterances against a domain language model trained on registered intent samples.

### Configuration

```json
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
```

### How It Works

**Auto-trains** by listening on `padatious:register_intent` and `register_vocab` bus messages. Rebuilds the domain LM when `mycroft.skills.trained` fires.

**Approach A — N-best rescoring**: When STT provides multiple hypotheses, reorders by `(1-α)*position_score + α/perplexity`. Higher `alpha` = more LM influence.

**Approach B — Word correction**: For single-hypothesis STT, checks if replacing any non-domain word with a phonetically similar domain word (edit distance ≤ 2) reduces perplexity significantly.

### Recommendations

- `alpha=0.3-0.5` for balanced acoustic/LM weighting
- `correction_threshold=0.5` means the corrected PPX must be <50% of original to trigger
- Works best on domain-specific vocabulary that STT models haven't seen

---

## Language Detection

**Entry point**: `opm.lang.detect` → `ovos-markov-lang-detect`

Detects language by character-level perplexity ensemble.

### Configuration

```json
{
  "lang_detect": {
    "module": "ovos-markov-lang-detect",
    "ovos-markov-lang-detect": {
      "order": 3,
      "training_data": {
        "en": "/path/to/english_samples.txt",
        "fr": "/path/to/french_samples.txt",
        "de": "/path/to/german_samples.txt"
      }
    }
  }
}
```

### Training

**From config**: provide `training_data` mapping language codes to text files. Trains automatically on init.

**Programmatic**:
```python
from markovonnx.opm import MarkovLangDetector
det = MarkovLangDetector()
det.train_from_samples("en", ["the quick brown fox"] * 20)
det.train_from_samples("fr", ["le petit prince est"] * 20)
print(det.detect("the weather is nice"))  # "en"
```

### Recommendations

- 25+ training sentences per language for reliable detection
- `order=3` works well for European languages
- Char-level captures morphological patterns better than word-level

---

## POS Tagging

**Entry point**: `opm.postag` → `ovos-markov-postag`

HMM-based part-of-speech tagger using Viterbi decoding.

### Configuration

```json
{
  "postag": {
    "module": "ovos-markov-postag",
    "ovos-markov-postag": {
      "smoothing": 1e-5
    }
  }
}
```

### Training

Requires labelled (word, tag) sequences. No pretrained model ships with the plugin.

```python
from markovonnx.opm import MarkovPosTagger
tagger = MarkovPosTagger()
tagger.train("en",
    word_sequences=[["the", "cat", "sat"], ["a", "dog", "ran"]],
    tag_sequences=[["DT", "NN", "VBD"], ["DT", "NN", "VBD"]])
tags = tagger.postag([(0, 3, "the"), (4, 7, "cat"), (8, 11, "sat")], "en")
```

---

## Keyword Extraction

**Entry point**: `opm.keywords` → `ovos-markov-keyword-extract`

Scores words by inverse frequency in a background language model. Rare (surprising) words = keywords.

### Configuration

```json
{
  "keyword_extract": {
    "module": "ovos-markov-keyword-extract",
    "ovos-markov-keyword-extract": {
      "order": 1,
      "top_k": 5
    }
  }
}
```

### Training

Train on general-domain text to establish "background" word frequencies:

```python
from markovonnx.opm import MarkovKeywordExtractor
ext = MarkovKeywordExtractor({"top_k": 3})
ext.train("en", ["the cat sat on the mat", ...] * 100)
print(ext.extract("the quantum physicist discovered antimatter", lang="en"))
# {"quantum": 1.0, "physicist": 1.0, "antimatter": 1.0}
```

---

## Sentence Segmentation

**Entry point**: `opm.segmentation` → `ovos-markov-segmentation`

Detects sentence boundaries by character-level boundary probability scoring.

### Configuration

```json
{
  "segmentation": {
    "module": "ovos-markov-segmentation",
    "ovos-markov-segmentation": {
      "order": 4,
      "threshold": 0.3
    }
  }
}
```

### Training

Train on individual sentences to learn boundary patterns:

```python
from markovonnx.opm import MarkovSegmenter
seg = MarkovSegmenter({"order": 4, "threshold": 0.3})
seg.train("en", ["The cat sat.", "The dog ran.", ...] * 50)
print(seg.segment("Hello world. How are you?", lang="en"))
# ["Hello world.", "How are you?"]
```

Falls back to regex splitting when no model is trained.

---

## Grapheme-to-Phoneme

**Entry point**: `opm.g2p` → `ovos-markov-g2p`

HMM-based character-to-phoneme mapping via Viterbi decoding.

### Configuration

```json
{
  "g2p": {
    "module": "ovos-markov-g2p",
    "ovos-markov-g2p": {
      "smoothing": 1e-5
    }
  }
}
```

### Training

Requires aligned (character, phoneme) pairs of equal length:

```python
from markovonnx.opm import MarkovG2P
g2p = MarkovG2P()
g2p.train("en",
    grapheme_sequences=[list("cat"), list("bat"), list("hat")],
    phoneme_sequences=[["K","AE","T"], ["B","AE","T"], ["HH","AE","T"]])
print(g2p.get_ipa("cat", "en"))  # ["K", "AE", "T"]
```

---

## Persona Chat

**Entry point**: `opm.agents.chat` → `ovos-markov-chat`

Text generation by Markov chain sampling trained on a persona's corpus. Produces stylistically consistent but not semantically coherent responses — creative and entertaining.

### Configuration

```json
{
  "module": "ovos-markov-chat",
  "ovos-markov-chat": {
    "corpus": "/path/to/death_metal_lyrics.txt",
    "mode": "word",
    "order": 2,
    "temperature": 0.7,
    "response_length": 20,
    "kneser_ney": true,
    "seed_from_input": true
  }
}
```

### Persona Ideas

| Persona | Corpus | Mode | Order | Temp |
|---------|--------|------|-------|------|
| Death metal | Lyrics collection | word | 2 | 0.7 |
| Shakespeare | Complete works | char | 5 | 0.6 |
| Fortune cookie | Fortune database | word | 1 | 0.8 |
| Pirate | Pirate phrases | word | 2 | 0.7 |

### Training

**From file** (auto on init):
```json
{"corpus": "/path/to/corpus.txt"}
```

**Programmatic**:
```python
from markovonnx.opm import MarkovChatEngine
engine = MarkovChatEngine({"mode": "word", "order": 2})
engine.train_from_samples(["darkness falls across the land", ...])
```

### Tips

- 100+ training lines for decent output, 1000+ for great output
- Lower temperature (0.3-0.5) for safer, more repetitive text
- Higher temperature (0.8-1.0) for more creative, surprising output
- Word mode for coherent phrases, char mode for creative neologisms
