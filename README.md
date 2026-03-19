# markovonnx

Markov chains and Hidden Markov Models with ONNX export and Kneser-Ney smoothing.

## Let's Be Honest

This library implements algorithms from the 1990s. In 2026, a fine-tuned transformer will beat it on every benchmark that matters. You should probably use sentence-transformers or a small LLM for anything production-critical.

**So why does this exist?**

1. **Baseline benchmarking.** You can't claim your fancy neural model is good unless you know how much better it is than the dumbest thing that could possibly work. markovonnx _is_ that dumb baseline. If your 400MB model only beats an n-gram chain by 3%, you have a problem.

2. **Running on a potato.** Raspberry Pi Zero. ESP32. A browser via WASM. A 15-year-old Android phone. Places where "just pip install torch" isn't an option. markovonnx models are kilobytes and train in milliseconds. They run anywhere ONNX Runtime runs, which is basically everywhere.

3. **When you have 10 training examples.** Neural models need hundreds or thousands of examples. Markov chains give you _something_ useful from 5-50 samples. It won't be great, but it'll be better than random — and for a voice assistant skill with 8 example utterances, that's the reality.

4. **Teaching.** If you're learning NLP, Markov chains and HMMs are where you start. This library gives you a complete, tested, documented implementation with 25 runnable examples.

5. **Fun.** Training a character-level Markov chain on black metal lyrics and letting it generate text is entertaining. The persona chat agent exists because it's funny, not because it's useful.

### What It's Actually Good At

- **Language detection** — char n-gram perplexity is genuinely competitive for this task
- **POS tagging** — HMM Viterbi gets 97.5% on Brown corpus, which was state-of-the-art in 1995
- **Anomaly detection** — "does this log line look normal?" is a legitimate use case for perplexity
- **Providing a floor** — every benchmark needs a lower bound

### What You Should Use Instead

| Task | Use markovonnx if... | Use this instead for production |
|------|---------------------|--------------------------------|
| Intent classification | You have <20 examples, no GPU | fine-tuned BERT, Model2Vec     |
| Language detection | Truly offline, no model downloads | fastText langdetect  |
| POS tagging | Teaching, embedded systems | spaCy, Stanza, flair           |
| Text generation | Fun/creative, persona chatbot | Any LLM                        |
| G2P | Tiny model requirement | espeak, DeepPhonemizer             |
| STT rescoring | Domain-specific vocab boost | Neural LM rescoring, KenLM     |

## Install

```bash
pip install markovonnx              # core (numpy, onnx, onnxruntime)
pip install markovonnx[ovos]        # + OVOS voice assistant plugins
pip install markovonnx[bpe]         # + BPE tokenizer support
pip install markovonnx[quantize]    # + INT8 quantization
```

## Quick Start

```python
from markovonnx import Vocabulary, MarkovChain, export_markov_onnx, MarkovONNXRuntime

# Train
vocab = Vocabulary()
vocab.build_from_sequences(corpus)
mc = MarkovChain(order=2, vocab=vocab, kneser_ney=True, backoff=True)
mc.fit(corpus)

# Export to ONNX
export_markov_onnx(mc, "model.onnx")

# Inference (3.5x faster than Python)
rt = MarkovONNXRuntime("model.onnx", vocab, order=2)
probs = rt.predict_probs(["the", "cat"])
```

## Features

| Feature | Description |
|---------|-------------|
| **Markov Chains** | N-gram models with sparse storage, Modified Kneser-Ney, interpolated backoff |
| **Hidden Markov Models** | Supervised MLE + log-space Baum-Welch, vectorized Viterbi |
| **ONNX Export** | Dense and sparse graphs, INT8 quantization, ~75% compression |
| **Pretrained Models** | save/load for MarkovChain and HMM, NLTK training scripts |
| **CLI** | `markovonnx train/generate/info` |
| **OVOS Plugins** | 8 native OPM entry points |

## 25 Examples

| Category | Examples |
|----------|----------|
| **Text Generation** | char-level, word-level, predictive text, name generator, music |
| **Classification** | language ID, intent, spam/ham, code language, authorship, style scoring |
| **Sequence Tagging** | POS tagging, weather/activity HMM, G2P, slot extraction |
| **Analysis** | perplexity comparison, DNA motif discovery, keyword extraction |
| **Audio** | audio event classifier (speech, bark, doorbell, knock, music) |
| **Operations** | anomaly detection, streaming training, quantization benchmark |

See [examples/](examples/) for all 25 runnable scripts with toy datasets.

## OVOS Integration

8 native [OpenVoiceOS](https://openvoiceos.org) plugins via `markovonnx.opm`:

| Plugin | OPM Entry Point | What It Does |
|--------|-----------------|--------------|
| `MarkovPipeline` | `opm.pipeline` | Intent matching via perplexity ensemble |
| `MarkovUtteranceTransformer` | `opm.transformer.text` | STT rescoring with domain LM |
| `MarkovLangDetector` | `opm.lang.detect` | Language detection |
| `MarkovPosTagger` | `opm.postag` | POS tagging (97.5% on Brown corpus) |
| `MarkovKeywordExtractor` | `opm.keywords` | Keyword extraction |
| `MarkovSegmenter` | `opm.segmentation` | Sentence boundary detection |
| `MarkovG2P` | `opm.g2p` | Grapheme-to-phoneme |
| `MarkovChatEngine` | `opm.agents.chat` | Persona chat (metal lyricist, Shakespeare, pirate) |

All OVOS imports confined to `markovonnx/opm.py`. Core library has zero OVOS deps.

See [docs/ovos-integration.md](docs/ovos-integration.md) for configuration, training, and pretrained model setup.

## Training Scripts

```bash
# POS tagger from NLTK Brown corpus (97.5% accuracy)
python scripts/train_pos_tagger.py --corpus brown -o models/en_pos.json

# Language detection from NLTK UDHR corpus
python scripts/train_lang_detect.py --nltk --langs en fr de es pt -o models/

# G2P from CMUDict
python scripts/train_g2p.py -o models/en_g2p.json
```

See [scripts/README.md](scripts/README.md) for details.

## Documentation

16 pages in [docs/](docs/) covering [getting started](docs/getting-started.md), [usage guide](docs/guide.md) (when to use, when to avoid), [OVOS integration](docs/ovos-integration.md), [API reference](docs/api-reference.md), [architecture](docs/architecture.md), [CLI](docs/cli.md), and more.

## AI Transparency

This project was built entirely by an AI agent with human oversight.

### AI Model

**Claude Opus 4.6** (Anthropic) via Claude Code CLI. All code, tests, documentation, and examples were generated in a single extended conversation.

### Human Oversight

- **Architecture decisions** were human-approved before implementation (plan mode review)
- **Feature lists** were proposed by the AI, reviewed and approved by the human before building
- **Every commit** was staged locally — the human controls all `git push` operations
- **Bug fixes and improvements** were guided by human feedback

### What the AI Did

- Extracted a Jupyter notebook into a standalone library (9 source modules)
- Wrote 150 unit tests achieving 99% code coverage
- Created 25 runnable examples across 10+ domains
- Built 8 OVOS plugins with full OPM integration and E2E tests
- Wrote 16 documentation pages including API reference and usage guide
- Performed comprehensive code audit (31 issues found and fixed)
- Implemented Modified Kneser-Ney smoothing, log-space Baum-Welch, vectorized Viterbi, sparse ONNX export, domain LM STT rescoring

### What the Human Did

- Provided the original notebook with the core algorithms
- Directed the overall vision and feature priorities
- Reviewed and approved plans before implementation
- Made final decisions on architecture
- Controls publishing, pushing, and deployment

Every AI-generated commit includes an `AI-Generated Change` footer with the model name, intent, impact, and verification method.

## License

Apache 2.0
