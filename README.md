# markovonnx

Markov chains and Hidden Markov Models with ONNX export, Kneser-Ney smoothing, and native OVOS plugin support.

## Why This Exists

In an era of billion-parameter transformers, why build a Markov chain library?

**Because not every problem needs a GPU.** Language detection, intent classification, POS tagging, anomaly detection, text generation — these tasks were solved by statistical models decades before deep learning. Those solutions are still valid when you need:

- **Millisecond training** — retrain on new data instantly, no GPU warmup
- **Kilobyte models** — deploy on microcontrollers, phones, browsers (via ONNX/WASM)
- **Zero dependencies at inference** — ONNX Runtime is the only runtime dep, available everywhere
- **Complete offline operation** — no API calls, no cloud, no data leaves the device
- **Interpretable decisions** — perplexity scores are transparent, not black-box embeddings
- **5-50 training examples** — useful accuracy with tiny datasets where neural models can't even overfit

The key insight: **perplexity-as-classifier**. Train one Markov chain per class, classify by lowest perplexity. This one pattern handles language detection, spam filtering, intent classification, authorship attribution, anomaly detection, and audio event classification — all with the same simple mechanism.

markovonnx makes this pattern first-class: train in Python, export to ONNX (dense or sparse), run anywhere.

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
| **Markov Chains** | N-gram models with sparse storage, Kneser-Ney smoothing, interpolated backoff |
| **Hidden Markov Models** | Supervised MLE + log-space Baum-Welch, vectorized Viterbi |
| **ONNX Export** | Dense and sparse graphs, INT8 quantization, ~75% compression |
| **Portable Archives** | `.markov` ZIP format bundles model + vocab + config |
| **CLI** | `markovonnx train/generate/info` |
| **OVOS Plugins** | 7 native OPM entry points (pipeline, lang detect, POS, keywords, segmentation, G2P, chat agent) |

## 25 Examples

| Category | Examples |
|----------|----------|
| **Text Generation** | char-level, word-level, predictive text, name generator, music |
| **Classification** | language ID, intent, spam/ham, code language, authorship, style scoring |
| **Sequence Tagging** | POS tagging, weather/activity HMM, G2P, slot extraction |
| **Analysis** | perplexity comparison, DNA motif discovery, keyword extraction |
| **Audio** | audio event classifier (speech, bark, doorbell, knock, music) |
| **Operations** | anomaly detection, streaming training, quantization benchmark |
| **Infrastructure** | config from env, ONNX export, sentence segmentation |

See [examples/](examples/) for all 25 runnable scripts with toy datasets.

## OVOS Integration

markovonnx provides 7 native [OpenVoiceOS](https://openvoiceos.org) plugins via `markovonnx.opm`:

```bash
pip install markovonnx[ovos]
```

| Plugin | OPM Entry Point | Use Case |
|--------|-----------------|----------|
| `MarkovPipeline` | `opm.pipeline` | Intent matching via perplexity ensemble |
| `MarkovLangDetector` | `opm.lang.detect` | Language detection |
| `MarkovPosTagger` | `opm.postag` | POS tagging via HMM Viterbi |
| `MarkovKeywordExtractor` | `opm.keywords` | Keyword extraction |
| `MarkovSegmenter` | `opm.segmentation` | Sentence boundary detection |
| `MarkovG2P` | `opm.g2p` | Grapheme-to-phoneme conversion |
| `MarkovChatEngine` | `opm.agents.chat` | Persona chat (poet, metal lyricist, pirate) |

All OVOS imports are confined to `markovonnx/opm.py` — the core library has zero OVOS dependencies.

## Documentation

See [docs/](docs/) for 15 pages covering getting started, API reference, architecture, CLI, archive format, and a comprehensive [usage guide](docs/guide.md) with recommendations on when to use (and when not to use) this library.

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
- Wrote 144 unit tests achieving 99% code coverage
- Created 25 runnable examples across 10+ domains
- Built 7 OVOS plugins with full OPM integration and E2E tests
- Wrote 15 documentation pages including API reference and usage guide
- Performed comprehensive code audit (31 issues found and fixed)
- Implemented Modified Kneser-Ney smoothing, log-space Baum-Welch, vectorized Viterbi, sparse ONNX export

### What the Human Did

- Provided the original notebook with the core algorithms
- Directed the overall vision and feature priorities
- Reviewed and approved plans before implementation
- Made final decisions on architecture
- Controls publishing, pushing, and deployment

Every AI-generated commit includes a `AI-Generated Change` footer with the model name, intent, impact, and verification method.

## License

Apache 2.0
