# markovonnx

Markov chains and Hidden Markov Models with ONNX export and Kneser-Ney smoothing.

## What It Is

markovonnx trains classical n-gram Markov chains and Hidden Markov Models and exports them to ONNX for fast inference. It does not use neural networks or a GPU. For most tasks, a fine-tuned transformer gets better accuracy than an n-gram model. Use markovonnx where a transformer is not practical, or as a baseline to measure a larger model against.

### When to Use markovonnx

1. **Baseline benchmarking.** Compare a larger model against the simplest thing that could work. If a 400 MB model only beats an n-gram chain by 3%, that result tells you something about the larger model.
2. **Constrained hardware.** Raspberry Pi Zero, ESP32, a browser through WASM, or an older Android phone. On these targets, installing a full deep learning stack is not an option. markovonnx models are kilobyte-sized and train in milliseconds. They run anywhere ONNX Runtime runs.
3. **Small datasets.** Neural models need hundreds or thousands of examples. A Markov chain gives a usable result from 5-50 samples, for example a voice assistant skill with 8 example utterances.
4. **Teaching.** The library gives a complete, tested, documented implementation of Markov chains and HMMs, with 25 runnable examples.
5. **Text generation for fun.** Training a character-level Markov chain on a small corpus and sampling from it produces stylistically consistent but not meaningful text.

### What It Is Good At

- **Language detection**: character n-gram perplexity is a standard, competitive approach for this task.
- **POS tagging**: HMM Viterbi decoding reaches 97.5% accuracy on the Brown corpus.
- **Anomaly detection**: perplexity is a reasonable score for "does this line look normal".
- **Providing a baseline**: every benchmark needs a lower bound to compare against.

### When to Use Something Else

| Task | Use markovonnx if... | Use this instead for production |
|------|---------------------|--------------------------------|
| Intent classification | You have <20 examples, no GPU | fine-tuned BERT, Model2Vec |
| Language detection | Truly offline, no model downloads | fastText, langdetect |
| POS tagging | Teaching, embedded systems | spaCy, Stanza, flair |
| Text generation | Creative or persona chatbot | Any LLM |
| G2P | Tiny model requirement | espeak, DeepPhonemizer |
| STT rescoring | Domain-specific vocabulary boost | Neural LM rescoring, KenLM |

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
| **Hidden Markov Models** | Supervised MLE and log-space Baum-Welch, vectorized Viterbi |
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

markovonnx provides 7 native [OpenVoiceOS](https://openvoiceos.org) plugins through `markovonnx.opm`.

| Plugin | OPM Entry Point | What It Does |
|--------|-----------------|---------------|
| `MarkovUtteranceTransformer` | `opm.transformer.text` | STT rescoring with domain LM |
| `MarkovLangDetector` | `opm.lang.detect` | Language detection |
| `MarkovPosTagger` | `opm.postag` | POS tagging (97.5% on Brown corpus) |
| `MarkovKeywordExtractor` | `opm.keywords` | Keyword extraction |
| `MarkovSegmenter` | `opm.segmentation` | Sentence boundary detection |
| `MarkovG2P` | `opm.g2p` | Grapheme-to-phoneme |
| `MarkovChatEngine` | `opm.agents.chat` | Persona chat (metal lyricist, Shakespeare, pirate) |

All OVOS imports stay confined to `markovonnx/opm.py`. The core library has no OVOS dependencies.

The OVOS intent pipeline plugin ships as a separate package, [`ovos-markov-pipeline-plugin`](https://github.com/TigreGotico/ovos-markov-pipeline-plugin).

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

16 pages in [docs/](docs/) cover the library end to end. Start with [getting started](docs/getting-started.md) and the [usage guide](docs/guide.md) (when to use markovonnx, when to avoid it). See also [OVOS integration](docs/ovos-integration.md), the [API reference](docs/api-reference.md), [architecture](docs/architecture.md), and the [CLI](docs/cli.md).

## Related Projects

- [`ovos-markov-pipeline-plugin`](https://github.com/TigreGotico/ovos-markov-pipeline-plugin): OVOS intent pipeline built on markovonnx.
- [OpenVoiceOS](https://openvoiceos.org): the voice assistant platform the OVOS plugins integrate with.

## AI Transparency

An AI agent built this project under human oversight.

### AI Model

**Claude Opus 4.6** (Anthropic), through the Claude Code CLI. The AI generated all code, tests, documentation, and examples in a single extended conversation.

### Human Oversight

- The human approved architecture decisions before implementation (plan mode review).
- The AI proposed feature lists. The human reviewed and approved them before building.
- The human staged every commit locally and controls all `git push` operations.
- Human feedback guided bug fixes and improvements.

### What the AI Did

- Extracted a Jupyter notebook into a standalone library (9 source modules).
- Wrote 150 unit tests reaching 99% code coverage.
- Created 25 runnable examples across 10+ domains.
- Built 8 OVOS plugins with full OPM integration and end-to-end tests.
- Wrote 16 documentation pages, including an API reference and usage guide.
- Audited the code and fixed 31 issues found in that audit.
- Implemented Modified Kneser-Ney smoothing, log-space Baum-Welch, vectorized Viterbi, sparse ONNX export, and domain LM STT rescoring.

### What the Human Did

- Provided the original notebook with the core algorithms.
- Directed the overall vision and feature priorities.
- Reviewed and approved plans before implementation.
- Made final decisions on architecture.
- Controls publishing, pushing, and deployment.

Every AI-generated commit includes an `AI-Generated Change` footer with the model name, intent, impact, and verification method.

## License

Apache 2.0
