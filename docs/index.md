# markovonnx

Markov chains and Hidden Markov Models with ONNX export and inference.

## Installation

```bash
pip install -e .            # core only
pip install -e ".[bpe]"     # + BPE tokenizer support
pip install -e ".[quantize]" # + INT8 quantization
pip install -e ".[test]"    # + pytest
```

**Requirements**: Python 3.10+, numpy, onnx, onnxruntime.

## Documentation

| Document | Description |
|----------|-------------|
| [When to Use](guide.md) | Use cases, recommendations, when to avoid, alternatives |
| [Getting Started](getting-started.md) | First steps: install, train, export, generate |
| [Configuration](configuration.md) | `MarkovConfig` fields and environment variables |
| [Tokenization](tokenization.md) | Character, word, and BPE tokenization modes |
| [Vocabulary](vocabulary.md) | `Vocabulary` class: building, encoding, decoding |
| [Markov Chains](markov-chains.md) | `MarkovChain` training, sampling, perplexity |
| [Hidden Markov Models](hmm.md) | `HiddenMarkovModel`: supervised, Baum-Welch, Viterbi |
| [ONNX Export](onnx-export.md) | Dense, sparse, and INT8 quantized ONNX export |
| [ONNX Inference](onnx-inference.md) | `MarkovONNXRuntime`, `from_file`, batch inference |
| [Text Generation](text-generation.md) | `generate_markov` auto-regressive generation |
| [Portable Archives](archive.md) | `.markov` save/load format for model distribution |
| [CLI](cli.md) | `markovonnx train/generate/info` command-line interface |
| [OVOS Integration](ovos-integration.md) | All 8 OVOS plugins: config, training, recommendations |
| [API Reference](api-reference.md) | Complete public API with signatures and source links |
| [Architecture](architecture.md) | ONNX graph structure and design decisions |

## Module Map

| Module | Key Symbols | Source |
|--------|-------------|--------|
| `config` | `MarkovConfig` | `markovonnx/config.py` |
| `tokenizers` | `SubwordTokenizer`, `char_tokenize`, `word_tokenize`, `corpus_iter`, `get_tokenize_fn` | `markovonnx/tokenizers.py` |
| `vocabulary` | `Vocabulary` | `markovonnx/vocabulary.py` |
| `markov` | `MarkovChain` | `markovonnx/markov.py` |
| `hmm` | `HiddenMarkovModel` | `markovonnx/hmm.py` |
| `onnx_export` | `export_markov_onnx`, `export_markov_sparse_onnx`, `export_hmm_onnx`, `quantize_model` | `markovonnx/onnx_export.py` |
| `onnx_runtime` | `MarkovONNXRuntime`, `HMMONNXRuntime` | `markovonnx/onnx_runtime.py` |
| `generate` | `generate_markov` | `markovonnx/generate.py` |
| `archive` | `save_markov_archive`, `load_markov_archive` | `markovonnx/archive.py` |
| `cli` | `main` (entry point: `markovonnx`) | `markovonnx/cli.py` |
