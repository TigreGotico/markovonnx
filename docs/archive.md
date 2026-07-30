# Portable Archives

`save_markov_archive` / `load_markov_archive` (`markovonnx/archive.py`)

## `.markov` Format

A `.markov` file is a ZIP archive containing:

| File | Description |
|------|-------------|
| `model.onnx` | Exported ONNX model |
| `vocab.json` | Serialized `Vocabulary` (full token list) |
| `config.json` | Metadata: model_type, order, smoothing, vocab_size, backoff |

This bundles everything needed to load and run a model in a single portable file.

## Saving

```python
from markovonnx import MarkovChain, Vocabulary, save_markov_archive

vocab = Vocabulary()
vocab.build_from_sequences(corpus)
mc = MarkovChain(order=2, vocab=vocab, kneser_ney=True)
mc.fit(corpus)

save_markov_archive(mc, "model.markov")
```

Works with both `MarkovChain` and `HiddenMarkovModel`.

## Loading

```python
from markovonnx import load_markov_archive, generate_markov

loaded = load_markov_archive("model.markov")

rt = loaded["runtime"]       # MarkovONNXRuntime or HMMONNXRuntime
vocab = loaded["vocab"]      # Vocabulary
config = loaded["config"]    # dict with model_type, order, etc.
onnx_path = loaded["onnx_path"]  # path to extracted ONNX file

# Use for inference
text = generate_markov(rt, "the", length=50, mode="char", order=config["order"])
```

## Config Metadata

For Markov chains:
```json
{
  "model_type": "markov_chain",
  "order": 2,
  "smoothing": 1e-05,
  "vocab_size": 324,
  "backoff": true
}
```

For HMMs:
```json
{
  "model_type": "hmm",
  "n_states": 3,
  "smoothing": 1e-05,
  "vocab_size": 4
}
```

## Cleanup

Extracted temp files are automatically cleaned up on process exit via `atexit`. The ONNX session holds the file open during the process lifetime.

## vs `MarkovONNXRuntime.from_file`

| Feature | `.markov` archive | `from_file` |
|---------|-------------------|-------------|
| Full vocab | Yes | Yes (v0.3+) |
| Config metadata | Yes | Partial (order, vocab_size) |
| Single file | Yes | Yes |
| HMM support | Yes | No |
| Requires onnx import | No | Yes (for metadata parsing) |

---
[← Text Generation](text-generation.md) · [Home](index.md) · [CLI →](cli.md)
