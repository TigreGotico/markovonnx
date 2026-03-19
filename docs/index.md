# markovonnx

Markov chains and Hidden Markov Models with ONNX export and inference.

## Modules

| Module | Key Classes/Functions | Description |
|---|---|---|
| `config` | `MarkovConfig` | Dataclass configuration with env var defaults |
| `tokenizers` | `SubwordTokenizer`, `char_tokenize`, `word_tokenize`, `corpus_iter`, `get_tokenize_fn` | Tokenization and corpus streaming |
| `vocabulary` | `Vocabulary` | Symbol-to-integer mapping with frequency pruning |
| `markov` | `MarkovChain` | N-gram Markov chain (sparse storage, dense export) |
| `hmm` | `HiddenMarkovModel` | Discrete HMM (supervised MLE + Baum-Welch) |
| `onnx_export` | `export_markov_onnx`, `export_hmm_onnx`, `quantize_model` | ONNX model creation and INT8 quantization |
| `onnx_runtime` | `MarkovONNXRuntime`, `HMMONNXRuntime` | ONNX Runtime inference wrappers |
| `generate` | `generate_markov` | Auto-regressive text generation |

## Usage

```python
from markovonnx import Vocabulary, MarkovChain, export_markov_onnx, MarkovONNXRuntime

vocab = Vocabulary()
vocab.build_from_sequences([list("abcabc")] * 100)

mc = MarkovChain(order=2, vocab=vocab)
mc.fit([list("abcabc")] * 100)

export_markov_onnx(mc, "model.onnx")
rt = MarkovONNXRuntime("model.onnx", vocab, order=2)
print(rt.argmax(["a", "b"]))
```

## Dependencies

- **Core**: `numpy`, `onnx`, `onnxruntime`
- **Optional**: `tokenizers` (BPE), `matplotlib` (viz), `onnxruntime-tools` (quantization)
