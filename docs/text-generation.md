# Text Generation

`generate_markov` (`markovonnx/generate.py:9`)

Auto-regressive text generation using a `MarkovONNXRuntime`.

## Signature

```python
generate_markov(
    ort_model: MarkovONNXRuntime,
    seed: str,
    length: int,
    temperature: float = 1.0,
    mode: str = "char",
    order: int = 2,
    bpe_tokenizer: Optional[SubwordTokenizer] = None,
) -> str
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ort_model` | `MarkovONNXRuntime` | Loaded ONNX inference session |
| `seed` | `str` | Initial text to seed generation |
| `length` | `int` | Number of tokens to generate |
| `temperature` | `float` | Sampling temperature (default 1.0) |
| `mode` | `str` | `"char"`, `"word"`, or `"bpe"` |
| `order` | `int` | Context window (must match model) |
| `bpe_tokenizer` | `SubwordTokenizer` | Required for `mode="bpe"` |

## Mode Behaviour

### Character mode (`mode="char"`)

- Seed is split into characters: `"the" → ["t", "h", "e"]`
- Default seed: `"the "`
- Output: characters joined directly (`"".join(result)`)

### Word mode (`mode="word"`)

- Seed is split on whitespace: `"the cat" → ["the", "cat"]`
- Default seed: `["the"]`
- Output: words joined with spaces (`" ".join(result)`)

### BPE mode (`mode="bpe"`)

- Seed is split on whitespace (token IDs as strings)
- Requires `bpe_tokenizer` for decoding output
- Output: `bpe_tokenizer.decode_bpe(result)`
- Raises `ValueError` if `bpe_tokenizer` is `None`

## Examples

```python
from markovonnx import MarkovONNXRuntime, generate_markov

rt = MarkovONNXRuntime("model.onnx", vocab, order=2)

# Character-level
text = generate_markov(rt, "the ", 100, temperature=0.7, mode="char", order=2)

# Word-level
text = generate_markov(rt, "the cat", 20, temperature=0.8, mode="word", order=2)

# BPE
from markovonnx import SubwordTokenizer
bpe = SubwordTokenizer("tokenizer.json")
text = generate_markov(rt, "", 50, mode="bpe", order=2, bpe_tokenizer=bpe)
```

## Temperature Guide

| Temperature | Effect |
|-------------|--------|
| 0.1-0.3 | Nearly deterministic, repetitive |
| 0.5-0.7 | Balanced variety and coherence |
| 1.0 | True learned distribution |
| 1.5-2.0 | High randomness, less coherent |

---
[← ONNX Inference](onnx-inference.md) · [Home](index.md) · [Portable Archives →](archive.md)
