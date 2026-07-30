# Vocabulary

`Vocabulary` (`markovonnx/vocabulary.py:7`)

Maps tokens (strings or ints) to integer IDs and back. Supports frequency-based pruning via `max_vocab`.

## Construction

```python
from markovonnx import Vocabulary

# Unlimited vocabulary
vocab = Vocabulary()
vocab.build_from_sequences([["the", "cat", "sat"], ["the", "dog", "ran"]])
print(vocab.size)  # 6 (5 tokens + <UNK>)

# Pruned to top-100 tokens
vocab = Vocabulary(max_vocab=100)
vocab.build_from_sequences(corpus)
```

## Building

### `build_from_sequences(sequences: List[List]) -> None` (`vocabulary.py:24`)

Counts all tokens in memory, then finalises the mapping. Tokens are ordered by descending frequency; `<UNK>` is always at index 0.

### `build_streaming(path, tokenize_fn, max_lines=0) -> None` (`vocabulary.py:30`)

Streams a corpus file to count tokens without loading everything into RAM. Requires a `tokenize_fn` callable.

```python
from markovonnx import Vocabulary, char_tokenize

vocab = Vocabulary(max_vocab=500)
vocab.build_streaming("large_corpus.txt", tokenize_fn=char_tokenize)
```

## Encoding and Decoding

### `encode(tokens: List) -> List[int]` (`vocabulary.py:56`)

Maps tokens to integer IDs. Unknown tokens map to the `<UNK>` ID (always 0).

### `decode(ids: List[int]) -> List` (`vocabulary.py:61`)

Maps integer IDs back to tokens.

```python
ids = vocab.encode(["the", "cat", "UNKNOWN"])
# ids[2] == 0 (UNK)
tokens = vocab.decode(ids)
# tokens[2] == "<UNK>"
```

## Properties and Constants

| Symbol | Type | Description |
|--------|------|-------------|
| `UNK` | `str` | Class constant: `"<UNK>"` |
| `size` | `int` (property) | Total token count including `<UNK>` |
| `tok2id` | `Dict[str, int]` | Token-to-ID mapping |
| `id2tok` | `List[str]` | ID-to-token mapping (ordered) |

## Token Ordering

After `_finalise()` (`vocabulary.py:49`), `id2tok` is:
1. Index 0: `<UNK>`
2. Index 1..N: tokens sorted by descending frequency

When `max_vocab > 0`, only the top `max_vocab` tokens are kept (plus `<UNK>`), so `size == max_vocab + 1`.

## Memory Considerations

The vocabulary itself is lightweight. The memory-intensive component is the transition matrix built during ONNX export, which scales as `V^order × V × 4` bytes (float32). For `V=324, order=2`: ~130 MB.

---
[← Tokenization](tokenization.md) · [Home](index.md) · [Markov Chains →](markov-chains.md)
