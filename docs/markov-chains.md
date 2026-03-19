# Markov Chains

`MarkovChain` — `markovonnx/markov.py:13`

N-gram Markov chain with sparse internal storage and dense ONNX export.

## How It Works

An order-N Markov chain predicts the next token based on the preceding N tokens (the "context"). Internally, counts are stored as a sparse `Dict[int, np.ndarray]` mapping context indices to count vectors. The context index is computed via base-V encoding (`_ctx_idx`, `markov.py:33`):

```
index = ctx[0] * V^(order-1) + ctx[1] * V^(order-2) + ... + ctx[order-1]
```

## Constructor

```python
MarkovChain(order: int, vocab: Vocabulary, smoothing: float = 1e-5)
```

| Parameter | Description |
|-----------|-------------|
| `order` | Context window size (e.g., 2 for bigram contexts) |
| `vocab` | `Vocabulary` instance |
| `smoothing` | Laplace smoothing alpha added to all transitions |

## Training

### In-Memory: `fit(sequences)` — `markov.py:53`

```python
mc = MarkovChain(order=2, vocab=vocab)
mc.fit(corpus)  # List[List[str]]
```

Encodes each sequence via `vocab.encode()`, then increments transition counts for every `(context, next_token)` pair.

### Streaming: `fit_streaming(path, tokenize_fn, max_lines=0)` — `markov.py:65`

```python
mc.fit_streaming("corpus.txt", tokenize_fn=char_tokenize)
```

Reads the corpus line-by-line via `corpus_iter`. Never loads the full file.

## Sampling

### `sample(context, temperature=1.0) -> token` — `markov.py:106`

Samples the next token from the learned distribution.

- `temperature < 1.0`: Sharper distribution (more deterministic)
- `temperature = 1.0`: Unscaled distribution
- `temperature > 1.0`: Flatter distribution (more random)

If the context has never been seen, returns a random token.

```python
next_token = mc.sample(["t", "h", "e"], temperature=0.7)
```

## Perplexity

### `perplexity(sequences) -> float` — `markov.py:131`

Computes perplexity over evaluation sequences. Lower values indicate a better fit. Uses smoothed probabilities to avoid log(0).

```python
ppx = mc.perplexity(eval_sequences)
```

## Dense Matrix

### `dense_matrix() -> np.ndarray` — `markov.py:92`

Builds the full transition matrix `T[V^order, V]` with Laplace smoothing applied. Each row sums to 1.0. This is called internally by `export_markov_onnx`.

**Memory**: `V^order × V × 4` bytes. For V=324, order=2: ~130 MB. The library raises `ValueError` (in notebook usage) if the matrix would exceed 2 GB.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `order` | `int` | N-gram order |
| `vocab` | `Vocabulary` | Associated vocabulary |
| `smoothing` | `float` | Laplace alpha |
| `_counts` | `Dict[int, np.ndarray]` | Sparse context→count-vector storage |
