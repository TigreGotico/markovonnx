# Markov Chains

`MarkovChain` — `markovonnx/markov.py:13`

N-gram Markov chain with sparse internal storage, interpolated backoff, Kneser-Ney smoothing, and dense/sparse ONNX export.

## How It Works

An order-N Markov chain predicts the next token based on the preceding N tokens (the "context"). Internally, counts are stored as a sparse `Dict[int, np.ndarray]` mapping context indices to count vectors. The context index is computed via base-V encoding (`_ctx_idx`, `markov.py:55`):

```
index = ctx[0] * V^(order-1) + ctx[1] * V^(order-2) + ... + ctx[order-1]
```

## Constructor

```python
MarkovChain(
    order: int,
    vocab: Vocabulary,
    smoothing: float = 1e-5,
    backoff: bool = False,
    kneser_ney: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `order` | Context window size (e.g., 2 for bigram contexts) |
| `vocab` | `Vocabulary` instance |
| `smoothing` | Laplace smoothing alpha (used when `kneser_ney=False`) |
| `backoff` | Train and use lower-order models as fallback for unseen contexts |
| `kneser_ney` | Use Kneser-Ney smoothing instead of Laplace |

## Smoothing Strategies

### Laplace (default)

Adds a small constant `smoothing` to all transition counts before normalizing. Simple but assigns too much probability mass to rare events.

```python
mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
```

### Kneser-Ney

Subtracts a discount `d` from observed counts and redistributes the mass uniformly. The discount is estimated automatically from count-of-counts after training: `d = n1 / (n1 + 2 * n2)` where n1/n2 are the number of n-grams appearing exactly once/twice.

```python
mc = MarkovChain(order=2, vocab=vocab, kneser_ney=True)
mc.fit(corpus)
print(mc._kn_discount)  # e.g. 0.73
```

Kneser-Ney typically gives **10-15% better accuracy** than Laplace on intent classification tasks.

### Interpolated Backoff

When `backoff=True`, lower-order models (order N-1, N-2, ... 1) are trained alongside the primary model. If a context is unseen at order N, the model falls back to shorter contexts. Reduces perplexity on data with rare contexts.

```python
mc = MarkovChain(order=3, vocab=vocab, kneser_ney=True, backoff=True)
mc.fit(corpus)
# mc._lower is an order-2 chain, mc._lower._lower is order-1
```

## Training

### In-Memory: `fit(sequences)` — `markov.py:90`

```python
mc = MarkovChain(order=2, vocab=vocab, kneser_ney=True, backoff=True)
mc.fit(corpus)  # List[List[str]]
```

### Streaming: `fit_streaming(path, tokenize_fn, max_lines=0)` — `markov.py:110`

```python
mc.fit_streaming("corpus.txt", tokenize_fn=char_tokenize)
```

Reads the corpus line-by-line via `corpus_iter`. Never loads the full file.

## Sampling

### `sample(context, temperature=1.0) -> token` — `markov.py:218`

Samples the next token from the learned distribution. Uses backoff for unseen contexts if enabled.

- `temperature < 1.0`: Sharper distribution (more deterministic)
- `temperature = 1.0`: Unscaled distribution
- `temperature > 1.0`: Flatter distribution (more random)

```python
next_token = mc.sample(["t", "h", "e"], temperature=0.7)
```

## Perplexity

### `perplexity(sequences) -> float` — `markov.py:234`

Computes perplexity over evaluation sequences. Uses backoff-aware probabilities. Lower values indicate a better fit.

```python
ppx = mc.perplexity(eval_sequences)
```

## Dense Matrix

### `dense_matrix() -> np.ndarray` — `markov.py:139`

Builds the full transition matrix `T[V^order, V]` with smoothing applied (Laplace or Kneser-Ney). Each row sums to 1.0. Called internally by `export_markov_onnx`.

Raises `MemoryError` if the matrix would exceed 2 GB. Use `export_markov_sparse_onnx()` for large models.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `order` | `int` | N-gram order |
| `vocab` | `Vocabulary` | Associated vocabulary |
| `smoothing` | `float` | Laplace alpha |
| `backoff` | `bool` | Whether backoff is enabled |
| `kneser_ney` | `bool` | Whether Kneser-Ney is enabled |
| `_kn_discount` | `float` | Estimated Kneser-Ney discount (after training) |
| `_lower` | `MarkovChain` or `None` | Lower-order backoff model |
| `_counts` | `Dict[int, np.ndarray]` | Sparse context→count-vector storage |
