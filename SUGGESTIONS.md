# Suggestions

1. **Sparse ONNX export** — For large vocab/order, explore sparse matrix representation to avoid dense matrix allocation.
2. **Kneser-Ney smoothing** — Replace Laplace with modified Kneser-Ney for better perplexity on small corpora.
3. **Log-space HMM** — Full log-semiring forward-backward for numerical stability with long sequences.
4. **Streaming backoff training** — `fit_streaming` doesn't currently train backoff models. Add support.
