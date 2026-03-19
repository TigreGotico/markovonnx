# Audit

## Known Issues

1. **Dense matrix memory** — `MarkovChain.dense_matrix()` allocates `V^order * V * 4` bytes. Use `export_markov_sparse_onnx()` for large models. — `markovonnx/markov.py:113`
2. **Sparse export linear scan** — `export_markov_sparse_onnx` uses brute-force Equal scan over all keys. For very large sparse tables (>100K rows), inference may be slower than dense Gather. — `markovonnx/onnx_export.py:192`
3. **from_file vocab truncation** — `MarkovONNXRuntime.from_file` stores only first 500 vocab tokens in ONNX metadata. Use `.markov` archives for full vocab preservation. — `markovonnx/onnx_runtime.py:67-69`

## Resolved Issues

1. ~~**CUDA provider warning**~~ — Fixed in v0.2.0.
2. ~~**BPE generate seed bug**~~ — Fixed in v0.2.0.
3. ~~**HMM underflow on long sequences**~~ — Fixed in v0.3.0 with log-space Baum-Welch.
4. ~~**Laplace-only smoothing**~~ — Kneser-Ney added in v0.3.0.
5. ~~**Dense-only ONNX export**~~ — Sparse export added in v0.3.0.
