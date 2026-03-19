# Audit

## Known Issues

1. **Sparse export linear scan** — `export_markov_sparse_onnx` uses brute-force Equal scan over all keys. For very large sparse tables (>100K rows), inference may be slower than dense Gather. Not fixable in ONNX opset 13 without custom ops. — `markovonnx/onnx_export.py:192`

## Resolved Issues

1. ~~**CUDA provider warning**~~ — Fixed in v0.2.0.
2. ~~**BPE generate seed bug**~~ — Fixed in v0.2.0.
3. ~~**HMM underflow on long sequences**~~ — Fixed in v0.3.0 with log-space Baum-Welch.
4. ~~**Laplace-only smoothing**~~ — Kneser-Ney added in v0.3.0.
5. ~~**Dense-only ONNX export**~~ — Sparse export added in v0.3.0.
6. ~~**Dense matrix memory**~~ — Fixed: `dense_matrix()` raises `MemoryError` if >2 GB.
7. ~~**Sparse ONNX shape [1,V]**~~ — Fixed: Squeeze node added after Gather.
8. ~~**from_file vocab truncation**~~ — Fixed: full vocab stored in ONNX metadata.
