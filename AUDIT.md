# Audit

## Known Issues

1. **Sparse export linear scan (large vocabs)** — `export_markov_sparse_onnx` uses flat-index O(1) Gather for `V^order <= 5_000_000`. Above that threshold it falls back to O(N) linear scan (Equal/ArgMax). Custom CUDA hash op would fix this for large models. — `markovonnx/onnx_export.py:_FLAT_INDEX_MAX_ENTRIES`
2. ~~**C export: one backoff level only**~~ — Fixed: all backoff levels exported with per-level arrays and lookup functions. — `markovonnx/c_export.py:_render_header`

## Resolved Issues

1. ~~**CUDA provider warning**~~ — Fixed in v0.2.0.
2. ~~**BPE generate seed bug**~~ — Fixed in v0.2.0.
3. ~~**HMM underflow on long sequences**~~ — Fixed in v0.3.0.
4. ~~**Laplace-only smoothing**~~ — Kneser-Ney added in v0.3.0.
5. ~~**Dense-only ONNX export**~~ — Sparse export added in v0.3.0.
6. ~~**Dense matrix memory**~~ — MemoryError raised if >2 GB.
7. ~~**Sparse ONNX shape [1,V]**~~ — Squeeze node added.
8. ~~**from_file vocab truncation**~~ — Full vocab stored; legacy padding with warning.
9. ~~**Division by zero in dense_matrix**~~ — row_sums clamped to 1e-30.
10. ~~**Division by zero in _kn_probs**~~ — Returns uniform if total==0.
11. ~~**IndexError in Vocabulary.decode**~~ — Returns <UNK> for out-of-range IDs.
12. ~~**KeyError in decode_bpe**~~ — Uses dict.get with fallback to unk_token.
13. ~~**UnicodeEncodeError in decode_bpe**~~ — Catches both encode and decode errors.
14. ~~**NaN in _logsumexp**~~ — Handles all-(-inf) inputs cleanly.
15. ~~**Short context crash in predict_probs**~~ — Raises ValueError.
16. ~~**Empty BPE context crash**~~ — Pads to order length with UNK IDs.
17. ~~**Temp dir leak in load_markov_archive**~~ — atexit cleanup registered.
