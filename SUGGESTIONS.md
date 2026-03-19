# Suggestions

1. **GPU-accelerated sparse lookup** — Replace the brute-force Equal scan in sparse export with a hash-based approach or custom ONNX operator for O(1) lookup.
2. **Streaming backoff training** — `fit_streaming` doesn't currently train backoff models. Add support by collecting sequences into batches.
3. **Vectorized Viterbi** — Current HMM Viterbi has a Python loop over states. Vectorizing the inner loop with NumPy broadcasting would improve speed.
4. **Interpolated Kneser-Ney** — Current implementation uses absolute discounting. Modified KN with three discount levels (n1, n2, n3+) would improve quality.
