# Maintenance Report

## 2026-03-19 — Sparse export, Kneser-Ney, log-space HMM (v0.3.0)

- **AI Model**: Claude Opus 4.6
- **Actions Taken**:
  - Added sparse ONNX export (`export_markov_sparse_onnx`) with hash-table-style key lookup
  - Added Kneser-Ney smoothing (`MarkovChain(kneser_ney=True)`) with auto-estimated discount
  - Rewrote HMM Baum-Welch to operate entirely in log-space (logsumexp) for numerical stability
  - 127 tests, 99% coverage (736 statements)
- **Oversight**: Human-reviewed TODO list before implementation.

## 2026-03-19 — Major improvements (v0.2.0)

- **AI Model**: Claude Opus 4.6
- **Actions Taken**:
  - Fixed BPE generate seed bug, suppressed CUDA warning, fixed HMM archive float32 cast
  - Added backoff, batch inference, .markov archive, metadata loading, CLI, vocab serialization
  - Added property-based tests (Hypothesis) and integration test
  - 114 tests, 99% coverage
- **Oversight**: Human-reviewed improvement list before implementation.

## 2026-03-19 — Initial library extraction (v0.1.0)

- **AI Model**: Claude Opus 4.6
- **Actions Taken**: Extracted all classes and functions from `notebooks/markov_chains/onnxmarkov.ipynb` into a standalone `markovonnx` package.
- **Oversight**: Human-reviewed plan before implementation.
