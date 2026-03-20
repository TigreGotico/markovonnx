# Maintenance Report

## 2026-03-20 — ESP32 C header export (v0.4.0)

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - Added `markovonnx/c_export.py` — `export_markov_c_header()` generating self-contained C99 headers
  - INT8 quantisation (4× smaller than float32), PROGMEM `.rodata` annotation, binary-search lookup, CDF-walk sampler
  - Context packing: N token IDs → single `uint64_t` key, sorted for O(log N) lookup
  - Added `export_markov_c_header` to public API (`__init__.py`)
  - Added `--export-c`, `--no-quantize`, `--progmem` flags to CLI `train` subcommand
  - 27 new tests in `test/unittests/test_c_export.py` (syntax check via gcc, lookup correctness, quantisation precision)
  - Created `docs/esp32.md` with memory budget table, Arduino sketch example, API reference
  - 177 tests passing, 100% coverage on `c_export.py`
- **Oversight**: All tests run and validated locally; human review before push

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
