# Maintenance Report

## 2026-03-20 — v0.5.0 follow-up: public API + archive round-trip + CLI guards

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - **feat**: `MarkovChain.predict_probs()` — public wrapper for `_get_probs()` (`markov.py:254`)
  - **feat**: `MarkovChain.save()`/`load()` now persist `_lower` backoff chain inline under `"lower"` key (`markov.py:325`)
  - **feat**: `.markov` archives include `chain.json`; `load_markov_archive` returns `"chain"` key with full `MarkovChain` (`archive.py:43,112`)
  - **feat**: `size-report --max-bytes N` CI guard — exits 1 when model exceeds budget (`cli.py:368`, `size_report.py:113`)
  - **feat**: `train-hmm --unsupervised` — plain-token corpus parsed, Baum-Welch via `fit_unsupervised()`; `--n-iter` controls EM iterations (`cli.py:102`)
  - 247 tests passing (8 new tests across `test_archive.py`, `test_cli.py`, `test_markov.py`)
- **Oversight**: All tests run and validated locally; human review before push

## 2026-03-20 — C export enhancements + new CLI subcommands (v0.5.0)

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - **fix**: `_collect_rows` normalization guard — if all uint8 entries round to 0, set `row[argmax] = 1` (`c_export.py:158`)
  - **feat**: Recursive backoff export — all `_lower` chain levels emitted as `MARKOV_KEYS_{k}`, `MARKOV_PROBS_{k}`, `markov_lookup_{k}()`. `markov_sample_backoff()` cascades through all levels. Legacy aliases kept (`c_export.py:_render_header`)
  - **feat**: HMM C header now emits linear-domain `HMM_PI`, `HMM_A`, `HMM_B` + `hmm_forward_init()`, `hmm_forward_step()`, `hmm_best_state()` for constant-memory forward filtering (`c_export.py:_render_hmm_header`)
  - **feat**: New module `markovonnx/size_report.py` — `markov_c_sizes()`, `hmm_c_sizes()`, `format_markov_report()`, `format_hmm_report()` with ESP32 fit checks
  - **feat**: `markovonnx size-report` CLI subcommand (`cli.py:cmd_size_report`)
  - **feat**: `markovonnx train-hmm` CLI subcommand — CoNLL corpus, `--n-states`, `--smoothing`, `--export-c`, `--max-lines` (`cli.py:cmd_train_hmm`)
  - **feat**: `--platformio` flag on `export` and `train` — generates `platformio.ini` + `src/main.cpp` (`cli.py:_write_platformio_files`)
  - 239 tests passing (38 new tests across `test_c_export.py`, `test_size_report.py`, `test_cli.py`)
- **Oversight**: All tests run and validated locally; human review before push

## 2026-03-20 — Backlog tasks (v0.4.1)

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - **fix**: `fit_streaming` now recursively trains backoff lower-order chains (was silently a no-op)
  - **perf**: sparse ONNX lookup replaced with O(1) flat-index Gather (6 nodes vs 11); linear-scan fallback kept for V^order > 5M
  - **feat**: C header export includes backoff chain arrays + `markov_sample_backoff()` when chain has `_lower`
  - **feat**: `markovonnx export` CLI subcommand for post-training C header generation from saved JSON files
  - **chore**: SUGGESTIONS.md pruned (vectorised Viterbi and KN smoothing were already implemented)
  - 199 tests passing (12 new tests added across 3 test files)
- **Oversight**: All tests run and validated locally; human review before push

## 2026-03-20 — HMM + Viterbi C header export

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**:
  - Added `export_hmm_c_header()` to `c_export.py` — generates C99 header with
    pre-computed `HMM_LOG_PI`, `HMM_LOG_A`, `HMM_LOG_B` float arrays and inline
    `hmm_viterbi()` decoder (O(T·S²), no `logf()` at runtime)
  - HMMs without `state_vocab` (unsupervised) export numeric state names
  - `export_hmm_c_header` added to public API (`__init__.py`)
  - 10 new tests (HMM structure, C syntax via gcc, Viterbi correctness vs Python)
  - `docs/esp32.md` updated with HMM section, Arduino sketch, memory budget table
  - `FAQ.md` updated with HMM/Viterbi Q&A entries
  - 187 tests passing
- **Oversight**: All tests run and validated locally; human review before push

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
