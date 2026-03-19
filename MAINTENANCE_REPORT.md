# Maintenance Report

## 2026-03-19 — Major improvements (v0.2.0)

- **AI Model**: Claude Opus 4.6
- **Actions Taken**:
  - Fixed BPE generate seed bug (used `encode_bpe(seed)` instead of `seed.split()`)
  - Suppressed CUDA provider warning by querying available providers
  - Added interpolated backoff to MarkovChain (`backoff=True`)
  - Added batch inference (`predict_probs_batch`)
  - Added portable `.markov` archive format (`save_markov_archive`/`load_markov_archive`)
  - Added metadata-based loading (`MarkovONNXRuntime.from_file`)
  - Added CLI entry point (`markovonnx train/generate/info`)
  - Added Vocabulary serialization (`to_dict`/`from_dict`/`save`/`load`)
  - Added property-based tests via Hypothesis
  - Added integration test (full pipeline)
  - 114 tests, 99% coverage (642 statements)
- **Oversight**: Human-reviewed improvement list before implementation.

## 2026-03-19 — Initial library extraction (v0.1.0)

- **AI Model**: Claude Opus 4.6
- **Actions Taken**: Extracted all classes and functions from `notebooks/markov_chains/onnxmarkov.ipynb` into a standalone `markovonnx` package. Created 9 source modules, 7 test modules (38 tests, 83% coverage), pyproject.toml, docs, and documentation files.
- **Oversight**: Human-reviewed plan before implementation.
