# Suggestions

1. **GPU-accelerated sparse lookup** — Replace the ONNX flat-index Gather with a CUDA hash-table custom op for vocabularies where `V^order > 5 000 000` (currently falls back to linear scan).
2. ~~**C export: recursive backoff**~~ — Implemented: all levels exported.
3. ~~**`train-hmm` CLI command**~~ — Implemented: `markovonnx train-hmm` with CoNLL corpus support.
4. **HMM unsupervised training from CLI** — Add `train-hmm --unsupervised` (Baum-Welch) for unannotated observation sequences.
5. **size-report in CI** — Fail CI if total model size exceeds a configurable byte limit (add `--max-bytes` flag to `size-report`).
6. **PlatformIO integration test** — Run `pio run` in CI on the generated `platformio.ini` to verify the sketch compiles end-to-end.
