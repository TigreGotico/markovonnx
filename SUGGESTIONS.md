# Suggestions

1. **GPU-accelerated sparse lookup** — Replace the ONNX flat-index Gather with a CUDA hash-table custom op for vocabularies where `V^order > 5 000 000` (currently falls back to linear scan).
2. **C export: recursive backoff** — Currently only one level of backoff is exported to C headers. Add recursive embedding for order > 2 chains.
3. **`train-hmm` CLI command** — Add a supervised HMM training subcommand that reads a tagged corpus (TSV) and saves `HiddenMarkovModel.save()` JSON, with `--export-c` option.
