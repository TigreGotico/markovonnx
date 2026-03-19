# FAQ

## What is markovonnx?
A Python library for training Markov chains and HMMs, exporting them to ONNX, and running inference via ONNX Runtime (3.5x+ speedup over pure Python).

## What tokenization modes are supported?
Character-level (`char`), word-level (`word`), and BPE subword (`bpe`). BPE requires a HuggingFace `tokenizers` JSON file and the `SubwordTokenizer` class.

## How does ONNX export work?
`export_markov_onnx` builds a dense transition matrix and encodes context via base-V powers into a row index. The ONNX graph: `Mul -> ReduceSum -> Gather -> ArgMax`.

## How do I configure without code changes?
Use `MarkovConfig.from_env()` — all settings read from `MARKOV_*` environment variables.

## What's the memory footprint?
Dense transition matrix = `V^order * V * 4` bytes (float32). For V=324 and order=2, that's ~130 MB.

## Does quantization work?
Yes — `quantize_model()` applies INT8 dynamic quantization, typically achieving 75% size reduction.
