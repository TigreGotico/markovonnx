# FAQ

## What is markovonnx?
A Python library for training Markov chains and HMMs, exporting them to ONNX, and running inference via ONNX Runtime (3.5x+ speedup over pure Python).

## What tokenization modes are supported?
Character-level (`char`), word-level (`word`), and BPE subword (`bpe`). BPE requires a HuggingFace `tokenizers` JSON file and the `SubwordTokenizer` class.

## How does ONNX export work?
Two modes: **dense** (`export_markov_onnx`) stores the full `V^order × V` transition matrix; **sparse** (`export_markov_sparse_onnx`) stores only observed rows with a fallback uniform row.

## When should I use sparse export?
When `V^order * V * 4` bytes exceeds available RAM or you want a smaller model file. Sparse models are smaller but have O(N) key lookup instead of O(1) gather.

## What smoothing options are available?
**Laplace** (default, `smoothing=1e-5`) and **Kneser-Ney** (`kneser_ney=True`). KN estimates discount `d = n1/(n1 + 2*n2)` from count-of-counts automatically.

## What is backoff?
`MarkovChain(backoff=True)` trains lower-order models and falls back to shorter contexts for unseen n-grams. Reduces perplexity on data with rare contexts.

## How do I save and load a model portably?
`save_markov_archive(model, "model.markov")` and `load_markov_archive("model.markov")`. The `.markov` file is a ZIP containing ONNX model, vocabulary, and config.

## Can I load without the original Vocabulary?
Yes — `MarkovONNXRuntime.from_file("model.onnx")` reconstructs vocab from ONNX metadata (first 500 tokens; larger vocabs get placeholder padding).

## Is there a CLI?
Yes — `markovonnx train corpus.txt -o model.markov` and `markovonnx generate model.markov --seed "the" --length 100`.

## Is the HMM numerically stable on long sequences?
Yes — Baum-Welch operates entirely in log-space using logsumexp, preventing underflow even on sequences hundreds of tokens long.

## Can I run a model on ESP32 or other microcontrollers?
Yes — `export_markov_c_header(mc, "model.h")` generates a self-contained C99 header with no external dependencies. It includes static vocab and probability arrays, an inline binary-search lookup, and a CDF-walk sampler. Character order=1 models (~7 KB flash) fit comfortably; character order=2 with int8 quantisation (~84 KB) fits in 4 MB flash. See [docs/esp32.md](docs/esp32.md).

## Can I batch inference calls?
Yes — `rt.predict_probs_batch(contexts)` processes multiple contexts and returns shape `(N, vocab_size)`.
