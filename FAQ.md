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

## How do I save and load a model portably?
Use `save_markov_archive(model, "model.markov")` and `load_markov_archive("model.markov")`. The `.markov` file is a ZIP containing the ONNX model, vocabulary, and config.

## Can I load a model without the original Vocabulary object?
Yes — `MarkovONNXRuntime.from_file("model.onnx")` reconstructs the vocab from metadata embedded in the ONNX file. For vocabs > 500 tokens, tokens beyond 500 are padded as `<TOKEN_N>`.

## What is backoff and when should I use it?
`MarkovChain(backoff=True)` trains lower-order models alongside the primary one. When a context is unseen at order N, the model backs off to order N-1, then N-2, etc. This reduces perplexity on data with rare contexts. Enable it for small corpora or high orders.

## Is there a CLI?
Yes — `markovonnx train corpus.txt -o model.markov` and `markovonnx generate model.markov --seed "the" --length 100`. See `markovonnx --help`.

## Can I batch inference calls?
Yes — `rt.predict_probs_batch(contexts)` processes multiple contexts and returns shape `(N, vocab_size)`.
