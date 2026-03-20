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
Yes — two C header exporters are provided.  `export_markov_c_header(mc, "model.h")` generates a self-contained C99 header with no external dependencies: static vocab and probability arrays, inline binary-search lookup, and CDF-walk sampler. `export_hmm_c_header(hmm, "model.h")` exports pre-computed log-probability matrices and an inline Viterbi decoder. Character order=1 Markov models fit in ~7 KB flash; HMMs with 8 states / 50 obs fit in ~3 KB. See [docs/esp32.md](docs/esp32.md).

## Can I run Viterbi decoding on ESP32?
Yes — `export_hmm_c_header(hmm, "model.h")` embeds `hmm_viterbi()` as an inline C function. It takes caller-allocated `delta` (float), `psi` (int), and `path` (int) buffers and runs in O(T·S²). Log probabilities are pre-computed at export time so no `logf()` is called at inference. State labels are recovered via `HMM_STATE_VOCAB[path[t]]`.

## Can I run online forward filtering on ESP32 without T-length buffers?
Yes — the HMM C header now also includes linear-domain `HMM_PI`, `HMM_A`, `HMM_B` arrays and three inline functions: `hmm_forward_init(obs_id, alpha)` (initialise), `hmm_forward_step(obs_id, alpha)` (update in-place with a stack buffer), and `hmm_best_state(alpha)` (argmax). These use constant memory and avoid `expf()`.

## How are all backoff levels exported to C?
`export_markov_c_header` now traverses the full `_lower` chain and emits `MARKOV_KEYS_{k}`, `MARKOV_PROBS_{k}`, and `markov_lookup_{k}()` for every order *k*. `markov_sample_backoff(key, r)` tries levels from highest to lowest. Legacy `MARKOV_KEYS_LOWER` and `markov_lookup_lower()` aliases are kept for backward compatibility.

## How do I check if my model fits in ESP32 flash?
Use `markovonnx size-report model.json [--format markov|hmm] [--no-quantize] [--progmem]`. The `markovonnx.size_report` module provides `markov_c_sizes()`, `hmm_c_sizes()`, `format_markov_report()`, and `format_hmm_report()` for programmatic use.

## How do I train an HMM from a tagged corpus via CLI?
Use `markovonnx train-hmm corpus.tsv -o model.hmm.json [--n-states 8] [--smoothing 1e-5] [--export-c model.h] [--progmem] [--max-lines 0]`. Corpus format: one `obs TAB tag` per line, blank lines between sequences (CoNLL-style).

## How do I generate a PlatformIO project for ESP32?
Pass `--platformio` to `markovonnx export` or `markovonnx train --export-c`. This writes `platformio.ini` and `src/main.cpp` next to the `.h` file with a ready-to-compile Arduino sketch.

## How do I use predict_probs() vs _get_probs()?
`MarkovChain.predict_probs(context)` is the public API — it is a direct wrapper of `_get_probs()` and applies backoff automatically. Use `predict_probs` in application code; `_get_probs` is internal.

## How do I train an HMM without labelled data (unsupervised)?
Use `markovonnx train-hmm corpus.txt --unsupervised [--n-iter 10]`. The corpus must have one token per line with blank lines between sequences (no tags required). Training uses Baum-Welch EM (`HiddenMarkovModel.fit_unsupervised`). Supervised CoNLL format is still the default when `--unsupervised` is absent.

## How do I set a maximum model size for CI?
Pass `--max-bytes N` to `markovonnx size-report`. The command exits with code 1 if the estimated C header size exceeds N bytes, enabling automated CI gates on model size budgets.

## Do .markov archives preserve the backoff chain for C export?
Yes — as of v0.5.0, `save_markov_archive` embeds `chain.json` with all counts and `_lower` backoff levels. `load_markov_archive` returns a `"chain"` key with the reconstructed `MarkovChain`, which can be passed directly to `export_markov_c_header`.

## Does `--backoff` work with streaming training?
Yes — `fit_streaming` now recursively trains lower-order models (one extra pass per order level). Before v0.4.1, `--backoff` was silently ignored when streaming; only `fit()` built the lower chain.

## Why is sparse ONNX inference faster now?
The sparse export uses a flat-index array (`flat_index[V^order]`) for O(1) context lookup via a single `Gather` op, replacing the O(N) linear scan (Equal + ArgMax). Falls back to linear scan only when `V^order > 5_000_000`.

## Can I batch inference calls?
Yes — `rt.predict_probs_batch(contexts)` processes multiple contexts and returns shape `(N, vocab_size)`.
