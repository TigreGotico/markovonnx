# Audit

## Known Issues

1. **Dense matrix memory** — `MarkovChain.dense_matrix()` allocates `V^order * V * 4` bytes. Large vocabularies with high order will OOM. — `markovonnx/markov.py:108`
2. **HMM Baum-Welch precision** — Forward-backward uses scaling but not full log-semiring. May lose precision with very long sequences. — `markovonnx/hmm.py:99-131`
3. **from_file vocab truncation** — `MarkovONNXRuntime.from_file` stores only first 500 vocab tokens in ONNX metadata. Larger vocabs get padded placeholders — usable for inference but not for text decoding. Use `.markov` archives for full vocab preservation. — `markovonnx/onnx_runtime.py:67-69`

## Resolved Issues

1. ~~**CUDA provider warning** — noisy warning on CPU-only machines~~ — Fixed in v0.2.0 by querying available providers.
2. ~~**BPE generate seed bug** — `seed.split()` produced strings, not int IDs~~ — Fixed in v0.2.0 using `encode_bpe(seed)`.
3. ~~**SubwordTokenizer not tested**~~ — Full test coverage added in v0.1.0 patch.
