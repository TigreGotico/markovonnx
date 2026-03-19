# Audit

## Known Issues

1. **Dense matrix memory** — `MarkovChain.dense_matrix()` allocates `V^order * V * 4` bytes. Large vocabularies with high order will OOM. — `markovonnx/markov.py:96`
2. **SubwordTokenizer not tested** — `SubwordTokenizer` class requires a JSON file; no unit test covers it (accounts for most uncovered lines in `tokenizers.py`). — `markovonnx/tokenizers.py:48-102`
3. **CUDA provider warning** — `MarkovONNXRuntime` requests CUDA provider which may not be available; warning is harmless but noisy. — `markovonnx/onnx_runtime.py:33`
