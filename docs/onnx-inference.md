# ONNX Inference

Runtime wrappers for fast inference via ONNX Runtime — `markovonnx/onnx_runtime.py`.

## MarkovONNXRuntime

`MarkovONNXRuntime` — `onnx_runtime.py:13`

### Constructor

```python
MarkovONNXRuntime(onnx_path: str, vocab: Vocabulary, order: int)
```

Loads an ONNX session with automatic provider selection (CUDA if available, else CPU). Sets `intra_op_num_threads` to CPU count and enables all graph optimisations.

### Methods

#### `predict_probs(context) -> np.ndarray` — `onnx_runtime.py:35`

Returns the full probability distribution over the vocabulary for a given context.

```python
probs = rt.predict_probs(["t", "h"])  # shape: (vocab_size,)
print(probs.sum())  # ~1.0
```

The context is truncated to the last `order` tokens automatically.

#### `sample(context, temperature=1.0) -> token` — `onnx_runtime.py:44`

Samples the next token probabilistically.

```python
token = rt.sample(["t", "h"], temperature=0.5)
```

When `temperature != 1.0`, logits are rescaled before converting to probabilities.

#### `argmax(context) -> token` — `onnx_runtime.py:60`

Returns the most likely next token (deterministic, greedy).

```python
token = rt.argmax(["t", "h"])  # always returns the same token
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vocab` | `Vocabulary` | Token mapping |
| `order` | `int` | Context length |
| `sess` | `ort.InferenceSession` | ONNX Runtime session |
| `provider` | `str` | Active execution provider name |

## HMMONNXRuntime

`HMMONNXRuntime` — `onnx_runtime.py:71`

### Constructor

```python
HMMONNXRuntime(onnx_path: str, hmm: HiddenMarkovModel)
```

Loads an ONNX session. Requires the original `HiddenMarkovModel` for `pi` (initial state distribution) and vocabulary access.

### Methods

#### `decode(obs_seq) -> List[str]` — `onnx_runtime.py:85`

Greedy forward-pass decoding over a full observation sequence.

```python
states = rt.decode(["walk", "shop", "clean"])
```

Processes one observation at a time, feeding `alpha_out` from each step as `alpha_in` to the next. Returns state names if `state_vocab` is set.

**Note**: This is a greedy forward pass, not Viterbi. Results may differ from `hmm.viterbi()`. Use `hmm.viterbi()` for optimal decoding; use `HMMONNXRuntime.decode()` for fast approximate decoding.

## Performance

ONNX Runtime typically achieves 3-4x speedup over the pure Python model for Markov chain inference. The improvement comes from:

1. Optimised native matrix operations
2. Graph-level optimisations (operator fusion)
3. Multi-threaded execution
4. Optional CUDA acceleration
