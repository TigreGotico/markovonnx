# Architecture

## Design Overview

markovonnx separates training (Python) from inference (ONNX Runtime). Models are trained using NumPy, exported to ONNX format, and loaded via ONNX Runtime for fast inference.

```
Training Pipeline:
  corpus → tokenize → Vocabulary → MarkovChain/HMM → fit()

Export Pipeline:
  trained model → dense_matrix() → ONNX graph → .onnx file → quantize (optional)

Inference Pipeline:
  .onnx file → MarkovONNXRuntime/HMMONNXRuntime → predict_probs/sample/argmax/decode
```

## Sparse vs Dense Storage

During training, `MarkovChain` stores counts sparsely as `Dict[int, np.ndarray]`. Only contexts that appear in the corpus get an entry. This keeps memory proportional to data, not `V^order`.

On ONNX export, `dense_matrix()` (`markov.py:92`) materialises the full `V^order × V` matrix with Laplace smoothing. This is necessary because ONNX requires fixed-size tensors.

## Markov Chain ONNX Graph

```
input_ids [order]
    │
    ├── Mul ──→ input_ids × powers ──→ mul_out
    │
    ├── ReduceSum ──→ mul_out ──→ index (scalar)
    │
    ├── Gather ──→ T[index] ──→ probs [vocab_size]
    │
    └── ArgMax ──→ probs ──→ next_id [1]
```

**Weights stored in ONNX:**
- `T`: Transition matrix `[V^order, V]` (float32)
- `powers`: Base-V encoding powers `[order]` (int64), e.g. `[V, 1]` for order=2

**Context encoding**: `index = sum(input_ids * powers)` maps an N-token context to a single row index using base-V positional encoding.

## HMM ONNX Graph

The HMM ONNX model computes one forward step:

```
obs_id [1], alpha_in [n_states]
    │
    ├── Gather ──→ B[:, obs_id] ──→ b_vec ──→ Squeeze ──→ b_sq [n_states]
    │
    ├── MatMul ──→ alpha_in @ A ──→ fwd [n_states]
    │
    ├── Mul ──→ fwd × b_sq ──→ alpha_raw [n_states]
    │
    ├── ReduceSum ──→ sum(alpha_raw) ──→ norm_val
    │
    ├── Div ──→ alpha_raw / norm_val ──→ alpha_out [n_states]
    │
    └── ArgMax ──→ alpha_out ──→ best_state [1]
```

**Weights stored in ONNX:**
- `A`: Transition matrix `[n_states, n_states]`
- `B`: Emission matrix `[n_states, obs_vocab_size]`
- `pi`: Initial distribution `[n_states]` (stored but not used in forward step)

For full-sequence decoding, `HMMONNXRuntime.decode()` (`onnx_runtime.py:85`) loops over observations, feeding `alpha_out` back as `alpha_in`.

## Sparse Markov Chain ONNX Graph

For large vocabularies, `export_markov_sparse_onnx` stores only observed rows:

```
input_ids [order]
    │
    ├── Mul + ReduceSum ──→ index (context row index)
    │
    ├── Equal(keys, index) ──→ match_bool [n_sparse]
    │
    ├── Cast + ReduceMax ──→ has_match (bool scalar)
    │
    ├── ArgMax(match_int) ──→ pos_found
    │
    ├── Where(has_match, pos_found, fallback_idx) ──→ pos
    │
    ├── Gather(sparse_table, pos) + Squeeze ──→ probs [V]
    │
    └── ArgMax ──→ next_id [1]
```

**Weights stored:**
- `sparse_table`: `[n_sparse + 1, V]`, observed rows plus a uniform fallback row
- `keys`: `[n_sparse]`, context indices for lookup
- `powers`: `[order]`, base-V encoding
- `fallback_idx`: `[1]`, index of the uniform row

## Module Dependency Graph

```
config  (standalone)
tokenizers  (standalone)
vocabulary  ──→ tokenizers (for corpus_iter in streaming)
markov  ──→ vocabulary, tokenizers
hmm  ──→ vocabulary
onnx_export  ──→ markov, hmm
onnx_runtime  ──→ vocabulary, hmm
generate  ──→ onnx_runtime, tokenizers
archive  ──→ onnx_export, onnx_runtime, vocabulary
cli  ──→ archive, markov, tokenizers, vocabulary
```

## ONNX Opset and IR Version

All exported models use **opset 13** and **IR version 8**, compatible with ONNX Runtime 1.10+.

---
[← API Reference](api-reference.md) · [Home](index.md)
