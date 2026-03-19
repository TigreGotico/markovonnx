# ONNX Export

Functions for exporting trained models to ONNX format — `markovonnx/onnx_export.py`.

## Markov Chain Export

### `export_markov_onnx(mc, path)` — `onnx_export.py:15`

Exports a trained `MarkovChain` to an ONNX model file.

```python
from markovonnx import MarkovChain, export_markov_onnx

mc = MarkovChain(order=2, vocab=vocab)
mc.fit(corpus)
export_markov_onnx(mc, "output/markov.onnx")
```

**What it does:**
1. Calls `mc.dense_matrix()` to build the full transition matrix `T[V^order, V]`
2. Computes base-V powers for context encoding
3. Builds an ONNX graph with 4 nodes (see [Architecture](architecture.md))
4. Stores metadata: `model_type`, `order`, `vocab` (first 500 tokens), `vocab_size`
5. Validates with `onnx.checker.check_model`
6. Creates parent directories automatically

**ONNX Model I/O:**

| Direction | Name | Type | Shape |
|-----------|------|------|-------|
| Input | `input_ids` | INT64 | `[order]` |
| Output | `probs` | FLOAT | `[vocab_size]` |
| Output | `next_id` | INT64 | `[1]` |

## HMM Export

### `export_hmm_onnx(hmm, path)` — `onnx_export.py:82`

Exports a trained `HiddenMarkovModel` to an ONNX model that computes one forward step.

```python
from markovonnx import HiddenMarkovModel, export_hmm_onnx

export_hmm_onnx(hmm, "output/hmm.onnx")
```

**ONNX Model I/O:**

| Direction | Name | Type | Shape |
|-----------|------|------|-------|
| Input | `obs_id` | INT64 | `[1]` |
| Input | `alpha_in` | FLOAT | `[n_states]` |
| Output | `alpha_out` | FLOAT | `[n_states]` |
| Output | `best_state` | INT64 | `[1]` |

The HMM ONNX model processes one observation at a time. For a full sequence, call it in a loop (which `HMMONNXRuntime.decode` does automatically).

## INT8 Quantization

### `quantize_model(onnx_path, quant_path) -> Optional[str]` — `onnx_export.py:147`

Applies dynamic INT8 quantization to reduce model size.

```python
from markovonnx import quantize_model

result = quantize_model("markov.onnx", "markov_int8.onnx")
```

| Parameter | Description |
|-----------|-------------|
| `onnx_path` | Path to full-precision model |
| `quant_path` | Destination for quantized model |
| **Returns** | `quant_path` on success, `None` on failure |

**Requires**: `onnxruntime.quantization` (install with `pip install onnxruntime-tools onnxconverter-common`).

Typical compression: ~75% size reduction (e.g., 130 MB → 33 MB).

## Sparse Export

### `export_markov_sparse_onnx(mc, path)` — `onnx_export.py:82`

Exports a Markov chain using a sparse lookup table instead of the full dense matrix. Only stores rows with observed counts, plus a uniform fallback row.

```python
from markovonnx import MarkovChain, export_markov_sparse_onnx

export_markov_sparse_onnx(mc, "sparse_model.onnx")
```

**When to use**: When `V^order × V × 4` bytes exceeds available RAM or you want a smaller model file. For a model with 1000 unique contexts out of 100,000 possible, the sparse export stores ~1% of the data.

**How it works**: The ONNX graph computes the context index, then searches a key array via brute-force `Equal` scan. If found, looks up the corresponding row in the sparse table; otherwise returns a uniform distribution.

**Trade-off**: Sparse models are smaller but have O(N) key lookup instead of O(1) gather. For >100K sparse rows, dense may be faster at inference time.

**Same I/O as dense export**:

| Direction | Name | Type | Shape |
|-----------|------|------|-------|
| Input | `input_ids` | INT64 | `[order]` |
| Output | `probs` | FLOAT | `[vocab_size]` |
| Output | `next_id` | INT64 | `[1]` |

## Metadata

All export functions embed metadata in the ONNX model's `metadata_props`:

| Key | Dense | Sparse | HMM |
|-----|-------|--------|-----|
| `model_type` | `"markov_chain"` | `"markov_chain_sparse"` | `"hmm"` |
| `order` | Yes | Yes | — |
| `vocab_size` | Yes | Yes | — |
| `vocab` | Full token list (JSON) | Full token list (JSON) | — |
| `n_sparse_rows` | — | Number of stored rows | — |
| `n_states` | — | — | Yes |

Access metadata after loading:

```python
import onnx
model = onnx.load("markov.onnx")
meta = {p.key: p.value for p in model.metadata_props}
print(meta["order"])  # "2"
```
