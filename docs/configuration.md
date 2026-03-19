# Configuration

`MarkovConfig` — `markovonnx/config.py:14`

A dataclass that centralises all training and export parameters. Every field has a matching `MARKOV_*` environment variable fallback.

## Fields

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `data_path` | str | `""` | `MARKOV_DATA_PATH` | Path to corpus file |
| `data_mode` | str | `"char"` | `MARKOV_DATA_MODE` | `"char"`, `"word"`, or `"bpe"` |
| `order` | int | `2` | `MARKOV_ORDER` | N-gram order (context length) |
| `hmm_states` | int | `16` | `MARKOV_HMM_STATES` | Number of hidden states for HMM |
| `smoothing` | float | `1e-5` | `MARKOV_SMOOTHING` | Laplace smoothing alpha |
| `max_vocab` | int | `0` | `MARKOV_MAX_VOCAB` | Max vocabulary size (0 = unlimited) |
| `max_lines` | int | `0` | `MARKOV_MAX_LINES` | Max corpus lines to process (0 = unlimited) |
| `stream_threshold_mb` | int | `1024` | `MARKOV_STREAM_MB` | Stream files larger than this (MB) |
| `onnx_path` | str | `"markov.onnx"` | `MARKOV_ONNX_PATH` | ONNX output filename |
| `quantize` | bool | `True` | `MARKOV_QUANTIZE` | Enable INT8 quantization (`1`/`true`/`yes`) |
| `quant_path` | str | `"markov_int8.onnx"` | `MARKOV_QUANT_PATH` | Quantized model filename |
| `temperature` | float | `0.5` | `MARKOV_TEMP` | Sampling temperature |
| `gen_length` | int | `10` | `MARKOV_GEN_LEN` | Tokens to generate |
| `seed` | str | `""` | `MARKOV_SEED` | Seed text for generation |
| `outdir` | str | `"./markov"` | `MARKOV_OUTDIR` | Output directory |

## Constructing

```python
from markovonnx import MarkovConfig

# Programmatic
cfg = MarkovConfig(data_mode="word", order=3, smoothing=1e-4)

# From environment variables
cfg = MarkovConfig.from_env()
```

## Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `outdir_path` | `Path` | `outdir` as a `pathlib.Path` |
| `resolved_onnx_path` | `str` | `outdir / onnx_path` |
| `resolved_quant_path` | `str` | `outdir / quant_path` |

## Environment Variable Precedence

`MarkovConfig.from_env()` reads each `MARKOV_*` variable. If the variable is empty or unset, the dataclass default is used. The `_env` helper (`config.py:8`) casts the string value to the target type.

For `MARKOV_QUANTIZE`, any of `"1"`, `"true"`, `"yes"` enables quantization; anything else disables it.
