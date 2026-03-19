# Command-Line Interface

markovonnx includes a CLI for training, generating, and inspecting models without writing Python.

## Install

```bash
pip install -e .
```

The `markovonnx` command is registered as a console script entry point.

## Commands

### `markovonnx train`

Train a Markov chain from a text corpus and save as a `.markov` archive.

```bash
markovonnx train corpus.txt -o model.markov --mode char --order 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `corpus` | (required) | Path to text file (one sequence per line) |
| `-o, --output` | `model.markov` | Output `.markov` archive path |
| `--mode` | `char` | Tokenization: `char` or `word` |
| `--order` | `3` | N-gram order |
| `--smoothing` | `1e-5` | Laplace smoothing alpha |
| `--max-vocab` | `0` | Max vocabulary size (0=unlimited) |
| `--max-lines` | `0` | Max corpus lines (0=unlimited) |
| `--backoff` | off | Enable interpolated backoff |

```bash
# Word-level with backoff
markovonnx train corpus.txt -o word_model.markov --mode word --order 2 --backoff

# Character-level, limited vocab
markovonnx train corpus.txt -o small.markov --max-vocab 100 --max-lines 1000
```

### `markovonnx generate`

Generate text from a trained `.markov` archive.

```bash
markovonnx generate model.markov --seed "the" --length 100 --temperature 0.7
```

| Flag | Default | Description |
|------|---------|-------------|
| `archive` | (required) | Path to `.markov` archive |
| `--seed` | `""` | Initial text |
| `--length` | `100` | Tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--mode` | `char` | Tokenization mode |

### `markovonnx info`

Display metadata from a `.markov` archive.

```bash
markovonnx info model.markov
```

Output:
```
Archive:    model.markov (5.4 KB)
  model_type: markov_chain
  order: 3
  smoothing: 1e-05
  vocab_size: 25
  backoff: True
```

## End-to-End Example

```bash
# Train on nursery rhymes
markovonnx train examples/data/nursery_rhymes.txt -o nursery.markov --order 3 --backoff

# Check metadata
markovonnx info nursery.markov

# Generate text
markovonnx generate nursery.markov --seed "the" --length 200 --temperature 0.5
```
