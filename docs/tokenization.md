# Tokenization

All tokenization utilities live in `markovonnx/tokenizers.py`.

## Modes

| Mode | Function | Input | Output | Use Case |
|------|----------|-------|--------|----------|
| `char` | `char_tokenize` | `"hello"` | `["h","e","l","l","o"]` | Character-level language modelling |
| `word` | `word_tokenize` | `"Hello World"` | `["hello","world"]` | Word-level modelling |
| `bpe` | `SubwordTokenizer.encode_bpe` | `"hello"` | `[42, 17, ...]` (int IDs) | Subword modelling (BPE) |

## Functions

### `char_tokenize(line: str) -> List[str]` — `tokenizers.py:7`

Strips whitespace, returns individual characters.

### `word_tokenize(line: str) -> List[str]` — `tokenizers.py:12`

Strips whitespace, lowercases, splits on whitespace.

### `corpus_iter(path, tokenize_fn, max_lines=0) -> Iterator[List]` — `tokenizers.py:17`

Streams a text file line-by-line, applying `tokenize_fn` to each line. Yields only non-empty token lists. Use `max_lines` to cap the number of sequences.

```python
from markovonnx import corpus_iter, word_tokenize

for tokens in corpus_iter("corpus.txt", word_tokenize, max_lines=1000):
    print(tokens)
```

### `get_tokenize_fn(mode, bpe_tokenizer=None) -> Callable` — `tokenizers.py:105`

Dispatcher that returns the appropriate tokenization function for a mode string.

```python
from markovonnx import get_tokenize_fn

fn = get_tokenize_fn("char")
fn("hello")  # ["h", "e", "l", "l", "o"]
```

Raises `ValueError` if `mode` is `"bpe"` and `bpe_tokenizer` is `None`.

## SubwordTokenizer

`SubwordTokenizer` — `tokenizers.py:40`

A dependency-free BPE tokenizer that loads a HuggingFace `tokenizers` JSON file. No runtime dependency on the `tokenizers` package — it re-implements BPE merge application in pure Python.

### Training a tokenizer (requires `tokenizers` package)

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=512,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    max_token_length=6,
)
tokenizer.train(["corpus.txt"], trainer)
tokenizer.save("subword_tokenizer.json")
```

### Using the tokenizer (no dependencies)

```python
from markovonnx import SubwordTokenizer

bpe = SubwordTokenizer("subword_tokenizer.json")

# Encode text to token IDs
ids = bpe.encode_bpe("hello world")

# Decode back to text
text = bpe.decode_bpe(ids)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(tokenizer_json: str)` | Load vocab and merges from JSON |
| `text_to_initial_tokens` | `(s: str) -> List[str]` | UTF-8 bytes as characters (static) |
| `encode_bpe` | `(s: str) -> List[int]` | Text to BPE token IDs |
| `decode_bpe` | `(ids: List[int]) -> str` | Token IDs back to text |
| `apply_bpe` | `(tokens: List[str]) -> List[str]` | Apply merge rules to character tokens |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vocab` | `dict` | Token string to ID mapping |
| `id_to_token` | `dict` | ID to token string mapping |
| `merges` | `list` | Ordered BPE merge pairs |
| `unk_token` | `str` | Unknown token string (default `"[UNK]"`) |
| `unk_id` | `int` | Unknown token ID |
