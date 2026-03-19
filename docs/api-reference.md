# API Reference

Complete public API with signatures and source locations.

## markovonnx.config

### `MarkovConfig` — `config.py:14`

```python
@dataclasses.dataclass
class MarkovConfig:
    data_path: str = ""
    data_mode: str = "char"
    order: int = 2
    hmm_states: int = 16
    smoothing: float = 1e-5
    max_vocab: int = 0
    max_lines: int = 0
    stream_threshold_mb: int = 1024
    onnx_path: str = "markov.onnx"
    quantize: bool = True
    quant_path: str = "markov_int8.onnx"
    temperature: float = 0.5
    gen_length: int = 10
    seed: str = ""
    outdir: str = "./markov"

    @classmethod
    def from_env(cls) -> "MarkovConfig": ...

    @property
    def outdir_path(self) -> Path: ...
    @property
    def resolved_onnx_path(self) -> str: ...
    @property
    def resolved_quant_path(self) -> str: ...
```

---

## markovonnx.tokenizers

### `char_tokenize(line: str) -> List[str]` — `tokenizers.py:7`
### `word_tokenize(line: str) -> List[str]` — `tokenizers.py:12`
### `corpus_iter(path: str, tokenize_fn: Callable, max_lines: int = 0) -> Iterator[List]` — `tokenizers.py:17`
### `get_tokenize_fn(mode: str, bpe_tokenizer: Optional[SubwordTokenizer] = None) -> Callable` — `tokenizers.py:105`

### `SubwordTokenizer` — `tokenizers.py:40`

```python
class SubwordTokenizer:
    vocab: dict
    id_to_token: dict
    merges: list
    unk_token: str
    unk_id: int

    def __init__(self, tokenizer_json: str = "subword_tokenizer.json"): ...
    @staticmethod
    def text_to_initial_tokens(s: str) -> List[str]: ...
    def encode_bpe(self, s: str) -> List[int]: ...
    def decode_bpe(self, ids: List[int]) -> str: ...
    def apply_bpe(self, tokens: List[str]) -> List[str]: ...
```

---

## markovonnx.vocabulary

### `Vocabulary` — `vocabulary.py:7`

```python
class Vocabulary:
    UNK: str = "<UNK>"
    tok2id: Dict[str, int]
    id2tok: List[str]

    def __init__(self, max_vocab: int = 0): ...
    def build_from_sequences(self, sequences: List[List]) -> None: ...
    def build_streaming(self, path: str, tokenize_fn: Callable, max_lines: int = 0) -> None: ...
    def encode(self, tokens: List) -> List[int]: ...
    def decode(self, ids: List[int]) -> List: ...
    @property
    def size(self) -> int: ...
```

---

## markovonnx.markov

### `MarkovChain` — `markov.py:13`

```python
class MarkovChain:
    order: int
    vocab: Vocabulary
    smoothing: float

    def __init__(self, order: int, vocab: Vocabulary, smoothing: float = 1e-5): ...
    def fit(self, sequences: List[List]) -> None: ...
    def fit_streaming(self, path: str, tokenize_fn: Callable, max_lines: int = 0) -> None: ...
    def dense_matrix(self) -> np.ndarray: ...
    def sample(self, context: List, temperature: float = 1.0) -> object: ...
    def perplexity(self, sequences: List[List]) -> float: ...
```

---

## markovonnx.hmm

### `HiddenMarkovModel` — `hmm.py:10`

```python
class HiddenMarkovModel:
    n_states: int
    obs_vocab: Vocabulary
    state_vocab: Optional[Vocabulary]
    smoothing: float
    pi: np.ndarray       # [n_states]
    A: np.ndarray        # [n_states, n_states]
    B: np.ndarray        # [n_states, obs_vocab.size]

    def __init__(self, n_states: int, obs_vocab: Vocabulary,
                 state_vocab: Optional[Vocabulary] = None,
                 smoothing: float = 1e-5): ...
    def fit_supervised(self, obs_seqs: List[List[str]], tag_seqs: List[List[str]]) -> None: ...
    def fit_unsupervised(self, obs_seqs: List[List[str]], n_iter: int = 10) -> None: ...
    def viterbi(self, obs_seq: List[str]) -> List[str]: ...
```

---

## markovonnx.onnx_export

### `export_markov_onnx(mc: MarkovChain, path: str) -> None` — `onnx_export.py:15`
### `export_hmm_onnx(hmm: HiddenMarkovModel, path: str) -> None` — `onnx_export.py:82`
### `quantize_model(onnx_path: str, quant_path: str) -> Optional[str]` — `onnx_export.py:147`

---

## markovonnx.onnx_runtime

### `MarkovONNXRuntime` — `onnx_runtime.py:13`

```python
class MarkovONNXRuntime:
    vocab: Vocabulary
    order: int
    sess: ort.InferenceSession
    provider: str

    def __init__(self, onnx_path: str, vocab: Vocabulary, order: int): ...
    def predict_probs(self, context: List) -> np.ndarray: ...
    def sample(self, context: List, temperature: float = 1.0) -> object: ...
    def argmax(self, context: List) -> object: ...
```

### `HMMONNXRuntime` — `onnx_runtime.py:71`

```python
class HMMONNXRuntime:
    hmm: HiddenMarkovModel
    sess: ort.InferenceSession

    def __init__(self, onnx_path: str, hmm: HiddenMarkovModel): ...
    def decode(self, obs_seq: List[str]) -> List[str]: ...
```

---

## markovonnx.generate

### `generate_markov(...)` — `generate.py:9`

```python
def generate_markov(
    ort_model: MarkovONNXRuntime,
    seed: str,
    length: int,
    temperature: float = 1.0,
    mode: str = "char",
    order: int = 2,
    bpe_tokenizer: Optional[SubwordTokenizer] = None,
) -> str: ...
```
