"""Symbol-to-integer vocabulary with optional max-size pruning."""

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List


class Vocabulary:
    """Symbol <-> integer mapping with optional max-size pruning.

    Args:
        max_vocab: Keep only the *max_vocab* most frequent tokens (0 = unlimited).
    """

    UNK = "<UNK>"

    def __init__(self, max_vocab: int = 0):
        self.max_vocab = max_vocab
        self.tok2id: Dict[str, int] = {}
        self.id2tok: List[str] = []
        self._counts: Counter = Counter()

    # -- Build ----------------------------------------------------------------

    def build_from_sequences(self, sequences: List[List]) -> None:
        """Build vocabulary from an in-memory list of token sequences."""
        for seq in sequences:
            self._counts.update(seq)
        self._finalise()

    def build_streaming(
        self,
        path: str,
        tokenize_fn: Callable[[str], List],
        max_lines: int = 0,
    ) -> None:
        """Build vocabulary by streaming a corpus file.

        Args:
            path: Path to a text corpus.
            tokenize_fn: Callable that converts a raw line to tokens.
            max_lines: Stop after this many non-empty lines (0 = unlimited).
        """
        from markovonnx.tokenizers import corpus_iter

        for seq in corpus_iter(path, tokenize_fn, max_lines):
            self._counts.update(seq)
        self._finalise()

    def _finalise(self) -> None:
        top = self._counts.most_common(self.max_vocab if self.max_vocab else None)
        self.id2tok = [self.UNK] + [tok for tok, _ in top]
        self.tok2id = {tok: i for i, tok in enumerate(self.id2tok)}

    # -- Encode / decode ------------------------------------------------------

    def encode(self, tokens: List) -> List[int]:
        """Map tokens to integer IDs (unknown tokens map to ``<UNK>``)."""
        unk = self.tok2id[self.UNK]
        return [self.tok2id.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List:
        """Map integer IDs back to tokens."""
        return [self.id2tok[i] for i in ids]

    @property
    def size(self) -> int:
        """Total number of symbols (including ``<UNK>``)."""
        return len(self.id2tok)

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "max_vocab": self.max_vocab,
            "id2tok": self.id2tok,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        vocab = cls(max_vocab=data.get("max_vocab", 0))
        vocab.id2tok = data["id2tok"]
        vocab.tok2id = {tok: i for i, tok in enumerate(vocab.id2tok)}
        return vocab

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
