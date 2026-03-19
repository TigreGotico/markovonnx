"""Symbol-to-integer vocabulary with optional max-size pruning."""

from collections import Counter
from typing import Callable, Dict, Iterator, List


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
