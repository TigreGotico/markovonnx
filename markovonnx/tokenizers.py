"""Tokenization utilities: character, word, BPE, and corpus iteration."""

import json
from typing import Callable, Iterator, List, Optional


def char_tokenize(line: str) -> List[str]:
    """Split a line into individual characters (stripping whitespace)."""
    return list(line.strip())


def word_tokenize(line: str) -> List[str]:
    """Split a line into lower-cased words."""
    return line.strip().lower().split()


def corpus_iter(
    path: str,
    tokenize_fn: Callable[[str], List],
    max_lines: int = 0,
) -> Iterator[List]:
    """Yield tokenised sequences one line at a time.

    Args:
        path: Path to a text corpus (one sentence/document per line).
        tokenize_fn: Callable that converts a raw line to a list of tokens.
        max_lines: Stop after this many non-empty lines (0 = unlimited).
    """
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            tokens = tokenize_fn(line)
            if tokens:
                yield tokens
                n += 1
                if max_lines and n >= max_lines:
                    break


class SubwordTokenizer:
    """Dependency-free BPE tokenizer that loads a HuggingFace ``tokenizers`` JSON.

    Args:
        tokenizer_json: Path to a ``tokenizers``-compatible JSON file.
    """

    def __init__(self, tokenizer_json: str = "subword_tokenizer.json"):
        with open(tokenizer_json, "r") as f:
            tok = json.load(f)

        self.vocab: dict = tok["model"]["vocab"]
        self.id_to_token: dict = {i: t for t, i in self.vocab.items()}
        self.merges: list = tok["model"]["merges"]
        self.unk_token: str = tok["model"].get("unk_token", "[UNK]")
        self.unk_id: int = self.vocab[self.unk_token]

    @staticmethod
    def text_to_initial_tokens(s: str) -> List[str]:
        """Convert text to byte-level character tokens."""
        return [chr(b) for b in s.encode("utf-8")]

    def encode_bpe(self, s: str) -> List[int]:
        """Encode a string to a list of BPE token IDs."""
        tokens = self.text_to_initial_tokens(s)
        tokens = self.apply_bpe(tokens)
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def decode_bpe(self, ids: List[int]) -> str:
        """Decode a list of BPE token IDs back to a string.

        Unknown IDs are replaced with the unknown token.
        """
        unk = self.unk_token
        tokens = [self.id_to_token.get(i, unk) for i in ids]
        s = "".join(tokens)
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return s

    def apply_bpe(self, tokens: List[str]) -> List[str]:
        """Apply BPE merges to a list of character-level tokens."""
        pairs = lambda ts: {(ts[i], ts[i + 1]) for i in range(len(ts) - 1)}

        while True:
            ps = pairs(tokens)
            candidate = None
            for m in self.merges:
                if tuple(m) in ps:
                    candidate = tuple(m)
                    break
            if candidate is None:
                break

            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == candidate:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens


def get_tokenize_fn(
    mode: str,
    bpe_tokenizer: Optional[SubwordTokenizer] = None,
) -> Callable[[str], List]:
    """Return the appropriate tokenization function for *mode*.

    Args:
        mode: One of ``'char'``, ``'word'``, ``'bpe'`` / ``'subword'``.
        bpe_tokenizer: Required when *mode* is ``'bpe'`` or ``'subword'``.

    Returns:
        A callable ``(str) -> List[str | int]``.
    """
    if mode in ("bpe", "subword"):
        if bpe_tokenizer is None:
            raise ValueError("bpe_tokenizer is required for BPE/subword mode")
        return bpe_tokenizer.encode_bpe
    if mode == "char":
        return char_tokenize
    return word_tokenize
