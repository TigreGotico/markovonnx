"""N-gram Markov chain with sparse storage, backoff, and dense export."""

import math
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from markovonnx.vocabulary import Vocabulary


class MarkovChain:
    """N-gram Markov chain with optional interpolated backoff.

    Stores counts in a dict-of-arrays to stay sparse, converting to a dense
    matrix only on export.  This keeps RAM reasonable for large vocabularies.

    When *backoff* is enabled, lower-order models are trained alongside
    the primary model.  During sampling and perplexity computation, if the
    full-order context is unseen the model backs off to shorter contexts.

    Args:
        order: N-gram order (context length).
        vocab: :class:`Vocabulary` instance.
        smoothing: Laplace smoothing alpha.
        backoff: If ``True``, train and use lower-order models as fallback.
    """

    def __init__(
        self,
        order: int,
        vocab: Vocabulary,
        smoothing: float = 1e-5,
        backoff: bool = False,
    ):
        self.order = order
        self.vocab = vocab
        self.smoothing = smoothing
        self.backoff = backoff
        self._counts: Dict[int, np.ndarray] = {}
        self._lower: Optional["MarkovChain"] = None

    # -- Context encoding -----------------------------------------------------

    def _ctx_idx(self, ctx_ids: List[int]) -> int:
        """Map a context (list of IDs) to a single row index via base-V encoding."""
        V = self.vocab.size
        idx = 0
        for t in ctx_ids[-self.order :]:
            idx = idx * V + t
        return idx

    # -- Training -------------------------------------------------------------

    def _update_from_sequence(self, ids: List[int]) -> None:
        V = self.vocab.size
        for i in range(len(ids) - self.order):
            ctx = ids[i : i + self.order]
            nxt = ids[i + self.order]
            ci = self._ctx_idx(ctx)
            if ci not in self._counts:
                self._counts[ci] = np.zeros(V, dtype=np.float32)
            self._counts[ci][nxt] += 1.0

    def fit(self, sequences: List[List]) -> None:
        """Train on an in-memory list of token sequences."""
        t0 = time.time()
        for seq in sequences:
            self._update_from_sequence(self.vocab.encode(seq))
        elapsed = time.time() - t0
        print(
            f"MarkovChain(order={self.order}) trained on "
            f"{len(sequences):,} sequences in {elapsed:.2f}s  |  "
            f"unique contexts: {len(self._counts):,}"
        )
        if self.backoff and self.order > 1:
            self._lower = MarkovChain(
                order=self.order - 1,
                vocab=self.vocab,
                smoothing=self.smoothing,
                backoff=True,
            )
            self._lower.fit(sequences)

    def fit_streaming(
        self,
        path: str,
        tokenize_fn: Callable[[str], List],
        max_lines: int = 0,
    ) -> None:
        """Train by streaming a corpus file.

        Args:
            path: Path to a text corpus.
            tokenize_fn: Callable that converts a raw line to tokens.
            max_lines: Stop after this many non-empty lines (0 = unlimited).
        """
        from markovonnx.tokenizers import corpus_iter

        t0 = time.time()
        n = 0
        for seq in corpus_iter(path, tokenize_fn, max_lines):
            self._update_from_sequence(self.vocab.encode(seq))
            n += 1
        print(
            f"Streaming fit done: {n:,} sequences, "
            f"{len(self._counts):,} contexts  ({time.time() - t0:.1f}s)"
        )

    # -- Dense matrix (needed for ONNX export) --------------------------------

    def dense_matrix(self) -> np.ndarray:
        """Build full transition matrix ``T[V^order, V]`` with Laplace smoothing."""
        V = self.vocab.size
        total_rows = V ** self.order
        T = np.full((total_rows, V), self.smoothing, dtype=np.float32)
        for ci, row in self._counts.items():
            if ci < total_rows:
                T[ci] += row
        row_sums = T.sum(axis=1, keepdims=True)
        T /= row_sums
        return T

    # -- Probability lookup with backoff --------------------------------------

    def _get_probs(self, context: List) -> np.ndarray:
        """Return smoothed probability vector for *context*, with backoff."""
        V = self.vocab.size
        ids = self.vocab.encode(context[-self.order :])
        ci = self._ctx_idx(ids)
        row = self._counts.get(ci, None)
        if row is not None and row.sum() > 0:
            return (row + self.smoothing) / (row.sum() + self.smoothing * V)
        # Backoff to lower-order model
        if self._lower is not None and len(context) > 1:
            return self._lower._get_probs(context[1:])
        # Uniform fallback
        return np.full(V, 1.0 / V, dtype=np.float32)

    # -- Sampling -------------------------------------------------------------

    def sample(self, context: List, temperature: float = 1.0) -> object:
        """Sample the next token given a context.

        Uses backoff to lower-order models when context is unseen
        (if *backoff* was enabled during training).

        Args:
            context: Token sequence (at least *order* tokens).
            temperature: Sampling temperature (1.0 = unscaled).

        Returns:
            A single token from the vocabulary.
        """
        V = self.vocab.size
        probs = self._get_probs(context)
        if temperature != 1.0:
            logits = np.log(probs + 1e-30) / temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
        return self.vocab.id2tok[int(np.random.choice(V, p=probs))]

    # -- Perplexity -----------------------------------------------------------

    def perplexity(self, sequences: List[List]) -> float:
        """Compute perplexity of the model over a list of token sequences.

        Uses backoff for unseen contexts if enabled.
        """
        total_log = 0.0
        total_n = 0
        for seq in sequences:
            ids = self.vocab.encode(seq)
            for i in range(len(ids) - self.order):
                ctx = ids[i : i + self.order]
                nxt = ids[i + self.order]
                # Use backoff-aware probability
                probs = self._get_probs(
                    self.vocab.decode(ctx)
                )
                p = float(probs[nxt])
                total_log += math.log(max(p, 1e-30))
                total_n += 1
        return math.exp(-total_log / max(total_n, 1))
