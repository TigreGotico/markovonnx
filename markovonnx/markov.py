"""N-gram Markov chain with sparse storage, backoff, and dense export."""

import math
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from markovonnx.vocabulary import Vocabulary


class MarkovChain:
    """N-gram Markov chain with optional interpolated backoff and Kneser-Ney smoothing.

    Stores counts in a dict-of-arrays to stay sparse, converting to a dense
    matrix only on export.  This keeps RAM reasonable for large vocabularies.

    When *backoff* is enabled, lower-order models are trained alongside
    the primary model.  During sampling and perplexity computation, if the
    full-order context is unseen the model backs off to shorter contexts.

    Args:
        order: N-gram order (context length).
        vocab: :class:`Vocabulary` instance.
        smoothing: Laplace smoothing alpha (used when *kneser_ney* is ``False``).
        backoff: If ``True``, train and use lower-order models as fallback.
        kneser_ney: If ``True``, use modified Kneser-Ney smoothing instead
            of Laplace.  The discount *d* is estimated from counts automatically.
    """

    def __init__(
        self,
        order: int,
        vocab: Vocabulary,
        smoothing: float = 1e-5,
        backoff: bool = False,
        kneser_ney: bool = False,
    ):
        self.order = order
        self.vocab = vocab
        self.smoothing = smoothing
        self.backoff = backoff
        self.kneser_ney = kneser_ney
        self._counts: Dict[int, np.ndarray] = {}
        self._lower: Optional["MarkovChain"] = None
        # Modified KN: three discount levels (d1 for count==1, d2 for count==2, d3+ for count>=3)
        self._kn_d1: float = 0.75
        self._kn_d2: float = 0.75
        self._kn_d3: float = 0.75
        self._kn_discount: float = 0.75  # legacy single discount (average)

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

    def _estimate_kn_discounts(self) -> None:
        """Estimate Modified Kneser-Ney discounts from count-of-counts.

        Uses three discount levels:
        - d1 for n-grams with count == 1
        - d2 for n-grams with count == 2
        - d3 for n-grams with count >= 3

        Formula (Chen & Goodman 1999):
            Y = n1 / (n1 + 2*n2)
            d1 = 1 - 2*Y*(n2/n1)
            d2 = 2 - 3*Y*(n3/n2)
            d3 = 3 - 4*Y*(n4/n3)
        """
        n1 = n2 = n3 = n4 = 0
        for row in self._counts.values():
            n1 += int((row == 1).sum())
            n2 += int((row == 2).sum())
            n3 += int((row == 3).sum())
            n4 += int((row == 4).sum())

        if n1 == 0 or n1 + 2 * n2 == 0:
            self._kn_d1 = self._kn_d2 = self._kn_d3 = 0.75
        else:
            Y = n1 / (n1 + 2 * n2)
            self._kn_d1 = max(0.0, 1.0 - 2.0 * Y * (n2 / n1)) if n1 > 0 else 0.5
            self._kn_d2 = max(0.0, 2.0 - 3.0 * Y * (n3 / n2)) if n2 > 0 else 1.0
            self._kn_d3 = max(0.0, 3.0 - 4.0 * Y * (n4 / n3)) if n3 > 0 else 1.5

        # Legacy single discount (weighted average for dense_matrix compatibility)
        self._kn_discount = (self._kn_d1 + self._kn_d2 + self._kn_d3) / 3.0

    def fit(self, sequences: List[List]) -> None:
        """Train on an in-memory list of token sequences."""
        t0 = time.time()
        for seq in sequences:
            self._update_from_sequence(self.vocab.encode(seq))
        elapsed = time.time() - t0
        if self.kneser_ney:
            self._estimate_kn_discounts()
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
                kneser_ney=self.kneser_ney,
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
        if self.kneser_ney:
            self._estimate_kn_discounts()
        print(
            f"Streaming fit done: {n:,} sequences, "
            f"{len(self._counts):,} contexts  ({time.time() - t0:.1f}s)"
        )

    # -- Dense matrix (needed for ONNX export) --------------------------------

    def dense_matrix(self) -> np.ndarray:
        """Build full transition matrix ``T[V^order, V]`` with smoothing.

        Uses Kneser-Ney discounting if *kneser_ney* was set, otherwise Laplace.

        Raises:
            MemoryError: If the matrix would exceed 2 GB.
        """
        V = self.vocab.size
        total_rows = V ** self.order
        matrix_bytes = total_rows * V * 4
        if matrix_bytes > 2 * 1024 ** 3:
            raise MemoryError(
                f"Dense transition matrix would be {matrix_bytes / (1024 ** 3):.1f} GB "
                f"(V={V}, order={self.order}). Use export_markov_sparse_onnx() instead."
            )
        if self.kneser_ney:
            T = self._dense_kneser_ney(total_rows, V)
        else:
            T = np.full((total_rows, V), self.smoothing, dtype=np.float32)
            for ci, row in self._counts.items():
                if ci < total_rows:
                    T[ci] += row
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-30)  # prevent division by zero
            T /= row_sums
        return T

    def _dense_kneser_ney(self, total_rows: int, V: int) -> np.ndarray:
        """Build dense matrix with absolute-discount Kneser-Ney smoothing."""
        d = self._kn_discount
        T = np.full((total_rows, V), 1.0 / V, dtype=np.float32)  # uniform backoff
        for ci, row in self._counts.items():
            if ci >= total_rows:
                continue
            total = row.sum()
            if total == 0:
                continue
            n_positive = float((row > 0).sum())
            lam = d * n_positive / total  # interpolation weight
            discounted = np.maximum(row - d, 0.0) / total
            T[ci] = discounted + lam * (1.0 / V)
        return T

    # -- Probability lookup with backoff --------------------------------------

    def _get_probs(self, context: List) -> np.ndarray:
        """Return smoothed probability vector for *context*, with backoff."""
        V = self.vocab.size
        ids = self.vocab.encode(context[-self.order :])
        ci = self._ctx_idx(ids)
        row = self._counts.get(ci, None)
        if row is not None and row.sum() > 0:
            if self.kneser_ney:
                return self._kn_probs(row)
            return (row + self.smoothing) / (row.sum() + self.smoothing * V)
        # Backoff to lower-order model
        if self._lower is not None and len(context) > 1:
            return self._lower._get_probs(context[1:])
        # Uniform fallback
        return np.full(V, 1.0 / V, dtype=np.float32)

    def _kn_probs(self, row: np.ndarray) -> np.ndarray:
        """Compute Modified Kneser-Ney smoothed probabilities for a single row.

        Uses three discount levels: d1 (count==1), d2 (count==2), d3+ (count>=3).
        """
        V = self.vocab.size
        total = row.sum()
        if total == 0:
            return np.full(V, 1.0 / V, dtype=np.float32)

        # Apply count-specific discounts
        discounts = np.where(
            row >= 3, self._kn_d3,
            np.where(row == 2, self._kn_d2,
                     np.where(row == 1, self._kn_d1, 0.0))
        )
        discounted = np.maximum(row - discounts, 0.0) / total

        # Interpolation weight from total discount mass
        discount_mass = float(np.sum(np.minimum(discounts, row))) / total
        return discounted + discount_mass * (1.0 / V)

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
