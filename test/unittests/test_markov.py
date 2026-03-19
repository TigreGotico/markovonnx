"""Tests for markovonnx.markov."""

import tempfile

from markovonnx.markov import MarkovChain
from markovonnx.tokenizers import char_tokenize
from markovonnx.vocabulary import Vocabulary


def _build_chain() -> MarkovChain:
    """Build a small Markov chain on deterministic data."""
    vocab = Vocabulary()
    # "a b a b a b" repeated — strong a->b and b->a transitions
    seqs = [["a", "b", "a", "b", "a", "b"]] * 20
    vocab.build_from_sequences(seqs)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-8)
    mc.fit(seqs)
    return mc


class TestMarkovChain:
    def test_fit_populates_counts(self) -> None:
        mc = _build_chain()
        assert len(mc._counts) > 0

    def test_sample_returns_valid_token(self) -> None:
        mc = _build_chain()
        token = mc.sample(["a"])
        assert token in mc.vocab.tok2id

    def test_deterministic_transition(self) -> None:
        mc = _build_chain()
        # With very low smoothing, a->b should be dominant
        counts = {}
        for _ in range(100):
            t = mc.sample(["a"], temperature=0.01)
            counts[t] = counts.get(t, 0) + 1
        assert counts.get("b", 0) > 90  # should be nearly always "b"

    def test_dense_matrix_shape(self) -> None:
        mc = _build_chain()
        T = mc.dense_matrix()
        V = mc.vocab.size
        assert T.shape == (V, V)  # order=1 so V^1 x V

    def test_dense_matrix_rows_sum_to_one(self) -> None:
        mc = _build_chain()
        T = mc.dense_matrix()
        row_sums = T.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-5

    def test_perplexity_finite(self) -> None:
        mc = _build_chain()
        ppx = mc.perplexity([["a", "b", "a", "b"]])
        assert ppx > 0
        assert ppx < 1000

    def test_order2(self) -> None:
        vocab = Vocabulary()
        seqs = [["a", "b", "c", "a", "b", "c"]] * 10
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-8)
        mc.fit(seqs)
        T = mc.dense_matrix()
        V = vocab.size
        assert T.shape == (V ** 2, V)

    def test_fit_streaming(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for _ in range(50):
                f.write("abcabc\n")
            path = f.name
        vocab = Vocabulary()
        vocab.build_streaming(path, tokenize_fn=char_tokenize)
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
        mc.fit_streaming(path, tokenize_fn=char_tokenize)
        assert len(mc._counts) > 0
        # Verify sampling works
        token = mc.sample(["a"])
        assert token in vocab.tok2id

    def test_sample_unseen_context(self) -> None:
        mc = _build_chain()
        # "<UNK>" context has no counts -> should return random token
        token = mc.sample(["<UNK>"])
        assert token in mc.vocab.tok2id

    def test_sample_with_temperature(self) -> None:
        mc = _build_chain()
        # Just verify it runs with temperature=1.0 (no rescaling path)
        token = mc.sample(["a"], temperature=1.0)
        assert token in mc.vocab.tok2id

    def test_perplexity_unseen_context(self) -> None:
        mc = _build_chain()
        # Sequence with tokens that create unseen contexts
        ppx = mc.perplexity([["<UNK>", "<UNK>", "<UNK>"]])
        assert ppx > 0

    def test_backoff_trains_lower_order(self) -> None:
        vocab = Vocabulary()
        seqs = [["a", "b", "c", "a", "b", "c"]] * 10
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5, backoff=True)
        mc.fit(seqs)
        assert mc._lower is not None
        assert mc._lower.order == 1
        # order=1 doesn't create a lower model (order must be > 1)
        assert mc._lower._lower is None

    def test_backoff_sampling_unseen(self) -> None:
        vocab = Vocabulary()
        seqs = [["a", "b", "a", "b"]] * 20
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5, backoff=True)
        mc.fit(seqs)
        # Context ["<UNK>", "a"] is unseen at order=2, should backoff to order=1
        token = mc.sample(["<UNK>", "a"])
        assert token in vocab.tok2id

    def test_get_probs_uniform_fallback(self) -> None:
        vocab = Vocabulary()
        seqs = [["a", "b"]] * 5
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5, backoff=False)
        mc.fit(seqs)
        # Unseen context with no backoff -> uniform
        probs = mc._get_probs(["<UNK>"])
        assert abs(probs.sum() - 1.0) < 1e-5
