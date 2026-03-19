"""Property-based tests using Hypothesis."""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from markovonnx.tokenizers import char_tokenize, word_tokenize
from markovonnx.vocabulary import Vocabulary
from markovonnx.markov import MarkovChain


# -- Tokenizer properties ----------------------------------------------------

@given(st.text(min_size=0, max_size=200))
def test_char_tokenize_length(text: str) -> None:
    """char_tokenize output length == len(stripped text)."""
    result = char_tokenize(text)
    assert len(result) == len(text.strip())


@given(st.text(min_size=0, max_size=200))
def test_word_tokenize_lowercase(text: str) -> None:
    """word_tokenize always returns lowercase tokens."""
    result = word_tokenize(text)
    for token in result:
        assert token == token.lower()


# -- Vocabulary properties ---------------------------------------------------

@given(st.lists(
    st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=10),
    min_size=1, max_size=20,
))
def test_vocab_encode_decode_roundtrip(sequences: list) -> None:
    """Encoding then decoding should return original tokens (if in vocab)."""
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)
    for seq in sequences:
        ids = vocab.encode(seq)
        decoded = vocab.decode(ids)
        assert decoded == seq


@given(st.integers(min_value=1, max_value=10))
def test_vocab_max_vocab_constraint(max_vocab: int) -> None:
    """Vocabulary size never exceeds max_vocab + 1 (for UNK)."""
    vocab = Vocabulary(max_vocab=max_vocab)
    seqs = [list("abcdefghijklmnop")]
    vocab.build_from_sequences(seqs)
    assert vocab.size <= max_vocab + 1


@given(st.lists(
    st.lists(st.sampled_from(["x", "y", "z"]), min_size=1, max_size=5),
    min_size=1, max_size=10,
))
def test_vocab_unk_at_zero(sequences: list) -> None:
    """UNK is always at index 0."""
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)
    assert vocab.id2tok[0] == "<UNK>"
    assert vocab.tok2id["<UNK>"] == 0


# -- Vocabulary serialization roundtrip ---------------------------------------

@given(st.lists(
    st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=5),
    min_size=1, max_size=5,
))
def test_vocab_serialization_roundtrip(sequences: list) -> None:
    """to_dict / from_dict preserves the mapping."""
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.id2tok == vocab.id2tok
    assert restored.tok2id == vocab.tok2id
    assert restored.size == vocab.size


# -- MarkovChain properties --------------------------------------------------

@given(st.lists(
    st.lists(st.sampled_from(["a", "b", "c"]), min_size=3, max_size=10),
    min_size=2, max_size=10,
))
def test_dense_matrix_rows_sum_to_one(sequences: list) -> None:
    """Every row of the dense transition matrix sums to 1."""
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(sequences)
    T = mc.dense_matrix()
    row_sums = T.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


@given(st.lists(
    st.lists(st.sampled_from(["a", "b"]), min_size=3, max_size=8),
    min_size=2, max_size=5,
))
def test_perplexity_always_positive(sequences: list) -> None:
    """Perplexity is always positive and finite."""
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(sequences)
    ppx = mc.perplexity(sequences)
    assert ppx > 0
    assert np.isfinite(ppx)
