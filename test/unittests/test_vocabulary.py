"""Tests for markovonnx.vocabulary."""

from markovonnx.vocabulary import Vocabulary


class TestVocabulary:
    def test_build_and_size(self) -> None:
        vocab = Vocabulary()
        vocab.build_from_sequences([["a", "b", "c"], ["a", "b"]])
        assert vocab.size == 4  # UNK + a + b + c

    def test_encode_decode_roundtrip(self) -> None:
        vocab = Vocabulary()
        vocab.build_from_sequences([["x", "y", "z"]])
        tokens = ["x", "y", "z"]
        ids = vocab.encode(tokens)
        assert vocab.decode(ids) == tokens

    def test_unknown_token(self) -> None:
        vocab = Vocabulary()
        vocab.build_from_sequences([["a"]])
        ids = vocab.encode(["a", "UNKNOWN"])
        assert ids[1] == vocab.tok2id[Vocabulary.UNK]

    def test_max_vocab(self) -> None:
        vocab = Vocabulary(max_vocab=2)
        vocab.build_from_sequences([["a", "a", "b", "b", "c"]])
        # Should keep only top 2 + UNK
        assert vocab.size == 3

    def test_ordering_by_frequency(self) -> None:
        vocab = Vocabulary()
        vocab.build_from_sequences([["rare", "common", "common", "common"]])
        # UNK at 0, common at 1, rare at 2
        assert vocab.id2tok[0] == "<UNK>"
        assert vocab.id2tok[1] == "common"
        assert vocab.id2tok[2] == "rare"
