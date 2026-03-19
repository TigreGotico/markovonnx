"""Tests for markovonnx.vocabulary."""

import tempfile

from markovonnx.tokenizers import char_tokenize
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

    def test_build_streaming(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nfoo bar\n")
            path = f.name
        vocab = Vocabulary()
        vocab.build_streaming(path, tokenize_fn=char_tokenize)
        assert vocab.size > 1  # at least UNK + some chars
        assert "<UNK>" in vocab.tok2id
        # Verify all characters are present
        for c in "helo wrdfobar":
            assert c in vocab.tok2id, f"'{c}' not in vocabulary"

    def test_build_streaming_with_max_lines(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aaa\nbbb\nccc\n")
            path = f.name
        vocab = Vocabulary()
        vocab.build_streaming(path, tokenize_fn=char_tokenize, max_lines=1)
        # Only first line "aaa" -> vocab has UNK + "a"
        assert "a" in vocab.tok2id
        assert "b" not in vocab.tok2id

    def test_to_dict_from_dict_roundtrip(self) -> None:
        vocab = Vocabulary(max_vocab=2)
        vocab.build_from_sequences([["a", "a", "b", "b", "c"]])
        data = vocab.to_dict()
        restored = Vocabulary.from_dict(data)
        assert restored.id2tok == vocab.id2tok
        assert restored.tok2id == vocab.tok2id
        assert restored.size == vocab.size
        assert restored.max_vocab == vocab.max_vocab

    def test_save_load_roundtrip(self) -> None:
        vocab = Vocabulary()
        vocab.build_from_sequences([["x", "y", "z"]])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vocab.save(path)
        loaded = Vocabulary.load(path)
        assert loaded.id2tok == vocab.id2tok
        assert loaded.tok2id == vocab.tok2id
