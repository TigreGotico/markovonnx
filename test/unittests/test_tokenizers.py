"""Tests for markovonnx.tokenizers."""

import json
import tempfile
from pathlib import Path

from markovonnx.tokenizers import (
    SubwordTokenizer,
    char_tokenize,
    corpus_iter,
    get_tokenize_fn,
    word_tokenize,
)


class TestCharTokenize:
    def test_basic(self) -> None:
        assert char_tokenize("hello") == ["h", "e", "l", "l", "o"]

    def test_strips_whitespace(self) -> None:
        assert char_tokenize("  ab \n") == ["a", "b"]

    def test_empty(self) -> None:
        assert char_tokenize("") == []


class TestWordTokenize:
    def test_basic(self) -> None:
        assert word_tokenize("Hello World") == ["hello", "world"]

    def test_lowercases(self) -> None:
        assert word_tokenize("FOO BAR") == ["foo", "bar"]

    def test_empty(self) -> None:
        assert word_tokenize("") == []


class TestCorpusIter:
    def test_basic(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nfoo bar\nbaz\n")
            path = f.name
        lines = list(corpus_iter(path, word_tokenize))
        assert len(lines) == 3
        assert lines[0] == ["hello", "world"]

    def test_max_lines(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\nd\n")
            path = f.name
        lines = list(corpus_iter(path, char_tokenize, max_lines=2))
        assert len(lines) == 2

    def test_skips_empty_lines(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n\n\nworld\n")
            path = f.name
        lines = list(corpus_iter(path, word_tokenize))
        assert len(lines) == 2


class TestGetTokenizeFn:
    def test_char_mode(self) -> None:
        fn = get_tokenize_fn("char")
        assert fn("hi") == ["h", "i"]

    def test_word_mode(self) -> None:
        fn = get_tokenize_fn("word")
        assert fn("Hello World") == ["hello", "world"]

    def test_bpe_requires_tokenizer(self) -> None:
        try:
            get_tokenize_fn("bpe")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_subword_mode_requires_tokenizer(self) -> None:
        try:
            get_tokenize_fn("subword")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_default_is_word(self) -> None:
        fn = get_tokenize_fn("anything_else")
        assert fn("Hello World") == ["hello", "world"]

    def test_bpe_with_tokenizer(self) -> None:
        bpe = _make_bpe_tokenizer()
        fn = get_tokenize_fn("bpe", bpe_tokenizer=bpe)
        result = fn("ab")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)


def _make_bpe_tokenizer() -> SubwordTokenizer:
    """Create a minimal SubwordTokenizer from a synthetic JSON."""
    tok_data = {
        "model": {
            "vocab": {
                "[UNK]": 0,
                "a": 1,
                "b": 2,
                "c": 3,
                "ab": 4,
            },
            "merges": [["a", "b"]],
            "unk_token": "[UNK]",
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(tok_data, f)
        path = f.name
    return SubwordTokenizer(path)


class TestSubwordTokenizer:
    def test_init_loads_vocab(self) -> None:
        bpe = _make_bpe_tokenizer()
        assert bpe.vocab == {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "ab": 4}
        assert bpe.unk_token == "[UNK]"
        assert bpe.unk_id == 0

    def test_id_to_token(self) -> None:
        bpe = _make_bpe_tokenizer()
        assert bpe.id_to_token[4] == "ab"
        assert bpe.id_to_token[0] == "[UNK]"

    def test_text_to_initial_tokens(self) -> None:
        tokens = SubwordTokenizer.text_to_initial_tokens("abc")
        assert tokens == ["a", "b", "c"]

    def test_text_to_initial_tokens_unicode(self) -> None:
        tokens = SubwordTokenizer.text_to_initial_tokens("é")
        # UTF-8 for é is 0xc3 0xa9 -> chr(195), chr(169)
        assert len(tokens) == 2

    def test_apply_bpe_merges(self) -> None:
        bpe = _make_bpe_tokenizer()
        result = bpe.apply_bpe(["a", "b", "c"])
        assert result == ["ab", "c"]

    def test_apply_bpe_no_merge(self) -> None:
        bpe = _make_bpe_tokenizer()
        result = bpe.apply_bpe(["c", "c"])
        assert result == ["c", "c"]

    def test_encode_bpe(self) -> None:
        bpe = _make_bpe_tokenizer()
        ids = bpe.encode_bpe("abc")
        assert ids == [4, 3]  # "ab"=4, "c"=3

    def test_encode_bpe_unknown(self) -> None:
        bpe = _make_bpe_tokenizer()
        ids = bpe.encode_bpe("z")  # 'z' not in vocab
        assert ids == [0]  # UNK

    def test_decode_bpe(self) -> None:
        bpe = _make_bpe_tokenizer()
        text = bpe.decode_bpe([4, 3])
        assert text == "abc"

    def test_decode_bpe_unicode_decode_error_fallback(self) -> None:
        """Test decode_bpe fallback when latin-1 bytes aren't valid UTF-8."""
        # Characters \x80\x81 are valid latin-1 but NOT valid UTF-8
        tok_data = {
            "model": {
                "vocab": {"[UNK]": 0, "\x80": 1, "\x81": 2},
                "merges": [],
                "unk_token": "[UNK]",
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(tok_data, f)
            path = f.name
        bpe = SubwordTokenizer(path)
        result = bpe.decode_bpe([1, 2])
        # latin-1 encode succeeds, but UTF-8 decode fails -> raw join fallback
        assert result == "\x80\x81"
