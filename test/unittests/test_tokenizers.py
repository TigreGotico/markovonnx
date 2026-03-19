"""Tests for markovonnx.tokenizers."""

import tempfile

from markovonnx.tokenizers import char_tokenize, corpus_iter, get_tokenize_fn, word_tokenize


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
