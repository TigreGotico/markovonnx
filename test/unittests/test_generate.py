"""Tests for markovonnx.generate."""

import tempfile
from pathlib import Path

from markovonnx.generate import generate_markov
from markovonnx.markov import MarkovChain
from markovonnx.onnx_export import export_markov_onnx
from markovonnx.onnx_runtime import MarkovONNXRuntime
from markovonnx.vocabulary import Vocabulary


def _build_rt(tmpdir: str) -> tuple:
    """Build a small trained runtime for char mode."""
    vocab = Vocabulary()
    seqs = [list("abcabc")] * 20
    vocab.build_from_sequences(seqs)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(seqs)
    path = str(Path(tmpdir) / "mc.onnx")
    export_markov_onnx(mc, path)
    rt = MarkovONNXRuntime(path, vocab, 1)
    return rt


class TestGenerateMarkov:
    def test_char_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rt = _build_rt(tmpdir)
            text = generate_markov(rt, "a", length=5, mode="char", order=1)
            assert isinstance(text, str)
            assert len(text) >= 2  # at least seed + some generated

    def test_word_mode(self) -> None:
        vocab = Vocabulary()
        seqs = [["the", "cat", "sat"]] * 20
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
        mc.fit(seqs)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mc.onnx")
            export_markov_onnx(mc, path)
            rt = MarkovONNXRuntime(path, vocab, 1)
            text = generate_markov(rt, "the", length=3, mode="word", order=1)
            assert isinstance(text, str)
            words = text.split()
            assert len(words) >= 2

    def test_bpe_requires_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rt = _build_rt(tmpdir)
            try:
                generate_markov(rt, "a", length=5, mode="bpe", order=1)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass
