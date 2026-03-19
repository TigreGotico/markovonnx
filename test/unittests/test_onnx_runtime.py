"""Tests for markovonnx.onnx_runtime."""

import tempfile
from pathlib import Path

import numpy as np

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.onnx_export import export_hmm_onnx, export_markov_onnx
from markovonnx.onnx_runtime import HMMONNXRuntime, MarkovONNXRuntime
from markovonnx.vocabulary import Vocabulary


def _export_markov(tmpdir: str) -> tuple:
    """Train, export, and return (path, vocab, order)."""
    vocab = Vocabulary()
    seqs = [["a", "b", "a", "b"]] * 20
    vocab.build_from_sequences(seqs)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(seqs)
    path = str(Path(tmpdir) / "mc.onnx")
    export_markov_onnx(mc, path)
    return path, vocab, 1


class TestMarkovONNXRuntime:
    def test_predict_probs_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path, vocab, order = _export_markov(tmpdir)
            rt = MarkovONNXRuntime(path, vocab, order)
            probs = rt.predict_probs(["a"])
            assert probs.shape == (vocab.size,)

    def test_probs_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path, vocab, order = _export_markov(tmpdir)
            rt = MarkovONNXRuntime(path, vocab, order)
            probs = rt.predict_probs(["a"])
            assert abs(probs.sum() - 1.0) < 1e-4

    def test_sample_returns_valid_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path, vocab, order = _export_markov(tmpdir)
            rt = MarkovONNXRuntime(path, vocab, order)
            token = rt.sample(["a"])
            assert token in vocab.tok2id

    def test_sample_with_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path, vocab, order = _export_markov(tmpdir)
            rt = MarkovONNXRuntime(path, vocab, order)
            token = rt.sample(["a"], temperature=0.5)
            assert token in vocab.tok2id

    def test_argmax_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path, vocab, order = _export_markov(tmpdir)
            rt = MarkovONNXRuntime(path, vocab, order)
            t1 = rt.argmax(["a"])
            t2 = rt.argmax(["a"])
            assert t1 == t2


class TestHMMONNXRuntime:
    def test_decode_returns_correct_length(self) -> None:
        vocab = Vocabulary()
        vocab.build_from_sequences([["a", "b", "c"]])
        hmm = HiddenMarkovModel(n_states=3, obs_vocab=vocab)
        hmm.fit_supervised(
            [["a", "b", "c"]] * 5,
            [["X", "Y", "Z"]] * 5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "hmm.onnx")
            export_hmm_onnx(hmm, path)
            rt = HMMONNXRuntime(path, hmm)
            states = rt.decode(["a", "b", "c"])
            assert len(states) == 3

    def test_decode_without_state_vocab(self) -> None:
        """HMM without state_vocab returns stringified state indices."""
        vocab = Vocabulary()
        vocab.build_from_sequences([["a", "b", "c"]])
        hmm = HiddenMarkovModel(n_states=2, obs_vocab=vocab)
        hmm.fit_unsupervised([["a", "b", "a", "b"]] * 10, n_iter=3)
        assert hmm.state_vocab is None
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "hmm.onnx")
            export_hmm_onnx(hmm, path)
            rt = HMMONNXRuntime(path, hmm)
            states = rt.decode(["a", "b", "c"])
            assert len(states) == 3
            # Should be stringified indices
            for s in states:
                assert s in ("0", "1")
