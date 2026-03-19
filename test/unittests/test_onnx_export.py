"""Tests for markovonnx.onnx_export."""

import tempfile
from pathlib import Path

import onnx

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.onnx_export import export_hmm_onnx, export_markov_onnx, quantize_model
from markovonnx.vocabulary import Vocabulary


def _trained_markov() -> MarkovChain:
    vocab = Vocabulary()
    seqs = [["a", "b", "a", "b"]] * 10
    vocab.build_from_sequences(seqs)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(seqs)
    return mc


def _trained_hmm() -> HiddenMarkovModel:
    vocab = Vocabulary()
    vocab.build_from_sequences([["a", "b", "c"]])
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=vocab)
    obs_seqs = [["a", "b", "c"]] * 5
    tag_seqs = [["X", "Y", "Z"]] * 5
    hmm.fit_supervised(obs_seqs, tag_seqs)
    return hmm


class TestExportMarkov:
    def test_creates_valid_onnx(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.onnx")
            export_markov_onnx(mc, path)
            assert Path(path).exists()
            model = onnx.load(path)
            onnx.checker.check_model(model)

    def test_metadata(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.onnx")
            export_markov_onnx(mc, path)
            model = onnx.load(path)
            meta = {p.key: p.value for p in model.metadata_props}
            assert meta["model_type"] == "markov_chain"
            assert meta["order"] == "1"

    def test_creates_parent_dirs(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "test.onnx")
            export_markov_onnx(mc, path)
            assert Path(path).exists()


class TestExportHMM:
    def test_creates_valid_onnx(self) -> None:
        hmm = _trained_hmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "hmm.onnx")
            export_hmm_onnx(hmm, path)
            assert Path(path).exists()
            model = onnx.load(path)
            onnx.checker.check_model(model)


class TestQuantizeModel:
    def test_quantize_success(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = str(Path(tmpdir) / "full.onnx")
            q_path = str(Path(tmpdir) / "int8.onnx")
            export_markov_onnx(mc, fp_path)
            result = quantize_model(fp_path, q_path)
            assert result == q_path
            assert Path(q_path).exists()
            # Quantized model should be loadable
            model = onnx.load(q_path)
            assert model is not None

    def test_quantize_failure_returns_none(self) -> None:
        result = quantize_model("/nonexistent/model.onnx", "/tmp/out.onnx")
        assert result is None
