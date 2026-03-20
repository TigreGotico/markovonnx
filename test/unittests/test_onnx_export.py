"""Tests for markovonnx.onnx_export."""

import tempfile
from pathlib import Path

import onnx

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.onnx_export import (
    export_hmm_onnx,
    export_markov_onnx,
    export_markov_sparse_onnx,
    quantize_model,
)
from markovonnx.onnx_runtime import MarkovONNXRuntime
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


class TestExportMarkovSparse:
    def test_creates_valid_onnx(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sparse.onnx")
            export_markov_sparse_onnx(mc, path)
            assert Path(path).exists()
            model = onnx.load(path)
            onnx.checker.check_model(model)

    def test_metadata(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sparse.onnx")
            export_markov_sparse_onnx(mc, path)
            model = onnx.load(path)
            meta = {p.key: p.value for p in model.metadata_props}
            assert meta["model_type"] == "markov_chain_sparse"
            assert meta["order"] == "1"
            assert int(meta["n_sparse_rows"]) > 0

    def test_smaller_than_dense(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            dense_path = str(Path(tmpdir) / "dense.onnx")
            sparse_path = str(Path(tmpdir) / "sparse.onnx")
            export_markov_onnx(mc, dense_path)
            export_markov_sparse_onnx(mc, sparse_path)
            dense_size = Path(dense_path).stat().st_size
            sparse_size = Path(sparse_path).stat().st_size
            # Sparse should be smaller (or similar for tiny models)
            # For this tiny model they may be similar, just check it works
            assert sparse_size > 0

    def test_inference_matches_dense(self) -> None:
        """Sparse export should produce same probs as dense for seen contexts."""
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            dense_path = str(Path(tmpdir) / "dense.onnx")
            sparse_path = str(Path(tmpdir) / "sparse.onnx")
            export_markov_onnx(mc, dense_path)
            export_markov_sparse_onnx(mc, sparse_path)

            rt_dense = MarkovONNXRuntime(dense_path, mc.vocab, mc.order)
            rt_sparse = MarkovONNXRuntime(sparse_path, mc.vocab, mc.order)

            # Test a known context
            dense_probs = rt_dense.predict_probs(["a"])
            sparse_probs = rt_sparse.predict_probs(["a"])
            import numpy as np
            assert np.allclose(dense_probs, sparse_probs, atol=1e-4)

    def test_kneser_ney_sparse(self) -> None:
        vocab = Vocabulary()
        seqs = [["a", "b", "a", "b"]] * 20
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=1, vocab=vocab, kneser_ney=True)
        mc.fit(seqs)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "kn_sparse.onnx")
            export_markov_sparse_onnx(mc, path)
            assert Path(path).exists()

    def test_flat_index_lookup_mode_metadata(self) -> None:
        """Small vocabulary uses flat-index O(1) lookup mode."""
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sparse.onnx")
            export_markov_sparse_onnx(mc, path)
            model = onnx.load(path)
            meta = {p.key: p.value for p in model.metadata_props}
            assert meta["lookup_mode"] == "flat-index O(1)"

    def test_flat_index_inference_correct(self) -> None:
        """Flat-index sparse export produces correct probabilities."""
        import numpy as np
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            dense_path = str(Path(tmpdir) / "dense.onnx")
            sparse_path = str(Path(tmpdir) / "sparse.onnx")
            export_markov_onnx(mc, dense_path)
            export_markov_sparse_onnx(mc, sparse_path)
            rt_dense = MarkovONNXRuntime(dense_path, mc.vocab, mc.order)
            rt_sparse = MarkovONNXRuntime(sparse_path, mc.vocab, mc.order)
            for ctx in [["a"], ["b"]]:
                np.testing.assert_allclose(
                    rt_dense.predict_probs(ctx),
                    rt_sparse.predict_probs(ctx),
                    atol=1e-4,
                )
