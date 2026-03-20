"""Tests for markovonnx.archive."""

import tempfile
from pathlib import Path

from markovonnx.archive import load_markov_archive, save_markov_archive
from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.vocabulary import Vocabulary


def _trained_markov() -> MarkovChain:
    vocab = Vocabulary()
    seqs = [["a", "b", "a", "b"]] * 20
    vocab.build_from_sequences(seqs)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(seqs)
    return mc


def _trained_hmm() -> HiddenMarkovModel:
    vocab = Vocabulary()
    vocab.build_from_sequences([["a", "b", "c"]])
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=vocab)
    hmm.fit_supervised(
        [["a", "b", "c"]] * 5,
        [["X", "Y", "Z"]] * 5,
    )
    return hmm


class TestMarkovArchive:
    def test_save_and_load_roundtrip(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "test.markov")
            save_markov_archive(mc, archive_path)
            assert Path(archive_path).exists()

            loaded = load_markov_archive(archive_path)
            assert loaded["config"]["model_type"] == "markov_chain"
            assert loaded["config"]["order"] == 1
            assert loaded["vocab"].size == mc.vocab.size

            rt = loaded["runtime"]
            token = rt.argmax(["a"])
            assert token in mc.vocab.tok2id

    def test_save_creates_parent_dirs(self) -> None:
        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "sub" / "dir" / "test.markov")
            save_markov_archive(mc, archive_path)
            assert Path(archive_path).exists()


class TestHMMArchive:
    def test_save_and_load_roundtrip(self) -> None:
        hmm = _trained_hmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "hmm.markov")
            save_markov_archive(hmm, archive_path)

            loaded = load_markov_archive(archive_path)
            assert loaded["config"]["model_type"] == "hmm"
            assert loaded["config"]["n_states"] == hmm.n_states

            rt = loaded["runtime"]
            states = rt.decode(["a", "b", "c"])
            assert len(states) == 3


class TestMarkovArchiveBackoff:
    def test_backoff_chain_round_trip(self) -> None:
        """Archives saved from a backoff MarkovChain include chain.json and restore _lower."""
        import zipfile

        vocab = Vocabulary()
        seqs = [["a", "b", "c", "a", "b"]] * 20
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5, backoff=True)
        mc.fit(seqs)
        assert mc._lower is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "backoff.markov")
            save_markov_archive(mc, archive_path)

            # chain.json must be present in the ZIP
            with zipfile.ZipFile(archive_path, "r") as zf:
                assert "chain.json" in zf.namelist()

            loaded = load_markov_archive(archive_path)
            assert "chain" in loaded
            restored: MarkovChain = loaded["chain"]
            assert restored.order == mc.order
            assert restored.backoff is True
            assert restored._lower is not None
            assert restored._lower.order == 1

    def test_no_backoff_archive_has_chain_json(self) -> None:
        """Non-backoff archives also contain chain.json for predict_probs round-trip."""
        import zipfile

        mc = _trained_markov()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "simple.markov")
            save_markov_archive(mc, archive_path)

            with zipfile.ZipFile(archive_path, "r") as zf:
                assert "chain.json" in zf.namelist()

            loaded = load_markov_archive(archive_path)
            assert "chain" in loaded
            chain = loaded["chain"]
            probs = chain.predict_probs(["a"])
            assert probs.sum() > 0.99


class TestArchiveErrors:
    def test_unsupported_model_type(self) -> None:
        try:
            save_markov_archive("not_a_model", "/tmp/bad.markov")
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_unknown_model_type_in_archive(self) -> None:
        """Archive with unknown model_type should raise ValueError."""
        import json
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "bad.markov")
            # Create a fake archive with unknown model_type
            mc = _trained_markov()
            real_archive = str(Path(tmpdir) / "real.markov")
            save_markov_archive(mc, real_archive)

            # Tamper with config.json
            with zipfile.ZipFile(real_archive, "r") as zf:
                zf.extractall(tmpdir + "/extracted")
            config_path = str(Path(tmpdir) / "extracted" / "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            config["model_type"] = "unknown_type"
            with open(config_path, "w") as f:
                json.dump(config, f)
            with zipfile.ZipFile(archive_path, "w") as zf:
                for name in ["model.onnx", "vocab.json", "config.json"]:
                    zf.write(str(Path(tmpdir) / "extracted" / name), name)

            try:
                load_markov_archive(archive_path)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass
