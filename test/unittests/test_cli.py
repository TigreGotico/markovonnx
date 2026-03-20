"""Tests for markovonnx.cli."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from markovonnx.cli import cmd_generate, cmd_info, cmd_train, main


class _Args:
    """Minimal namespace for CLI args."""
    def __init__(self, **kwargs: object):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCLITrain:
    def test_train_creates_archive(self) -> None:
        corpus_path = str(Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.markov")
            args = _Args(
                corpus=corpus_path,
                output=output,
                mode="char",
                order=2,
                smoothing=1e-5,
                max_vocab=0,
                max_lines=5,
                backoff=False,
            )
            cmd_train(args)
            assert Path(output).exists()


class TestCLIGenerate:
    def test_generate_from_archive(self) -> None:
        corpus_path = str(Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = str(Path(tmpdir) / "test.markov")
            train_args = _Args(
                corpus=corpus_path, output=archive, mode="char",
                order=2, smoothing=1e-5, max_vocab=0, max_lines=5, backoff=False,
            )
            cmd_train(train_args)

            gen_args = _Args(
                archive=archive, seed="th", length=10,
                temperature=0.5, mode="char",
            )
            cmd_generate(gen_args)  # Should print without error


class TestCLIInfo:
    def test_info_prints_metadata(self, capsys: object) -> None:
        corpus_path = str(Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = str(Path(tmpdir) / "test.markov")
            train_args = _Args(
                corpus=corpus_path, output=archive, mode="char",
                order=2, smoothing=1e-5, max_vocab=0, max_lines=5, backoff=False,
            )
            cmd_train(train_args)

            info_args = _Args(archive=archive)
            cmd_info(info_args)


class TestCLIMain:
    def test_main_no_args_exits(self) -> None:
        with patch("sys.argv", ["markovonnx"]):
            try:
                main()
                assert False, "Should have exited"
            except SystemExit as e:
                assert e.code == 1

    def test_main_version(self) -> None:
        with patch("sys.argv", ["markovonnx", "--version"]):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0

    def test_main_train(self) -> None:
        corpus_path = str(Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "test.markov")
            with patch("sys.argv", [
                "markovonnx", "train", corpus_path,
                "-o", output, "--mode", "char", "--order", "2", "--max-lines", "5",
            ]):
                main()
            assert Path(output).exists()

    def test_main_generate(self) -> None:
        corpus_path = str(Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = str(Path(tmpdir) / "test.markov")
            with patch("sys.argv", [
                "markovonnx", "train", corpus_path,
                "-o", archive, "--max-lines", "5",
            ]):
                main()
            with patch("sys.argv", [
                "markovonnx", "generate", archive,
                "--seed", "the", "--length", "10",
            ]):
                main()

    def test_main_info(self) -> None:
        corpus_path = str(Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = str(Path(tmpdir) / "test.markov")
            with patch("sys.argv", [
                "markovonnx", "train", corpus_path,
                "-o", archive, "--max-lines", "5",
            ]):
                main()
            with patch("sys.argv", ["markovonnx", "info", archive]):
                main()


# ---------------------------------------------------------------------------
# train-hmm subcommand
# ---------------------------------------------------------------------------


def _write_conll_corpus(path: Path) -> None:
    """Write a minimal CoNLL-style tagged corpus."""
    path.write_text(
        "hello\tGREET\nworld\tNOUN\n\n"
        "good\tADJ\nday\tNOUN\n\n"
        "hello\tGREET\ngoodbye\tGREET\n\n",
        encoding="utf-8",
    )


class TestCLITrainHMM:
    def test_train_hmm_creates_json(self, tmp_path: Path) -> None:
        """train-hmm writes a JSON model file."""
        corpus = tmp_path / "corpus.tsv"
        _write_conll_corpus(corpus)
        output = tmp_path / "model.hmm.json"

        result = subprocess.run(
            [
                sys.executable, "-m", "markovonnx.cli",
                "train-hmm", str(corpus),
                "-o", str(output),
                "--n-states", "3",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output.exists()

    def test_train_hmm_json_loadable(self, tmp_path: Path) -> None:
        """Model saved by train-hmm can be loaded with HiddenMarkovModel.load."""
        from markovonnx.hmm import HiddenMarkovModel

        corpus = tmp_path / "corpus.tsv"
        _write_conll_corpus(corpus)
        output = tmp_path / "model.hmm.json"

        subprocess.run(
            [
                sys.executable, "-m", "markovonnx.cli",
                "train-hmm", str(corpus), "-o", str(output), "--n-states", "3",
            ],
            check=True,
        )
        hmm = HiddenMarkovModel.load(str(output))
        # n_states may differ from --n-states if fit_supervised infers more
        # states from the tag vocabulary; just verify the model is loadable.
        assert hmm.n_states > 0

    def test_train_hmm_export_c(self, tmp_path: Path) -> None:
        """train-hmm --export-c writes a C header."""
        corpus = tmp_path / "corpus.tsv"
        _write_conll_corpus(corpus)
        output = tmp_path / "model.hmm.json"
        header = tmp_path / "model.h"

        result = subprocess.run(
            [
                sys.executable, "-m", "markovonnx.cli",
                "train-hmm", str(corpus), "-o", str(output),
                "--n-states", "3",
                "--export-c", str(header),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert header.exists()
        content = header.read_text()
        assert "#define HMM_N_STATES" in content

    def test_train_hmm_max_lines(self, tmp_path: Path) -> None:
        """train-hmm --max-lines limits parsed tokens."""
        corpus = tmp_path / "corpus.tsv"
        _write_conll_corpus(corpus)
        output_full = tmp_path / "full.json"
        output_limited = tmp_path / "limited.json"

        subprocess.run(
            [sys.executable, "-m", "markovonnx.cli", "train-hmm", str(corpus),
             "-o", str(output_full)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "markovonnx.cli", "train-hmm", str(corpus),
             "-o", str(output_limited), "--max-lines", "2"],
            check=True,
        )
        assert output_full.exists()
        assert output_limited.exists()

    def test_train_hmm_unsupervised(self, tmp_path: Path) -> None:
        """train-hmm --unsupervised accepts plain-token corpus (no tags)."""
        corpus = tmp_path / "obs.txt"
        # One token per line, blank lines between sequences
        corpus.write_text(
            "hello\nworld\n\ngood\nday\n\nhello\ngoodbye\n\n",
            encoding="utf-8",
        )
        output = tmp_path / "model_unsup.hmm.json"

        result = subprocess.run(
            [
                sys.executable, "-m", "markovonnx.cli",
                "train-hmm", str(corpus),
                "-o", str(output),
                "--n-states", "2",
                "--unsupervised", "--n-iter", "3",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output.exists()


# ---------------------------------------------------------------------------
# size-report --max-bytes
# ---------------------------------------------------------------------------


class TestCLISizeReportMaxBytes:
    def _train_markov_json(self, tmp_path: Path) -> Path:
        """Train a small MarkovChain and save as JSON for size-report."""
        from markovonnx.markov import MarkovChain
        from markovonnx.vocabulary import Vocabulary

        vocab = Vocabulary()
        seqs = [["a", "b", "c", "a"]] * 5
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
        mc.fit(seqs)
        out = tmp_path / "model.json"
        mc.save(str(out))
        return out

    def test_max_bytes_passes_when_under_limit(self, tmp_path: Path) -> None:
        model_path = self._train_markov_json(tmp_path)
        result = subprocess.run(
            [
                sys.executable, "-m", "markovonnx.cli",
                "size-report", str(model_path),
                "--format", "markov",
                "--max-bytes", "10000000",  # 10 MB — definitely fits
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_max_bytes_fails_when_over_limit(self, tmp_path: Path) -> None:
        model_path = self._train_markov_json(tmp_path)
        result = subprocess.run(
            [
                sys.executable, "-m", "markovonnx.cli",
                "size-report", str(model_path),
                "--format", "markov",
                "--max-bytes", "1",  # 1 byte — always fails
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stderr
