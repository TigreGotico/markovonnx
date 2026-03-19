"""Tests for markovonnx.cli."""

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
