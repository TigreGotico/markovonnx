"""Tests for markovonnx.config."""

import os
from pathlib import Path

from markovonnx.config import MarkovConfig, _env


class TestEnvHelper:
    def test_returns_default_when_unset(self) -> None:
        key = "MARKOV_TEST_NONEXISTENT_KEY_12345"
        assert _env(key, "fallback") == "fallback"

    def test_returns_cast_value_when_set(self) -> None:
        os.environ["MARKOV_TEST_KEY"] = "42"
        try:
            result = _env("MARKOV_TEST_KEY", 0, int)
            assert result == 42
        finally:
            os.environ.pop("MARKOV_TEST_KEY")

    def test_returns_default_when_empty(self) -> None:
        os.environ["MARKOV_TEST_KEY"] = ""
        try:
            result = _env("MARKOV_TEST_KEY", "default")
            assert result == "default"
        finally:
            os.environ.pop("MARKOV_TEST_KEY")


class TestMarkovConfig:
    def test_defaults(self) -> None:
        cfg = MarkovConfig()
        assert cfg.data_path == ""
        assert cfg.data_mode == "char"
        assert cfg.order == 2
        assert cfg.smoothing == 1e-5
        assert cfg.quantize is True

    def test_from_env(self) -> None:
        os.environ["MARKOV_DATA_MODE"] = "word"
        os.environ["MARKOV_ORDER"] = "5"
        os.environ["MARKOV_SMOOTHING"] = "0.01"
        os.environ["MARKOV_QUANTIZE"] = "false"
        try:
            cfg = MarkovConfig.from_env()
            assert cfg.data_mode == "word"
            assert cfg.order == 5
            assert cfg.smoothing == 0.01
            assert cfg.quantize is False
        finally:
            for key in ["MARKOV_DATA_MODE", "MARKOV_ORDER", "MARKOV_SMOOTHING", "MARKOV_QUANTIZE"]:
                os.environ.pop(key, None)

    def test_from_env_defaults(self) -> None:
        # Ensure no MARKOV_ vars interfere
        keys = [
            "MARKOV_DATA_PATH", "MARKOV_DATA_MODE", "MARKOV_ORDER",
            "MARKOV_HMM_STATES", "MARKOV_SMOOTHING", "MARKOV_MAX_VOCAB",
            "MARKOV_MAX_LINES", "MARKOV_STREAM_MB", "MARKOV_ONNX_PATH",
            "MARKOV_QUANTIZE", "MARKOV_QUANT_PATH", "MARKOV_TEMP",
            "MARKOV_GEN_LEN", "MARKOV_SEED", "MARKOV_OUTDIR",
        ]
        saved = {}
        for k in keys:
            if k in os.environ:
                saved[k] = os.environ.pop(k)
        try:
            cfg = MarkovConfig.from_env()
            assert cfg.data_mode == "char"
            assert cfg.order == 2
        finally:
            for k, v in saved.items():
                os.environ[k] = v

    def test_outdir_path(self) -> None:
        cfg = MarkovConfig(outdir="/tmp/test_dir")
        assert cfg.outdir_path == Path("/tmp/test_dir")

    def test_resolved_onnx_path(self) -> None:
        cfg = MarkovConfig(outdir="/tmp/out", onnx_path="model.onnx")
        assert cfg.resolved_onnx_path == "/tmp/out/model.onnx"

    def test_resolved_quant_path(self) -> None:
        cfg = MarkovConfig(outdir="/tmp/out", quant_path="q.onnx")
        assert cfg.resolved_quant_path == "/tmp/out/q.onnx"
