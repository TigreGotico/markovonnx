"""Unit tests for markovonnx.size_report — ESP32 memory size estimation."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.size_report import (
    ESP32_FLASH_BYTES,
    ESP32_IRAM_BYTES,
    format_hmm_report,
    format_markov_report,
    hmm_c_sizes,
    markov_c_sizes,
)
from markovonnx.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chain(order: int = 1, backoff: bool = False) -> MarkovChain:
    """Build a small trained MarkovChain."""
    corpus: List[List[str]] = [list("abcabc"), list("bbbccc"), list("abcbc")]
    vocab = Vocabulary(max_vocab=0)
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5, backoff=backoff)
    mc.fit(corpus)
    return mc


def _make_hmm() -> HiddenMarkovModel:
    """Build a small trained HMM."""
    obs_seqs = [["a", "b", "c"], ["b", "c", "a"], ["a", "a", "b"]]
    tag_seqs = [["X", "Y", "Z"], ["Y", "Z", "X"], ["X", "X", "Y"]]
    obs_vocab = Vocabulary(max_vocab=0)
    obs_vocab.build_from_sequences(obs_seqs)
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab, smoothing=1e-5)
    hmm.fit_supervised(obs_seqs, tag_seqs)
    return hmm


# ---------------------------------------------------------------------------
# markov_c_sizes
# ---------------------------------------------------------------------------


def test_markov_sizes_positive() -> None:
    """All size values are positive integers."""
    mc = _make_chain()
    sizes = markov_c_sizes(mc)
    for key, val in sizes.items():
        if key in {"n_sparse_rows", "vocab_size", "quantized"}:
            continue
        assert isinstance(val, int), f"{key} is not int: {type(val)}"
        assert val > 0, f"{key} is not positive: {val}"


def test_markov_sizes_total_equals_sum() -> None:
    """total equals the sum of all array entries."""
    mc = _make_chain()
    sizes = markov_c_sizes(mc)
    meta = {"total", "n_sparse_rows", "vocab_size", "quantized"}
    expected = sum(v for k, v in sizes.items() if k not in meta)
    assert sizes["total"] == expected


def test_markov_sizes_quantize_smaller_than_float() -> None:
    """uint8 sizes are smaller than float32 sizes for MARKOV_PROBS."""
    mc = _make_chain()
    q_sizes = markov_c_sizes(mc, quantize=True)
    f_sizes = markov_c_sizes(mc, quantize=False)
    assert q_sizes["MARKOV_PROBS"] < f_sizes["MARKOV_PROBS"]
    assert q_sizes["total"] < f_sizes["total"]


def test_markov_sizes_backoff_includes_lower_level() -> None:
    """Backoff chain includes keys for the lower-order level."""
    mc = _make_chain(order=2, backoff=True)
    sizes = markov_c_sizes(mc)
    # Should have MARKOV_KEYS_1 and MARKOV_PROBS_1
    assert "MARKOV_KEYS_1" in sizes
    assert "MARKOV_PROBS_1" in sizes
    assert sizes["MARKOV_KEYS_1"] > 0
    assert sizes["MARKOV_PROBS_1"] > 0


def test_markov_sizes_metadata() -> None:
    """Metadata fields are correctly populated."""
    mc = _make_chain()
    sizes = markov_c_sizes(mc, quantize=True)
    assert sizes["vocab_size"] == mc.vocab.size
    assert sizes["n_sparse_rows"] == len(mc._counts)
    assert sizes["quantized"] == 1


# ---------------------------------------------------------------------------
# hmm_c_sizes
# ---------------------------------------------------------------------------


def test_hmm_sizes_positive() -> None:
    """All HMM size values are positive integers."""
    hmm = _make_hmm()
    sizes = hmm_c_sizes(hmm)
    for key, val in sizes.items():
        if key in {"n_states", "obs_vocab_size"}:
            continue
        assert isinstance(val, int), f"{key} is not int"
        assert val > 0, f"{key} is not positive"


def test_hmm_sizes_total_equals_sum() -> None:
    """HMM total equals sum of all array entries."""
    hmm = _make_hmm()
    sizes = hmm_c_sizes(hmm)
    meta = {"total", "n_states", "obs_vocab_size"}
    expected = sum(v for k, v in sizes.items() if k not in meta)
    assert sizes["total"] == expected


def test_hmm_sizes_contains_all_arrays() -> None:
    """hmm_c_sizes returns entries for all 6 matrix arrays."""
    hmm = _make_hmm()
    sizes = hmm_c_sizes(hmm)
    for key in ["obs_vocab", "state_vocab", "HMM_LOG_PI", "HMM_LOG_A", "HMM_LOG_B",
                "HMM_PI", "HMM_A", "HMM_B"]:
        assert key in sizes, f"Missing key: {key}"


def test_hmm_sizes_log_equals_linear() -> None:
    """Log and linear arrays have the same byte size (both float32)."""
    hmm = _make_hmm()
    sizes = hmm_c_sizes(hmm)
    assert sizes["HMM_LOG_PI"] == sizes["HMM_PI"]
    assert sizes["HMM_LOG_A"] == sizes["HMM_A"]
    assert sizes["HMM_LOG_B"] == sizes["HMM_B"]


def test_hmm_sizes_metadata() -> None:
    """Metadata n_states and obs_vocab_size are correct."""
    hmm = _make_hmm()
    sizes = hmm_c_sizes(hmm)
    assert sizes["n_states"] == hmm.n_states
    assert sizes["obs_vocab_size"] == hmm.obs_vocab.size


# ---------------------------------------------------------------------------
# ESP32 fit logic
# ---------------------------------------------------------------------------


def test_fit_logic_small_model_fits_flash() -> None:
    """A small model should fit in ESP32 flash."""
    mc = _make_chain()
    sizes = markov_c_sizes(mc)
    assert sizes["total"] < ESP32_FLASH_BYTES


def test_fit_logic_constants_sane() -> None:
    """ESP32 constants are positive and flash > IRAM."""
    assert ESP32_FLASH_BYTES > 0
    assert ESP32_IRAM_BYTES > 0
    assert ESP32_FLASH_BYTES > ESP32_IRAM_BYTES


# ---------------------------------------------------------------------------
# format_markov_report
# ---------------------------------------------------------------------------


def test_format_markov_report_returns_string() -> None:
    """format_markov_report returns a non-empty string."""
    mc = _make_chain()
    report = format_markov_report(mc)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_markov_report_contains_esp32_check() -> None:
    """Report contains Flash fit check."""
    mc = _make_chain()
    report = format_markov_report(mc)
    assert "Flash" in report
    assert ("FITS" in report or "TOO LARGE" in report)


def test_format_markov_report_progmem_hides_iram() -> None:
    """progmem=True omits IRAM line."""
    mc = _make_chain()
    report_progmem = format_markov_report(mc, progmem=True)
    report_no_progmem = format_markov_report(mc, progmem=False)
    assert "IRAM" not in report_progmem
    assert "IRAM" in report_no_progmem


def test_format_markov_report_contains_total() -> None:
    """Report contains TOTAL row."""
    mc = _make_chain()
    report = format_markov_report(mc)
    assert "TOTAL" in report


# ---------------------------------------------------------------------------
# format_hmm_report
# ---------------------------------------------------------------------------


def test_format_hmm_report_returns_string() -> None:
    """format_hmm_report returns a non-empty string."""
    hmm = _make_hmm()
    report = format_hmm_report(hmm)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_hmm_report_contains_esp32_check() -> None:
    """HMM report contains Flash fit check."""
    hmm = _make_hmm()
    report = format_hmm_report(hmm)
    assert "Flash" in report
    assert ("FITS" in report or "TOO LARGE" in report)


def test_format_hmm_report_contains_total() -> None:
    """HMM report contains TOTAL row."""
    hmm = _make_hmm()
    report = format_hmm_report(hmm)
    assert "TOTAL" in report


# ---------------------------------------------------------------------------
# CLI size-report subcommand
# ---------------------------------------------------------------------------


def test_cli_size_report_markov(tmp_path: Path) -> None:
    """markovonnx size-report --format markov prints a report."""
    mc = _make_chain(order=1)
    json_path = str(tmp_path / "model.json")
    mc.save(json_path)

    result = subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "size-report", json_path, "--format", "markov"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Flash" in result.stdout
    assert "TOTAL" in result.stdout


def test_cli_size_report_hmm(tmp_path: Path) -> None:
    """markovonnx size-report --format hmm prints a report."""
    hmm = _make_hmm()
    json_path = str(tmp_path / "hmm.json")
    hmm.save(json_path)

    result = subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "size-report", json_path, "--format", "hmm"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Flash" in result.stdout
    assert "TOTAL" in result.stdout


def test_cli_size_report_no_quantize(tmp_path: Path) -> None:
    """size-report --no-quantize uses float32 sizes."""
    mc = _make_chain(order=1)
    json_path = str(tmp_path / "model.json")
    mc.save(json_path)

    result_q = subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "size-report", json_path, "--format", "markov"],
        capture_output=True, text=True,
    )
    result_f = subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "size-report", json_path,
         "--format", "markov", "--no-quantize"],
        capture_output=True, text=True,
    )
    assert result_q.returncode == 0
    assert result_f.returncode == 0
    # Float32 report total should be larger — check it's a different output
    assert result_q.stdout != result_f.stdout
