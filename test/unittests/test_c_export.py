"""Unit tests for markovonnx.c_export — C header generation for ESP32."""

import os
import re
import subprocess
import sys
import tempfile
from typing import List

import numpy as np
import pytest

import numpy as np_mod  # aliased to avoid shadowing

from markovonnx.c_export import (
    _collect_rows,
    _ctx_ids_from_index,
    _fmt_float_row,
    _pack_key,
    _render_hmm_header,
    export_hmm_c_header,
    export_markov_c_header,
)
from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chain(order: int = 1, corpus: List[List[str]] = None) -> MarkovChain:
    """Build a small trained MarkovChain for testing."""
    if corpus is None:
        corpus = [list("abcabc"), list("bbbccc"), list("abcbc")]
    vocab = Vocabulary(max_vocab=0)
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)
    return mc


# ---------------------------------------------------------------------------
# _pack_key / _ctx_ids_from_index round-trip
# ---------------------------------------------------------------------------


def test_pack_key_order1() -> None:
    """Single token context packs into low 16 bits."""
    assert _pack_key([42], 1) == 42


def test_pack_key_order2() -> None:
    """Two-token context packs into two 16-bit slots."""
    key = _pack_key([1, 2], 2)
    assert key == (1 << 16) | 2


def test_pack_key_order3() -> None:
    """Three-token context is reversible."""
    ids = [3, 7, 11]
    key = _pack_key(ids, 3)
    assert key == (3 << 32) | (7 << 16) | 11


def test_ctx_ids_round_trip_order1() -> None:
    """Base-V decode matches pack_key for order=1."""
    mc = _make_chain(order=1)
    V = mc.vocab.size
    for ci in list(mc._counts.keys())[:5]:
        ids = _ctx_ids_from_index(ci, 1, V)
        key = _pack_key(ids, 1)
        # key should equal ci for order=1 (single token, base-V == ID)
        assert key == ci % (2 ** 16)


def test_ctx_ids_round_trip_order2() -> None:
    """Base-V decode is self-consistent for order=2."""
    mc = _make_chain(order=2)
    V = mc.vocab.size
    for ci in list(mc._counts.keys())[:5]:
        ids = _ctx_ids_from_index(ci, 2, V)
        assert len(ids) == 2
        assert all(0 <= i < V for i in ids)


# ---------------------------------------------------------------------------
# _collect_rows
# ---------------------------------------------------------------------------


def test_collect_rows_sorted() -> None:
    """Rows are sorted by packed key."""
    mc = _make_chain(order=1)
    rows = _collect_rows(mc, mc.vocab.size, quantize=True)
    keys = [k for k, _ in rows]
    assert keys == sorted(keys)


def test_collect_rows_uint8_dtype() -> None:
    """quantize=True produces uint8 rows."""
    mc = _make_chain(order=1)
    rows = _collect_rows(mc, mc.vocab.size, quantize=True)
    for _, row in rows:
        assert row.dtype == np.uint8


def test_collect_rows_float_dtype() -> None:
    """quantize=False produces float32 rows."""
    mc = _make_chain(order=1)
    rows = _collect_rows(mc, mc.vocab.size, quantize=False)
    for _, row in rows:
        assert row.dtype == np.float32


def test_collect_rows_probs_sum() -> None:
    """Float32 rows sum to approximately 1.0."""
    mc = _make_chain(order=1)
    rows = _collect_rows(mc, mc.vocab.size, quantize=False)
    for _, row in rows:
        assert abs(float(row.sum()) - 1.0) < 1e-5


def test_collect_rows_count() -> None:
    """Number of rows equals number of observed contexts."""
    mc = _make_chain(order=1)
    rows = _collect_rows(mc, mc.vocab.size, quantize=True)
    assert len(rows) == len(mc._counts)


def test_collect_rows_no_all_zero_uint8_rows() -> None:
    """Normalization guard: no uint8 row should sum to zero after quantization."""
    mc = _make_chain(order=1)
    rows = _collect_rows(mc, mc.vocab.size, quantize=True)
    for _, row in rows:
        assert row.sum() > 0, "All-zero quantized row violates normalization guard"


def test_collect_rows_normalization_guard_forced() -> None:
    """Guard sets argmax token to 1 when all entries round to 0."""
    # Build a chain where one row has near-zero probabilities
    mc = _make_chain(order=1)
    V = mc.vocab.size
    # Manually inject a counts entry where all probs are tiny so quantization
    # would otherwise round everything to 0.
    ci = next(iter(mc._counts))
    mc._counts[ci] = np.full(V, 1e-10)
    rows = _collect_rows(mc, V, quantize=True)
    for _, row in rows:
        assert row.sum() > 0


# ---------------------------------------------------------------------------
# export_markov_c_header — file generation
# ---------------------------------------------------------------------------


def test_export_creates_file() -> None:
    """export_markov_c_header writes a file at the given path."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_export_contains_required_symbols() -> None:
    """Generated header defines the mandatory macros and arrays."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False, mode="w") as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        assert "#define MARKOV_VOCAB_SIZE" in content
        assert "#define MARKOV_ORDER" in content
        assert "#define MARKOV_SPARSE_ROWS" in content
        assert "MARKOV_VOCAB[" in content
        assert "MARKOV_KEYS[" in content
        assert "MARKOV_PROBS[" in content
        assert "markov_lookup(" in content
        assert "markov_sample(" in content
    finally:
        os.unlink(path)


def test_export_pragma_once() -> None:
    """Header includes #pragma once guard."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        assert "#pragma once" in content
    finally:
        os.unlink(path)


def test_export_quantize_uint8_type() -> None:
    """Quantized header uses uint8_t for MARKOV_PROBS."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path, quantize=True)
        content = open(path).read()
        assert "uint8_t" in content
        assert "MARKOV_PROBS" in content
    finally:
        os.unlink(path)


def test_export_no_quantize_float_type() -> None:
    """Non-quantized header uses float for MARKOV_PROBS."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path, quantize=False)
        content = open(path).read()
        assert "float" in content
        # Must NOT have uint8_t in probs array line
        lines = [l for l in content.splitlines() if "MARKOV_PROBS" in l]
        assert any("float" in l for l in lines)
    finally:
        os.unlink(path)


def test_export_progmem_attribute() -> None:
    """progmem=True inserts .rodata section attribute."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path, progmem=True)
        content = open(path).read()
        assert '.rodata' in content
    finally:
        os.unlink(path)


def test_export_progmem_false_no_attribute() -> None:
    """progmem=False (default) has no section annotation."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path, progmem=False)
        content = open(path).read()
        assert '.rodata' not in content
    finally:
        os.unlink(path)


def test_export_vocab_size_macro_correct() -> None:
    """MARKOV_VOCAB_SIZE macro matches vocabulary size."""
    mc = _make_chain()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        m = re.search(r"#define MARKOV_VOCAB_SIZE\s+(\d+)", content)
        assert m is not None
        assert int(m.group(1)) == mc.vocab.size
    finally:
        os.unlink(path)


def test_export_order_macro_correct() -> None:
    """MARKOV_ORDER macro matches chain order."""
    mc = _make_chain(order=2)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        m = re.search(r"#define MARKOV_ORDER\s+(\d+)", content)
        assert m is not None
        assert int(m.group(1)) == 2
    finally:
        os.unlink(path)


def test_export_sparse_rows_macro_correct() -> None:
    """MARKOV_SPARSE_ROWS macro matches number of observed contexts."""
    mc = _make_chain(order=1)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        m = re.search(r"#define MARKOV_SPARSE_ROWS\s+(\d+)", content)
        assert m is not None
        assert int(m.group(1)) == len(mc._counts)
    finally:
        os.unlink(path)


def test_export_vocab_too_large_raises() -> None:
    """Vocabularies over 65535 tokens raise ValueError."""
    mc = _make_chain()
    # Monkey-patch vocab size to exceed limit
    mc.vocab.id2tok = ["x"] * 70000
    mc.vocab.tok2id = {f"x{i}": i for i in range(70000)}
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="65535"):
            export_markov_c_header(mc, path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# C syntax validity via gcc (skipped when gcc unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
    reason="gcc not available",
)
def test_export_valid_c_syntax_quantized() -> None:
    """Generated quantized header passes gcc -fsyntax-only."""
    mc = _make_chain(order=1)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path, quantize=True)
        # Wrap in a minimal translation unit
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
            cf.write(f'#include "{path}"\nint main(void) {{ return 0; }}\n')
            c_path = cf.name
        result = subprocess.run(
            ["gcc", "-std=c99", "-fsyntax-only", c_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)
        if os.path.exists(c_path):
            os.unlink(c_path)


@pytest.mark.skipif(
    subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
    reason="gcc not available",
)
def test_export_valid_c_syntax_float() -> None:
    """Generated float32 header passes gcc -fsyntax-only."""
    mc = _make_chain(order=1)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    c_path = ""
    try:
        export_markov_c_header(mc, path, quantize=False)
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
            cf.write(f'#include "{path}"\nint main(void) {{ return 0; }}\n')
            c_path = cf.name
        result = subprocess.run(
            ["gcc", "-std=c99", "-fsyntax-only", c_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)
        if c_path and os.path.exists(c_path):
            os.unlink(c_path)


# ---------------------------------------------------------------------------
# Lookup correctness — compare Python prediction to C header data
# ---------------------------------------------------------------------------


def _python_lookup(
    mc: MarkovChain, context: List[str]
) -> np.ndarray:
    """Return normalised float32 probability row from chain._counts."""
    V = mc.vocab.size
    ids = mc.vocab.encode(context[-mc.order:])
    ci = mc._ctx_idx(ids)
    counts = mc._counts.get(ci, None)
    if counts is None:
        return np.full(V, 1.0 / V, dtype=np.float32)
    return (counts + mc.smoothing) / (counts.sum() + mc.smoothing * V)


def _c_header_lookup(
    pairs: list, key: int, V: int, quantize: bool
) -> np.ndarray:
    """Simulate the markov_lookup binary search and return probability row."""
    lo, hi = 0, len(pairs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        k, row = pairs[mid]
        if k == key:
            if quantize:
                return row.astype(np.float32) / 255.0
            return row
        elif k < key:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def test_lookup_matches_python_probs_order1() -> None:
    """C header lookup returns same probabilities as Python chain for observed contexts."""
    mc = _make_chain(order=1)
    V = mc.vocab.size
    pairs = _collect_rows(mc, V, quantize=False)

    for ctx_id in range(V):
        tok = mc.vocab.id2tok[ctx_id]
        key = _pack_key([ctx_id], 1)
        c_probs = _c_header_lookup(pairs, key, V, quantize=False)
        if c_probs is None:
            continue
        py_probs = _python_lookup(mc, [tok])
        np.testing.assert_allclose(c_probs, py_probs, atol=1e-6)


def test_lookup_quantized_approx_probs() -> None:
    """uint8 quantised lookup is within 1/255 of float probabilities."""
    mc = _make_chain(order=1)
    V = mc.vocab.size
    pairs_q = _collect_rows(mc, V, quantize=True)
    pairs_f = _collect_rows(mc, V, quantize=False)

    for (key, _) in pairs_f:
        c_float = _c_header_lookup(pairs_f, key, V, quantize=False)
        c_quant = _c_header_lookup(pairs_q, key, V, quantize=True)
        if c_float is None or c_quant is None:
            continue
        np.testing.assert_allclose(c_quant, c_float, atol=1.0 / 255.0 + 1e-6)


def test_unseen_context_returns_none() -> None:
    """Binary search returns None for a key not in MARKOV_KEYS."""
    mc = _make_chain(order=1)
    V = mc.vocab.size
    pairs = _collect_rows(mc, V, quantize=True)
    # Use a key that is certainly not in the table (large value)
    missing_key = 0xFFFFFFFFFFFFFFFF
    result = _c_header_lookup(pairs, missing_key, V, quantize=True)
    assert result is None


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_train_export_c(tmp_path) -> None:
    """markovonnx train --export-c writes a valid C header file."""
    import subprocess

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\nbbbccc\nabcbc\n")
    archive = tmp_path / "model.markov"
    header = tmp_path / "model.h"

    result = subprocess.run(
        [
            sys.executable, "-m", "markovonnx.cli",
            "train", str(corpus),
            "-o", str(archive),
            "--mode", "char",
            "--order", "1",
            "--export-c", str(header),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert header.exists()
    content = header.read_text()
    assert "#define MARKOV_VOCAB_SIZE" in content


# ---------------------------------------------------------------------------
# HMM C header export
# ---------------------------------------------------------------------------


def _make_hmm() -> HiddenMarkovModel:
    """Build a small trained HMM for testing."""
    obs_seqs = [["a", "b", "c"], ["b", "c", "a"], ["a", "a", "b"]]
    tag_seqs = [["X", "Y", "Z"], ["Y", "Z", "X"], ["X", "X", "Y"]]
    obs_vocab = Vocabulary(max_vocab=0)
    obs_vocab.build_from_sequences(obs_seqs)
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab, smoothing=1e-5)
    hmm.fit_supervised(obs_seqs, tag_seqs)
    return hmm


# --- file structure ---


def test_hmm_export_creates_file() -> None:
    """export_hmm_c_header writes a file at the given path."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_hmm_export_required_symbols() -> None:
    """Generated HMM header contains all mandatory macros and arrays."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        content = open(path).read()
        for sym in [
            "#pragma once",
            "#define HMM_N_STATES",
            "#define HMM_OBS_SIZE",
            "HMM_OBS_VOCAB",
            "HMM_STATE_VOCAB",
            "HMM_LOG_PI",
            "HMM_LOG_A",
            "HMM_LOG_B",
            "hmm_viterbi(",
        ]:
            assert sym in content, f"Missing: {sym}"
    finally:
        os.unlink(path)


def test_hmm_export_n_states_macro() -> None:
    """HMM_N_STATES macro matches hmm.n_states."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        content = open(path).read()
        m = re.search(r"#define HMM_N_STATES\s+(\d+)", content)
        assert m is not None
        assert int(m.group(1)) == hmm.n_states
    finally:
        os.unlink(path)


def test_hmm_export_obs_size_macro() -> None:
    """HMM_OBS_SIZE macro matches obs_vocab.size."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        content = open(path).read()
        m = re.search(r"#define HMM_OBS_SIZE\s+(\d+)", content)
        assert m is not None
        assert int(m.group(1)) == hmm.obs_vocab.size
    finally:
        os.unlink(path)


def test_hmm_export_progmem() -> None:
    """progmem=True inserts .rodata annotation."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path, progmem=True)
        content = open(path).read()
        assert ".rodata" in content
    finally:
        os.unlink(path)


def test_hmm_export_no_state_vocab_uses_numeric() -> None:
    """HMM without state_vocab exports numeric state names."""
    obs_seqs = [["a", "b", "c"]]
    obs_vocab = Vocabulary(max_vocab=0)
    obs_vocab.build_from_sequences(obs_seqs)
    hmm = HiddenMarkovModel(n_states=2, obs_vocab=obs_vocab)
    # No fit_supervised → state_vocab is None
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        content = open(path).read()
        # Should contain "0" and "1" as state labels
        assert '"0"' in content or '"1"' in content
    finally:
        os.unlink(path)


# --- C syntax validity ---


@pytest.mark.skipif(
    subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
    reason="gcc not available",
)
def test_hmm_export_valid_c_syntax() -> None:
    """Generated HMM header passes gcc -fsyntax-only."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    c_path = ""
    try:
        export_hmm_c_header(hmm, path)
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
            cf.write(f'#include "{path}"\nint main(void) {{ return 0; }}\n')
            c_path = cf.name
        result = subprocess.run(
            ["gcc", "-std=c99", "-fsyntax-only", c_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)
        if c_path and os.path.exists(c_path):
            os.unlink(c_path)


# --- Viterbi correctness ---


def _py_viterbi(hmm: HiddenMarkovModel, obs_seq: List[str]) -> List[str]:
    """Run Python Viterbi and return decoded state sequence."""
    return hmm.viterbi(obs_seq)


def _c_viterbi(hmm: HiddenMarkovModel, obs_seq: List[str]) -> List[int]:
    """Simulate the C Viterbi algorithm in Python using HMM log matrices."""
    import numpy as np
    obs_ids = hmm.obs_vocab.encode(obs_seq)
    T = len(obs_ids)
    S = hmm.n_states
    log_pi = np.log(np.asarray(hmm.pi, dtype=np.float64) + 1e-30)
    log_A = np.log(np.asarray(hmm.A, dtype=np.float64) + 1e-30)
    log_B = np.log(np.asarray(hmm.B, dtype=np.float64) + 1e-30)

    delta = np.full((T, S), -np.inf)
    psi = np.zeros((T, S), dtype=int)
    delta[0] = log_pi + log_B[:, obs_ids[0]]

    for t in range(1, T):
        for s in range(S):
            scores = delta[t - 1] + log_A[:, s]
            best_q = int(scores.argmax())
            delta[t, s] = scores[best_q] + log_B[s, obs_ids[t]]
            psi[t, s] = best_q

    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(psi[t, path[-1]])
    path.reverse()
    return path


def test_viterbi_c_matches_python() -> None:
    """C-simulated Viterbi produces same state path as Python HMM.viterbi()."""
    hmm = _make_hmm()
    for obs_seq in [["a", "b", "c"], ["b", "b", "a"], ["c", "a", "b"]]:
        py_labels = _py_viterbi(hmm, obs_seq)
        c_ids = _c_viterbi(hmm, obs_seq)
        c_labels = hmm.state_vocab.decode(c_ids) if hmm.state_vocab else [str(i) for i in c_ids]
        assert py_labels == c_labels, f"obs={obs_seq}: python={py_labels} c={c_labels}"


def test_viterbi_single_token() -> None:
    """Viterbi on a single-token sequence returns the argmax of log_pi + log_B[:,obs]."""
    import numpy as np
    hmm = _make_hmm()
    obs_seq = ["a"]
    py_labels = _py_viterbi(hmm, obs_seq)
    c_ids = _c_viterbi(hmm, obs_seq)
    c_labels = hmm.state_vocab.decode(c_ids) if hmm.state_vocab else [str(i) for i in c_ids]
    assert py_labels == c_labels


def test_fmt_float_row_format() -> None:
    """_fmt_float_row produces valid C float literals."""
    import numpy as np
    row = np.array([0.1, 0.5, 0.4], dtype=np.float32)
    result = _fmt_float_row(row)
    parts = result.split(", ")
    assert len(parts) == 3
    assert all(p.endswith("f") for p in parts)


# ---------------------------------------------------------------------------
# C export — backoff chain
# ---------------------------------------------------------------------------


def _make_chain_backoff(order: int = 2) -> MarkovChain:
    """Build a trained MarkovChain with backoff enabled."""
    corpus = [list("abcabc"), list("bbbccc"), list("abcbc")]
    vocab = Vocabulary(max_vocab=0)
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5, backoff=True)
    mc.fit(corpus)
    return mc


def test_backoff_export_contains_lower_arrays() -> None:
    """Chain with backoff exports MARKOV_KEYS_LOWER and MARKOV_PROBS_LOWER."""
    mc = _make_chain_backoff(order=2)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        assert "MARKOV_KEYS_LOWER" in content
        assert "MARKOV_PROBS_LOWER" in content
        assert "MARKOV_SPARSE_ROWS_LOWER" in content
        assert "markov_lookup_lower(" in content
        assert "markov_sample_backoff(" in content
    finally:
        os.unlink(path)


def test_no_backoff_export_no_lower_arrays() -> None:
    """Chain without backoff does not emit lower-order arrays."""
    mc = _make_chain(order=2)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        assert "MARKOV_KEYS_LOWER" not in content
        assert "markov_sample_backoff" not in content
    finally:
        os.unlink(path)


def test_backoff_header_comment_flags_backoff_true() -> None:
    """Header comment line says backoff=true when backoff chain present."""
    mc = _make_chain_backoff(order=2)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        assert "backoff=true" in content
    finally:
        os.unlink(path)


@pytest.mark.skipif(
    subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
    reason="gcc not available",
)
def test_backoff_header_valid_c_syntax() -> None:
    """Backoff C header passes gcc -fsyntax-only."""
    mc = _make_chain_backoff(order=2)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    c_path = ""
    try:
        export_markov_c_header(mc, path)
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
            cf.write(f'#include "{path}"\nint main(void) {{ return 0; }}\n')
            c_path = cf.name
        result = subprocess.run(
            ["gcc", "-std=c99", "-fsyntax-only", c_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)
        if c_path and os.path.exists(c_path):
            os.unlink(c_path)


# ---------------------------------------------------------------------------
# CLI export subcommand
# ---------------------------------------------------------------------------


def test_cli_export_markov_c(tmp_path) -> None:
    """markovonnx export --format markov-c writes a C header from MarkovChain JSON."""
    mc = _make_chain(order=1)
    json_path = str(tmp_path / "model.json")
    header_path = str(tmp_path / "model.h")
    mc.save(json_path)

    result = subprocess.run(
        [
            sys.executable, "-m", "markovonnx.cli",
            "export", json_path,
            "--format", "markov-c",
            "-o", header_path,
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(header_path)
    content = open(header_path).read()
    assert "#define MARKOV_VOCAB_SIZE" in content


def test_cli_export_hmm_c(tmp_path) -> None:
    """markovonnx export --format hmm-c writes a C header from HMM JSON."""
    obs_seqs = [["a", "b", "c"]] * 3
    tag_seqs = [["X", "Y", "Z"]] * 3
    obs_vocab = Vocabulary(max_vocab=0)
    obs_vocab.build_from_sequences(obs_seqs)
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
    hmm.fit_supervised(obs_seqs, tag_seqs)

    json_path = str(tmp_path / "hmm.json")
    header_path = str(tmp_path / "hmm.h")
    hmm.save(json_path)

    result = subprocess.run(
        [
            sys.executable, "-m", "markovonnx.cli",
            "export", json_path,
            "--format", "hmm-c",
            "-o", header_path,
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(header_path)
    content = open(header_path).read()
    assert "#define HMM_N_STATES" in content


# ---------------------------------------------------------------------------
# --platformio flag
# ---------------------------------------------------------------------------


def test_platformio_markov_c_creates_files(tmp_path) -> None:
    """export --platformio creates platformio.ini and src/main.cpp for markov-c."""
    mc = _make_chain(order=1)
    json_path = str(tmp_path / "model.json")
    header_path = str(tmp_path / "model.h")
    mc.save(json_path)

    result = subprocess.run(
        [
            sys.executable, "-m", "markovonnx.cli",
            "export", json_path,
            "--format", "markov-c",
            "-o", header_path,
            "--platformio",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(str(tmp_path / "platformio.ini"))
    assert os.path.exists(str(tmp_path / "src" / "main.cpp"))


def test_platformio_ini_content(tmp_path) -> None:
    """platformio.ini contains [env:esp32dev] section."""
    mc = _make_chain(order=1)
    json_path = str(tmp_path / "model.json")
    header_path = str(tmp_path / "model.h")
    mc.save(json_path)

    subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "export", json_path,
         "--format", "markov-c", "-o", header_path, "--platformio"],
        check=True,
    )
    ini_content = open(str(tmp_path / "platformio.ini")).read()
    assert "[env:esp32dev]" in ini_content
    assert "espressif32" in ini_content


def test_platformio_markov_main_cpp_content(tmp_path) -> None:
    """src/main.cpp for markov-c references markov_sample."""
    mc = _make_chain(order=1)
    json_path = str(tmp_path / "model.json")
    header_path = str(tmp_path / "model.h")
    mc.save(json_path)

    subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "export", json_path,
         "--format", "markov-c", "-o", header_path, "--platformio"],
        check=True,
    )
    cpp_content = open(str(tmp_path / "src" / "main.cpp")).read()
    assert "markov_sample" in cpp_content


def test_platformio_hmm_c_creates_files(tmp_path) -> None:
    """export --platformio creates platformio files for hmm-c."""
    obs_seqs = [["a", "b", "c"]] * 3
    tag_seqs = [["X", "Y", "Z"]] * 3
    obs_vocab = Vocabulary(max_vocab=0)
    obs_vocab.build_from_sequences(obs_seqs)
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
    hmm.fit_supervised(obs_seqs, tag_seqs)

    json_path = str(tmp_path / "hmm.json")
    header_path = str(tmp_path / "hmm.h")
    hmm.save(json_path)

    result = subprocess.run(
        [sys.executable, "-m", "markovonnx.cli", "export", json_path,
         "--format", "hmm-c", "-o", header_path, "--platformio"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(str(tmp_path / "platformio.ini"))
    cpp_content = open(str(tmp_path / "src" / "main.cpp")).read()
    assert "hmm_forward_init" in cpp_content
    assert "hmm_forward_step" in cpp_content
    assert "hmm_best_state" in cpp_content


# ---------------------------------------------------------------------------
# HMM forward step: linear arrays and hmm_forward_init/step/best_state
# ---------------------------------------------------------------------------


def test_hmm_export_linear_arrays_present() -> None:
    """Generated HMM header includes linear-domain HMM_PI, HMM_A, HMM_B."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        content = open(path).read()
        assert "HMM_PI[" in content
        assert "HMM_A[" in content
        assert "HMM_B[" in content
    finally:
        os.unlink(path)


def test_hmm_export_forward_functions_present() -> None:
    """Generated HMM header includes hmm_forward_init, hmm_forward_step, hmm_best_state."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_hmm_c_header(hmm, path)
        content = open(path).read()
        assert "hmm_forward_init(" in content
        assert "hmm_forward_step(" in content
        assert "hmm_best_state(" in content
    finally:
        os.unlink(path)


def _py_forward(hmm: HiddenMarkovModel, obs_seq: List[str]) -> np.ndarray:
    """Run forward algorithm in Python, return normalized alpha after last obs."""
    obs_ids = hmm.obs_vocab.encode(obs_seq)
    S = hmm.n_states
    pi = np.asarray(hmm.pi, dtype=np.float64)
    A = np.asarray(hmm.A, dtype=np.float64)
    B = np.asarray(hmm.B, dtype=np.float64)

    alpha = pi * B[:, obs_ids[0]]
    s_sum = alpha.sum()
    if s_sum > 0:
        alpha /= s_sum
    else:
        alpha[:] = 1.0 / S

    for oid in obs_ids[1:]:
        new_alpha = (alpha @ A) * B[:, oid]
        s_sum = new_alpha.sum()
        if s_sum > 0:
            new_alpha /= s_sum
        else:
            new_alpha[:] = 1.0 / S
        alpha = new_alpha
    return alpha


def _c_forward(hmm: HiddenMarkovModel, obs_seq: List[str]) -> np.ndarray:
    """Simulate hmm_forward_init/step in Python using the HMM's linear arrays."""
    obs_ids = hmm.obs_vocab.encode(obs_seq)
    S = hmm.n_states
    pi = np.asarray(hmm.pi, dtype=np.float32)
    A = np.asarray(hmm.A, dtype=np.float32)
    B = np.asarray(hmm.B, dtype=np.float32)

    # hmm_forward_init
    alpha = pi * B[:, obs_ids[0]]
    s_sum = float(alpha.sum())
    if s_sum > 0.0:
        alpha /= s_sum
    else:
        alpha[:] = 1.0 / S

    # hmm_forward_step for each subsequent obs
    for oid in obs_ids[1:]:
        tmp = np.array([(alpha @ A[..., s]) * B[s, oid] for s in range(S)], dtype=np.float32)
        # Correct: tmp[s] = sum_q alpha[q]*A[q,s] * B[s,oid]
        tmp = np.array(
            [(alpha * A[:, s]).sum() * B[s, oid] for s in range(S)],
            dtype=np.float32,
        )
        s_sum = float(tmp.sum())
        if s_sum > 0.0:
            alpha = tmp / s_sum
        else:
            alpha = np.full(S, 1.0 / S, dtype=np.float32)

    return alpha


def test_hmm_forward_step_matches_python() -> None:
    """C-simulated forward step produces same best state as Python forward."""
    hmm = _make_hmm()
    for obs_seq in [["a", "b", "c"], ["b", "c", "a"], ["a", "a", "b"]]:
        py_alpha = _py_forward(hmm, obs_seq)
        c_alpha = _c_forward(hmm, obs_seq)
        py_best = int(py_alpha.argmax())
        c_best = int(c_alpha.argmax())
        assert py_best == c_best, f"obs={obs_seq}: py_best={py_best} c_best={c_best}"


def test_hmm_forward_step_probabilities_sum_to_one() -> None:
    """Normalized forward alpha sums to ~1.0."""
    hmm = _make_hmm()
    alpha = _c_forward(hmm, ["a", "b", "c"])
    assert abs(float(alpha.sum()) - 1.0) < 1e-5


@pytest.mark.skipif(
    subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
    reason="gcc not available",
)
def test_hmm_forward_step_valid_c_syntax() -> None:
    """HMM header with forward step functions passes gcc -fsyntax-only."""
    hmm = _make_hmm()
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    c_path = ""
    try:
        export_hmm_c_header(hmm, path)
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
            cf.write(f'#include "{path}"\nint main(void) {{ return 0; }}\n')
            c_path = cf.name
        result = subprocess.run(
            ["gcc", "-std=c99", "-fsyntax-only", c_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)
        if c_path and os.path.exists(c_path):
            os.unlink(c_path)


# ---------------------------------------------------------------------------
# Recursive (3-level) backoff export
# ---------------------------------------------------------------------------


def _make_chain_backoff3(order: int = 3) -> MarkovChain:
    """Build a trained MarkovChain with 3-level backoff (order 3 -> 2 -> 1)."""
    corpus = [list("abcabc"), list("bbbccc"), list("abcbc")]
    vocab = Vocabulary(max_vocab=0)
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5, backoff=True)
    mc.fit(corpus)
    return mc


def test_recursive_backoff_all_levels_exported() -> None:
    """Order-3 chain with backoff exports arrays for order 2 and order 1."""
    mc = _make_chain_backoff3(order=3)
    assert mc._lower is not None
    assert mc._lower._lower is not None  # three levels exist

    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        # Order-2 arrays
        assert "MARKOV_SPARSE_ROWS_2" in content
        assert "MARKOV_KEYS_2[" in content
        assert "MARKOV_PROBS_2[" in content
        assert "markov_lookup_2(" in content
        # Order-1 arrays
        assert "MARKOV_SPARSE_ROWS_1" in content
        assert "MARKOV_KEYS_1[" in content
        assert "MARKOV_PROBS_1[" in content
        assert "markov_lookup_1(" in content
        # Combined sample_backoff
        assert "markov_sample_backoff(" in content
    finally:
        os.unlink(path)


def test_recursive_backoff_legacy_alias_present() -> None:
    """Legacy MARKOV_KEYS_LOWER alias points to first backoff level."""
    mc = _make_chain_backoff3(order=3)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    try:
        export_markov_c_header(mc, path)
        content = open(path).read()
        assert "MARKOV_KEYS_LOWER" in content
        assert "markov_lookup_lower(" in content
    finally:
        os.unlink(path)


@pytest.mark.skipif(
    subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
    reason="gcc not available",
)
def test_recursive_backoff_valid_c_syntax() -> None:
    """3-level backoff C header passes gcc -fsyntax-only."""
    mc = _make_chain_backoff3(order=3)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as f:
        path = f.name
    c_path = ""
    try:
        export_markov_c_header(mc, path)
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as cf:
            cf.write(f'#include "{path}"\nint main(void) {{ return 0; }}\n')
            c_path = cf.name
        result = subprocess.run(
            ["gcc", "-std=c99", "-fsyntax-only", c_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)
        if c_path and os.path.exists(c_path):
            os.unlink(c_path)
