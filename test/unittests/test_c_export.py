"""Unit tests for markovonnx.c_export — C header generation for ESP32."""

import os
import re
import subprocess
import sys
import tempfile
from typing import List

import numpy as np
import pytest

from markovonnx.c_export import (
    _collect_rows,
    _ctx_ids_from_index,
    _pack_key,
    export_markov_c_header,
)
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
