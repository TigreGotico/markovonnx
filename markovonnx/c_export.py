"""C header export for ESP32/embedded targets.

Generates a self-contained ``*.h`` file from a trained :class:`~markovonnx.markov.MarkovChain`
that can be compiled with any C99 toolchain (Arduino, ESP-IDF, bare-metal).

The generated header has **zero external dependencies** — only ``<stdint.h>``.
All model data lives in static arrays suitable for placement in flash
(``PROGMEM`` / ``.rodata``).

Inference algorithm: binary search over sorted packed context keys → CDF walk over
quantized probability row.
"""

from __future__ import annotations

import struct
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_markov_c_header(
    chain: "MarkovChain",  # type: ignore[name-defined]  # noqa: F821
    path: str,
    quantize: bool = True,
    progmem: bool = False,
) -> None:
    """Export a trained MarkovChain as a self-contained C99 header file.

    The generated file contains:

    * ``MARKOV_VOCAB`` — ``const char*`` array, token_id → string.
    * ``MARKOV_KEYS`` — sorted ``uint64_t`` packed context keys.
    * ``MARKOV_PROBS`` — per-row probability arrays (``uint8_t`` when
      *quantize* is ``True``, ``float`` otherwise).
    * ``markov_lookup()`` — inline binary search returning the probability row.
    * ``markov_sample()`` — inline CDF walk returning a token id.

    Context packing: for order *N*, token IDs are packed into a single
    ``uint64_t`` by shifting each ID by 16 bits.  Supports vocabularies up to
    65 535 tokens.

    Args:
        chain: Trained :class:`~markovonnx.markov.MarkovChain`.
        path: Destination ``.h`` file path.
        quantize: If ``True`` (default), store probabilities as ``uint8_t``
            (multiply by 255, round). Four times smaller than float32.
            Precision is adequate for multinomial sampling.
        progmem: If ``True``, annotate arrays with
            ``__attribute__((section(".rodata")))`` for Arduino/ESP32 SDK
            placement in flash memory.
    """
    vocab = chain.vocab
    V = vocab.size
    order = chain.order

    if V > 65535:
        raise ValueError(
            f"Vocabulary size {V} exceeds uint16_t range (65535). "
            "Reduce max_vocab before exporting."
        )

    rows = _collect_rows(chain, V, quantize)
    header = _render_header(rows, vocab, V, order, quantize, progmem)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pack_key(ctx_ids: List[int], order: int) -> int:
    """Pack up to *order* token IDs into a single uint64_t key.

    Each slot occupies 16 bits.  Supports vocab size < 65535.

    Args:
        ctx_ids: List of token IDs for the context (length == order).
        order: Model order (must be <= 4 for uint64_t packing).

    Returns:
        Packed integer key.
    """
    key = 0
    for tok_id in ctx_ids[-order:]:
        key = (key << 16) | (tok_id & 0xFFFF)
    return key


def _ctx_ids_from_index(ctx_index: int, order: int, V: int) -> List[int]:
    """Reverse base-V context index back to a list of token IDs.

    Args:
        ctx_index: Base-V encoded context index (from ``MarkovChain._counts``).
        order: N-gram order.
        V: Vocabulary size.

    Returns:
        List of *order* token IDs (most-recent last).
    """
    ids: List[int] = []
    for _ in range(order):
        ids.append(ctx_index % V)
        ctx_index //= V
    ids.reverse()
    return ids


def _collect_rows(
    chain: "MarkovChain",  # type: ignore[name-defined]  # noqa: F821
    V: int,
    quantize: bool,
) -> List[Tuple[int, np.ndarray]]:
    """Build sorted (packed_key, probability_row) pairs from chain._counts.

    Normalises raw counts with Laplace smoothing, optionally quantises to
    uint8, and sorts by packed key for binary search.

    Args:
        chain: Trained MarkovChain.
        V: Vocabulary size.
        quantize: Whether to quantise probabilities to uint8.

    Returns:
        Sorted list of (packed_uint64_key, row_array) tuples.
        Row dtype is uint8 when *quantize* is True, float32 otherwise.
    """
    order = chain.order
    smoothing = chain.smoothing
    pairs: List[Tuple[int, np.ndarray]] = []

    for ci, counts in chain._counts.items():
        ctx_ids = _ctx_ids_from_index(ci, order, V)
        key = _pack_key(ctx_ids, order)
        probs = (counts + smoothing) / (counts.sum() + smoothing * V)
        if quantize:
            row = np.clip(np.round(probs * 255.0), 0, 255).astype(np.uint8)
        else:
            row = probs.astype(np.float32)
        pairs.append((key, row))

    # Sort by packed key for binary search in generated C code
    pairs.sort(key=lambda x: x[0])
    return pairs


def _fmt_vocab(tokens: List[str]) -> str:
    """Format vocabulary tokens as a C string array initialiser.

    Args:
        tokens: List of token strings (id2tok).

    Returns:
        Multi-line C array content string.
    """
    parts: List[str] = []
    for tok in tokens:
        escaped = tok.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        parts.append(f'"{escaped}"')
    return ", ".join(parts)


def _fmt_keys(pairs: List[Tuple[int, np.ndarray]]) -> str:
    """Format packed context keys as a C uint64_t array initialiser.

    Args:
        pairs: Sorted (key, row) pairs.

    Returns:
        Comma-separated hex literal string.
    """
    return ", ".join(f"0x{k:016X}ULL" for k, _ in pairs)


def _fmt_probs_uint8(pairs: List[Tuple[int, np.ndarray]]) -> str:
    """Format uint8 probability rows as a 2-D C array initialiser.

    Args:
        pairs: Sorted (key, row) pairs where row.dtype == uint8.

    Returns:
        Multi-line nested braces string.
    """
    rows_str: List[str] = []
    for _, row in pairs:
        inner = ", ".join(str(int(v)) for v in row)
        rows_str.append(f"  {{{inner}}}")
    return ",\n".join(rows_str)


def _fmt_probs_float(pairs: List[Tuple[int, np.ndarray]]) -> str:
    """Format float32 probability rows as a 2-D C array initialiser.

    Args:
        pairs: Sorted (key, row) pairs where row.dtype == float32.

    Returns:
        Multi-line nested braces string.
    """
    rows_str: List[str] = []
    for _, row in pairs:
        inner = ", ".join(f"{v:.8f}f" for v in row)
        rows_str.append(f"  {{{inner}}}")
    return ",\n".join(rows_str)


def _render_header(
    pairs: List[Tuple[int, np.ndarray]],
    vocab: "Vocabulary",  # type: ignore[name-defined]  # noqa: F821
    V: int,
    order: int,
    quantize: bool,
    progmem: bool,
) -> str:
    """Render the complete C header string.

    Args:
        pairs: Sorted (packed_key, row) pairs.
        vocab: Vocabulary instance with id2tok.
        V: Vocabulary size.
        order: Markov chain order.
        quantize: Whether probs are uint8 (True) or float32 (False).
        progmem: Whether to add ESP32 PROGMEM section attribute.

    Returns:
        Complete C header file content as a string.
    """
    n_rows = len(pairs)
    prog_attr = ' __attribute__((section(".rodata")))' if progmem else ""
    prob_type = "uint8_t" if quantize else "float"

    vocab_str = _fmt_vocab(vocab.id2tok)
    keys_str = _fmt_keys(pairs)
    probs_str = _fmt_probs_uint8(pairs) if quantize else _fmt_probs_float(pairs)

    lookup_scale = "/ 255.0f" if quantize else ""
    prob_cast = f"(float)row[i] {lookup_scale}" if quantize else "row[i]"

    header = f"""// generated by markovonnx — DO NOT EDIT
// Model: order={order}, vocab_size={V}, sparse_rows={n_rows}, quantized={str(quantize).lower()}
#pragma once
#include <stdint.h>

#define MARKOV_VOCAB_SIZE  {V}
#define MARKOV_ORDER       {order}
#define MARKOV_SPARSE_ROWS {n_rows}

static const char*{prog_attr} MARKOV_VOCAB[MARKOV_VOCAB_SIZE] = {{ {vocab_str} }};

static const uint64_t{prog_attr} MARKOV_KEYS[MARKOV_SPARSE_ROWS] = {{
  {keys_str}
}};

static const {prob_type}{prog_attr} MARKOV_PROBS[MARKOV_SPARSE_ROWS][MARKOV_VOCAB_SIZE] = {{
{probs_str}
}};

// Binary search: returns pointer to probability row for key, or NULL if unseen.
static inline const {prob_type}* markov_lookup(uint64_t key) {{
  int lo = 0, hi = MARKOV_SPARSE_ROWS - 1;
  while (lo <= hi) {{
    int mid = lo + (hi - lo) / 2;
    if (MARKOV_KEYS[mid] == key) return MARKOV_PROBS[mid];
    if (MARKOV_KEYS[mid] < key) lo = mid + 1;
    else hi = mid - 1;
  }}
  return 0;
}}

// CDF walk: r must be uniform in [0, 1). Returns token id in [0, MARKOV_VOCAB_SIZE).
// Falls back to token 0 if row is NULL (unseen context) or CDF doesn't reach 1.
static inline int markov_sample(const {prob_type}* row, float r) {{
  if (!row) return 0;
  float cum = 0.0f;
  for (int i = 0; i < MARKOV_VOCAB_SIZE; i++) {{
    cum += {prob_cast};
    if (r < cum) return i;
  }}
  return MARKOV_VOCAB_SIZE - 1;
}}
"""
    return header
