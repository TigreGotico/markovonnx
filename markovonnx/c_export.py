"""C header export for ESP32/embedded targets.

Generates self-contained ``*.h`` files from trained models that can be compiled
with any C99 toolchain (Arduino, ESP-IDF, bare-metal).

Two export functions are provided:

* :func:`export_markov_c_header` — :class:`~markovonnx.markov.MarkovChain` →
  binary-search lookup + CDF-walk sampler.
* :func:`export_hmm_c_header` — :class:`~markovonnx.hmm.HiddenMarkovModel` →
  pre-computed log-probability matrices + inline Viterbi decoder.

Generated headers have **zero external dependencies** — only ``<stdint.h>`` and
``<math.h>`` (for ``logf``).  All model data lives in static arrays suitable for
placement in flash (``PROGMEM`` / ``.rodata``).
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

    If the chain was trained with ``backoff=True``, **all** backoff levels are
    exported.  For each lower-order chain of order *k*, the header includes
    ``MARKOV_KEYS_{k}``, ``MARKOV_PROBS_{k}``, and ``markov_lookup_{k}()``.
    ``markov_sample_backoff()`` tries levels from highest to lowest order.

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

    # Collect ALL backoff levels, not just one.
    backoff_levels: List[Tuple[int, List[Tuple[int, np.ndarray]]]] = []
    lower = chain._lower
    while lower is not None:
        bp = _collect_rows(lower, V, quantize)
        backoff_levels.append((lower.order, bp))
        lower = lower._lower

    header = _render_header(rows, vocab, V, order, quantize, progmem, backoff_levels)

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
            # Guard: if all entries round to 0 (can happen for large uniform
            # distributions), ensure at least one entry is non-zero so that
            # markov_sample() can always satisfy r < cum.
            if row.sum() == 0:
                row[probs.argmax()] = 1
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
    backoff_levels: Optional[List[Tuple[int, List[Tuple[int, np.ndarray]]]]] = None,
) -> str:
    """Render the complete C header string.

    Args:
        pairs: Sorted (packed_key, row) pairs for the top-order chain.
        vocab: Vocabulary instance with id2tok.
        V: Vocabulary size.
        order: Markov chain order.
        quantize: Whether probs are uint8 (True) or float32 (False).
        progmem: Whether to add ESP32 PROGMEM section attribute.
        backoff_levels: List of (level_order, pairs) for each backoff chain,
            from highest to lowest order. All levels are exported.

    Returns:
        Complete C header file content as a string.
    """
    if backoff_levels is None:
        backoff_levels = []

    n_rows = len(pairs)
    prog_attr = ' __attribute__((section(".rodata")))' if progmem else ""
    prob_type = "uint8_t" if quantize else "float"

    vocab_str = _fmt_vocab(vocab.id2tok)
    keys_str = _fmt_keys(pairs)
    probs_str = _fmt_probs_uint8(pairs) if quantize else _fmt_probs_float(pairs)

    lookup_scale = "/ 255.0f" if quantize else ""
    prob_cast = f"(float)row[i] {lookup_scale}" if quantize else "row[i]"

    has_backoff = len(backoff_levels) > 0
    has_backoff_str = str(has_backoff).lower()

    header = f"""// generated by markovonnx (MarkovChain) — DO NOT EDIT
// Model: order={order}, vocab_size={V}, sparse_rows={n_rows}, quantized={str(quantize).lower()}, backoff={has_backoff_str}
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
// Falls back to MARKOV_VOCAB_SIZE-1 if row is NULL or CDF doesn't reach 1.
// Guarantee: exported rows always have at least one non-zero entry so the loop
// will terminate early for any r < 1.0.
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

    # Emit all backoff levels (previously only the first level was exported).
    for level_order, level_pairs in backoff_levels:
        n_level = len(level_pairs)
        level_keys_str = _fmt_keys(level_pairs)
        level_probs_str = _fmt_probs_uint8(level_pairs) if quantize else _fmt_probs_float(level_pairs)
        level_mask = (1 << (16 * level_order)) - 1
        header += f"""
// Backoff chain (order={level_order}) — used when higher-order context is unseen.
#define MARKOV_SPARSE_ROWS_{level_order} {n_level}

static const uint64_t{prog_attr} MARKOV_KEYS_{level_order}[MARKOV_SPARSE_ROWS_{level_order}] = {{
  {level_keys_str}
}};

static const {prob_type}{prog_attr} MARKOV_PROBS_{level_order}[MARKOV_SPARSE_ROWS_{level_order}][MARKOV_VOCAB_SIZE] = {{
{level_probs_str}
}};

// Backoff lookup for order={level_order}: uses the last {level_order} token(s) of key.
static inline const {prob_type}* markov_lookup_{level_order}(uint64_t key) {{
  uint64_t lkey = key & 0x{level_mask:016X}ULL;
  int lo = 0, hi = MARKOV_SPARSE_ROWS_{level_order} - 1;
  while (lo <= hi) {{
    int mid = lo + (hi - lo) / 2;
    if (MARKOV_KEYS_{level_order}[mid] == lkey) return MARKOV_PROBS_{level_order}[mid];
    if (MARKOV_KEYS_{level_order}[mid] < lkey) lo = mid + 1;
    else hi = mid - 1;
  }}
  return 0;
}}
"""

    if has_backoff:
        # Build the fallback chain: try full order first, then each backoff level.
        fallback_lines = ["  const {pt}* row = markov_lookup(key);".format(pt=prob_type)]
        for level_order, _ in backoff_levels:
            fallback_lines.append(f"  if (!row) row = markov_lookup_{level_order}(key);")
        fallback_body = "\n".join(fallback_lines)
        header += f"""
// Backoff sample: tries full-order lookup, then each backoff level in order.
static inline int markov_sample_backoff(uint64_t key, float r) {{
{fallback_body}
  return markov_sample(row, r);
}}
"""

    # Legacy aliases for single-level backoff (backwards compat with existing code).
    if len(backoff_levels) >= 1:
        first_order = backoff_levels[0][0]
        header += f"""
// Legacy aliases — use markov_lookup_{first_order} / markov_sample_backoff instead.
#define MARKOV_SPARSE_ROWS_LOWER MARKOV_SPARSE_ROWS_{first_order}
#define MARKOV_KEYS_LOWER MARKOV_KEYS_{first_order}
#define MARKOV_PROBS_LOWER MARKOV_PROBS_{first_order}
static inline const {prob_type}* markov_lookup_lower(uint64_t key) {{
  return markov_lookup_{first_order}(key);
}}
"""

    return header


# ---------------------------------------------------------------------------
# HMM C header export
# ---------------------------------------------------------------------------


def export_hmm_c_header(
    hmm: "HiddenMarkovModel",  # type: ignore[name-defined]  # noqa: F821
    path: str,
    progmem: bool = False,
) -> None:
    """Export a trained HiddenMarkovModel as a self-contained C99 header.

    The generated file contains:

    * ``HMM_OBS_VOCAB`` / ``HMM_STATE_VOCAB`` — observation and state string arrays.
    * ``HMM_LOG_PI``, ``HMM_LOG_A``, ``HMM_LOG_B`` — pre-computed log-probability
      matrices (float32), stored as static C arrays.
    * ``hmm_viterbi()`` — inline Viterbi decoder.  Caller allocates ``delta``
      (``float[T * HMM_N_STATES]``) and ``psi`` / ``path`` (``int[T * HMM_N_STATES]``
      / ``int[T]``) buffers.  Returns the best-path log-probability.

    Using pre-computed log probabilities avoids calling ``logf`` at inference
    time, keeping the inner Viterbi loop cheap on microcontrollers without FPUs.

    Args:
        hmm: Trained :class:`~markovonnx.hmm.HiddenMarkovModel`.
        path: Destination ``.h`` file path.
        progmem: If ``True``, annotate arrays with
            ``__attribute__((section(".rodata")))`` for ESP32 flash placement.

    Raises:
        ValueError: If the HMM has no ``state_vocab`` (unsupervised model with
            integer state indices is still exported, but state names are numeric).
    """
    import numpy as np

    S = hmm.n_states
    O = hmm.obs_vocab.size

    pi_lin = np.asarray(hmm.pi, dtype=np.float32)
    A_lin = np.asarray(hmm.A, dtype=np.float32)
    B_lin = np.asarray(hmm.B, dtype=np.float32)

    log_pi = np.log(pi_lin + 1e-30)
    log_A = np.log(A_lin + 1e-30)
    log_B = np.log(B_lin + 1e-30)

    obs_vocab = hmm.obs_vocab.id2tok
    state_vocab = (
        hmm.state_vocab.id2tok
        if hmm.state_vocab is not None
        else [str(i) for i in range(S)]
    )

    header = _render_hmm_header(log_pi, log_A, log_B, pi_lin, A_lin, B_lin, obs_vocab, state_vocab, S, O, progmem)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)


def _fmt_float_row(arr: "np.ndarray") -> str:  # type: ignore[name-defined]  # noqa: F821
    """Format a 1-D float array as a C array initialiser row.

    Args:
        arr: 1-D float array.

    Returns:
        Comma-separated float literal string.
    """
    return ", ".join(f"{float(v):.8f}f" for v in arr)


def _render_hmm_header(
    log_pi: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    log_A: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    log_B: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    pi_lin: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    A_lin: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    B_lin: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    obs_vocab: List[str],
    state_vocab: List[str],
    S: int,
    O: int,
    progmem: bool,
) -> str:
    """Render the complete HMM C header string.

    Emits both log-domain arrays (for Viterbi) and linear-domain arrays
    (for the constant-memory forward algorithm).

    Args:
        log_pi: Log initial state probabilities, shape (S,).
        log_A: Log transition matrix, shape (S, S).
        log_B: Log emission matrix, shape (S, O).
        pi_lin: Linear initial state probabilities, shape (S,).
        A_lin: Linear transition matrix, shape (S, S).
        B_lin: Linear emission matrix, shape (S, O).
        obs_vocab: Observation token strings (id2tok).
        state_vocab: State label strings (id2tok or numeric).
        S: Number of hidden states.
        O: Observation vocabulary size.
        progmem: Whether to add ESP32 PROGMEM section attribute.

    Returns:
        Complete C header file content as a string.
    """
    prog_attr = ' __attribute__((section(".rodata")))' if progmem else ""

    obs_str = _fmt_vocab(obs_vocab)
    state_str = _fmt_vocab(state_vocab)
    pi_str = _fmt_float_row(log_pi)
    a_rows = ",\n".join(f"  {{ {_fmt_float_row(log_A[i])} }}" for i in range(S))
    b_rows = ",\n".join(f"  {{ {_fmt_float_row(log_B[i])} }}" for i in range(S))

    lin_pi_str = _fmt_float_row(pi_lin)
    lin_a_rows = ",\n".join(f"  {{ {_fmt_float_row(A_lin[i])} }}" for i in range(S))
    lin_b_rows = ",\n".join(f"  {{ {_fmt_float_row(B_lin[i])} }}" for i in range(S))

    header = f"""// generated by markovonnx (HiddenMarkovModel) — DO NOT EDIT
// Model: n_states={S}, obs_vocab_size={O}
#pragma once
#include <stdint.h>

#define HMM_N_STATES  {S}
#define HMM_OBS_SIZE  {O}

static const char*{prog_attr} HMM_OBS_VOCAB[HMM_OBS_SIZE] = {{ {obs_str} }};
static const char*{prog_attr} HMM_STATE_VOCAB[HMM_N_STATES] = {{ {state_str} }};

// Pre-computed log-probabilities (natural log, float32).
// Using log-domain avoids logf() calls inside the Viterbi inner loop.
static const float{prog_attr} HMM_LOG_PI[HMM_N_STATES] = {{ {pi_str} }};

static const float{prog_attr} HMM_LOG_A[HMM_N_STATES][HMM_N_STATES] = {{
{a_rows}
}};

static const float{prog_attr} HMM_LOG_B[HMM_N_STATES][HMM_OBS_SIZE] = {{
{b_rows}
}};

// Linear-domain probabilities (float32) — used by hmm_forward_init/step.
// These avoid expf() calls and enable constant-memory forward filtering.
static const float{prog_attr} HMM_PI[HMM_N_STATES] = {{ {lin_pi_str} }};

static const float{prog_attr} HMM_A[HMM_N_STATES][HMM_N_STATES] = {{
{lin_a_rows}
}};

static const float{prog_attr} HMM_B[HMM_N_STATES][HMM_OBS_SIZE] = {{
{lin_b_rows}
}};

// Viterbi decoder.
//
// obs_ids : array of T observation ids (each in [0, HMM_OBS_SIZE))
// T       : sequence length
// delta   : caller-allocated float[T * HMM_N_STATES]  (log-viterbi scores)
// psi     : caller-allocated int[T * HMM_N_STATES]    (backpointer table)
// path    : caller-allocated int[T]                   (decoded state ids)
//
// Returns the log-probability of the best path.
// State string: HMM_STATE_VOCAB[path[t]] for t in [0, T).
static inline float hmm_viterbi(
    const int* obs_ids, int T,
    float* delta, int* psi, int* path)
{{
  int s, q, t;
  // Initialise t=0
  for (s = 0; s < HMM_N_STATES; s++) {{
    delta[s] = HMM_LOG_PI[s] + HMM_LOG_B[s][obs_ids[0]];
    psi[s] = 0;
  }}
  // Recursion
  for (t = 1; t < T; t++) {{
    for (s = 0; s < HMM_N_STATES; s++) {{
      float best = -3.402823466e+38f;  /* -FLT_MAX */
      int best_q = 0;
      for (q = 0; q < HMM_N_STATES; q++) {{
        float score = delta[(t-1)*HMM_N_STATES + q] + HMM_LOG_A[q][s];
        if (score > best) {{ best = score; best_q = q; }}
      }}
      delta[t*HMM_N_STATES + s] = best + HMM_LOG_B[s][obs_ids[t]];
      psi[t*HMM_N_STATES + s] = best_q;
    }}
  }}
  // Termination: find best final state
  float best_score = -3.402823466e+38f;
  int best_s = 0;
  for (s = 0; s < HMM_N_STATES; s++) {{
    if (delta[(T-1)*HMM_N_STATES + s] > best_score) {{
      best_score = delta[(T-1)*HMM_N_STATES + s];
      best_s = s;
    }}
  }}
  // Backtrack
  path[T-1] = best_s;
  for (t = T-2; t >= 0; t--) {{
    path[t] = psi[(t+1)*HMM_N_STATES + path[t+1]];
  }}
  return best_score;
}}

// Forward algorithm (constant memory — no T-length buffers needed).
// alpha: caller-allocated float[HMM_N_STATES] (modified in place).

// Initialize alpha for the first observation.
static inline void hmm_forward_init(int obs_id, float* alpha) {{
  float sum = 0.0f;
  int s;
  for (s = 0; s < HMM_N_STATES; s++) {{
    alpha[s] = HMM_PI[s] * HMM_B[s][obs_id];
    sum += alpha[s];
  }}
  if (sum > 0.0f) for (s = 0; s < HMM_N_STATES; s++) alpha[s] /= sum;
  else for (s = 0; s < HMM_N_STATES; s++) alpha[s] = 1.0f / HMM_N_STATES;
}}

// Update alpha for the next observation (in place). Uses stack buffer.
static inline void hmm_forward_step(int obs_id, float* alpha) {{
  float tmp[HMM_N_STATES];
  float sum = 0.0f;
  int s, q;
  for (s = 0; s < HMM_N_STATES; s++) {{
    float acc = 0.0f;
    for (q = 0; q < HMM_N_STATES; q++) acc += alpha[q] * HMM_A[q][s];
    tmp[s] = acc * HMM_B[s][obs_id];
    sum += tmp[s];
  }}
  if (sum > 0.0f) for (s = 0; s < HMM_N_STATES; s++) alpha[s] = tmp[s] / sum;
  else for (s = 0; s < HMM_N_STATES; s++) alpha[s] = 1.0f / HMM_N_STATES;
}}

// Returns the state id with the highest alpha value.
static inline int hmm_best_state(const float* alpha) {{
  int best = 0, s;
  for (s = 1; s < HMM_N_STATES; s++) if (alpha[s] > alpha[best]) best = s;
  return best;
}}
"""
    return header
