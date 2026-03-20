"""Memory size estimation for MarkovChain and HMM C header exports.

Provides byte-accurate size breakdowns for each generated C array and
reports whether the model fits in ESP32 flash or IRAM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from markovonnx.hmm import HiddenMarkovModel
    from markovonnx.markov import MarkovChain

ESP32_FLASH_BYTES: int = 4 * 1024 * 1024   # 4 MB
ESP32_IRAM_BYTES: int = 520 * 1024          # 520 KB


def markov_c_sizes(chain: "MarkovChain", quantize: bool = True) -> Dict[str, int]:
    """Return byte sizes of each C array that would be generated for a MarkovChain.

    Args:
        chain: Trained :class:`~markovonnx.markov.MarkovChain`.
        quantize: Whether uint8 (``True``) or float32 (``False``) probs are used.

    Returns:
        Dict with keys: ``vocab``, ``MARKOV_KEYS``, ``MARKOV_PROBS``,
        one ``MARKOV_KEYS_{k}`` + ``MARKOV_PROBS_{k}`` per backoff level,
        ``total``, plus metadata keys ``n_sparse_rows``, ``vocab_size``,
        ``quantized``.
    """
    V = chain.vocab.size
    n_sparse = len(chain._counts)
    prob_bytes = 1 if quantize else 4

    # Vocabulary: sum of UTF-8 encoded string lengths + null terminator
    vocab_bytes = sum(len(tok.encode("utf-8")) + 1 for tok in chain.vocab.id2tok)

    keys_bytes = n_sparse * 8                    # uint64_t per row
    probs_bytes = n_sparse * V * prob_bytes       # prob_type[n_sparse][V]

    sizes: Dict[str, int] = {
        "vocab": vocab_bytes,
        "MARKOV_KEYS": keys_bytes,
        "MARKOV_PROBS": probs_bytes,
    }

    lower = chain._lower
    while lower is not None:
        k = lower.order
        n_lower = len(lower._counts)
        sizes[f"MARKOV_KEYS_{k}"] = n_lower * 8
        sizes[f"MARKOV_PROBS_{k}"] = n_lower * V * prob_bytes
        lower = lower._lower

    total = sum(v for k, v in sizes.items())
    sizes["total"] = total
    sizes["n_sparse_rows"] = n_sparse
    sizes["vocab_size"] = V
    sizes["quantized"] = int(quantize)
    return sizes


def hmm_c_sizes(hmm: "HiddenMarkovModel") -> Dict[str, int]:
    """Return byte sizes of each C array that would be generated for an HMM.

    Args:
        hmm: Trained :class:`~markovonnx.hmm.HiddenMarkovModel`.

    Returns:
        Dict with keys: ``obs_vocab``, ``state_vocab``, ``HMM_LOG_PI``,
        ``HMM_LOG_A``, ``HMM_LOG_B``, ``HMM_PI``, ``HMM_A``, ``HMM_B``,
        ``total``, plus metadata ``n_states``, ``obs_vocab_size``.
    """
    S = hmm.n_states
    O = hmm.obs_vocab.size
    state_vocab = (
        hmm.state_vocab.id2tok
        if hmm.state_vocab is not None
        else [str(i) for i in range(S)]
    )

    obs_vocab_bytes = sum(len(tok.encode("utf-8")) + 1 for tok in hmm.obs_vocab.id2tok)
    state_vocab_bytes = sum(len(tok.encode("utf-8")) + 1 for tok in state_vocab)

    log_pi_bytes = S * 4
    log_a_bytes = S * S * 4
    log_b_bytes = S * O * 4

    # Linear-domain duplicates (same shape)
    pi_bytes = S * 4
    a_bytes = S * S * 4
    b_bytes = S * O * 4

    sizes: Dict[str, int] = {
        "obs_vocab": obs_vocab_bytes,
        "state_vocab": state_vocab_bytes,
        "HMM_LOG_PI": log_pi_bytes,
        "HMM_LOG_A": log_a_bytes,
        "HMM_LOG_B": log_b_bytes,
        "HMM_PI": pi_bytes,
        "HMM_A": a_bytes,
        "HMM_B": b_bytes,
    }

    total = sum(v for k, v in sizes.items())
    sizes["total"] = total
    sizes["n_states"] = S
    sizes["obs_vocab_size"] = O
    return sizes


def _fit_line(label: str, total: int, budget: int) -> str:
    """Return a single ESP32 fit-check line.

    Args:
        label: Human-readable budget name (e.g. ``"Flash (4 MB)"``).
        total: Total bytes used.
        budget: Budget ceiling in bytes.

    Returns:
        String like ``"Flash (4 MB): 3.2% — FITS"`` or ``"... TOO LARGE"``.
    """
    pct = total / budget * 100.0
    status = "FITS" if total <= budget else "TOO LARGE"
    return f"{label}: {pct:.1f}% — {status}"


def format_markov_report(
    chain: "MarkovChain",
    quantize: bool = True,
    progmem: bool = False,
) -> str:
    """Return a human-readable size report string for a MarkovChain.

    Args:
        chain: Trained :class:`~markovonnx.markov.MarkovChain`.
        quantize: Whether to assume uint8 (``True``) or float32 (``False``) probs.
        progmem: If ``True``, only Flash (PROGMEM) comparison is shown;
            otherwise also shows IRAM.

    Returns:
        Multi-line string with per-array breakdown, total, ESP32 fit check,
        and suggestions if the model is too large.
    """
    sizes = markov_c_sizes(chain, quantize)
    total = sizes["total"]
    skip = {"total", "n_sparse_rows", "vocab_size", "quantized"}

    lines = ["MarkovChain C header size report"]
    lines.append(f"  order={chain.order}  vocab={sizes['vocab_size']}  "
                 f"sparse_rows={sizes['n_sparse_rows']}  quantized={bool(quantize)}")
    lines.append("")
    lines.append(f"  {'Array':<30} {'Bytes':>10}  {'KB':>8}")
    lines.append("  " + "-" * 52)
    for key, val in sizes.items():
        if key in skip:
            continue
        lines.append(f"  {key:<30} {val:>10}  {val / 1024:>8.2f}")
    lines.append("  " + "-" * 52)
    lines.append(f"  {'TOTAL':<30} {total:>10}  {total / 1024:>8.2f}")
    lines.append("")
    lines.append("  ESP32 fit check:")
    lines.append("    " + _fit_line("Flash (4 MB)", total, ESP32_FLASH_BYTES))
    if not progmem:
        lines.append("    " + _fit_line("IRAM  (520 KB)", total, ESP32_IRAM_BYTES))

    if total > ESP32_FLASH_BYTES:
        lines.append("")
        lines.append("  Suggestions:")
        lines.append("    - Reduce vocabulary (--max-vocab)")
        lines.append("    - Reduce chain order (--order)")
        lines.append("    - Use quantization (quantize=True, default)")
    elif not progmem and total > ESP32_IRAM_BYTES:
        lines.append("")
        lines.append("  Suggestions:")
        lines.append("    - Use --progmem to place arrays in flash (.rodata)")
        lines.append("    - Reduce vocabulary (--max-vocab)")

    return "\n".join(lines)


def format_hmm_report(
    hmm: "HiddenMarkovModel",
    progmem: bool = False,
) -> str:
    """Return a human-readable size report string for an HMM.

    Args:
        hmm: Trained :class:`~markovonnx.hmm.HiddenMarkovModel`.
        progmem: If ``True``, only Flash (PROGMEM) comparison is shown;
            otherwise also shows IRAM.

    Returns:
        Multi-line string with per-array breakdown, total, ESP32 fit check,
        and suggestions if the model is too large.
    """
    sizes = hmm_c_sizes(hmm)
    total = sizes["total"]
    skip = {"total", "n_states", "obs_vocab_size"}

    lines = ["HiddenMarkovModel C header size report"]
    lines.append(f"  n_states={sizes['n_states']}  obs_vocab={sizes['obs_vocab_size']}")
    lines.append("")
    lines.append(f"  {'Array':<30} {'Bytes':>10}  {'KB':>8}")
    lines.append("  " + "-" * 52)
    for key, val in sizes.items():
        if key in skip:
            continue
        lines.append(f"  {key:<30} {val:>10}  {val / 1024:>8.2f}")
    lines.append("  " + "-" * 52)
    lines.append(f"  {'TOTAL':<30} {total:>10}  {total / 1024:>8.2f}")
    lines.append("")
    lines.append("  ESP32 fit check:")
    lines.append("    " + _fit_line("Flash (4 MB)", total, ESP32_FLASH_BYTES))
    if not progmem:
        lines.append("    " + _fit_line("IRAM  (520 KB)", total, ESP32_IRAM_BYTES))

    if total > ESP32_FLASH_BYTES:
        lines.append("")
        lines.append("  Suggestions:")
        lines.append("    - Reduce vocabulary (--max-vocab)")
        lines.append("    - Reduce n_states")
    elif not progmem and total > ESP32_IRAM_BYTES:
        lines.append("")
        lines.append("  Suggestions:")
        lines.append("    - Use --progmem to place arrays in flash (.rodata)")
        lines.append("    - Reduce vocabulary or n_states")

    return "\n".join(lines)
