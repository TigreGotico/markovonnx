"""Text generation using a Markov ONNX Runtime model."""

from typing import List, Optional

from markovonnx.onnx_runtime import MarkovONNXRuntime
from markovonnx.tokenizers import SubwordTokenizer


def generate_markov(
    ort_model: MarkovONNXRuntime,
    seed: str,
    length: int,
    temperature: float = 1.0,
    mode: str = "char",
    order: int = 2,
    bpe_tokenizer: Optional[SubwordTokenizer] = None,
) -> str:
    """Generate text auto-regressively using an ONNX Markov model.

    Args:
        ort_model: A loaded :class:`MarkovONNXRuntime`.
        seed: Initial text to seed generation.
        length: Number of tokens to generate.
        temperature: Sampling temperature.
        mode: Tokenization mode (``'char'``, ``'word'``, ``'bpe'``).
        order: N-gram order (context window size).
        bpe_tokenizer: Required when *mode* is ``'bpe'``.

    Returns:
        Generated text string.
    """
    if mode == "char":
        context: List = list(seed) if seed else list("the ")
        result = list(context)
        for _ in range(length):
            nxt = ort_model.sample(context[-order:], temperature)
            result.append(nxt)
            context = (context + [nxt])[-order:]
        return "".join(result)

    if mode in ("bpe", "subword"):
        if bpe_tokenizer is None:
            raise ValueError("bpe_tokenizer is required for BPE generation")
        context: List = list(bpe_tokenizer.encode_bpe(seed)) if seed else []
        result = list(context)
        for _ in range(length):
            nxt = ort_model.sample(context[-order:], temperature)
            result.append(nxt)
            context = (context + [nxt])[-order:]
        return bpe_tokenizer.decode_bpe(result)

    # word mode
    context = seed.split() if seed else ["the"]
    result = list(context)
    for _ in range(length):
        nxt = ort_model.sample(context[-order:], temperature)
        result.append(nxt)
        context = (context + [nxt])[-order:]
    return " ".join(result)
