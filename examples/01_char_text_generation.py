#!/usr/bin/env python3
"""Example: Character-level text generation from nursery rhymes.

Trains a character-level Markov chain on nursery rhymes, exports to ONNX,
and generates text.
"""

import tempfile
from pathlib import Path

from markovonnx import (
    MarkovChain,
    MarkovONNXRuntime,
    Vocabulary,
    char_tokenize,
    export_markov_onnx,
    generate_markov,
)


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "nursery_rhymes.txt")

    # 1. Load corpus
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    corpus = [char_tokenize(line) for line in lines if line.strip()]
    print(f"Loaded {len(corpus)} lines, {sum(len(s) for s in corpus)} characters")

    # 2. Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    print(f"Vocabulary size: {vocab.size}")

    # 3. Train Markov chain (order=3 for character-level)
    mc = MarkovChain(order=3, vocab=vocab, smoothing=1e-6)
    mc.fit(corpus)

    # 4. Evaluate perplexity
    ppx = mc.perplexity(corpus[:5])
    print(f"Perplexity (train sample): {ppx:.2f}")

    # 5. Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "nursery_char.onnx")
        export_markov_onnx(mc, onnx_path)

        # 6. Load runtime and generate
        rt = MarkovONNXRuntime(onnx_path, vocab, order=3)

        print("\n--- Generated Text (temperature=0.5) ---")
        text = generate_markov(rt, "the ", length=200, temperature=0.5, mode="char", order=3)
        print(text)

        print("\n--- Generated Text (temperature=1.0) ---")
        text = generate_markov(rt, "mary", length=200, temperature=1.0, mode="char", order=3)
        print(text)

        print("\n--- Generated Text (temperature=0.2, more deterministic) ---")
        text = generate_markov(rt, "jack", length=200, temperature=0.2, mode="char", order=3)
        print(text)


if __name__ == "__main__":
    main()
