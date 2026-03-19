#!/usr/bin/env python3
"""Example: Word-level text generation from nursery rhymes.

Trains a word-level Markov chain and generates text with different temperatures.
"""

import tempfile
from pathlib import Path

from markovonnx import (
    MarkovChain,
    MarkovONNXRuntime,
    Vocabulary,
    export_markov_onnx,
    generate_markov,
    word_tokenize,
)


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "nursery_rhymes.txt")

    # 1. Load and tokenize
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    corpus = [word_tokenize(line) for line in lines if line.strip()]
    print(f"Loaded {len(corpus)} lines, {sum(len(s) for s in corpus)} words")

    # 2. Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    print(f"Vocabulary size: {vocab.size}")

    # 3. Train (order=2 for bigram word model)
    mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    # 4. Python-side sampling (no ONNX)
    print("\n--- Python sampling ---")
    context = ["the", "little"]
    for _ in range(5):
        token = mc.sample(context, temperature=0.7)
        print(f"  context={context} → {token}")
        context = [context[-1], token]

    # 5. Export and use ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "nursery_word.onnx")
        export_markov_onnx(mc, onnx_path)
        rt = MarkovONNXRuntime(onnx_path, vocab, order=2)

        print("\n--- ONNX generation (temperature=0.7) ---")
        text = generate_markov(rt, "the little", length=30, temperature=0.7, mode="word", order=2)
        print(text)

        print("\n--- ONNX generation (temperature=0.3) ---")
        text = generate_markov(rt, "mary had", length=30, temperature=0.3, mode="word", order=2)
        print(text)

        # 6. Compare argmax (greedy)
        print("\n--- Greedy next-token predictions ---")
        for seed in [["the", "little"], ["jack", "and"], ["mary", "had"]]:
            token = rt.argmax(seed)
            print(f"  {seed} → {token}")


if __name__ == "__main__":
    main()
