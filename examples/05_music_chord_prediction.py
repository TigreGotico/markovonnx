#!/usr/bin/env python3
"""Example: Predicting the next chord in a music progression.

Trains a word-level Markov chain on chord progressions and predicts/generates
chord sequences.
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
    data_path = str(Path(__file__).parent / "data" / "music_chords.txt")

    # 1. Load chord progressions
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    corpus = [word_tokenize(line) for line in lines if line.strip()]
    print(f"Loaded {len(corpus)} chord progressions")

    # 2. Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    print(f"Chords: {vocab.id2tok[1:]}")  # skip <UNK>

    # 3. Train order-2 chain (predict next chord from 2 preceding chords)
    mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    # 4. Predict next chord for common progressions
    print("\n--- Next chord predictions ---")
    progressions = [
        ["C", "G"],
        ["Am", "F"],
        ["F", "G"],
        ["C", "Am"],
        ["G", "Am"],
    ]
    for prog in progressions:
        next_chord = mc.sample(prog, temperature=0.3)
        print(f"  {' → '.join(prog)} → {next_chord}")

    # 5. Export and generate full progressions
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "chords.onnx")
        export_markov_onnx(mc, onnx_path)
        rt = MarkovONNXRuntime(onnx_path, vocab, order=2)

        print("\n--- Generated chord progressions ---")
        for seed in ["C G", "Am F", "C Am"]:
            progression = generate_markov(
                rt, seed, length=14, temperature=0.5, mode="word", order=2
            )
            print(f"  {progression}")

        # 6. Probability analysis
        print("\n--- Probability distribution after C → G ---")
        probs = rt.predict_probs(["c", "g"])
        top_indices = probs.argsort()[-5:][::-1]
        for idx in top_indices:
            token = vocab.id2tok[idx]
            print(f"  {token}: {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
