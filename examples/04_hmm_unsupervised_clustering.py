#!/usr/bin/env python3
"""Example: Unsupervised HMM for discovering patterns in DNA sequences.

Uses Baum-Welch to find hidden states in DNA data without labels.
"""

from pathlib import Path

from markovonnx import HiddenMarkovModel, Vocabulary, char_tokenize


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "dna_sequences.txt")

    # 1. Load DNA sequences as character tokens
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    corpus = [char_tokenize(line) for line in lines if line.strip()]
    print(f"Loaded {len(corpus)} sequences")
    print(f"Alphabet sample: {sorted(set(corpus[0]))}")

    # 2. Build observation vocabulary (A, C, G, T)
    obs_vocab = Vocabulary()
    obs_vocab.build_from_sequences(corpus)
    print(f"Observation vocab: {obs_vocab.id2tok}")

    # 3. Train unsupervised HMM with 3 hidden states
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
    hmm.fit_unsupervised(corpus, n_iter=15)

    # 4. Examine learned parameters
    print(f"\nInitial state probs (pi): {hmm.pi}")
    print(f"\nTransition matrix (A):")
    for i, row in enumerate(hmm.A):
        print(f"  State {i}: {row}")
    print(f"\nEmission matrix (B):")
    for i, row in enumerate(hmm.B):
        tokens = obs_vocab.id2tok
        emission = {tokens[j]: f"{row[j]:.3f}" for j in range(len(tokens))}
        print(f"  State {i}: {emission}")

    # 5. Decode some sequences
    print("\n--- State assignments ---")
    for seq in corpus[:5]:
        states = hmm.viterbi(seq)
        # Show first 20 characters with their states
        pairs = list(zip(seq[:20], states[:20]))
        print(f"  {''.join(c for c, _ in pairs)}")
        print(f"  {''.join(s for _, s in pairs)}")
        print()


if __name__ == "__main__":
    main()
