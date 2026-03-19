#!/usr/bin/env python3
"""Example: POS tagging with supervised HMM.

Trains an HMM on (word, POS-tag) pairs and decodes new sentences
with Viterbi. Classic approach to part-of-speech tagging.
"""

from markovonnx import HiddenMarkovModel, Vocabulary


def main() -> None:
    # Training data: simple (word, tag) sequences
    word_seqs = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["a", "dog", "ran", "in", "the", "park"],
        ["the", "bird", "flew", "over", "the", "house"],
        ["a", "fish", "swam", "under", "the", "bridge"],
        ["the", "boy", "walked", "to", "the", "store"],
        ["a", "girl", "jumped", "over", "the", "fence"],
        ["the", "man", "drove", "to", "the", "office"],
        ["a", "woman", "ran", "through", "the", "forest"],
    ]
    tag_seqs = [
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
        ["DT", "NN", "VBD", "IN", "DT", "NN"],
    ]

    # Build vocabulary
    obs_vocab = Vocabulary()
    obs_vocab.build_from_sequences(word_seqs)

    # Train HMM
    hmm = HiddenMarkovModel(n_states=5, obs_vocab=obs_vocab)
    hmm.fit_supervised(word_seqs, tag_seqs)

    print(f"States: {hmm.state_vocab.id2tok}")
    print(f"Vocab: {obs_vocab.size} words")

    # Tag new sentences
    test_sents = [
        ["the", "cat", "ran", "to", "the", "park"],
        ["a", "bird", "flew", "over", "the", "bridge"],
        ["the", "dog", "walked", "through", "the", "forest"],
    ]

    print("\n--- POS Tagging ---")
    for words in test_sents:
        tags = hmm.viterbi(words)
        pairs = [f"{w}/{t}" for w, t in zip(words, tags)]
        print(f"  {' '.join(pairs)}")


if __name__ == "__main__":
    main()
