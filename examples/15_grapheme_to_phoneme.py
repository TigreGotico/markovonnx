#!/usr/bin/env python3
"""Example: Grapheme-to-phoneme conversion with supervised HMM.

Maps character sequences to phoneme sequences using Viterbi decoding.
Each character is an observation, each phoneme is a hidden state.
Requires aligned (char, phoneme) training pairs of equal length.
"""

from markovonnx import HiddenMarkovModel, Vocabulary


def main() -> None:
    # Training data: aligned character → phoneme pairs
    # Each character maps to one phoneme (simplification)
    graphemes = [
        list("cat"),  list("bat"),  list("hat"),  list("mat"),  list("rat"),
        list("dog"),  list("fog"),  list("log"),  list("hog"),  list("bog"),
        list("sit"),  list("bit"),  list("fit"),  list("hit"),  list("kit"),
        list("cup"),  list("pup"),  list("sup"),
        list("pen"),  list("hen"),  list("ten"),  list("den"),  list("men"),
    ]
    phonemes = [
        ["K","AE","T"], ["B","AE","T"], ["HH","AE","T"], ["M","AE","T"], ["R","AE","T"],
        ["D","AO","G"], ["F","AO","G"], ["L","AO","G"], ["HH","AO","G"], ["B","AO","G"],
        ["S","IH","T"], ["B","IH","T"], ["F","IH","T"], ["HH","IH","T"], ["K","IH","T"],
        ["K","AH","P"], ["P","AH","P"], ["S","AH","P"],
        ["P","EH","N"], ["HH","EH","N"], ["T","EH","N"], ["D","EH","N"], ["M","EH","N"],
    ]

    # Build vocab
    obs_vocab = Vocabulary()
    obs_vocab.build_from_sequences(graphemes)

    # Train HMM
    hmm = HiddenMarkovModel(n_states=20, obs_vocab=obs_vocab)
    hmm.fit_supervised(graphemes * 3, phonemes * 3)

    print(f"Grapheme vocab: {obs_vocab.id2tok[1:]}")
    print(f"Phoneme states: {hmm.state_vocab.id2tok[1:]}")

    # Convert new words
    test_words = ["cat", "dog", "bat", "fog", "pen", "cup", "hat", "log"]

    print("\n--- Grapheme to Phoneme ---")
    for word in test_words:
        chars = list(word)
        phones = hmm.viterbi(chars)
        print(f"  {word:>5s} → {' '.join(phones)}")


if __name__ == "__main__":
    main()
