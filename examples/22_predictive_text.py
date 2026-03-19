#!/usr/bin/env python3
"""Example: Predictive text / autocomplete using Markov chains.

Given a partial sentence, show the top-K most likely next words with
their probabilities. Demonstrates the core mechanism behind smartphone
keyboard prediction.
"""

import numpy as np

from markovonnx import MarkovChain, Vocabulary, word_tokenize

CORPUS = [
    "i want to go to the store",
    "i want to eat some pizza",
    "i want to play some music",
    "i want to watch a movie",
    "i want to read a book",
    "i need to go to work",
    "i need to buy some groceries",
    "i need to call the doctor",
    "i need to fix the car",
    "how do i get to the airport",
    "how do i make a reservation",
    "how is the weather today",
    "how are you doing today",
    "what time is the meeting",
    "what time does the store close",
    "what is the weather forecast",
    "what is your name",
    "can you help me with this",
    "can you play some music",
    "can you turn on the lights",
    "please set a timer for five minutes",
    "please remind me to call mom",
    "please turn off the television",
    "the weather is nice today",
    "the movie was really good",
    "the food at the restaurant was excellent",
    "the traffic is terrible this morning",
    "she went to the store to buy milk",
    "he needs to finish the report by friday",
    "they are going to the park this afternoon",
] * 3


def predict_next(
    mc: MarkovChain,
    context: list,
    top_k: int = 5,
) -> list:
    """Return top-K next word predictions with probabilities."""
    probs = mc._get_probs(context)
    # Get top-k indices
    top_indices = np.argsort(probs)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        token = mc.vocab.id2tok[idx]
        prob = float(probs[idx])
        if prob > 0.001 and token != "<UNK>":
            results.append((token, prob))
    return results


def main() -> None:
    sequences = [word_tokenize(s) for s in CORPUS]
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)

    mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5, kneser_ney=True)
    mc.fit(sequences)

    print("--- Predictive Text / Autocomplete ---\n")

    test_contexts = [
        ["i", "want"],
        ["i", "need"],
        ["how", "is"],
        ["what", "time"],
        ["can", "you"],
        ["please", "set"],
        ["the", "weather"],
        ["to", "the"],
        ["going", "to"],
        ["she", "went"],
    ]

    for ctx in test_contexts:
        predictions = predict_next(mc, ctx, top_k=5)
        ctx_str = " ".join(ctx)
        pred_str = "  ".join(f"{w}({p:.0%})" for w, p in predictions)
        print(f"  \"{ctx_str} ___\"")
        print(f"    → {pred_str}")
        print()


if __name__ == "__main__":
    main()
