#!/usr/bin/env python3
"""Example: Keyword extraction via perplexity surprise scoring.

Trains a background language model on general text, then scores words
in a query by inverse frequency — rare (surprising) words are keywords.
"""

import math
from typing import Dict

from markovonnx import MarkovChain, Vocabulary, word_tokenize


def train_background_model(corpus: list) -> tuple:
    """Train a word-level background model."""
    sequences = [word_tokenize(s.lower()) for s in corpus]
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(sequences)
    return mc, vocab


def extract_keywords(
    text: str,
    vocab: Vocabulary,
    top_k: int = 5,
) -> Dict[str, float]:
    """Extract keywords by inverse frequency in background model."""
    tokens = word_tokenize(text.lower())
    total_count = sum(vocab._counts.values())

    scores: Dict[str, float] = {}
    for tok in set(tokens):
        if len(tok) <= 2:
            continue
        count = vocab._counts.get(tok, 0)
        if count == 0:
            scores[tok] = 1.0  # OOV = max surprise
        else:
            scores[tok] = 1.0 / (1.0 + math.log1p(count))

    # Normalize
    if scores:
        mx = max(scores.values())
        scores = {k: v / mx for k, v in scores.items()}

    return dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])


def main() -> None:
    # Background corpus (general text)
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the house",
        "the fish swam under the bridge",
        "she went to the store to buy milk",
        "he drove to the office every morning",
        "they walked through the forest together",
        "the children played in the garden all day",
        "the sun was shining brightly in the sky",
        "the rain fell gently on the roof",
    ] * 10

    print("Training background language model...")
    mc, vocab = train_background_model(corpus)
    print(f"Vocab: {vocab.size} words, {sum(vocab._counts.values())} total tokens")

    # Extract keywords from various texts
    test_texts = [
        "the quantum physicist discovered antimatter in the laboratory",
        "machine learning algorithms optimize neural network parameters",
        "the cat sat on the mat in the garden",
        "cryptocurrency blockchain technology disrupts traditional banking",
        "the astronaut performed a spacewalk outside the station",
    ]

    print("\n--- Keyword Extraction ---")
    for text in test_texts:
        keywords = extract_keywords(text, vocab, top_k=3)
        kw_str = ", ".join(f"{w} ({s:.2f})" for w, s in keywords.items())
        print(f"\n  Text: {text}")
        print(f"  Keywords: {kw_str}")


if __name__ == "__main__":
    main()
