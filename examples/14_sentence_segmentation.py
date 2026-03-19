#!/usr/bin/env python3
"""Example: Sentence segmentation using character-level boundary detection.

Trains a char-level Markov model on sentence-final patterns. Detects
boundary positions where the model predicts a period/newline with
high probability.
"""

from markovonnx import MarkovChain, Vocabulary, char_tokenize


def train_boundary_model(sentences: list, order: int = 4) -> tuple:
    """Train a char-level model on sentences ending with boundary markers."""
    boundary = "\n"
    corpus = [char_tokenize(s.strip() + boundary) for s in sentences]
    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    # Identify boundary token IDs
    boundary_ids = []
    for ch in [".", "!", "?", "\n"]:
        if ch in vocab.tok2id:
            boundary_ids.append(vocab.tok2id[ch])

    return mc, vocab, boundary_ids


def segment(text: str, mc: MarkovChain, boundary_ids: list, threshold: float = 0.3) -> list:
    """Split text into sentences based on boundary probability."""
    tokens = char_tokenize(text)
    order = mc.order
    if len(tokens) < order:
        return [text.strip()] if text.strip() else []

    splits = []
    for i in range(order, len(tokens)):
        ctx = tokens[i - order:i]
        probs = mc._get_probs(ctx)
        bp = sum(float(probs[bid]) for bid in boundary_ids if bid < len(probs))
        if bp > threshold:
            splits.append(i)

    # Build segments
    segments = []
    prev = 0
    for sp in splits:
        seg = "".join(tokens[prev:sp]).strip()
        if seg:
            segments.append(seg)
        prev = sp
    last = "".join(tokens[prev:]).strip()
    if last:
        segments.append(last)

    return segments or [text.strip()]


def main() -> None:
    # Training sentences
    train_sents = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "Hello world.",
        "How are you doing today?",
        "This is a test sentence.",
        "She went to the store.",
        "He drove to the office.",
        "They walked through the forest.",
        "The sun was shining brightly.",
        "It was a beautiful day.",
    ] * 10

    print("Training boundary model...")
    mc, vocab, boundary_ids = train_boundary_model(train_sents, order=4)
    print(f"Vocab: {vocab.size} chars, {len(boundary_ids)} boundary tokens")

    # Test segmentation
    test_texts = [
        "Hello world. How are you?",
        "The cat sat on the mat. The dog ran in the park. It was sunny.",
        "She went to the store and bought milk then she came home",
        "Good morning! How are you doing today? I hope you are well.",
    ]

    print("\n--- Sentence Segmentation ---")
    for text in test_texts:
        segments = segment(text, mc, boundary_ids, threshold=0.25)
        print(f"\n  Input: {text}")
        for i, seg in enumerate(segments, 1):
            print(f"    [{i}] {seg}")


if __name__ == "__main__":
    main()
