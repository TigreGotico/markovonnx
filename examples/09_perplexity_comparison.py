#!/usr/bin/env python3
"""Example: Comparing perplexity across different model orders.

Trains Markov chains of order 1 through 4 and compares perplexity
on held-out evaluation data.
"""

from pathlib import Path

from markovonnx import MarkovChain, Vocabulary, char_tokenize


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "nursery_rhymes.txt")

    # 1. Load corpus
    with open(data_path, encoding="utf-8") as f:
        corpus = [char_tokenize(line) for line in f if line.strip()]

    # Split: 80% train, 20% eval
    split = int(len(corpus) * 0.8)
    train = corpus[:split]
    eval_data = corpus[split:]
    print(f"Train: {len(train)} sequences, Eval: {len(eval_data)} sequences")

    # 2. Build vocabulary from training data
    vocab = Vocabulary()
    vocab.build_from_sequences(train)
    print(f"Vocabulary: {vocab.size} symbols\n")

    # 3. Train and evaluate for each order
    print(f"{'Order':<8} {'Train PPX':<15} {'Eval PPX':<15} {'Contexts':<12}")
    print("-" * 50)

    for order in range(1, 5):
        # Filter sequences long enough for this order
        train_filtered = [s for s in train if len(s) > order]
        eval_filtered = [s for s in eval_data if len(s) > order]

        if not train_filtered or not eval_filtered:
            print(f"{order:<8} (insufficient data)")
            continue

        mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5)
        mc.fit(train_filtered)

        train_ppx = mc.perplexity(train_filtered[:50])
        eval_ppx = mc.perplexity(eval_filtered)
        n_contexts = len(mc._counts)

        print(f"{order:<8} {train_ppx:<15.2f} {eval_ppx:<15.2f} {n_contexts:<12}")

    print("\nLower perplexity = better model fit.")
    print("Watch for eval perplexity increasing at higher orders (overfitting).")


if __name__ == "__main__":
    main()
