#!/usr/bin/env python3
"""Example: Streaming training for large corpora.

Demonstrates vocabulary building and model training without loading
the entire corpus into memory.
"""

import tempfile
from pathlib import Path

from markovonnx import (
    MarkovChain,
    MarkovONNXRuntime,
    Vocabulary,
    char_tokenize,
    corpus_iter,
    export_markov_onnx,
    generate_markov,
)


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "nursery_rhymes.txt")

    # 1. Stream vocabulary building (never loads full file)
    vocab = Vocabulary(max_vocab=50)
    vocab.build_streaming(data_path, tokenize_fn=char_tokenize)
    print(f"Vocabulary (top 50 + UNK): {vocab.size} symbols")
    print(f"Top 10: {vocab.id2tok[1:11]}")

    # 2. Stream training
    mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
    mc.fit_streaming(data_path, tokenize_fn=char_tokenize)

    # 3. Count tokens via streaming iterator
    total_tokens = 0
    for tokens in corpus_iter(data_path, char_tokenize):
        total_tokens += len(tokens)
    print(f"Total tokens in corpus: {total_tokens}")

    # 4. Demonstrate max_lines limit
    limited = list(corpus_iter(data_path, char_tokenize, max_lines=5))
    print(f"First 5 lines: {len(limited)} sequences")

    # 5. Export and generate
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "streamed.onnx")
        export_markov_onnx(mc, onnx_path)
        rt = MarkovONNXRuntime(onnx_path, vocab, order=2)

        text = generate_markov(rt, "th", length=100, temperature=0.6, mode="char", order=2)
        print(f"\nGenerated: {text}")


if __name__ == "__main__":
    main()
