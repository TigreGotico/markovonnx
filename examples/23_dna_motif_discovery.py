#!/usr/bin/env python3
"""Example: DNA sequence motif discovery using Markov chain statistics.

Train on DNA sequences and identify conserved vs variable positions
by examining transition probabilities. Positions where one nucleotide
dominates (high probability) are conserved; positions with uniform
distribution are variable.

Use cases: transcription factor binding sites, protein domain signatures,
phylogenetic analysis, primer design.
"""

import numpy as np

from markovonnx import MarkovChain, Vocabulary, char_tokenize

# Simulated DNA sequences with a conserved motif at positions 10-16
# Motif: TATAAAT (TATA box — a real promoter element)
import random


def generate_sequence(length: int = 30, motif: str = "TATAAAT",
                      motif_pos: int = 10) -> str:
    """Generate a DNA sequence with a conserved motif."""
    bases = "ACGT"
    seq = [random.choice(bases) for _ in range(length)]
    # Insert motif with slight variation
    for i, base in enumerate(motif):
        pos = motif_pos + i
        if pos < length:
            if random.random() < 0.85:  # 85% conservation
                seq[pos] = base
            else:
                seq[pos] = random.choice(bases)
    return "".join(seq)


def main() -> None:
    random.seed(42)

    # Generate training sequences
    n_seqs = 100
    motif = "TATAAAT"
    motif_pos = 10
    seq_len = 30

    sequences = [generate_sequence(seq_len, motif, motif_pos) for _ in range(n_seqs)]
    print(f"--- DNA Motif Discovery ---\n")
    print(f"  Generated {n_seqs} sequences of length {seq_len}")
    print(f"  Hidden motif: {motif} at position {motif_pos}")
    print(f"  Conservation rate: ~85%")
    print(f"\n  Sample sequences:")
    for seq in sequences[:5]:
        # Highlight motif region
        before = seq[:motif_pos]
        region = seq[motif_pos:motif_pos + len(motif)]
        after = seq[motif_pos + len(motif):]
        print(f"    {before}[{region}]{after}")

    # Train order-1 model
    corpus = [char_tokenize(s) for s in sequences]
    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    # Analyze per-position entropy (conservation)
    print(f"\n  Position-wise conservation analysis:")
    print(f"  {'Pos':<5s} {'A':>6s} {'C':>6s} {'G':>6s} {'T':>6s} {'Entropy':>8s} {'Cons?':>6s}")
    print(f"  {'─' * 43}")

    for pos in range(seq_len):
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        for seq in sequences:
            if pos < len(seq):
                counts[seq[pos]] = counts.get(seq[pos], 0) + 1
        total = sum(counts.values())
        probs = {b: c / total for b, c in counts.items()}

        # Shannon entropy
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * np.log2(p)

        # Max entropy for 4 bases = 2.0
        conserved = "***" if entropy < 1.0 else ""
        prob_str = "  ".join(f"{probs[b]:.2f}" for b in "ACGT")
        print(f"  {pos:<5d} {prob_str} {entropy:>8.2f} {conserved:>6s}")

    # Find the motif region
    print(f"\n  Detected conserved region (entropy < 1.0):")
    conserved_positions = []
    for pos in range(seq_len):
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        for seq in sequences:
            if pos < len(seq):
                counts[seq[pos]] += 1
        total = sum(counts.values())
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * np.log2(p)
        if entropy < 1.0:
            dominant = max(counts, key=counts.get)
            conserved_positions.append((pos, dominant))

    if conserved_positions:
        start = conserved_positions[0][0]
        end = conserved_positions[-1][0]
        motif_found = "".join(b for _, b in conserved_positions)
        print(f"    Positions {start}-{end}: {motif_found}")
        print(f"    Expected:             {motif}")
    else:
        print(f"    No strongly conserved region found")


if __name__ == "__main__":
    main()
