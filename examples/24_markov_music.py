#!/usr/bin/env python3
"""Example: Music generation using Markov chains on note sequences.

Encodes melodies as sequences of note tokens (C4, D4, E4, etc.) and
trains a Markov chain to generate new melodies that follow the
statistical patterns of the training data.
"""

import random

from markovonnx import MarkovChain, Vocabulary

# Simple melodies encoded as note sequences
# Format: note + octave (e.g., C4 = middle C)
MELODIES = [
    # Twinkle Twinkle
    ["C4", "C4", "G4", "G4", "A4", "A4", "G4", "R",
     "F4", "F4", "E4", "E4", "D4", "D4", "C4", "R"],
    # Mary Had a Little Lamb
    ["E4", "D4", "C4", "D4", "E4", "E4", "E4", "R",
     "D4", "D4", "D4", "R", "E4", "G4", "G4", "R"],
    # Ode to Joy (simplified)
    ["E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
     "C4", "C4", "D4", "E4", "E4", "D4", "D4", "R"],
    # Simple blues riff
    ["C4", "E4", "G4", "A4", "G4", "E4", "C4", "R",
     "C4", "D4", "E4", "G4", "E4", "D4", "C4", "R"],
    # Ascending scale pattern
    ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
     "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4"],
    # Pentatonic melody
    ["C4", "D4", "E4", "G4", "A4", "G4", "E4", "D4",
     "C4", "E4", "G4", "A4", "C5", "A4", "G4", "E4"],
    # Simple waltz
    ["C4", "E4", "G4", "E4", "C4", "E4", "G4", "R",
     "D4", "F4", "A4", "F4", "D4", "F4", "A4", "R"],
    # Minor key melody
    ["A3", "B3", "C4", "D4", "E4", "D4", "C4", "B3",
     "A3", "C4", "E4", "A4", "E4", "C4", "A3", "R"],
    # Jazz-ish pattern
    ["C4", "E4", "G4", "B4", "A4", "F4", "D4", "R",
     "D4", "F4", "A4", "C5", "B4", "G4", "E4", "R"],
    # Folk-style
    ["G3", "A3", "B3", "D4", "G4", "D4", "B3", "A3",
     "G3", "B3", "D4", "G4", "A4", "G4", "D4", "B3"],
]


def note_to_display(notes: list) -> str:
    """Format notes for display with bars."""
    bars = []
    current_bar = []
    for n in notes:
        current_bar.append(n)
        if len(current_bar) == 4:
            bars.append(" ".join(f"{x:>3s}" for x in current_bar))
            current_bar = []
    if current_bar:
        bars.append(" ".join(f"{x:>3s}" for x in current_bar))
    return " | ".join(bars)


def main() -> None:
    random.seed(42)

    vocab = Vocabulary()
    vocab.build_from_sequences(MELODIES)
    print(f"--- Markov Music Generator ---\n")
    print(f"  Notes: {sorted(set(n for m in MELODIES for n in m))}")
    print(f"  Training melodies: {len(MELODIES)}")

    # Train with different orders
    for order in [1, 2, 3]:
        mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5, kneser_ney=True)
        mc.fit(MELODIES)

        print(f"\n  Order {order} — generated melodies:")
        for i in range(3):
            # Start with a random note
            seed = random.choice(["C4", "E4", "G4", "A3"])
            context = [seed] * order
            melody = list(context)

            for _ in range(16 - order):
                nxt = mc.sample(context[-order:], temperature=0.7)
                melody.append(nxt)
                context = (context + [nxt])[-order:]

            # Take exactly 16 notes
            melody = melody[:16]
            print(f"    {i + 1}. {note_to_display(melody)}")

    # Transition analysis
    print(f"\n  --- Note Transition Analysis (order=1) ---")
    mc1 = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
    mc1.fit(MELODIES)

    common_notes = ["C4", "D4", "E4", "F4", "G4", "A4"]
    print(f"\n  {'From':<6s}", end="")
    for n in common_notes:
        print(f"{n:>6s}", end="")
    print()
    for src in common_notes:
        probs = mc1._get_probs([src])
        print(f"  {src:<6s}", end="")
        for dst in common_notes:
            idx = vocab.tok2id.get(dst, 0)
            p = float(probs[idx]) if idx < len(probs) else 0
            print(f"{p:>6.0%}", end="")
        print()


if __name__ == "__main__":
    main()
