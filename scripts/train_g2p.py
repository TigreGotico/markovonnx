#!/usr/bin/env python3
"""Train a G2P model from CMUDict (NLTK) and save as pretrained HMM.

Usage:
    pip install nltk
    python scripts/train_g2p.py --output models/en_g2p.json

Output: a JSON file loadable by MarkovG2P via config:
    {"pretrained": {"en": "models/en_g2p.json"}}

Note: CMUDict uses ARPABET phonemes, not IPA. The HMM maps single
characters to phonemes. Words where len(graphemes) != len(phonemes)
are aligned by padding the shorter sequence.
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def load_cmudict(max_words: int = 0) -> Tuple[List[List[str]], List[List[str]]]:
    """Load CMUDict entries and align grapheme-phoneme pairs.

    Returns (grapheme_sequences, phoneme_sequences) where each pair
    has equal length (padded if necessary).
    """
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict

    entries = cmudict.entries()
    graphemes: List[List[str]] = []
    phonemes: List[List[str]] = []

    for word, phones in entries:
        if max_words and len(graphemes) >= max_words:
            break
        chars = list(word.lower())
        # Strip stress markers from ARPABET
        clean_phones = [p.rstrip("012") for p in phones]

        # Simple alignment: pad shorter to match longer
        if len(chars) == len(clean_phones):
            graphemes.append(chars)
            phonemes.append(clean_phones)
        elif len(chars) < len(clean_phones):
            # Pad graphemes with placeholder
            padded = chars + ["_"] * (len(clean_phones) - len(chars))
            graphemes.append(padded)
            phonemes.append(clean_phones)
        else:
            # Pad phonemes
            padded = clean_phones + ["_"] * (len(chars) - len(clean_phones))
            graphemes.append(chars)
            phonemes.append(padded)

    return graphemes, phonemes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G2P from CMUDict")
    parser.add_argument("-o", "--output", default="models/en_g2p.json")
    parser.add_argument("--max-words", type=int, default=10000, help="Max dictionary entries")
    args = parser.parse_args()

    print("Loading CMUDict...")
    graphemes, phonemes = load_cmudict(args.max_words)
    print(f"  Entries: {len(graphemes)}")
    print(f"  Unique graphemes: {len(set(c for g in graphemes for c in g))}")
    print(f"  Unique phonemes: {len(set(p for ps in phonemes for p in ps))}")
    print(f"  Sample: {''.join(graphemes[0])} → {' '.join(phonemes[0])}")

    # Train
    from markovonnx import HiddenMarkovModel, Vocabulary
    import time

    obs_vocab = Vocabulary()
    obs_vocab.build_from_sequences(graphemes)

    all_phones = set()
    for ps in phonemes:
        all_phones.update(ps)

    t0 = time.time()
    hmm = HiddenMarkovModel(n_states=len(all_phones) + 1, obs_vocab=obs_vocab)
    hmm.fit_supervised(graphemes, phonemes)
    print(f"  Trained in {time.time() - t0:.2f}s")

    # Quick eval
    correct = total = 0
    for g, p in zip(graphemes[-100:], phonemes[-100:]):
        pred = hmm.viterbi(g)
        for gold, predicted in zip(p, pred):
            total += 1
            if gold == predicted:
                correct += 1
    print(f"  Phoneme accuracy (last 100 words): {correct / max(total, 1):.1%}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    hmm.save(args.output)
    size_kb = Path(args.output).stat().st_size / 1024
    print(f"  Saved: {args.output} ({size_kb:.0f} KB)")

    # Demo
    print(f"\n  Demo predictions:")
    for word in ["hello", "world", "python", "computer", "music"]:
        chars = list(word)
        pred = hmm.viterbi(chars)
        print(f"    {word} → {' '.join(pred)}")

    print(f"\nOVOS config:")
    print(f'  {{"g2p": {{"module": "ovos-markov-g2p", "ovos-markov-g2p": {{"pretrained": {{"en": "{args.output}"}}}}}}}}')


if __name__ == "__main__":
    main()
