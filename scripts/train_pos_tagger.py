#!/usr/bin/env python3
"""Train a POS tagger from NLTK's tagged corpora and save as pretrained model.

Usage:
    pip install nltk
    python scripts/train_pos_tagger.py --output models/en_pos.json --corpus brown

Output: a JSON file loadable by MarkovPosTagger via config:
    {"pretrained": {"en": "models/en_pos.json"}}
"""

import argparse
import sys
from typing import List, Tuple


def load_nltk_corpus(corpus_name: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Load a tagged corpus from NLTK.

    Returns (word_sequences, tag_sequences).
    """
    import nltk
    nltk.download(corpus_name, quiet=True)
    nltk.download("universal_tagset", quiet=True)

    if corpus_name == "brown":
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents(tagset="universal")
    elif corpus_name == "treebank":
        nltk.download("treebank", quiet=True)
        from nltk.corpus import treebank
        tagged_sents = treebank.tagged_sents(tagset="universal")
    elif corpus_name == "conll2000":
        nltk.download("conll2000", quiet=True)
        from nltk.corpus import conll2000
        tagged_sents = conll2000.tagged_sents(tagset="universal")
    else:
        raise ValueError(f"Unknown corpus: {corpus_name}. Use: brown, treebank, conll2000")

    word_seqs: List[List[str]] = []
    tag_seqs: List[List[str]] = []
    for sent in tagged_sents:
        words = [w.lower() for w, t in sent]
        tags = [t for w, t in sent]
        if words and tags:
            word_seqs.append(words)
            tag_seqs.append(tags)

    return word_seqs, tag_seqs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train POS tagger from NLTK corpus")
    parser.add_argument("-o", "--output", default="models/en_pos.json", help="Output model path")
    parser.add_argument("--corpus", default="brown", choices=["brown", "treebank", "conll2000"])
    parser.add_argument("--max-sents", type=int, default=0, help="Max sentences (0=all)")
    args = parser.parse_args()

    print(f"Loading NLTK corpus: {args.corpus}...")
    word_seqs, tag_seqs = load_nltk_corpus(args.corpus)
    if args.max_sents > 0:
        word_seqs = word_seqs[:args.max_sents]
        tag_seqs = tag_seqs[:args.max_sents]
    print(f"  Sentences: {len(word_seqs)}")
    print(f"  Tags: {sorted(set(t for ts in tag_seqs for t in ts))}")

    # Train
    from markovonnx import HiddenMarkovModel, Vocabulary
    import time

    obs_vocab = Vocabulary()
    obs_vocab.build_from_sequences(word_seqs)
    print(f"  Vocabulary: {obs_vocab.size} words")

    all_tags = set()
    for ts in tag_seqs:
        all_tags.update(ts)

    t0 = time.time()
    hmm = HiddenMarkovModel(n_states=len(all_tags) + 1, obs_vocab=obs_vocab)
    hmm.fit_supervised(word_seqs, tag_seqs)
    print(f"  Trained in {time.time() - t0:.2f}s")

    # Evaluate on last 10% as held-out
    split = int(len(word_seqs) * 0.9)
    correct = total = 0
    for words, gold_tags in zip(word_seqs[split:], tag_seqs[split:]):
        pred_tags = hmm.viterbi(words)
        for g, p in zip(gold_tags, pred_tags):
            total += 1
            if g == p:
                correct += 1
    acc = correct / max(total, 1)
    print(f"  Accuracy (held-out 10%): {acc:.1%} ({correct}/{total})")

    # Save
    hmm.save(args.output)
    from pathlib import Path
    size_kb = Path(args.output).stat().st_size / 1024
    print(f"  Saved: {args.output} ({size_kb:.0f} KB)")
    print(f"\nTo use in OVOS config:")
    print(f'  {{"postag": {{"module": "ovos-markov-postag", "ovos-markov-postag": {{"pretrained": {{"en": "{args.output}"}}}}}}}}')


if __name__ == "__main__":
    main()
