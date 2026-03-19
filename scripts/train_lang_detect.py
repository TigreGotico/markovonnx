#!/usr/bin/env python3
"""Train language detection models from text files or NLTK corpora.

Usage:
    # From text files
    python scripts/train_lang_detect.py --langs en:data/en.txt fr:data/fr.txt -o models/

    # From NLTK (English + Europarl samples)
    python scripts/train_lang_detect.py --nltk --langs en es de fr pt -o models/

Output: one JSON per language, loadable by MarkovLangDetector via config:
    {"pretrained": {"en": "models/en_lm.json", "fr": "models/fr_lm.json"}}
"""

import argparse
import sys
from pathlib import Path


def train_from_file(lang: str, path: str, order: int, output_dir: str) -> None:
    """Train from a raw text file."""
    from markovonnx import MarkovChain, Vocabulary, char_tokenize

    with open(path, encoding="utf-8", errors="ignore") as f:
        corpus = [char_tokenize(line) for line in f if line.strip()]
    if not corpus:
        print(f"  {lang}: empty corpus, skipping")
        return

    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    out_path = str(Path(output_dir) / f"{lang}_lm.json")
    mc.save(out_path)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"  {lang}: {len(corpus)} lines, vocab={vocab.size}, saved {size_kb:.0f} KB")


def train_from_nltk(lang: str, order: int, output_dir: str, max_sents: int = 5000) -> None:
    """Train from NLTK's Europarl or other corpora."""
    import nltk
    from markovonnx import MarkovChain, Vocabulary, char_tokenize

    # Try udhr corpus (Universal Declaration of Human Rights — many languages)
    nltk.download("udhr2", quiet=True)
    from nltk.corpus import udhr

    lang_map = {
        "en": "English-Latin1", "fr": "French_Francais-Latin1",
        "de": "German_Deutsch-Latin1", "es": "Spanish-Latin1",
        "pt": "Portuguese_Portugues-Latin1", "it": "Italian-Latin1",
        "nl": "Dutch_Nederlands-Latin1", "da": "Danish_Dansk-Latin1",
        "sv": "Swedish_Svenska-Latin1", "fi": "Finnish_Suomi-Latin1",
        "ru": "Russian-UTF8", "zh": "Chinese_Mandarin-UTF8",
        "ar": "Arabic-UTF8", "ja": "Japanese-UTF8",
    }

    if lang not in lang_map:
        print(f"  {lang}: not available in UDHR, skipping. Available: {list(lang_map.keys())}")
        return

    try:
        text = udhr.raw(lang_map[lang])
    except Exception as e:
        print(f"  {lang}: failed to load UDHR ({e})")
        return

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if max_sents > 0:
        lines = lines[:max_sents]

    corpus = [char_tokenize(line) for line in lines]
    if not corpus:
        print(f"  {lang}: empty corpus, skipping")
        return

    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    out_path = str(Path(output_dir) / f"{lang}_lm.json")
    mc.save(out_path)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"  {lang}: {len(corpus)} lines, vocab={vocab.size}, saved {size_kb:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train language detection models")
    parser.add_argument("-o", "--output", default="models/", help="Output directory")
    parser.add_argument("--order", type=int, default=3, help="N-gram order")
    parser.add_argument("--langs", nargs="+", required=True,
                        help="Languages: 'en' (for NLTK) or 'en:path.txt' (for files)")
    parser.add_argument("--nltk", action="store_true", help="Use NLTK UDHR corpus")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"Training language models (order={args.order})...\n")

    for spec in args.langs:
        if ":" in spec and not args.nltk:
            lang, path = spec.split(":", 1)
            train_from_file(lang, path, args.order, args.output)
        else:
            lang = spec.split(":")[0]
            if args.nltk:
                train_from_nltk(lang, args.order, args.output)
            else:
                print(f"  {spec}: use --nltk or specify path as 'lang:path.txt'")

    # Print config snippet
    models = {p.stem.replace("_lm", ""): str(p)
              for p in Path(args.output).glob("*_lm.json")}
    if models:
        import json
        print(f"\nOVOS config snippet:")
        print(json.dumps({"pretrained": models}, indent=2))


if __name__ == "__main__":
    main()
