#!/usr/bin/env python3
"""Example: Language identification via perplexity ensemble.

Trains one character-level Markov chain per language. To classify a text
sample, compute perplexity under each model — the language whose model
assigns the lowest perplexity (= highest likelihood) wins.

Languages: English, French, German, Spanish, Portuguese.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from markovonnx import MarkovChain, Vocabulary, char_tokenize


DATA_DIR = Path(__file__).parent / "data"

LANGUAGES = {
    "en": DATA_DIR / "lang_train_en.txt",
    "fr": DATA_DIR / "lang_train_fr.txt",
    "de": DATA_DIR / "lang_train_de.txt",
    "es": DATA_DIR / "lang_train_es.txt",
    "pt": DATA_DIR / "lang_train_pt.txt",
}

# Test sentences (not in training data)
TEST_SENTENCES = [
    ("en", "the weather is beautiful today and the birds are singing"),
    ("en", "she went to the library to borrow some interesting books"),
    ("fr", "je vais au supermarche pour acheter du pain et du beurre"),
    ("fr", "les enfants sont alles a la piscine cet apres midi"),
    ("de", "er hat gestern einen langen spaziergang im park gemacht"),
    ("de", "die kinder gehen jeden tag in die schule und lernen viel"),
    ("es", "ella fue al supermercado a comprar pan y leche fresca"),
    ("es", "los estudiantes trabajan mucho para aprobar los examenes"),
    ("pt", "ele foi ao supermercado comprar pao e leite fresco"),
    ("pt", "os estudantes trabalham muito para passar nos exames finais"),
    # Ambiguous / mixed-cue sentences
    ("en", "information technology is transforming modern communication"),
    ("fr", "la revolution industrielle a transforme la societe moderne"),
    ("de", "die industrielle revolution hat die moderne gesellschaft veraendert"),
    ("es", "la revolucion industrial transformo la sociedad moderna"),
    ("pt", "a revolucao industrial transformou a sociedade moderna"),
]


def load_corpus(path: Path) -> List[List[str]]:
    """Load a text file as character-tokenised sequences."""
    with open(path, encoding="utf-8") as f:
        return [char_tokenize(line) for line in f if line.strip()]


def train_ensemble(
    order: int = 3,
    smoothing: float = 1e-5,
) -> Dict[str, Tuple[Vocabulary, MarkovChain]]:
    """Train one Markov chain per language, returning {lang: (vocab, model)}."""
    ensemble: Dict[str, Tuple[Vocabulary, MarkovChain]] = {}

    for lang, path in LANGUAGES.items():
        corpus = load_corpus(path)

        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)

        mc = MarkovChain(order=order, vocab=vocab, smoothing=smoothing)
        mc.fit(corpus)

        ensemble[lang] = (vocab, mc)

    return ensemble


def classify(
    text: str,
    ensemble: Dict[str, Tuple[Vocabulary, MarkovChain]],
) -> Tuple[str, Dict[str, float]]:
    """Classify text by lowest perplexity across the ensemble.

    Returns (predicted_lang, {lang: perplexity}).
    """
    tokens = char_tokenize(text)
    scores: Dict[str, float] = {}

    for lang, (vocab, mc) in ensemble.items():
        # Re-encode with this language's vocabulary
        scores[lang] = mc.perplexity([tokens])

    predicted = min(scores, key=scores.get)
    return predicted, scores


def main() -> None:
    print("Training language models...\n")
    ensemble = train_ensemble(order=3, smoothing=1e-5)

    for lang, (vocab, mc) in ensemble.items():
        print(f"  [{lang}] vocab={vocab.size:>3}  contexts={len(mc._counts):>5}")

    # Classify test sentences
    print(f"\n{'True':<6} {'Pred':<6} {'OK':<4} {'Perplexities'}")
    print("-" * 80)

    correct = 0
    total = len(TEST_SENTENCES)

    for true_lang, sentence in TEST_SENTENCES:
        pred_lang, scores = classify(sentence, ensemble)
        ok = pred_lang == true_lang
        correct += int(ok)

        # Format perplexities (highlight winner)
        ppx_str = "  ".join(
            f"{lang}:{scores[lang]:>8.1f}{'*' if lang == pred_lang else ' '}"
            for lang in sorted(scores)
        )
        marker = "+" if ok else "X"
        print(f"  {true_lang:<4}   {pred_lang:<4}   {marker:<3}  {ppx_str}")

    accuracy = 100 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.0f}%)")

    # Interactive mode
    print("\n--- Try your own sentences ---")
    print("(Enter a sentence, or 'quit' to exit)\n")

    try:
        while True:
            text = input("> ").strip()
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            pred, scores = classify(text, ensemble)
            ranked = sorted(scores.items(), key=lambda x: x[1])
            print(f"  Predicted: {pred}")
            for lang, ppx in ranked:
                bar = "#" * max(1, int(50 / (ppx / ranked[0][1])))
                print(f"    {lang}: {ppx:>8.1f}  {bar}")
            print()
    except (EOFError, KeyboardInterrupt):
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
