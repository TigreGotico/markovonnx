#!/usr/bin/env python3
"""Example: Authorship attribution — identify who wrote a text.

Trains one character-level Markov chain per author on their writing samples.
Classifies unknown text by lowest perplexity. Works because authors have
distinctive vocabulary, sentence structure, and character-level patterns.
"""

from typing import Dict, List, Tuple

from markovonnx import MarkovChain, Vocabulary, char_tokenize

AUTHORS = {
    "hemingway": [
        "the old man was thin and gaunt with deep wrinkles in the back of his neck",
        "he was an old man who fished alone in a skiff in the gulf stream",
        "everything about him was old except his eyes and they were the same color as the sea",
        "the sun was hot and he felt it on the back of his neck as he rowed",
        "he kept them moving steadily and it was not so much work because he was well in the current",
        "the fish pulled the line steady and the old man held it against his back",
        "he was happy feeling the gentle pulling and then he felt something hard and heavy",
        "the boy was sad looking at the old man come in each day with his skiff empty",
        "the old man had taught the boy to fish and the boy loved him",
        "now they sat on the terrace and many of the fishermen made fun of the old man",
    ],
    "poe": [
        "once upon a midnight dreary while i pondered weak and weary",
        "deep into that darkness peering long i stood there wondering fearing",
        "doubting dreaming dreams no mortal ever dared to dream before",
        "and the silken sad uncertain rustling of each purple curtain",
        "thrilled me filled me with fantastic terrors never felt before",
        "presently my soul grew stronger hesitating then no longer",
        "the raven sat upon the pallid bust of pallas just above my chamber door",
        "from the ashes of the dead the darkness crept into the chamber of my soul",
        "the boundaries between life and death are shadowy and vague",
        "all that we see or seem is but a dream within a dream",
    ],
    "austen": [
        "it is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife",
        "she was a woman of mean understanding little information and uncertain temper",
        "the business of her life was to get her daughters married",
        "i could easily forgive his pride if he had not mortified mine",
        "there are few people whom i really love and still fewer of whom i think well",
        "vanity and pride are different things though the words are often used synonymously",
        "a lady of high rank does not expect to be treated so shabbily",
        "i declare after all there is no enjoyment like reading",
        "happiness in marriage is entirely a matter of chance",
        "you must allow me to tell you how ardently i admire and love you",
    ],
}


def main() -> None:
    # Train per-author char-level models
    models: Dict[str, MarkovChain] = {}
    for author, samples in AUTHORS.items():
        corpus = [char_tokenize(s) for s in samples]
        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        mc = MarkovChain(order=4, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)
        models[author] = mc

    # Test passages (style-similar to training, not exact matches)
    test_passages = [
        ("the old man sat in the boat and the sea was calm and he waited", "hemingway"),
        ("he was tired but he kept rowing because that was what a man does", "hemingway"),
        ("and the darkness and decay held illimitable dominion over all", "poe"),
        ("the soul of man is restless and the shadows grow long at midnight", "poe"),
        ("she had neither beauty nor wit but she had a good heart and kind manner", "austen"),
        ("it is not what we say but what we do that reveals our true character", "austen"),
    ]

    print("--- Authorship Attribution ---\n")
    correct = 0
    for text, expected in test_passages:
        tokens = char_tokenize(text)
        scores: List[Tuple[str, float]] = []
        for author, mc in models.items():
            ppx = mc.perplexity([tokens])
            scores.append((author, ppx))
        scores.sort(key=lambda x: x[1])
        predicted = scores[0][0]
        ok = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1

        ppx_str = "  ".join(f"{a}={p:.0f}" for a, p in scores)
        print(f"  {ok} [{predicted:>10s}] \"{text[:60]}...\"")
        print(f"         {ppx_str}")

    print(f"\nAccuracy: {correct}/{len(test_passages)} ({100*correct/len(test_passages):.0f}%)")


if __name__ == "__main__":
    main()
