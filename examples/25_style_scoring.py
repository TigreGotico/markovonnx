#!/usr/bin/env python3
"""Example: Text style scoring — measure how well text matches a style.

Train one Markov chain per style (formal, casual, technical, poetic).
Score new text by perplexity under each model. The ratio between styles
tells you which style the text matches best.

Use cases: content moderation (is this formal enough?), writing
assistance (does this sound professional?), brand voice consistency.
"""

from typing import Dict, List, Tuple

from markovonnx import MarkovChain, Vocabulary, word_tokenize

STYLES = {
    "formal": [
        "we would like to inform you of the following changes",
        "please find attached the quarterly report for your review",
        "i am writing to express my sincere gratitude",
        "the committee has reached a unanimous decision regarding",
        "pursuant to our previous correspondence we wish to clarify",
        "it is with great pleasure that we announce the appointment",
        "we respectfully request your consideration of this proposal",
        "the board of directors has approved the following resolution",
        "your prompt attention to this matter would be appreciated",
        "we hereby acknowledge receipt of your communication dated",
        "the undersigned parties agree to the terms set forth herein",
        "we are pleased to confirm the successful completion of",
    ],
    "casual": [
        "hey whats up just checking in on you",
        "lol that was so funny you totally nailed it",
        "gonna grab some coffee want to come along",
        "dude that movie was absolutely insane last night",
        "nah i think im gonna skip that one honestly",
        "yeah for sure lets do it this weekend sounds fun",
        "oh man i totally forgot about that my bad",
        "haha no way thats crazy i cant believe it",
        "cool cool ill text you later about the plans",
        "btw did you see what happened at the game",
        "yo check this out its pretty awesome right",
        "whatever its fine dont worry about it seriously",
    ],
    "technical": [
        "the algorithm has time complexity of o n log n",
        "implement the interface using dependency injection pattern",
        "the api endpoint returns a json response with pagination",
        "configure the load balancer for round robin distribution",
        "the database schema requires normalization to third normal form",
        "deploy the microservice using containerized kubernetes pods",
        "the memory leak was caused by unreleased file handles",
        "refactor the legacy codebase to use async await pattern",
        "the regression test suite covers ninety five percent of branches",
        "optimize the query by adding a composite index on columns",
        "the cache invalidation strategy uses time to live expiry",
        "implement rate limiting using the token bucket algorithm",
    ],
    "poetic": [
        "the golden light of dawn spills across the silent meadow",
        "whispers of the ancient trees echo through the hollow vale",
        "beneath the silver moon the rivers sing their endless song",
        "petals fall like tears from blossoms weeping in the rain",
        "the wind carries memories of summers long since passed",
        "in the garden of forgotten dreams the roses still bloom",
        "shadows dance upon the wall like spirits of the night",
        "the ocean breathes a sigh that reaches every distant shore",
        "stars are scattered diamonds sewn upon the velvet sky",
        "time flows like a river carrying us to unknown seas",
        "the mountain stands eternal witness to the passing age",
        "morning dew upon the grass reflects a thousand tiny suns",
    ],
}


def main() -> None:
    # Build shared vocab and train per-style models
    all_seqs = []
    for samples in STYLES.values():
        all_seqs.extend([word_tokenize(s) for s in samples])
    vocab = Vocabulary()
    vocab.build_from_sequences(all_seqs)

    models: Dict[str, MarkovChain] = {}
    for style, samples in STYLES.items():
        seqs = [word_tokenize(s) for s in samples]
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5, kneser_ney=True)
        mc.fit(seqs)
        models[style] = mc

    # Test texts
    tests = [
        "we would like to schedule a meeting at your earliest convenience",
        "yo that party was wild you shoulda been there dude",
        "the distributed system uses consistent hashing for partition tolerance",
        "beneath the weeping willow the old pond reflects the clouds",
        "please review the attached document and provide your feedback",
        "lol im literally dying this is the funniest thing ever",
        "deploy the updated container image to the staging cluster",
        "the autumn leaves descend like whispered prayers to earth below",
    ]

    print("--- Text Style Scoring ---\n")
    print(f"  Styles: {', '.join(STYLES.keys())}")
    print(f"  Shared vocab: {vocab.size} words\n")

    for text in tests:
        tokens = word_tokenize(text)
        scores: List[Tuple[str, float]] = []
        for style, mc in models.items():
            ppx = mc.perplexity([tokens])
            scores.append((style, ppx))
        scores.sort(key=lambda x: x[1])
        best = scores[0][0]

        # Normalize to percentages (inverse perplexity)
        inv = [(s, 1.0 / max(p, 1e-10)) for s, p in scores]
        total = sum(v for _, v in inv)
        pcts = [(s, 100 * v / total) for s, v in inv]
        pcts.sort(key=lambda x: -x[1])

        pct_str = "  ".join(f"{s}={p:.0f}%" for s, p in pcts)
        print(f"  [{best:>9s}] \"{text[:55]}...\"")
        print(f"              {pct_str}\n")


if __name__ == "__main__":
    main()
