#!/usr/bin/env python3
"""Example: Intent classification benchmark on utterance_tags dataset.

Trains a perplexity ensemble classifier on 5504 utterances across 11
intent classes (COMMAND:ACTION, QUESTION:YESNO, etc.) and evaluates
accuracy, F1, and throughput across different model configurations.

Dataset: utterance_tags_v0.2.csv from guided-categorical-embeddings.
If the dataset is not found at the default path, a synthetic subset
is generated for demonstration.
"""

import csv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from markovonnx import MarkovChain, Vocabulary, word_tokenize

# Try to find the real dataset; fall back to synthetic
_REAL_DATA_PATH = (
    Path("/home/miro/PycharmProjects")
    / "Machine Learning Workspace"
    / "guided-categorical-embeddings"
    / "examples"
    / "questions_experiment_1"
    / "utterance_tags_v0.2.csv"
)

# Synthetic data for when real dataset is unavailable
_SYNTHETIC_DATA = [
    ("COMMAND:ACTION", [
        "turn on the lights", "activate the alarm", "start the dishwasher",
        "open the garage door", "lock the front door", "set the thermostat to 72",
        "turn off the television", "close the blinds", "start the washing machine",
        "activate the sprinklers", "switch on the fan", "dim the bedroom lights",
        "enable do not disturb mode", "turn up the volume", "mute the speakers",
        "play the next track", "pause the music", "start the coffee maker",
        "preheat the oven to 350", "activate night mode",
    ]),
    ("QUESTION:YESNO", [
        "is it going to rain today", "will the package arrive tomorrow",
        "can you set an alarm", "is the door locked", "are the lights on",
        "did I get any messages", "is the thermostat set correctly",
        "will it snow this weekend", "can you turn on the heater",
        "is my meeting still scheduled", "are there any updates",
        "has the laundry finished", "is the oven preheated", "can I cancel the order",
        "did the backup complete", "is the wifi working", "are there any alerts",
        "was the payment processed", "is there traffic on the highway",
        "should I bring an umbrella",
    ]),
    ("QUESTION:QUERY", [
        "what time is it", "what is the weather forecast",
        "how far is the nearest gas station", "what is the capital of france",
        "how many calories in a banana", "what is the exchange rate",
        "where is the nearest pharmacy", "who won the game last night",
        "when does the store close", "how long until my flight",
        "what temperature is it outside", "which route has less traffic",
        "how much battery is left", "what song is playing",
        "who is calling me", "when is the next bus", "where did I park",
        "what appointments do I have today", "how old is the earth",
        "what is the population of tokyo",
    ]),
    ("QUESTION:REQUEST", [
        "could you tell me the time", "would you check the weather",
        "can you find a restaurant nearby", "please look up this recipe",
        "could you read my messages", "would you play some jazz",
        "can you order more paper towels", "please remind me at five",
        "could you book a reservation", "would you call mom",
        "can you translate this to spanish", "please set a timer for ten minutes",
        "could you calculate the tip", "would you search for flights",
        "can you check my schedule", "please add milk to the grocery list",
        "could you summarize this article", "would you dim the lights",
        "can you recommend a movie", "please turn down the thermostat",
    ]),
    ("SENTENCE:STATEMENT", [
        "the weather is nice today", "I finished the report yesterday",
        "the meeting starts at three", "traffic was terrible this morning",
        "I left my keys on the table", "the new update looks great",
        "dinner is ready in twenty minutes", "the project deadline is friday",
        "I already bought the groceries", "the printer is out of ink",
        "my phone battery is low", "the package arrived this morning",
        "the restaurant was fully booked", "I sent the email an hour ago",
        "the garden needs watering", "my subscription expires next week",
        "the movie starts at eight", "I already have plans tonight",
        "the wifi password changed", "the dog needs to go for a walk",
    ]),
    ("COMMAND:DENIAL", [
        "no do not do that", "cancel that request", "stop playing music",
        "never mind forget it", "do not send that message",
        "cancel the alarm", "no I changed my mind", "stop the timer",
        "do not turn on the lights", "cancel my reservation",
        "no that is wrong", "stop the recording", "do not order that",
        "cancel the reminder", "no go back", "stop navigation",
        "do not call them", "cancel the download", "no I did not mean that",
        "skip this song",
    ]),
    ("SENTENCE:EXCLAMATION", [
        "that is amazing", "wow what a great deal", "I can not believe it",
        "this is incredible", "fantastic news", "oh no that is terrible",
        "what a beautiful sunset", "unbelievable performance",
        "how wonderful", "that is so cool", "absolutely brilliant",
        "oh my goodness", "what a surprise", "this is the best day ever",
        "incredible work everyone", "how exciting", "that was hilarious",
        "what a disaster", "simply outstanding", "I love this song",
    ]),
    ("SENTENCE:SOCIAL", [
        "good morning", "thank you so much", "have a great day",
        "nice to meet you", "see you later", "happy birthday",
        "congratulations on the promotion", "welcome home",
        "I appreciate your help", "take care of yourself",
        "enjoy your vacation", "it was nice talking to you",
        "please and thank you", "good luck with the interview",
        "thanks for letting me know", "well done on the presentation",
        "safe travels", "have a wonderful weekend", "cheers",
        "glad you made it",
    ]),
]


def load_real_data(path: str) -> List[Tuple[str, str]]:
    """Load (utterance, tag) pairs from CSV."""
    data: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                tag, sentence = row[0].strip(), row[1].strip()
                if tag and sentence:
                    data.append((sentence, tag))
    return data


def load_synthetic_data() -> List[Tuple[str, str]]:
    """Generate synthetic intent data for demonstration."""
    data: List[Tuple[str, str]] = []
    for tag, sentences in _SYNTHETIC_DATA:
        for s in sentences:
            data.append((s, tag))
    return data


def split_data(
    data: List[Tuple[str, str]], ratio: float = 0.8, seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Stratified train/test split."""
    rng = random.Random(seed)
    by_tag: Dict[str, List[str]] = {}
    for sentence, tag in data:
        by_tag.setdefault(tag, []).append(sentence)
    train, test = [], []
    for tag, sents in by_tag.items():
        rng.shuffle(sents)
        n = max(1, int(len(sents) * ratio))
        train.extend((s, tag) for s in sents[:n])
        test.extend((s, tag) for s in sents[n:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _normalize(text: str) -> str:
    return text.lower().strip()


def _ppx_to_confidence(ppx: float) -> float:
    import math
    if ppx <= 1.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.log(ppx))))


def train_and_evaluate(
    train: List[Tuple[str, str]],
    test: List[Tuple[str, str]],
    order: int,
    kneser_ney: bool,
    backoff: bool,
) -> Dict:
    """Train one MarkovChain per intent, evaluate on test set."""
    # Group by tag
    by_tag: Dict[str, List[List[str]]] = {}
    all_seqs: List[List[str]] = []
    for sentence, tag in train:
        tokens = word_tokenize(_normalize(sentence))
        by_tag.setdefault(tag, []).append(tokens)
        all_seqs.append(tokens)

    # Shared vocabulary
    vocab = Vocabulary()
    vocab.build_from_sequences(all_seqs)

    # Train models
    t0 = time.perf_counter()
    models: Dict[str, MarkovChain] = {}
    for tag, seqs in by_tag.items():
        mc = MarkovChain(
            order=order, vocab=vocab,
            smoothing=1e-5, kneser_ney=kneser_ney, backoff=backoff,
        )
        mc.fit(seqs)
        models[tag] = mc
    train_time = time.perf_counter() - t0

    # Evaluate
    correct = 0
    per_class_correct: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}

    t0 = time.perf_counter()
    for sentence, expected in test:
        tokens = word_tokenize(_normalize(sentence))
        if len(tokens) < order:
            per_class_total[expected] = per_class_total.get(expected, 0) + 1
            continue

        best_tag = None
        best_conf = -1.0
        for tag, mc in models.items():
            ppx = mc.perplexity([tokens])
            conf = _ppx_to_confidence(ppx)
            if conf > best_conf:
                best_conf = conf
                best_tag = tag

        per_class_total[expected] = per_class_total.get(expected, 0) + 1
        if best_tag == expected:
            correct += 1
            per_class_correct[expected] = per_class_correct.get(expected, 0) + 1
    eval_time = time.perf_counter() - t0

    accuracy = correct / max(len(test), 1)
    throughput = len(test) / max(eval_time, 1e-9)

    return {
        "order": order,
        "kneser_ney": kneser_ney,
        "backoff": backoff,
        "accuracy": accuracy,
        "train_ms": train_time * 1000,
        "throughput": throughput,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
        "vocab_size": vocab.size,
        "n_intents": len(models),
    }


def main() -> None:
    # Load data
    if _REAL_DATA_PATH.exists():
        print(f"Loading real dataset: {_REAL_DATA_PATH}")
        data = load_real_data(str(_REAL_DATA_PATH))
    else:
        print("Real dataset not found, using synthetic data for demo")
        data = load_synthetic_data()

    print(f"Total samples: {len(data)}")
    tag_counts = Counter(tag for _, tag in data)
    print(f"Classes: {len(tag_counts)}")
    for tag, count in tag_counts.most_common():
        print(f"  {tag:<30s} {count:>5d}")

    train, test = split_data(data)
    print(f"\nTrain: {len(train)}, Test: {len(test)}")

    # Benchmark configs
    configs = [
        (1, False, False, "order=1, Laplace"),
        (1, True,  False, "order=1, Kneser-Ney"),
        (2, False, False, "order=2, Laplace"),
        (2, True,  True,  "order=2, KN+backoff"),
    ]

    print(f"\n{'Config':<25s} {'Acc':<8s} {'Train':<10s} {'QPS':<8s} {'Vocab':<8s}")
    print("-" * 60)

    best_result = None
    for order, kn, bo, label in configs:
        r = train_and_evaluate(train, test, order, kn, bo)
        print(f"{label:<25s} {r['accuracy']:<8.1%} {r['train_ms']:<10.0f}ms "
              f"{r['throughput']:<8.0f} {r['vocab_size']:<8d}")
        if best_result is None or r["accuracy"] > best_result["accuracy"]:
            best_result = r

    # Per-class breakdown for best config
    print(f"\n--- Best config: order={best_result['order']}, "
          f"KN={best_result['kneser_ney']}, backoff={best_result['backoff']} ---")
    print(f"\n{'Intent':<30s} {'Correct':<10s} {'Total':<8s} {'Acc':<8s}")
    print("-" * 56)
    for tag in sorted(best_result["per_class_total"].keys()):
        c = best_result["per_class_correct"].get(tag, 0)
        t = best_result["per_class_total"][tag]
        print(f"{tag:<30s} {c:<10d} {t:<8d} {c/t:<8.1%}")

    print(f"\nOverall accuracy: {best_result['accuracy']:.1%}")
    print(f"Training time: {best_result['train_ms']:.0f}ms")
    print(f"Inference: {best_result['throughput']:.0f} queries/sec")


if __name__ == "__main__":
    main()
