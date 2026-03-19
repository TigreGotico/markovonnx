#!/usr/bin/env python3
"""Example: Spam/ham text classifier using perplexity ensemble.

Trains one word-level Markov chain on spam text and another on legitimate
(ham) text. Classifies new messages by which model assigns lower perplexity.
"""

from typing import Dict, List, Tuple

from markovonnx import MarkovChain, Vocabulary, word_tokenize

SPAM_SAMPLES = [
    "congratulations you have won a free iphone click here now",
    "act now limited time offer buy one get one free",
    "you have been selected for a special prize claim now",
    "free money no credit check apply today guaranteed approval",
    "earn thousands from home with this one simple trick",
    "urgent your account has been compromised click to verify",
    "hot singles in your area want to meet you tonight",
    "lose weight fast with this miracle pill order now",
    "make money online no experience needed start earning today",
    "your lottery ticket has won claim your million dollars now",
    "exclusive deal just for you save up to ninety percent",
    "work from home and earn six figures in just weeks",
    "free trial no obligation cancel anytime sign up now",
    "you are a winner click below to collect your reward",
    "limited offer get rich quick with zero investment needed",
    "congratulations you qualify for a free vacation package today",
    "double your income guaranteed with this secret method",
    "act fast this deal expires in twenty four hours only",
    "free gift card waiting for you claim before midnight tonight",
    "amazing discount on luxury watches buy now pay later",
]

HAM_SAMPLES = [
    "hey are you coming to the meeting at three today",
    "can you pick up some groceries on your way home",
    "the project deadline has been moved to next friday",
    "thanks for sending the report it looks great",
    "do you want to grab lunch tomorrow at the cafe",
    "i finished reviewing the pull request looks good to merge",
    "the kids have soccer practice at five this afternoon",
    "can we reschedule our call to wednesday morning instead",
    "just saw the news about the new office location",
    "happy birthday hope you have a wonderful day today",
    "the flight is delayed by two hours check the app",
    "did you see the email from the client about changes",
    "i will be working from home tomorrow let me know",
    "the restaurant reservation is for seven thirty tonight",
    "great presentation today the team really liked your slides",
    "can you send me the updated spreadsheet when ready",
    "the weather looks nice this weekend for a hike",
    "i just pushed the fix to the development branch",
    "mom called she wants to know about thanksgiving plans",
    "the plumber is coming between ten and noon tomorrow",
]


def main() -> None:
    # Build shared vocabulary
    all_seqs = ([word_tokenize(s) for s in SPAM_SAMPLES] +
                [word_tokenize(s) for s in HAM_SAMPLES])
    vocab = Vocabulary()
    vocab.build_from_sequences(all_seqs)

    # Train models
    spam_mc = MarkovChain(order=1, vocab=vocab, kneser_ney=True)
    spam_mc.fit([word_tokenize(s) for s in SPAM_SAMPLES])

    ham_mc = MarkovChain(order=1, vocab=vocab, kneser_ney=True)
    ham_mc.fit([word_tokenize(s) for s in HAM_SAMPLES])

    # Classify test messages
    test_messages = [
        ("free iphone giveaway click to claim your prize now", "spam"),
        ("hey can we meet for coffee tomorrow morning", "ham"),
        ("earn money fast no experience needed apply today", "spam"),
        ("the meeting has been rescheduled to four pm", "ham"),
        ("congratulations you won a free trip to hawaii", "spam"),
        ("i will pick up the kids after school today", "ham"),
        ("limited time discount on all electronics buy now", "spam"),
        ("thanks for helping with the presentation yesterday", "ham"),
        ("your account needs immediate verification click here", "spam"),
        ("do you want me to bring anything to the party", "ham"),
    ]

    print("--- Spam/Ham Classifier ---\n")
    correct = 0
    for text, expected in test_messages:
        tokens = word_tokenize(text)
        spam_ppx = spam_mc.perplexity([tokens])
        ham_ppx = ham_mc.perplexity([tokens])
        predicted = "spam" if spam_ppx < ham_ppx else "ham"
        ok = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        print(f"  {ok} [{predicted:>4s}] {text}")
        print(f"         spam_ppx={spam_ppx:.1f}  ham_ppx={ham_ppx:.1f}")

    print(f"\nAccuracy: {correct}/{len(test_messages)} ({100*correct/len(test_messages):.0f}%)")


if __name__ == "__main__":
    main()
