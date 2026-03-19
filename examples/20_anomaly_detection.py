#!/usr/bin/env python3
"""Example: Anomaly detection via perplexity thresholding.

Train a Markov chain on "normal" log lines. Lines with perplexity above
a threshold are flagged as anomalous. High perplexity = the model hasn't
seen patterns like this = suspicious.

Use cases: intrusion detection, error log monitoring, fraud detection,
manufacturing defect detection, network traffic analysis.
"""

from markovonnx import MarkovChain, Vocabulary, word_tokenize

NORMAL_LOGS = [
    "user alice logged in from 192.168.1.10",
    "user bob logged in from 192.168.1.20",
    "user alice accessed file report.pdf",
    "user bob accessed file budget.xlsx",
    "user alice logged out",
    "user bob logged out",
    "user carol logged in from 192.168.1.30",
    "user carol accessed file notes.txt",
    "user carol logged out",
    "system backup completed successfully",
    "system health check passed all tests",
    "user alice logged in from 192.168.1.10",
    "user alice changed password successfully",
    "user bob logged in from 192.168.1.20",
    "user bob accessed file presentation.pptx",
    "system disk usage at 45 percent normal",
    "system memory usage at 62 percent normal",
    "user dave logged in from 192.168.1.40",
    "user dave accessed file schedule.csv",
    "user dave logged out",
] * 5  # repeat for more training data

TEST_LINES = [
    # Normal
    ("user alice logged in from 192.168.1.10", False),
    ("user bob accessed file document.docx", False),
    ("system backup completed successfully", False),
    ("user carol logged out", False),
    ("system health check passed all tests", False),
    # Anomalous
    ("user root logged in from 10.0.0.1", True),
    ("user alice deleted database production", True),
    ("failed login attempt user admin from 45.33.22.11", True),
    ("system process unexpected binary executed", True),
    ("user unknown escalated privileges to root", True),
    ("port scan detected from external ip 203.0.113.5", True),
    ("user bob transferred 500gb to external server", True),
]


def main() -> None:
    # Train on normal logs
    sequences = [word_tokenize(line) for line in NORMAL_LOGS]
    vocab = Vocabulary()
    vocab.build_from_sequences(sequences)

    mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5, kneser_ney=True)
    mc.fit(sequences)

    # Compute perplexity threshold from training data
    train_ppxs = [mc.perplexity([word_tokenize(line)]) for line in NORMAL_LOGS[:20]]
    mean_ppx = sum(train_ppxs) / len(train_ppxs)
    threshold = mean_ppx * 3  # 3x mean = anomaly threshold

    print("--- Anomaly Detection via Perplexity ---\n")
    print(f"  Training: {len(NORMAL_LOGS)} normal log lines")
    print(f"  Vocab: {vocab.size} tokens")
    print(f"  Mean normal PPX: {mean_ppx:.1f}")
    print(f"  Anomaly threshold: {threshold:.1f} (3x mean)\n")

    tp = fp = tn = fn = 0
    for line, is_anomaly in TEST_LINES:
        tokens = word_tokenize(line)
        ppx = mc.perplexity([tokens])
        predicted_anomaly = ppx > threshold

        if predicted_anomaly and is_anomaly:
            tp += 1
        elif predicted_anomaly and not is_anomaly:
            fp += 1
        elif not predicted_anomaly and not is_anomaly:
            tn += 1
        else:
            fn += 1

        label = "ANOMALY" if predicted_anomaly else "normal "
        expected = "anomaly" if is_anomaly else "normal"
        ok = "✓" if predicted_anomaly == is_anomaly else "✗"
        print(f"  {ok} [{label}] ppx={ppx:>8.1f}  {line}")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    print(f"\n  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}")


if __name__ == "__main__":
    main()
