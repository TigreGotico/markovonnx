"""Measure the numbers that the docs quote and write docs/results/measured.json.

Run: python scripts/measure_docs.py  (needs nltk; it downloads the Brown and
Penn Treebank samples). The docs test compares the docs with that file.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "results" / "measured.json"


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True,
                          text=True, check=True, input="quit\n").stdout


def pos(corpus, tmp):
    out = run("scripts/train_pos_tagger.py", "--corpus", corpus, "-o", f"{tmp}/{corpus}.json")
    correct, total = map(int, re.search(r"\((\d+)/(\d+)\)", out).groups())
    return {"script": "scripts/train_pos_tagger.py", "corpus": corpus,
            "split": "held-out last 10% of sentences", "correct": correct, "total": total}


def quantize():
    out = run("examples/07_quantization_benchmark.py")
    full, int8 = (float(x) for x in re.search(r"Quantized: ([\d.]+) KB -> ([\d.]+) KB", out).groups())
    return {"script": "examples/07_quantization_benchmark.py", "full_kb": full, "int8_kb": int8}


def lang_detect():
    out = run("examples/10_language_identifier.py")
    correct, total = map(int, re.search(r"Accuracy: (\d+)/(\d+)", out).groups())
    return {"script": "examples/10_language_identifier.py", "correct": correct, "total": total}


def intent():
    out = run("examples/11_intent_classifier_benchmark.py")
    rows = re.findall(r"^(order=\d, [\w+-]+)\s+([\d.]+)%", out, re.M)
    return {"script": "examples/11_intent_classifier_benchmark.py", "data": "synthetic, 32 test utterances",
            "accuracy_pct": {name: float(acc) for name, acc in rows}}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data = {"pos_brown": pos("brown", tmp), "pos_treebank": pos("treebank", tmp),
                "quantize": quantize(), "lang_detect": lang_detect(), "intent": intent()}
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: {len(data)} measurements")


if __name__ == "__main__":
    main()
