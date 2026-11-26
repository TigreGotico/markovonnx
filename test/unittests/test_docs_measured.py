"""Every number the docs quote equals the value in docs/results/measured.json."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "docs/results/measured.json").read_text())


def pct(correct, total):
    return f"{100 * correct / total:.1f}%"


def claims():
    brown = pct(M["pos_brown"]["correct"], M["pos_brown"]["total"])
    tb = pct(M["pos_treebank"]["correct"], M["pos_treebank"]["total"])
    q = M["quantize"]
    smaller = f"{100 * (1 - q['int8_kb'] / q['full_kb']):.1f}% smaller"
    share = f"{100 * q['int8_kb'] / q['full_kb']:.1f}% of full precision"
    ld = M["lang_detect"]
    lang = f"{100 * ld['correct'] / ld['total']:.0f}% ({ld['correct']}/{ld['total']})"
    acc = M["intent"]["accuracy_pct"]
    kn = (f"order 1 with Kneser-Ney gets {acc['order=1, Kneser-Ney']}% accuracy and "
          f"order 1 with Laplace gets {acc['order=1, Laplace']}%")
    sizes = f"({q['full_kb']} KB to {q['int8_kb']} KB)"
    return [
        ("README.md", f"reaches {brown} accuracy on a held-out 10% of the Brown corpus"),
        ("README.md", f"POS tagging ({brown} on Brown corpus)"),
        ("README.md", f"NLTK Brown corpus ({brown} accuracy)"),
        ("README.md", smaller),
        ("docs/getting-started.md", f"{smaller} {sizes}"),
        ("docs/onnx-export.md", f"{smaller} {sizes}"),
        ("docs/getting-started.md", kn),
        ("docs/markov-chains.md", kn),
        ("docs/guide.md", f"reaching {tb} accuracy on the NLTK Penn Treebank sample"),
        ("docs/guide.md", share),
        ("docs/guide.md", lang),
        ("docs/guide.md", f"{acc['order=1, Kneser-Ney']}% with order 1 and Kneser-Ney"),
        ("docs/guide.md", f"{brown} on Brown and {tb} on the NLTK Penn Treebank sample"),
    ]


@pytest.mark.parametrize("path,text", claims())
def test_the_docs_quote_the_measured_value(path, text):
    assert text in (REPO / path).read_text()
