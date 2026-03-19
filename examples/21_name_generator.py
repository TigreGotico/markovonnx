#!/usr/bin/env python3
"""Example: Name generation using character-level Markov chains.

Train on lists of names (fantasy, baby names, place names) and generate
plausible new names that follow the statistical patterns of the originals.
"""

import random
from typing import List

from markovonnx import MarkovChain, Vocabulary, char_tokenize

FANTASY_NAMES = [
    "aragorn", "legolas", "gandalf", "gimli", "boromir", "faramir",
    "eowyn", "arwen", "galadriel", "elrond", "theoden", "eomer",
    "saruman", "sauron", "frodo", "samwise", "merry", "pippin",
    "celeborn", "haldir", "treebeard", "radagast", "glorfindel",
    "thranduil", "thorin", "balin", "dwalin", "bifur", "bofur",
    "denethor", "beregond", "imrahil", "lothiriel", "finduilas",
    "cirdan", "celebrimbor", "feanor", "fingolfin", "turgon",
    "idril", "earendil", "elwing", "luthien", "beren", "tuor",
    "morwen", "hurin", "turin", "nienor", "beleg", "mablung",
]

PLACE_NAMES = [
    "gondor", "rohan", "mordor", "rivendell", "lothlorien", "mirkwood",
    "isengard", "minas tirith", "minas morgul", "osgiliath", "edoras",
    "helms deep", "fangorn", "ithilien", "arnor", "angmar", "rhovanion",
    "harad", "umbar", "erebor", "dale", "laketown", "bree",
    "weathertop", "dunharrow", "pelargir", "dol amroth", "belfalas",
    "cirith ungol", "barad dur", "mount doom", "shelob lair",
    "grey havens", "mithlond", "forlindon", "shire", "hobbiton",
]

SCIFI_NAMES = [
    "zarkon", "voltron", "xerath", "pylex", "kronos", "nova",
    "andromeda", "cypher", "nexus", "zenith", "quasar", "vortex",
    "axiom", "synth", "helix", "prism", "flux", "nebula",
    "cortex", "zephyr", "apex", "onyx", "aether", "quantum",
    "phoenix", "hydra", "titan", "atlas", "orion", "lyra",
    "cassian", "jorah", "theron", "kaelen", "zara", "nyxa",
    "dragen", "kyros", "velara", "morinth", "aldric", "seraphine",
]


def train_and_generate(
    name: str,
    names: List[str],
    order: int = 3,
    n_generate: int = 15,
    temperature: float = 0.7,
) -> List[str]:
    """Train on names and generate new ones."""
    # Add start/end markers
    corpus = [char_tokenize("^" + n + "$") for n in names]
    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    generated = []
    attempts = 0
    while len(generated) < n_generate and attempts < 200:
        attempts += 1
        # Start with "^" marker
        context = ["^"] * order
        result = []
        for _ in range(25):  # max name length
            nxt = mc.sample(context[-order:], temperature)
            if nxt == "$":
                break
            if nxt != "^":
                result.append(nxt)
            context = (context + [nxt])[-order:]

        name_str = "".join(result)
        # Filter: reasonable length, not a duplicate
        if 3 <= len(name_str) <= 15 and name_str not in names and name_str not in generated:
            generated.append(name_str)

    return generated


def main() -> None:
    random.seed(42)

    datasets = [
        ("Fantasy Character Names", FANTASY_NAMES, 3, 0.7),
        ("Place Names", PLACE_NAMES, 3, 0.8),
        ("Sci-Fi Names", SCIFI_NAMES, 2, 0.8),
    ]

    for title, names, order, temp in datasets:
        print(f"\n--- {title} ---")
        print(f"  Training on {len(names)} names (order={order}, temp={temp})")
        generated = train_and_generate(title, names, order, n_generate=15, temperature=temp)
        for i, name in enumerate(generated, 1):
            print(f"  {i:>2d}. {name}")


if __name__ == "__main__":
    main()
