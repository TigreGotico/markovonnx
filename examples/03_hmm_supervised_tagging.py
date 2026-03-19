#!/usr/bin/env python3
"""Example: Supervised HMM for weather/activity tagging.

Trains an HMM from labelled (activity, weather) pairs and uses Viterbi
decoding + ONNX forward-pass to predict weather states from activities.
"""

import tempfile
from pathlib import Path
from typing import List, Tuple

from markovonnx import (
    HiddenMarkovModel,
    HMMONNXRuntime,
    Vocabulary,
    export_hmm_onnx,
)


def load_tagged_data(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Load pipe-separated tagged data: 'obs|tag obs|tag ...'."""
    obs_seqs: List[List[str]] = []
    tag_seqs: List[List[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs = line.split()
            obs = [p.split("|")[0] for p in pairs]
            tags = [p.split("|")[1] for p in pairs]
            obs_seqs.append(obs)
            tag_seqs.append(tags)
    return obs_seqs, tag_seqs


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "weather_tagged.txt")

    # 1. Load tagged data
    obs_seqs, tag_seqs = load_tagged_data(data_path)
    print(f"Loaded {len(obs_seqs)} sequences")
    print(f"Sample: obs={obs_seqs[0]}, tags={tag_seqs[0]}")

    # 2. Build observation vocabulary
    obs_vocab = Vocabulary()
    obs_vocab.build_from_sequences(obs_seqs)
    print(f"Observation vocab: {obs_vocab.id2tok}")

    # 3. Train supervised HMM
    hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
    hmm.fit_supervised(obs_seqs, tag_seqs)

    print(f"\nLearned parameters:")
    print(f"  States: {hmm.state_vocab.id2tok}")
    print(f"  pi (initial): {hmm.pi}")
    print(f"  A (transitions):\n{hmm.A}")
    print(f"  B (emissions):\n{hmm.B}")

    # 4. Viterbi decoding (Python)
    test_seq = ["walk", "walk", "shop", "clean", "clean"]
    states = hmm.viterbi(test_seq)
    print(f"\nViterbi decode: {test_seq} → {states}")

    # 5. Export to ONNX and decode
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "weather_hmm.onnx")
        export_hmm_onnx(hmm, onnx_path)

        rt = HMMONNXRuntime(onnx_path, hmm)
        ort_states = rt.decode(test_seq)
        print(f"ONNX decode:   {test_seq} → {ort_states}")

    # 6. Try different sequences
    for seq in [
        ["walk", "walk", "walk"],
        ["clean", "clean", "clean"],
        ["shop", "walk", "clean", "shop"],
    ]:
        states = hmm.viterbi(seq)
        print(f"  {seq} → {states}")


if __name__ == "__main__":
    main()
