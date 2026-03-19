"""Tests for markovonnx.hmm."""

import numpy as np

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.vocabulary import Vocabulary


def _obs_vocab() -> Vocabulary:
    vocab = Vocabulary()
    vocab.build_from_sequences([["a", "b", "c"]])
    return vocab


class TestHMMSupervised:
    def test_fit_supervised(self) -> None:
        obs_vocab = _obs_vocab()
        hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
        obs_seqs = [["a", "b", "c"], ["a", "a", "b"]]
        tag_seqs = [["X", "Y", "Z"], ["X", "X", "Y"]]
        hmm.fit_supervised(obs_seqs, tag_seqs)
        assert hmm.state_vocab is not None
        assert hmm.A.shape[0] == hmm.n_states

    def test_viterbi_returns_correct_length(self) -> None:
        obs_vocab = _obs_vocab()
        hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
        obs_seqs = [["a", "b", "c"]] * 5
        tag_seqs = [["X", "Y", "Z"]] * 5
        hmm.fit_supervised(obs_seqs, tag_seqs)
        result = hmm.viterbi(["a", "b", "c"])
        assert len(result) == 3

    def test_matrices_are_probability_distributions(self) -> None:
        obs_vocab = _obs_vocab()
        hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
        obs_seqs = [["a", "b", "c"]] * 5
        tag_seqs = [["X", "Y", "Z"]] * 5
        hmm.fit_supervised(obs_seqs, tag_seqs)
        assert abs(hmm.pi.sum() - 1.0) < 1e-5
        for row in hmm.A:
            assert abs(row.sum() - 1.0) < 1e-5
        for row in hmm.B:
            assert abs(row.sum() - 1.0) < 1e-5


class TestHMMUnsupervised:
    def test_fit_unsupervised_converges(self) -> None:
        obs_vocab = _obs_vocab()
        hmm = HiddenMarkovModel(n_states=2, obs_vocab=obs_vocab)
        obs_seqs = [["a", "b", "a", "b"]] * 10
        hmm.fit_unsupervised(obs_seqs, n_iter=3)
        # Matrices should still be valid probability distributions
        assert abs(hmm.pi.sum() - 1.0) < 1e-4
        for row in hmm.A:
            assert abs(row.sum() - 1.0) < 1e-4
