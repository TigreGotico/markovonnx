"""Discrete Hidden Markov Model for sequence tagging / G2P / NER."""

from typing import List, Optional

import numpy as np

from markovonnx.vocabulary import Vocabulary


class HiddenMarkovModel:
    """Discrete HMM with supervised (MLE) and unsupervised (Baum-Welch) training.

    Args:
        n_states: Number of hidden states.
        obs_vocab: Observation :class:`Vocabulary`.
        state_vocab: Optional state :class:`Vocabulary` (``None`` for unsupervised).
        smoothing: Laplace smoothing alpha.
    """

    def __init__(
        self,
        n_states: int,
        obs_vocab: Vocabulary,
        state_vocab: Optional[Vocabulary] = None,
        smoothing: float = 1e-5,
    ):
        self.n_states = n_states
        self.obs_vocab = obs_vocab
        self.smoothing = smoothing
        self.state_vocab = state_vocab

        self.pi = np.ones(n_states, dtype=np.float64) / n_states
        self.A = np.ones((n_states, n_states), dtype=np.float64) / n_states
        self.B = np.ones((n_states, obs_vocab.size), dtype=np.float64) / obs_vocab.size

    # -- Supervised MLE -------------------------------------------------------

    def fit_supervised(
        self,
        obs_seqs: List[List[str]],
        tag_seqs: List[List[str]],
    ) -> None:
        """Train from labelled (observation, tag) pairs via MLE.

        Args:
            obs_seqs: Lists of observation token sequences.
            tag_seqs: Corresponding lists of state/tag sequences.
        """
        if self.state_vocab is None:
            self.state_vocab = Vocabulary(max_vocab=0)
            self.state_vocab.build_from_sequences(tag_seqs)
        S = self.n_states = self.state_vocab.size
        O = self.obs_vocab.size

        pi_c = np.zeros(S) + self.smoothing
        A_c = np.zeros((S, S)) + self.smoothing
        B_c = np.zeros((S, O)) + self.smoothing

        for obs_seq, tag_seq in zip(obs_seqs, tag_seqs):
            obs_ids = self.obs_vocab.encode(obs_seq)
            tag_ids = self.state_vocab.encode(tag_seq)
            if not tag_ids:
                continue
            pi_c[tag_ids[0]] += 1
            for t in range(len(tag_ids) - 1):
                A_c[tag_ids[t], tag_ids[t + 1]] += 1
            for oid, sid in zip(obs_ids, tag_ids):
                B_c[sid, oid] += 1

        self.pi = (pi_c / pi_c.sum()).astype(np.float32)
        self.A = (A_c / A_c.sum(axis=1, keepdims=True)).astype(np.float32)
        self.B = (B_c / B_c.sum(axis=1, keepdims=True)).astype(np.float32)
        print(f"HMM supervised fit: {S} states, {O} observations")

    # -- Baum-Welch (unsupervised) --------------------------------------------

    def fit_unsupervised(
        self,
        obs_seqs: List[List[str]],
        n_iter: int = 10,
    ) -> None:
        """Train via Baum-Welch (forward-backward EM).

        Args:
            obs_seqs: Lists of observation token sequences.
            n_iter: Number of EM iterations.
        """
        S = self.n_states
        O = self.obs_vocab.size

        rng = np.random.default_rng(42)
        self.pi = rng.dirichlet(np.ones(S)).astype(np.float32)
        self.A = rng.dirichlet(np.ones(S), size=S).astype(np.float32)
        self.B = rng.dirichlet(np.ones(O), size=S).astype(np.float32)

        for it in range(n_iter):
            pi_num = np.zeros(S) + self.smoothing
            A_num = np.zeros((S, S)) + self.smoothing
            B_num = np.zeros((S, O)) + self.smoothing
            log_like = 0.0

            for obs_seq in obs_seqs:
                obs = self.obs_vocab.encode(obs_seq)
                T = len(obs)
                if T < 2:
                    continue
                # Forward
                alpha = np.zeros((T, S), dtype=np.float64)
                alpha[0] = self.pi * self.B[:, obs[0]]
                scale = np.zeros(T)
                scale[0] = alpha[0].sum() + 1e-300
                alpha[0] /= scale[0]
                for t in range(1, T):
                    alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
                    scale[t] = alpha[t].sum() + 1e-300
                    alpha[t] /= scale[t]
                log_like += np.log(scale + 1e-300).sum()
                # Backward
                beta = np.zeros((T, S), dtype=np.float64)
                beta[-1] = 1.0
                for t in range(T - 2, -1, -1):
                    beta[t] = (self.A * self.B[:, obs[t + 1]]) @ beta[t + 1]
                    beta[t] /= scale[t + 1]
                # Gamma / xi
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
                pi_num += gamma[0]
                for t in range(T - 1):
                    xi = (
                        alpha[t][:, None]
                        * self.A
                        * self.B[:, obs[t + 1]]
                        * beta[t + 1]
                    )
                    xi /= xi.sum() + 1e-300
                    A_num += xi
                for t in range(T):
                    B_num[:, obs[t]] += gamma[t]

            self.pi = (pi_num / pi_num.sum()).astype(np.float32)
            self.A = (A_num / A_num.sum(axis=1, keepdims=True)).astype(np.float32)
            self.B = (B_num / B_num.sum(axis=1, keepdims=True)).astype(np.float32)
            print(f"  Baum-Welch iter {it + 1}/{n_iter}  log-likelihood={log_like:.2f}")

    # -- Viterbi decoding -----------------------------------------------------

    def viterbi(self, obs_seq: List[str]) -> List[str]:
        """Find the most likely state sequence via Viterbi decoding.

        Args:
            obs_seq: Observation token sequence.

        Returns:
            Decoded state sequence (strings if *state_vocab* is set,
            otherwise stringified indices).
        """
        obs = self.obs_vocab.encode(obs_seq)
        T, S = len(obs), self.n_states
        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta = np.full((T, S), -np.inf)
        psi = np.zeros((T, S), dtype=int)
        delta[0] = log_pi + log_B[:, obs[0]]

        for t in range(1, T):
            for s in range(S):
                trans = delta[t - 1] + log_A[:, s]
                psi[t, s] = trans.argmax()
                delta[t, s] = trans.max() + log_B[s, obs[t]]

        path = [int(delta[-1].argmax())]
        for t in range(T - 1, 0, -1):
            path.append(psi[t, path[-1]])
        path.reverse()

        if self.state_vocab:
            return self.state_vocab.decode(path)
        return [str(s) for s in path]
