"""Discrete Hidden Markov Model for sequence tagging / G2P / NER."""

from typing import List, Optional

import numpy as np

from markovonnx.vocabulary import Vocabulary


def _logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp (like scipy.special.logsumexp).

    Returns ``-inf`` when all inputs are ``-inf`` (empty sum).
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    # Guard: if a_max is -inf, all values are -inf → result is -inf
    with np.errstate(invalid="ignore"):
        exp_sum = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
        out = a_max + np.log(exp_sum)
    # Replace NaN (from -inf + log(0)) with -inf
    out = np.where(np.isfinite(a_max), out, a_max)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


class HiddenMarkovModel:
    """Discrete HMM with supervised (MLE) and unsupervised (Baum-Welch) training.

    Baum-Welch operates entirely in log-space for numerical stability
    with long sequences.

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

    # -- Baum-Welch (unsupervised, log-space) ---------------------------------

    def fit_unsupervised(
        self,
        obs_seqs: List[List[str]],
        n_iter: int = 10,
    ) -> None:
        """Train via Baum-Welch (forward-backward EM) in log-space.

        All forward/backward computations use log-probabilities and
        :func:`_logsumexp` to avoid underflow on long sequences.

        Args:
            obs_seqs: Lists of observation token sequences.
            n_iter: Number of EM iterations.
        """
        S = self.n_states
        O = self.obs_vocab.size

        rng = np.random.default_rng(42)
        self.pi = rng.dirichlet(np.ones(S)).astype(np.float64)
        self.A = rng.dirichlet(np.ones(S), size=S).astype(np.float64)
        self.B = rng.dirichlet(np.ones(O), size=S).astype(np.float64)

        for it in range(n_iter):
            log_pi = np.log(self.pi + 1e-300)
            log_A = np.log(self.A + 1e-300)
            log_B = np.log(self.B + 1e-300)

            # Accumulators in log-space (initialised to log(smoothing))
            log_smooth = np.log(self.smoothing)
            log_pi_num = np.full(S, log_smooth)
            log_A_num = np.full((S, S), log_smooth)
            log_B_num = np.full((S, O), log_smooth)
            total_log_like = 0.0

            for obs_seq in obs_seqs:
                obs = self.obs_vocab.encode(obs_seq)
                T = len(obs)
                if T < 2:
                    continue

                # -- Log-space forward pass -----------------------------------
                log_alpha = np.full((T, S), -np.inf)
                log_alpha[0] = log_pi + log_B[:, obs[0]]

                for t in range(1, T):
                    # log_alpha[t, j] = log( sum_i alpha[t-1,i] * A[i,j] ) + log B[j, obs[t]]
                    # = logsumexp( log_alpha[t-1, :] + log_A[:, j] ) + log_B[j, obs[t]]
                    for j in range(S):
                        log_alpha[t, j] = (
                            _logsumexp(log_alpha[t - 1] + log_A[:, j])
                            + log_B[j, obs[t]]
                        )

                log_likelihood = float(_logsumexp(log_alpha[-1]))
                total_log_like += log_likelihood

                # -- Log-space backward pass ----------------------------------
                log_beta = np.full((T, S), -np.inf)
                log_beta[-1] = 0.0  # log(1)

                for t in range(T - 2, -1, -1):
                    for i in range(S):
                        log_beta[t, i] = _logsumexp(
                            log_A[i, :] + log_B[:, obs[t + 1]] + log_beta[t + 1]
                        )

                # -- Log-gamma: log P(state_t = i | observations) ------------
                log_gamma = log_alpha + log_beta
                log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)

                # Accumulate pi
                log_pi_num = np.logaddexp(log_pi_num, log_gamma[0])

                # Accumulate A: log_xi[t, i, j]
                for t in range(T - 1):
                    log_xi = (
                        log_alpha[t][:, None]
                        + log_A
                        + log_B[:, obs[t + 1]][None, :]
                        + log_beta[t + 1][None, :]
                    )
                    log_xi -= _logsumexp(log_xi.ravel())
                    log_A_num = np.logaddexp(log_A_num, log_xi)

                # Accumulate B
                for t in range(T):
                    log_B_num[:, obs[t]] = np.logaddexp(
                        log_B_num[:, obs[t]], log_gamma[t]
                    )

            # -- M-step: normalize accumulators in log-space ------------------
            self.pi = np.exp(log_pi_num - _logsumexp(log_pi_num)).astype(np.float32)
            self.A = np.exp(
                log_A_num - _logsumexp(log_A_num, axis=1, keepdims=True)
            ).astype(np.float32)
            self.B = np.exp(
                log_B_num - _logsumexp(log_B_num, axis=1, keepdims=True)
            ).astype(np.float32)
            print(
                f"  Baum-Welch iter {it + 1}/{n_iter}  "
                f"log-likelihood={total_log_like:.2f}"
            )

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
