"""ONNX Runtime wrappers for Markov chain and HMM inference."""

import os
import warnings
from typing import List

import numpy as np
import onnxruntime as ort

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.vocabulary import Vocabulary


def _available_providers() -> List[str]:
    """Return ORT providers available on this machine, preferring CUDA."""
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def _make_session(onnx_path: str, providers: List[str] = None) -> ort.InferenceSession:
    """Create an ORT session with sensible defaults."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.cpu_count() or 1)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        providers = _available_providers()
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


class MarkovONNXRuntime:
    """Wraps an ORT session for Markov chain next-token prediction.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        vocab: :class:`Vocabulary` used during training.
        order: N-gram order matching the exported model.
    """

    def __init__(self, onnx_path: str, vocab: Vocabulary, order: int):
        self.vocab = vocab
        self.order = order
        self.sess = _make_session(onnx_path)
        self.provider = self.sess.get_providers()[0]

    @classmethod
    def from_file(cls, onnx_path: str) -> "MarkovONNXRuntime":
        """Load from an ONNX file, reconstructing vocab from embedded metadata.

        The ONNX file must have been exported with :func:`export_markov_onnx`,
        which embeds ``order``, ``vocab_size``, and ``vocab`` (first 500 tokens)
        in the model's metadata.

        Args:
            onnx_path: Path to the ``.onnx`` file.

        Returns:
            A ready-to-use runtime instance.
        """
        import json
        import onnx
        model = onnx.load(onnx_path)
        meta = {p.key: p.value for p in model.metadata_props}
        order = int(meta["order"])
        id2tok = json.loads(meta["vocab"])
        vocab_size = int(meta["vocab_size"])
        # Pad if the stored vocab was truncated (legacy models exported with 500-token cap)
        if len(id2tok) < vocab_size:
            import warnings
            warnings.warn(
                f"ONNX metadata contains {len(id2tok)} vocab tokens but model expects "
                f"{vocab_size}. Padding with placeholders. Re-export for full vocab.",
                UserWarning,
                stacklevel=2,
            )
            while len(id2tok) < vocab_size:
                id2tok.append(f"<TOKEN_{len(id2tok)}>")
        vocab = Vocabulary()
        vocab.id2tok = id2tok
        vocab.tok2id = {tok: i for i, tok in enumerate(id2tok)}
        return cls(onnx_path, vocab, order)

    def predict_probs(self, context: List) -> np.ndarray:
        """Return the full probability vector over the vocabulary.

        Args:
            context: Token sequence (at least *order* tokens).
        """
        ids = np.array(self.vocab.encode(context[-self.order :]), dtype=np.int64)
        return self.sess.run(["probs"], {"input_ids": ids})[0]

    def predict_probs_batch(self, contexts: List[List]) -> np.ndarray:
        """Return probability vectors for a batch of contexts.

        Args:
            contexts: List of token sequences, each at least *order* tokens.

        Returns:
            Array of shape ``(len(contexts), vocab_size)``.
        """
        batch = np.array(
            [self.vocab.encode(ctx[-self.order :]) for ctx in contexts],
            dtype=np.int64,
        )
        return np.stack([
            self.sess.run(["probs"], {"input_ids": row})[0]
            for row in batch
        ])

    def sample(self, context: List, temperature: float = 1.0) -> object:
        """Sample the next token with optional temperature scaling.

        Args:
            context: Token sequence.
            temperature: Sampling temperature (1.0 = unscaled).
        """
        probs = self.predict_probs(context).astype(np.float64)
        if temperature != 1.0:
            logits = np.log(probs + 1e-30) / temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
        return self.vocab.id2tok[
            int(np.random.choice(len(probs), p=probs / probs.sum()))
        ]

    def argmax(self, context: List) -> object:
        """Return the greedy (deterministic) next token.

        Args:
            context: Token sequence.
        """
        ids = np.array(self.vocab.encode(context[-self.order :]), dtype=np.int64)
        nid = self.sess.run(["next_id"], {"input_ids": ids})[0]
        return self.vocab.id2tok[int(nid.flat[0])]


class HMMONNXRuntime:
    """Wraps an ORT session for HMM step-by-step forward-pass decoding.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        hmm: :class:`HiddenMarkovModel` used during training (for vocab / pi).
    """

    def __init__(self, onnx_path: str, hmm: HiddenMarkovModel):
        self.hmm = hmm
        self.sess = _make_session(onnx_path)

    def decode(self, obs_seq: List[str]) -> List[str]:
        """Decode an observation sequence via greedy forward pass.

        Args:
            obs_seq: Observation token sequence.

        Returns:
            Predicted state sequence.
        """
        alpha = self.hmm.pi.copy()
        states = []
        for tok in obs_seq:
            obs_id = np.array(
                [self.hmm.obs_vocab.tok2id.get(tok, 0)], dtype=np.int64
            )
            alpha, best = self.sess.run(
                ["alpha_out", "best_state"],
                {"obs_id": obs_id, "alpha_in": alpha},
            )
            states.append(int(best.flat[0]))
        if self.hmm.state_vocab:
            return self.hmm.state_vocab.decode(states)
        return [str(s) for s in states]
