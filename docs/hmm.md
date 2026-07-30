# Hidden Markov Models

`HiddenMarkovModel` (`markovonnx/hmm.py:10`)

Discrete HMM for sequence tagging, NER, G2P, and unsupervised clustering.

## Constructor

```python
HiddenMarkovModel(
    n_states: int,
    obs_vocab: Vocabulary,
    state_vocab: Optional[Vocabulary] = None,
    smoothing: float = 1e-5,
)
```

| Parameter | Description |
|-----------|-------------|
| `n_states` | Number of hidden states |
| `obs_vocab` | Observation `Vocabulary` |
| `state_vocab` | State `Vocabulary` (built automatically in supervised mode if `None`) |
| `smoothing` | Laplace alpha for count smoothing |

## Parameters (Learned)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `pi` | `[n_states]` | Initial state probabilities |
| `A` | `[n_states, n_states]` | State transition matrix |
| `B` | `[n_states, obs_vocab.size]` | Emission matrix |

All three are proper probability distributions (rows sum to 1.0) after training.

## Supervised Training

### `fit_supervised(obs_seqs, tag_seqs)` (`hmm.py:38`)

Trains via Maximum Likelihood Estimation from labelled `(observation, tag)` pairs.

```python
hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
hmm.fit_supervised(
    obs_seqs=[["walk", "shop", "clean"]],
    tag_seqs=[["sunny", "sunny", "rainy"]],
)
```

If `state_vocab` is `None`, it is built automatically from `tag_seqs`. After fitting, `n_states` is updated to match `state_vocab.size`.

## Unsupervised Training (Baum-Welch)

### `fit_unsupervised(obs_seqs, n_iter=10)` (`hmm.py:77`)

Trains via the Baum-Welch (forward-backward EM) algorithm. No labels required.

```python
hmm = HiddenMarkovModel(n_states=4, obs_vocab=obs_vocab)
hmm.fit_unsupervised(observation_sequences, n_iter=20)
```

Initialises parameters randomly (seed 42 for reproducibility). Prints log-likelihood at each iteration. This value should increase monotonically if the model is converging.

**Implementation details** (`hmm.py:96-143`):
1. Forward pass with scaling to prevent underflow
2. Backward pass
3. Gamma (state posteriors) and Xi (transition posteriors) computation
4. M-step: re-estimate pi, A, B from sufficient statistics

Sequences shorter than 2 tokens are skipped.

## Viterbi Decoding

### `viterbi(obs_seq) -> List[str]` (`hmm.py:147`)

Finds the most likely state sequence via the Viterbi algorithm (log-space).

```python
states = hmm.viterbi(["walk", "shop", "clean"])
# ['sunny', 'sunny', 'rainy']
```

Returns string state names if `state_vocab` is set, otherwise stringified indices.

## Supervised vs Unsupervised

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| Requires labels | Yes | No |
| Method | MLE counting | Baum-Welch EM |
| `state_vocab` | Auto-built or provided | Not used |
| Convergence | Single pass | Iterative (monitor log-likelihood) |
| Use case | POS tagging, NER, G2P | Clustering, pattern discovery |

---
[← Markov Chains](markov-chains.md) · [Home](index.md) · [ONNX Export →](onnx-export.md)
