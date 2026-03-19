# Getting Started

## Install

```bash
pip install -e ".[test]"
```

## Quickstart: Markov Chain

```python
from markovonnx import Vocabulary, MarkovChain, export_markov_onnx, MarkovONNXRuntime

# 1. Build vocabulary
vocab = Vocabulary()
corpus = [list("the cat sat on the mat")] * 100
vocab.build_from_sequences(corpus)

# 2. Train
mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
mc.fit(corpus)

# 3. Export to ONNX
export_markov_onnx(mc, "markov.onnx")

# 4. Inference via ONNX Runtime
rt = MarkovONNXRuntime("markov.onnx", vocab, order=2)

# Greedy next token
print(rt.argmax(["t", "h"]))

# Probabilistic sampling
print(rt.sample(["t", "h"], temperature=0.8))

# Full probability distribution
probs = rt.predict_probs(["t", "h"])
```

## Quickstart: HMM

```python
from markovonnx import Vocabulary, HiddenMarkovModel, export_hmm_onnx, HMMONNXRuntime

# 1. Build observation vocabulary
obs_vocab = Vocabulary()
obs_vocab.build_from_sequences([["walk", "shop", "clean"]])

# 2. Supervised training from labelled data
hmm = HiddenMarkovModel(n_states=3, obs_vocab=obs_vocab)
hmm.fit_supervised(
    obs_seqs=[["walk", "shop", "clean"], ["walk", "walk", "shop"]],
    tag_seqs=[["sunny", "sunny", "rainy"], ["sunny", "sunny", "sunny"]],
)

# 3. Viterbi decoding (pure Python)
states = hmm.viterbi(["walk", "shop", "clean"])
print(states)  # e.g. ['sunny', 'sunny', 'rainy']

# 4. Export + ONNX inference
export_hmm_onnx(hmm, "hmm.onnx")
rt = HMMONNXRuntime("hmm.onnx", hmm)
states = rt.decode(["walk", "shop", "clean"])
```

## Quickstart: Text Generation

```python
from markovonnx import generate_markov

text = generate_markov(
    ort_model=rt,       # MarkovONNXRuntime
    seed="the",
    length=50,
    temperature=0.7,
    mode="char",        # or "word" or "bpe"
    order=2,
)
print(text)
```

## Training on a File (Streaming)

For large corpora that don't fit in memory:

```python
from markovonnx import Vocabulary, MarkovChain, char_tokenize

vocab = Vocabulary(max_vocab=500)
vocab.build_streaming("corpus.txt", tokenize_fn=char_tokenize)

mc = MarkovChain(order=2, vocab=vocab)
mc.fit_streaming("corpus.txt", tokenize_fn=char_tokenize)
```

Both `Vocabulary.build_streaming` and `MarkovChain.fit_streaming` read the file line-by-line, never loading the full corpus into RAM.

## INT8 Quantization

```python
from markovonnx import quantize_model

result = quantize_model("markov.onnx", "markov_int8.onnx")
if result:
    print(f"Quantized model saved to {result}")
```

Typically achieves ~75% file size reduction with minimal accuracy loss.

## Kneser-Ney Smoothing + Backoff

For best accuracy on small datasets:

```python
mc = MarkovChain(order=2, vocab=vocab, kneser_ney=True, backoff=True)
mc.fit(corpus)
```

Kneser-Ney typically gives 10-15% better accuracy than Laplace. Backoff falls back to shorter contexts for unseen n-grams.

## Save / Load Archives

Bundle a trained model into a single portable file:

```python
from markovonnx import save_markov_archive, load_markov_archive

save_markov_archive(mc, "model.markov")

loaded = load_markov_archive("model.markov")
rt = loaded["runtime"]
text = generate_markov(rt, "the", length=50, mode="char", order=2)
```

## CLI

```bash
markovonnx train corpus.txt -o model.markov --order 3 --backoff
markovonnx generate model.markov --seed "the" --length 100
markovonnx info model.markov
```
