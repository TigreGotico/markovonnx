# When to Use markovonnx

## What It Is

markovonnx is a library for training n-gram Markov chains and Hidden Markov Models, and exporting them to ONNX for fast inference. It is a classical statistical NLP toolkit. It uses no neural networks, no GPU, and no large model downloads.

## What's Different

Most Markov chain implementations exist as teaching tools or one-off scripts. markovonnx combines:

1. **ONNX export.** No other Markov chain library exports to ONNX. This enables deployment anywhere ONNX Runtime runs (mobile, edge, browser through WASM, embedded) without Python.
2. **Sparse ONNX graph.** The sparse export stores only observed n-gram rows. This makes models practical for large vocabularies where the full `V^N × V` matrix will not fit in memory.
3. **Perplexity-as-classifier.** The library supports using Markov chains as classifiers: one model per class, classify by lowest perplexity. This pattern works for language detection, intent classification, spam filtering, and authorship attribution, for any problem where you have text samples per category.
4. **Full HMM with log-space Baum-Welch.** The HMM implementation operates entirely in log-space. This prevents the underflow that affects most textbook implementations on sequences longer than about 50 tokens.
5. **Kneser-Ney smoothing.** The discount is auto-estimated from count-of-counts. Most n-gram libraries only offer Laplace smoothing or no smoothing.

## Use Cases

### Strong fits (use markovonnx)

| Use Case | Why | Example |
|----------|-----|---------|
| **Language identification** | Character n-gram perplexity is the classic approach: fast, accurate, tiny models | Example 10, `MarkovLangDetector` |
| **Intent classification (small data)** | 5-50 examples per class is enough for a word-level perplexity ensemble | Example 11, OVOS plugin `ovos-markov-pipeline-plugin` |
| **POS tagging** | HMM Viterbi is the textbook solution, reaching 96% accuracy on Penn Treebank | Example 12, `MarkovPosTagger` |
| **Sequence tagging (NER, BIO)** | Supervised HMM works well for entity extraction with labeled data | Example 3, `SlotExtractor` |
| **Text generation (creative)** | Character-level Markov chains produce entertaining, style-mimicking text | Examples 1-2 |
| **G2P (grapheme-to-phoneme)** | HMM maps character sequences to phoneme sequences | Example 15, `MarkovG2P` |
| **Edge/embedded deployment** | ONNX models are KB-sized with zero runtime dependencies | `export_markov_sparse_onnx` |
| **Offline/privacy-critical** | Everything runs locally, no API calls, no data leaves the device | All features |
| **Rapid prototyping** | Train in milliseconds, iterate instantly | CLI: `markovonnx train` |

### Weak fits (consider alternatives)

| Use Case | Why | Use Instead |
|----------|-----|-------------|
| **Semantic similarity** | Markov chains capture word co-occurrence, not meaning. "dog" and "puppy" look unrelated. | Sentence embeddings (Model2Vec, sentence-transformers) |
| **Long-range dependencies** | Order-N context is strictly local. Can't capture "the man who... *he*" coreference. | Transformers, RNNs |
| **Intent classification (large scale)** | With 100+ intents and ambiguous boundaries, perplexity doesn't discriminate well. | Fine-tuned BERT, Padatious with templates |
| **Machine translation** | No alignment model, no attention, no subword handling. | seq2seq, transformers |
| **Summarization** | Can't understand or compress meaning. | LLMs, extractive models |
| **Conversational AI** | No dialogue state, no grounding, no reasoning. | LLMs, dialogue managers |

## Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Training speed | 100-500 ms for 5000 sentences |
| Inference speed | 300-1000 queries/sec (Python), 3-4x faster via ONNX |
| Model size (dense) | `V^order × V × 4` bytes |
| Model size (sparse) | Proportional to observed n-grams only |
| Model size (quantized) | ~25% of full precision |
| Memory at inference | Model size + trivial overhead |
| Accuracy (lang detect) | 95-100% on 5+ languages with 25 training sentences each |
| Accuracy (intent, 8 classes) | 70-75% with Kneser-Ney on natural utterances |
| Accuracy (POS tagging) | ~96% on Penn Treebank with supervised HMM |

## Recommendations

- **Always use Kneser-Ney.** It is better than Laplace on every metric measured.
- **Use sparse export** for any model where `V^order > 10,000` rows
- **Use `.markov` archives** for model distribution instead of raw ONNX files
- **Evaluate with `calibration.find_optimal_thresholds()`** before deploying as a classifier

---
[Home](index.md) · [Getting Started →](getting-started.md)
