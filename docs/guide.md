# When to Use markovonnx

## What It Is

markovonnx is a library for training n-gram Markov chains and Hidden Markov Models, exporting them to ONNX for fast inference. It's a classical statistical NLP toolkit — no neural networks, no GPU, no large model downloads.

## What's Novel

Most Markov chain implementations exist as teaching tools or one-off scripts. markovonnx is unique in combining:

1. **ONNX export** — No other Markov chain library exports to ONNX. This enables deployment anywhere ONNX Runtime runs (mobile, edge, browser via WASM, embedded) without Python.

2. **Sparse ONNX graph** — The sparse export stores only observed n-gram rows, making models practical for large vocabularies where the full `V^N × V` matrix won't fit in memory.

3. **Perplexity-as-classifier** — The library is designed for using Markov chains as classifiers (one model per class, classify by lowest perplexity). This pattern works for language detection, intent classification, spam filtering, authorship attribution — any problem where you have text samples per category.

4. **Full HMM with log-space Baum-Welch** — The HMM implementation operates entirely in log-space, preventing the underflow that plagues most textbook implementations on sequences longer than ~50 tokens.

5. **Kneser-Ney smoothing** — Auto-estimated discount from count-of-counts. Most n-gram libraries only offer Laplace or no smoothing.

## Use Cases

### Strong fits (use markovonnx)

| Use Case | Why | Example |
|----------|-----|---------|
| **Language identification** | Character n-gram perplexity is the classic approach; fast, accurate, tiny models | Example 10, `MarkovLangDetector` |
| **Intent classification (small data)** | 5-50 examples per class is enough for word-level perplexity ensemble | Example 11; OVOS plugin: `ovos-markov-pipeline-plugin` |
| **POS tagging** | HMM Viterbi is the textbook solution; 96% accuracy on Penn Treebank | Example 12, `MarkovPosTagger` |
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

- **Always use Kneser-Ney** — it's strictly better than Laplace on every metric
- **Use sparse export** for any model where `V^order > 10,000` rows
- **Use `.markov` archives** for model distribution instead of raw ONNX files
- **Evaluate with `calibration.find_optimal_thresholds()`** before deploying as a classifier
