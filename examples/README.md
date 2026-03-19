# Examples

Run any example from the repo root:

```bash
uv run python examples/01_char_text_generation.py
```

## Examples

| # | Script | Use Case |
|---|--------|----------|
| 01 | `01_char_text_generation.py` | Character-level text generation from nursery rhymes |
| 02 | `02_word_text_generation.py` | Word-level text generation and greedy predictions |
| 03 | `03_hmm_supervised_tagging.py` | Supervised HMM for weather/activity tagging |
| 04 | `04_hmm_unsupervised_clustering.py` | Baum-Welch HMM for DNA sequence clustering |
| 05 | `05_music_chord_prediction.py` | Chord progression prediction and generation |
| 06 | `06_streaming_large_corpus.py` | Memory-efficient streaming training |
| 07 | `07_quantization_benchmark.py` | INT8 quantization and Python vs ONNX benchmarks |
| 08 | `08_config_from_env.py` | Programmatic and environment variable configuration |
| 09 | `09_perplexity_comparison.py` | Perplexity comparison across model orders |
| 10 | `10_language_identifier.py` | Language ID via perplexity ensemble (EN/FR/DE/ES/PT) |
| 11 | `11_intent_classifier_benchmark.py` | Intent classification benchmark (5504 utterances, 11 classes) |
| 12 | `12_pos_tagger.py` | POS tagging with supervised HMM Viterbi decoding |
| 13 | `13_keyword_extraction.py` | Keyword extraction via inverse frequency surprise |
| 14 | `14_sentence_segmentation.py` | Sentence boundary detection with char-level models |
| 15 | `15_grapheme_to_phoneme.py` | G2P conversion with supervised HMM |

## Toy Datasets

| File | Description |
|------|-------------|
| `data/nursery_rhymes.txt` | 27 lines of nursery rhymes (character/word models) |
| `data/weather_tagged.txt` | 20 pipe-separated activity\|weather sequences (supervised HMM) |
| `data/dna_sequences.txt` | 20 DNA strings of A/C/G/T (unsupervised HMM) |
| `data/music_chords.txt` | 20 chord progressions using C/Am/F/G/Dm/Em (word model) |
| `data/lang_train_en.txt` | 25 English sentences (language ID training) |
| `data/lang_train_fr.txt` | 25 French sentences (language ID training) |
| `data/lang_train_de.txt` | 25 German sentences (language ID training) |
| `data/lang_train_es.txt` | 25 Spanish sentences (language ID training) |
| `data/lang_train_pt.txt` | 25 Portuguese sentences (language ID training) |
