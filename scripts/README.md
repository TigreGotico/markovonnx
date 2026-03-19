# Training Scripts

Scripts for training pretrained models that can be loaded by the OVOS plugins.

## Prerequisites

```bash
pip install nltk
```

## Scripts

### POS Tagger (from NLTK Brown/Treebank corpus)

```bash
python scripts/train_pos_tagger.py --corpus brown --output models/en_pos.json
```

Trains a supervised HMM on the NLTK Brown corpus with Universal POS tagset. Outputs a JSON file loadable by `MarkovPosTagger`.

**Corpora**: `brown` (57K sentences), `treebank` (3.9K), `conll2000` (10K)

### Language Detection (from NLTK UDHR or text files)

```bash
# From NLTK's Universal Declaration of Human Rights corpus
python scripts/train_lang_detect.py --nltk --langs en fr de es pt -o models/

# From custom text files
python scripts/train_lang_detect.py --langs en:data/english.txt fr:data/french.txt -o models/
```

### G2P (from CMUDict)

```bash
python scripts/train_g2p.py --output models/en_g2p.json --max-words 10000
```

Trains from CMUDict's 134K pronunciation entries. Uses ARPABET phonemes.

## Output Format

All scripts produce JSON files compatible with the `pretrained` config option:

```json
{
  "postag": {
    "module": "ovos-markov-postag",
    "ovos-markov-postag": {
      "pretrained": {"en": "models/en_pos.json"}
    }
  },
  "lang_detect": {
    "module": "ovos-markov-lang-detect",
    "ovos-markov-lang-detect": {
      "pretrained": {"en": "models/en_lm.json", "fr": "models/fr_lm.json"}
    }
  },
  "g2p": {
    "module": "ovos-markov-g2p",
    "ovos-markov-g2p": {
      "pretrained": {"en": "models/en_g2p.json"}
    }
  }
}
```
