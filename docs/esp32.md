# ESP32 / Embedded Deployment

`export_markov_c_header()` — `markovonnx/c_export.py:33`

Exports a trained `MarkovChain` as a self-contained C99 header with zero external dependencies. No ONNX Runtime required. Targets Arduino ESP32 SDK, ESP-IDF, or any bare-metal C99 toolchain.

## Memory budget

| Config | Sparse rows | Vocab | Probs (int8) | Keys | Total flash |
|--------|------------|-------|-------------|------|-------------|
| char order=1, EN | ~80 | 80 | 6.4 KB | 0.6 KB | **~7 KB** |
| char order=2, EN | ~3 000 | 80 | 60 KB | 24 KB | **~84 KB** |
| word order=1, 500 vocab | 500 | 500 | 62 KB | 4 KB | **~66 KB** |

Character order=1 fits in IRAM (520 KB). Order=2 with int8 fits in flash (4 MB).

## Usage

```python
from markovonnx import MarkovChain, Vocabulary, export_markov_c_header

mc = MarkovChain(order=1, vocab=vocab)
mc.fit(sequences)
export_markov_c_header(mc, "markov_model.h")                      # uint8, no PROGMEM
export_markov_c_header(mc, "markov_model.h", quantize=False)      # float32
export_markov_c_header(mc, "markov_model.h", progmem=True)        # ESP32 .rodata section
```

CLI:

```bash
markovonnx train corpus.txt -o model.markov --export-c markov_model.h
markovonnx train corpus.txt -o model.markov --export-c markov_model.h --no-quantize --progmem
```

## Generated header structure

```c
#pragma once
#include <stdint.h>

#define MARKOV_VOCAB_SIZE  80
#define MARKOV_ORDER       1
#define MARKOV_SPARSE_ROWS 76

static const char* MARKOV_VOCAB[MARKOV_VOCAB_SIZE] = { "<UNK>", "a", "b", ... };
static const uint64_t MARKOV_KEYS[MARKOV_SPARSE_ROWS] = { ... };   // sorted
static const uint8_t  MARKOV_PROBS[MARKOV_SPARSE_ROWS][MARKOV_VOCAB_SIZE] = { ... };

// Binary search — O(log N)
static inline const uint8_t* markov_lookup(uint64_t key);

// CDF walk — returns token id
static inline int markov_sample(const uint8_t* row, float r);
```

## Arduino/ESP32 sketch example

```cpp
#include "markov_model.h"

void setup() {
  Serial.begin(115200);
}

void loop() {
  // Pack context: last token id (order=1)
  uint64_t key = (uint64_t)last_token_id;
  const uint8_t* row = markov_lookup(key);
  // r must be in [0, 1)
  float r = (float)esp_random() / (float)UINT32_MAX;
  int next_id = markov_sample(row, r);
  Serial.println(MARKOV_VOCAB[next_id]);
  last_token_id = next_id;
  delay(500);
}
```

## Context key packing

For order *N*, each token ID occupies 16 bits in a `uint64_t`:

| Order | Bits used | Max vocab |
|-------|-----------|-----------|
| 1 | 16 | 65 535 |
| 2 | 32 | 65 535 |
| 3 | 48 | 65 535 |
| 4 | 64 | 65 535 |

Pack manually in C:

```c
// order=2: tokens [prev, curr]
uint64_t key = ((uint64_t)prev_id << 16) | (uint64_t)curr_id;
```

## API reference

`export_markov_c_header(chain, path, quantize=True, progmem=False)` — `c_export.py:33`

| Parameter | Default | Effect |
|-----------|---------|--------|
| `quantize` | `True` | `uint8_t` probs (4× smaller); within 1/255 of float32 |
| `progmem` | `False` | Adds `__attribute__((section(".rodata")))` to all arrays |

Raises `ValueError` if vocab size exceeds 65 535.

## HMM + Viterbi export

`export_hmm_c_header()` — `markovonnx/c_export.py:310`

Exports a trained `HiddenMarkovModel` as a C99 header containing pre-computed
log-probability matrices and an inline Viterbi decoder.

```python
from markovonnx import HiddenMarkovModel, export_hmm_c_header

hmm = HiddenMarkovModel(n_states=8, obs_vocab=obs_vocab)
hmm.fit_supervised(obs_seqs, tag_seqs)
export_hmm_c_header(hmm, "hmm_model.h")
export_hmm_c_header(hmm, "hmm_model.h", progmem=True)  # ESP32 flash
```

### Generated HMM header structure

```c
#define HMM_N_STATES  8
#define HMM_OBS_SIZE  50

static const char* HMM_OBS_VOCAB[HMM_OBS_SIZE] = { ... };
static const char* HMM_STATE_VOCAB[HMM_N_STATES] = { ... };

// Pre-computed log-probabilities (natural log, float32)
static const float HMM_LOG_PI[HMM_N_STATES] = { ... };
static const float HMM_LOG_A[HMM_N_STATES][HMM_N_STATES] = { ... };
static const float HMM_LOG_B[HMM_N_STATES][HMM_OBS_SIZE] = { ... };

// Inline Viterbi decoder — caller allocates delta, psi, path buffers
static inline float hmm_viterbi(
    const int* obs_ids, int T,
    float* delta, int* psi, int* path);
```

### Arduino/ESP32 sketch example (HMM)

```cpp
#include "hmm_model.h"

// Stack-allocate buffers for max sequence length
#define MAX_T 64
float delta[MAX_T * HMM_N_STATES];
int   psi  [MAX_T * HMM_N_STATES];
int   path [MAX_T];

void tag_sequence(int* obs_ids, int T) {
  float score = hmm_viterbi(obs_ids, T, delta, psi, path);
  for (int t = 0; t < T; t++) {
    Serial.printf("%s -> %s\n",
      HMM_OBS_VOCAB[obs_ids[t]],
      HMM_STATE_VOCAB[path[t]]);
  }
}
```

Log probabilities are pre-computed at export time — no `logf()` calls in the
Viterbi inner loop.  The HMM Viterbi function is O(T · S²) where S is the
number of states.

### HMM memory budget

| Config | HMM_LOG_A | HMM_LOG_B | Total flash |
|--------|-----------|-----------|-------------|
| 8 states, 50 obs | 1.3 KB | 1.6 KB | **~3 KB** |
| 16 states, 80 obs | 4 KB | 5 KB | **~9 KB** |
| 32 states, 200 obs | 16 KB | 25 KB | **~41 KB** |

## Limitations

- No dynamic allocation; unseen contexts fall back to token 0 (see `markov_sample` null check).
- No backoff chain in C; only the top-level model is exported.
- Vocab strings are UTF-8 char pointers; ensure flash encoding matches sketch charset.
- Use PlatformIO for ESP32 build integration; ESP-IDF CMake integration is out of scope.
