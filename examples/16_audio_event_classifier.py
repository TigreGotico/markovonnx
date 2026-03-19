#!/usr/bin/env python3
"""Example: Audio event classification using Markov chains on quantized features.

Trains one Markov chain per audio class on sequences of quantized energy
levels. Classifies new audio by comparing perplexity across all models —
the class whose model assigns the lowest perplexity wins.

This works because different audio events have characteristic energy
contour patterns:
  - Speech: sustained mid-high energy with rhythmic dips (syllables)
  - Dog bark: sharp burst → rapid decay → silence → burst
  - Doorbell: two tones at fixed intervals, clean energy envelope
  - Silence: sustained low energy with minor fluctuations
  - Music: sustained high energy with gradual, smooth transitions
  - Applause: noisy, sustained, mid-energy with fast fluctuations
  - Knock: very sharp, brief bursts with silence between

Use cases:
  - VAD (Voice Activity Detection): speech vs silence/noise
  - Smart home: doorbell, knock, glass break, smoke alarm
  - Pet monitoring: dog bark, cat meow
  - Baby monitor: crying vs babbling vs silence
  - Security: gunshot, glass break, scream
  - Wildlife: bird species by call pattern
  - Industrial: machine fault detection by vibration pattern
  - Music: genre classification by rhythm/energy patterns

The approach: quantize frame-level audio features (energy, spectral
centroid, etc.) into N discrete bins, then treat the bin sequence as
"text" for Markov chain training. Each audio class gets its own model.

This example uses synthetic energy contours (no real audio required).
Replace the synthetic data with real feature extraction for production.
"""

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from markovonnx import MarkovChain, Vocabulary


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic audio feature generators
#
# Each function returns a sequence of quantized energy levels (0-15)
# simulating the energy contour of that audio event.
#
# In production, replace these with real feature extraction:
#   librosa.feature.rms() → quantize → token sequence
# ═══════════════════════════════════════════════════════════════════════════

N_BINS = 16  # quantization levels (0 = silence, 15 = max energy)


def _quantize(values: np.ndarray) -> List[str]:
    """Quantize continuous values to discrete bin tokens."""
    clipped = np.clip(values, 0, 1)
    bins = (clipped * (N_BINS - 1)).astype(int)
    return [str(b) for b in bins]


def generate_speech(n_frames: int = 200) -> List[str]:
    """Simulate speech energy: rhythmic syllable pattern with pauses."""
    frames = []
    t = 0
    while t < n_frames:
        # Voiced segment (syllable): 5-15 frames of mid-high energy
        syllable_len = random.randint(5, 15)
        base_energy = random.uniform(0.4, 0.8)
        for i in range(min(syllable_len, n_frames - t)):
            # Smooth rise-fall within syllable
            phase = i / max(syllable_len - 1, 1)
            envelope = math.sin(phase * math.pi)
            frames.append(base_energy * envelope + random.gauss(0, 0.03))
        t += syllable_len

        # Inter-syllable dip: 2-5 frames of lower energy
        gap = random.randint(2, 5)
        for _ in range(min(gap, n_frames - t)):
            frames.append(random.uniform(0.05, 0.2))
        t += gap

        # Occasional pause (between words): 10-20 frames of near-silence
        if random.random() < 0.2:
            pause = random.randint(10, 20)
            for _ in range(min(pause, n_frames - t)):
                frames.append(random.uniform(0.0, 0.05))
            t += pause

    return _quantize(np.array(frames[:n_frames]))


def generate_silence(n_frames: int = 200) -> List[str]:
    """Simulate silence/ambient: low energy with minor noise."""
    base = random.uniform(0.01, 0.05)
    frames = [base + random.gauss(0, 0.01) for _ in range(n_frames)]
    return _quantize(np.array(frames))


def generate_dog_bark(n_frames: int = 200) -> List[str]:
    """Simulate dog barking: sharp bursts with silence between."""
    frames = []
    t = 0
    while t < n_frames:
        # Bark: sharp attack, fast decay (5-10 frames)
        bark_len = random.randint(5, 10)
        peak = random.uniform(0.7, 1.0)
        for i in range(min(bark_len, n_frames - t)):
            decay = math.exp(-3.0 * i / bark_len)
            frames.append(peak * decay + random.gauss(0, 0.02))
        t += bark_len

        # Silence between barks: 15-40 frames
        gap = random.randint(15, 40)
        for _ in range(min(gap, n_frames - t)):
            frames.append(random.uniform(0.0, 0.03))
        t += gap

    return _quantize(np.array(frames[:n_frames]))


def generate_doorbell(n_frames: int = 200) -> List[str]:
    """Simulate doorbell: two clean tones at fixed intervals."""
    frames = []
    t = 0
    for ring in range(2):
        # Tone: sustained energy, 20-30 frames
        tone_len = random.randint(20, 30)
        energy = random.uniform(0.5, 0.7)
        for i in range(min(tone_len, n_frames - t)):
            # Slight vibrato
            mod = 1.0 + 0.05 * math.sin(i * 0.5)
            frames.append(energy * mod)
        t += tone_len

        # Gap between tones: 15-25 frames
        gap = random.randint(15, 25)
        for _ in range(min(gap, n_frames - t)):
            frames.append(random.uniform(0.01, 0.04))
        t += gap

    # Trailing silence
    while len(frames) < n_frames:
        frames.append(random.uniform(0.0, 0.03))

    return _quantize(np.array(frames[:n_frames]))


def generate_music(n_frames: int = 200) -> List[str]:
    """Simulate music: sustained energy with smooth, slow transitions."""
    base = random.uniform(0.3, 0.6)
    frames = []
    energy = base
    for _ in range(n_frames):
        # Slow random walk
        energy += random.gauss(0, 0.015)
        energy = max(0.1, min(0.9, energy))
        frames.append(energy + random.gauss(0, 0.02))
    return _quantize(np.array(frames))


def generate_applause(n_frames: int = 200) -> List[str]:
    """Simulate applause: sustained noisy mid-energy."""
    base = random.uniform(0.3, 0.5)
    frames = [base + random.gauss(0, 0.08) for _ in range(n_frames)]
    return _quantize(np.array(frames))


def generate_knock(n_frames: int = 200) -> List[str]:
    """Simulate knocking: 2-3 very brief, sharp bursts."""
    frames = [random.uniform(0.0, 0.02) for _ in range(n_frames)]
    n_knocks = random.randint(2, 3)
    for k in range(n_knocks):
        pos = 20 + k * random.randint(25, 40)
        if pos < n_frames:
            # Very brief spike: 2-3 frames
            for i in range(min(3, n_frames - pos)):
                frames[pos + i] = random.uniform(0.6, 0.95) * math.exp(-1.5 * i)
    return _quantize(np.array(frames))


# ═══════════════════════════════════════════════════════════════════════════
# Training and classification
# ═══════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "speech": generate_speech,
    "silence": generate_silence,
    "dog_bark": generate_dog_bark,
    "doorbell": generate_doorbell,
    "music": generate_music,
    "applause": generate_applause,
    "knock": generate_knock,
}


def generate_dataset(
    n_samples_per_class: int = 50,
    n_frames: int = 200,
) -> Dict[str, List[List[str]]]:
    """Generate synthetic training data for all audio classes."""
    dataset: Dict[str, List[List[str]]] = {}
    for name, gen_fn in GENERATORS.items():
        dataset[name] = [gen_fn(n_frames) for _ in range(n_samples_per_class)]
    return dataset


def train_models(
    dataset: Dict[str, List[List[str]]],
    order: int = 3,
) -> Tuple[Dict[str, MarkovChain], Vocabulary]:
    """Train one Markov chain per audio class with shared vocabulary."""
    # Shared vocab (all quantization bins)
    all_seqs = []
    for seqs in dataset.values():
        all_seqs.extend(seqs)
    vocab = Vocabulary()
    vocab.build_from_sequences(all_seqs)

    models: Dict[str, MarkovChain] = {}
    for name, seqs in dataset.items():
        mc = MarkovChain(order=order, vocab=vocab, smoothing=1e-5,
                         kneser_ney=True)
        mc.fit(seqs)
        models[name] = mc

    return models, vocab


def classify(
    sequence: List[str],
    models: Dict[str, MarkovChain],
) -> List[Tuple[str, float]]:
    """Classify an audio sequence by perplexity across all models."""
    scores = []
    for name, mc in models.items():
        ppx = mc.perplexity([sequence])
        # Inverse perplexity as confidence
        conf = 1.0 / max(ppx, 1e-10)
        scores.append((name, conf))

    # Normalize
    total = sum(c for _, c in scores)
    if total > 0:
        scores = [(n, c / total) for n, c in scores]
    scores.sort(key=lambda x: -x[1])
    return scores


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print("═" * 60)
    print("  Audio Event Classifier — Markov Chains on Quantized Energy")
    print("═" * 60)

    # Generate training data
    print("\nGenerating synthetic training data...")
    n_train = 80
    n_test = 20
    train_data = generate_dataset(n_samples_per_class=n_train, n_frames=200)
    test_data = generate_dataset(n_samples_per_class=n_test, n_frames=200)

    for name, seqs in train_data.items():
        print(f"  {name:<12s}: {len(seqs)} training sequences")

    # Train models
    print(f"\nTraining (order=3, Kneser-Ney, {N_BINS} bins)...")
    models, vocab = train_models(train_data, order=3)
    print(f"  Vocabulary: {vocab.size} tokens (bin levels 0-{N_BINS - 1})")

    # Evaluate
    print("\n─── Classification Results ───")
    correct = 0
    total = 0
    per_class_correct: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}
    confusion: Dict[str, Dict[str, int]] = {}

    for true_class, seqs in test_data.items():
        per_class_total[true_class] = len(seqs)
        per_class_correct[true_class] = 0
        confusion[true_class] = {}
        for seq in seqs:
            scores = classify(seq, models)
            predicted = scores[0][0]
            total += 1
            confusion[true_class][predicted] = confusion[true_class].get(predicted, 0) + 1
            if predicted == true_class:
                correct += 1
                per_class_correct[true_class] += 1

    accuracy = correct / total
    print(f"\n  Overall accuracy: {accuracy:.1%} ({correct}/{total})")

    print(f"\n  {'Class':<12s} {'Correct':<10s} {'Total':<8s} {'Accuracy':<10s}")
    print(f"  {'─' * 40}")
    for cls in sorted(per_class_total.keys()):
        c = per_class_correct[cls]
        t = per_class_total[cls]
        print(f"  {cls:<12s} {c:<10d} {t:<8d} {c / t:<10.1%}")

    # Confusion matrix
    classes = sorted(GENERATORS.keys())
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "  " + " " * 12 + "".join(f"{c[:5]:>6s}" for c in classes)
    print(header)
    for true_cls in classes:
        row = f"  {true_cls:<12s}"
        for pred_cls in classes:
            count = confusion.get(true_cls, {}).get(pred_cls, 0)
            row += f"{count:>6d}"
        print(row)

    # Demo: classify a few individual samples
    print(f"\n─── Sample Classifications ───")
    for cls_name, gen_fn in list(GENERATORS.items())[:4]:
        seq = gen_fn(200)
        scores = classify(seq, models)
        top3 = ", ".join(f"{n}={c:.2f}" for n, c in scores[:3])
        marker = "✓" if scores[0][0] == cls_name else "✗"
        print(f"  {cls_name:<12s} → {scores[0][0]:<12s} {marker}  [{top3}]")

    # Data requirements guide
    print(f"\n─── Data Requirements Guide ───")
    print(f"""
  For real audio, replace synthetic generators with:
    import librosa
    y, sr = librosa.load("audio.wav", sr=16000)
    energy = librosa.feature.rms(y=y, frame_length=320, hop_length=160)[0]
    tokens = [str(int(min(e * {N_BINS}, {N_BINS - 1}))) for e in energy]

  Recommended data per class:
    Minimum:      5 min audio → ~15K frames → works for distinct events
    Good:        30 min audio → ~90K frames → robust across environments
    Production: 2+ hrs audio → ~360K frames → noise-robust, multi-speaker

  Tips:
    - Add noise augmentation (SNR 5-20 dB) to training data
    - Use multiple feature streams (energy + ZCR + spectral centroid)
      → concatenate as multi-character tokens: "E7_Z3_S5"
    - Order 3-5 works best (captures ~30-100ms of temporal context)
    - {N_BINS}=16 bins is a good starting point; 32 for finer resolution
    - For VAD specifically: just use 2 classes (speech, silence)
    - For smart home: train on recordings from the actual deployment room
""")


if __name__ == "__main__":
    main()
