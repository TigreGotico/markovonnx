#!/usr/bin/env python3
"""Example: ONNX quantization and inference benchmarking.

Compares full-precision vs INT8 quantized model size and performance.
Also benchmarks Python model vs ONNX Runtime.
"""

import tempfile
import time
from pathlib import Path

import numpy as np

from markovonnx import (
    MarkovChain,
    MarkovONNXRuntime,
    Vocabulary,
    char_tokenize,
    export_markov_onnx,
    quantize_model,
)


def main() -> None:
    data_path = str(Path(__file__).parent / "data" / "nursery_rhymes.txt")

    # 1. Train model
    with open(data_path, encoding="utf-8") as f:
        corpus = [char_tokenize(line) for line in f if line.strip()]

    vocab = Vocabulary()
    vocab.build_from_sequences(corpus)
    mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
    mc.fit(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp_path = str(Path(tmpdir) / "full.onnx")
        q_path = str(Path(tmpdir) / "int8.onnx")

        # 2. Export full precision
        export_markov_onnx(mc, fp_path)
        fp_size = Path(fp_path).stat().st_size

        # 3. Quantize
        result = quantize_model(fp_path, q_path)
        if result:
            q_size = Path(q_path).stat().st_size
            print(f"\n--- Size comparison ---")
            print(f"  Full precision: {fp_size / 1024:.1f} KB")
            print(f"  INT8 quantized: {q_size / 1024:.1f} KB")
            print(f"  Reduction:      {100 * (1 - q_size / fp_size):.1f}%")
        else:
            print("Quantization not available (install onnxruntime-tools)")
            q_path = fp_path

        # 4. Benchmark: Python vs ONNX Runtime
        rt = MarkovONNXRuntime(fp_path, vocab, order=2)

        # Build test contexts
        n_bench = 1000
        test_contexts = []
        for seq in corpus[:n_bench]:
            if len(seq) > 2:
                test_contexts.append(seq[:2])
        if len(test_contexts) < n_bench:
            test_contexts = test_contexts * (n_bench // len(test_contexts) + 1)
        test_contexts = test_contexts[:n_bench]

        # Python benchmark
        t0 = time.perf_counter()
        for ctx in test_contexts:
            mc.sample(ctx)
        py_time = time.perf_counter() - t0

        # ONNX benchmark (warmup)
        for ctx in test_contexts[:20]:
            rt.predict_probs(ctx)

        t0 = time.perf_counter()
        for ctx in test_contexts:
            rt.predict_probs(ctx)
        ort_time = time.perf_counter() - t0

        print(f"\n--- Inference benchmark ({n_bench} calls) ---")
        print(f"  Python model:  {py_time * 1000:.1f} ms ({n_bench / py_time:.0f} calls/s)")
        print(f"  ONNX Runtime:  {ort_time * 1000:.1f} ms ({n_bench / ort_time:.0f} calls/s)")
        print(f"  Speedup:       {py_time / ort_time:.2f}x")

        # 5. Verify consistency
        print(f"\n--- Verification ---")
        max_err = 0.0
        for ctx in test_contexts[:50]:
            # Python probs
            V = vocab.size
            ids = vocab.encode(ctx[-2:])
            ci = mc._ctx_idx(ids)
            row = mc._counts.get(ci, np.full(V, mc.smoothing, np.float32))
            row_sum = row.sum() + mc.smoothing * V
            py_probs = (row + mc.smoothing) / row_sum

            # ONNX probs
            ort_probs = rt.predict_probs(ctx)
            err = np.abs(py_probs - ort_probs).max()
            max_err = max(max_err, err)

        print(f"  Max absolute error (Python vs ONNX): {max_err:.2e}")
        print(f"  Status: {'PASS' if max_err < 0.01 else 'CHECK'}")


if __name__ == "__main__":
    main()
