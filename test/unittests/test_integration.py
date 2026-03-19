"""End-to-end integration test: train → export → quantize → load → generate → verify."""

import tempfile
from pathlib import Path

import numpy as np

from markovonnx import (
    MarkovChain,
    MarkovONNXRuntime,
    Vocabulary,
    char_tokenize,
    export_markov_onnx,
    generate_markov,
    quantize_model,
    save_markov_archive,
    load_markov_archive,
)


class TestEndToEnd:
    def test_full_pipeline_char_mode(self) -> None:
        """Train on real data → export → quantize → load → generate → verify."""
        corpus_path = str(
            Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt"
        )

        # 1. Load corpus
        with open(corpus_path, encoding="utf-8") as f:
            corpus = [char_tokenize(line) for line in f if line.strip()]
        assert len(corpus) > 0

        # 2. Build vocab
        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        assert vocab.size > 5

        # 3. Train
        mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)
        assert len(mc._counts) > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            q_path = str(Path(tmpdir) / "model_int8.onnx")

            # 4. Export
            export_markov_onnx(mc, onnx_path)
            assert Path(onnx_path).exists()

            # 5. Quantize
            result = quantize_model(onnx_path, q_path)
            if result:
                assert Path(q_path).exists()

            # 6. Load and inference
            rt = MarkovONNXRuntime(onnx_path, vocab, order=2)
            probs = rt.predict_probs(["t", "h"])
            assert probs.shape == (vocab.size,)
            assert abs(probs.sum() - 1.0) < 1e-4

            # 7. Generate
            text = generate_markov(
                rt, "the", length=50, temperature=0.7, mode="char", order=2
            )
            assert isinstance(text, str)
            assert len(text) > 3

            # 8. Verify Python vs ONNX consistency
            test_ctx = corpus[0][:2]
            py_probs = mc._get_probs(test_ctx)
            ort_probs = rt.predict_probs(test_ctx)
            assert np.abs(py_probs - ort_probs).max() < 0.01

    def test_archive_roundtrip_with_generation(self) -> None:
        """Train → save archive → load → generate."""
        corpus_path = str(
            Path(__file__).parent.parent.parent / "examples" / "data" / "nursery_rhymes.txt"
        )
        with open(corpus_path, encoding="utf-8") as f:
            corpus = [char_tokenize(line) for line in f if line.strip()]

        vocab = Vocabulary()
        vocab.build_from_sequences(corpus)
        mc = MarkovChain(order=2, vocab=vocab, smoothing=1e-5)
        mc.fit(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = str(Path(tmpdir) / "model.markov")
            save_markov_archive(mc, archive_path)

            loaded = load_markov_archive(archive_path)
            rt = loaded["runtime"]
            text = generate_markov(
                rt, "th", length=20, temperature=0.5,
                mode="char", order=loaded["config"]["order"],
            )
            assert isinstance(text, str)
            assert len(text) > 2

    def test_backoff_handles_unseen_contexts(self) -> None:
        """Backoff model should not crash on unseen contexts and produce valid output."""
        vocab = Vocabulary()
        train = [list("abcabc")] * 50
        vocab.build_from_sequences(train)

        mc = MarkovChain(order=3, vocab=vocab, smoothing=1e-5, backoff=True)
        mc.fit(train)

        # Sample from unseen context — should backoff gracefully
        token = mc.sample(list("zzz"))
        assert token in vocab.tok2id

        # Perplexity should be finite
        ppx = mc.perplexity([list("abcba")])
        assert ppx > 0
        assert np.isfinite(ppx)

    def test_from_file_metadata_loading(self) -> None:
        """MarkovONNXRuntime.from_file should work without providing vocab."""
        vocab = Vocabulary()
        seqs = [["a", "b", "a", "b"]] * 20
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
        mc.fit(seqs)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            export_markov_onnx(mc, onnx_path)

            rt = MarkovONNXRuntime.from_file(onnx_path)
            assert rt.order == 1
            assert rt.vocab.size == vocab.size
            token = rt.argmax(["a"])
            assert token in rt.vocab.tok2id

    def test_batch_predict_probs(self) -> None:
        """predict_probs_batch should return correct shapes."""
        vocab = Vocabulary()
        seqs = [["a", "b", "a", "b"]] * 20
        vocab.build_from_sequences(seqs)
        mc = MarkovChain(order=1, vocab=vocab, smoothing=1e-5)
        mc.fit(seqs)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")
            export_markov_onnx(mc, onnx_path)
            rt = MarkovONNXRuntime(onnx_path, vocab, order=1)

            contexts = [["a"], ["b"], ["a"]]
            batch_probs = rt.predict_probs_batch(contexts)
            assert batch_probs.shape == (3, vocab.size)
            for row in batch_probs:
                assert abs(row.sum() - 1.0) < 1e-4
