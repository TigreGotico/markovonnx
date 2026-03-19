"""Save and load portable .markov archive files.

A ``.markov`` archive is a ZIP file containing:
- ``model.onnx`` — the exported ONNX model
- ``vocab.json`` — the serialized :class:`Vocabulary`
- ``config.json`` — metadata (model_type, order, etc.)
"""

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Union

from markovonnx.config import MarkovConfig
from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain
from markovonnx.onnx_export import export_hmm_onnx, export_markov_onnx
from markovonnx.vocabulary import Vocabulary


def save_markov_archive(
    model: Union[MarkovChain, HiddenMarkovModel],
    archive_path: str,
) -> str:
    """Save a trained model + vocabulary to a portable ``.markov`` archive.

    Args:
        model: A trained :class:`MarkovChain` or :class:`HiddenMarkovModel`.
        archive_path: Destination path (should end in ``.markov``).

    Returns:
        The archive path.
    """
    Path(archive_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "model.onnx")
        vocab_path = str(Path(tmpdir) / "vocab.json")
        config_path = str(Path(tmpdir) / "config.json")

        if isinstance(model, MarkovChain):
            export_markov_onnx(model, onnx_path)
            model.vocab.save(vocab_path)
            config = {
                "model_type": "markov_chain",
                "order": model.order,
                "smoothing": model.smoothing,
                "vocab_size": model.vocab.size,
                "backoff": model.backoff,
            }
        elif isinstance(model, HiddenMarkovModel):
            export_hmm_onnx(model, onnx_path)
            model.obs_vocab.save(vocab_path)
            config = {
                "model_type": "hmm",
                "n_states": model.n_states,
                "smoothing": model.smoothing,
                "vocab_size": model.obs_vocab.size,
            }
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(onnx_path, "model.onnx")
            zf.write(vocab_path, "vocab.json")
            zf.write(config_path, "config.json")

    print(f"Saved archive: {archive_path} ({Path(archive_path).stat().st_size / 1024:.1f} KB)")
    return archive_path


def load_markov_archive(archive_path: str) -> dict:
    """Load a ``.markov`` archive and return runtime components.

    Args:
        archive_path: Path to a ``.markov`` file.

    Returns:
        A dict with keys:
        - ``"runtime"``: :class:`MarkovONNXRuntime` or :class:`HMMONNXRuntime`
        - ``"vocab"``: :class:`Vocabulary`
        - ``"config"``: dict of metadata (model_type, order, etc.)
        - ``"onnx_path"``: path to the extracted ONNX file (in temp dir)
    """
    # Lazy imports to avoid circular dependency
    from markovonnx.onnx_runtime import HMMONNXRuntime, MarkovONNXRuntime

    tmpdir = tempfile.mkdtemp(prefix="markovonnx_")

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(tmpdir)

    onnx_path = str(Path(tmpdir) / "model.onnx")
    vocab_path = str(Path(tmpdir) / "vocab.json")
    config_path = str(Path(tmpdir) / "config.json")

    vocab = Vocabulary.load(vocab_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config["model_type"]

    if model_type == "markov_chain":
        rt = MarkovONNXRuntime(onnx_path, vocab, config["order"])
    elif model_type == "hmm":
        # Create a minimal HMM for the runtime wrapper (needs pi, obs_vocab)
        import numpy as np
        hmm = HiddenMarkovModel(
            n_states=config["n_states"],
            obs_vocab=vocab,
            smoothing=config.get("smoothing", 1e-5),
        )
        # ORT expects float32 for alpha_in
        hmm.pi = hmm.pi.astype(np.float32)
        rt = HMMONNXRuntime(onnx_path, hmm)
    else:
        raise ValueError(f"Unknown model_type in archive: {model_type}")

    return {
        "runtime": rt,
        "vocab": vocab,
        "config": config,
        "onnx_path": onnx_path,
    }
