"""Configuration dataclass with environment variable defaults."""

import dataclasses
import os
from pathlib import Path


def _env(key: str, default: object, cast: type = str) -> object:
    """Read environment variable with typed default."""
    val = os.environ.get(key, "")
    return cast(val) if val else default


@dataclasses.dataclass
class MarkovConfig:
    """Configuration for markovonnx training and export.

    All fields fall back to ``MARKOV_*`` environment variables when not
    provided explicitly.
    """

    data_path: str = ""
    data_mode: str = "char"
    order: int = 2
    hmm_states: int = 16
    smoothing: float = 1e-5
    max_vocab: int = 0
    max_lines: int = 0
    stream_threshold_mb: int = 1024
    onnx_path: str = "markov.onnx"
    quantize: bool = True
    quant_path: str = "markov_int8.onnx"
    temperature: float = 0.5
    gen_length: int = 10
    seed: str = ""
    outdir: str = "./markov"

    @classmethod
    def from_env(cls) -> "MarkovConfig":
        """Build config from ``MARKOV_*`` environment variables."""
        return cls(
            data_path=_env("MARKOV_DATA_PATH", ""),
            data_mode=_env("MARKOV_DATA_MODE", "char"),
            order=_env("MARKOV_ORDER", 2, int),
            hmm_states=_env("MARKOV_HMM_STATES", 16, int),
            smoothing=_env("MARKOV_SMOOTHING", 1e-5, float),
            max_vocab=_env("MARKOV_MAX_VOCAB", 0, int),
            max_lines=_env("MARKOV_MAX_LINES", 0, int),
            stream_threshold_mb=_env("MARKOV_STREAM_MB", 1024, int),
            onnx_path=_env("MARKOV_ONNX_PATH", "markov.onnx"),
            quantize=_env("MARKOV_QUANTIZE", "1") in ("1", "true", "yes"),
            quant_path=_env("MARKOV_QUANT_PATH", "markov_int8.onnx"),
            temperature=_env("MARKOV_TEMP", 0.5, float),
            gen_length=_env("MARKOV_GEN_LEN", 10, int),
            seed=_env("MARKOV_SEED", ""),
            outdir=_env("MARKOV_OUTDIR", "./markov"),
        )

    @property
    def outdir_path(self) -> Path:
        """Return *outdir* as a :class:`~pathlib.Path`."""
        return Path(self.outdir)

    @property
    def resolved_onnx_path(self) -> str:
        """Full path combining *outdir* and *onnx_path*."""
        return str(self.outdir_path / self.onnx_path)

    @property
    def resolved_quant_path(self) -> str:
        """Full path combining *outdir* and *quant_path*."""
        return str(self.outdir_path / self.quant_path)
