#!/usr/bin/env python3
"""Example: Using MarkovConfig for configuration.

Demonstrates programmatic and environment-variable-based configuration.
"""

import os
from pathlib import Path

from markovonnx import MarkovConfig


def main() -> None:
    # 1. Programmatic configuration
    cfg = MarkovConfig(
        data_path=str(Path(__file__).parent / "data" / "nursery_rhymes.txt"),
        data_mode="word",
        order=2,
        smoothing=1e-4,
        max_vocab=100,
        quantize=True,
        temperature=0.7,
        gen_length=20,
        seed="the little",
        outdir="./output",
    )

    print("--- Programmatic config ---")
    print(f"  data_path:    {cfg.data_path}")
    print(f"  data_mode:    {cfg.data_mode}")
    print(f"  order:        {cfg.order}")
    print(f"  smoothing:    {cfg.smoothing}")
    print(f"  max_vocab:    {cfg.max_vocab}")
    print(f"  quantize:     {cfg.quantize}")
    print(f"  ONNX path:    {cfg.resolved_onnx_path}")
    print(f"  Quant path:   {cfg.resolved_quant_path}")
    print(f"  outdir_path:  {cfg.outdir_path}")

    # 2. Environment variable configuration
    os.environ["MARKOV_DATA_MODE"] = "char"
    os.environ["MARKOV_ORDER"] = "3"
    os.environ["MARKOV_SMOOTHING"] = "0.001"
    os.environ["MARKOV_QUANTIZE"] = "false"

    env_cfg = MarkovConfig.from_env()
    print("\n--- Environment config ---")
    print(f"  data_mode:    {env_cfg.data_mode}")
    print(f"  order:        {env_cfg.order}")
    print(f"  smoothing:    {env_cfg.smoothing}")
    print(f"  quantize:     {env_cfg.quantize}")

    # Clean up
    for key in ["MARKOV_DATA_MODE", "MARKOV_ORDER", "MARKOV_SMOOTHING", "MARKOV_QUANTIZE"]:
        os.environ.pop(key, None)

    # 3. Show all defaults
    defaults = MarkovConfig()
    print("\n--- All defaults ---")
    for field in defaults.__dataclass_fields__:
        print(f"  {field}: {getattr(defaults, field)}")


if __name__ == "__main__":
    main()
