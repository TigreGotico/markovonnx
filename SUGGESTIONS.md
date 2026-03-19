# Suggestions

1. **CLI entry point** — Add `markovonnx train` / `markovonnx generate` commands via argparse for notebook-free workflows.
2. **Sparse ONNX export** — For large vocab/order, explore sparse matrix representation to avoid dense matrix allocation.
3. **Provider selection** — Allow configuring ONNX Runtime providers via `MarkovConfig` instead of hardcoding CUDA+CPU.
