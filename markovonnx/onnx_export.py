"""Export MarkovChain and HiddenMarkovModel to ONNX format."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from markovonnx.hmm import HiddenMarkovModel
from markovonnx.markov import MarkovChain


def export_markov_onnx(mc: MarkovChain, path: str) -> None:
    """Export a :class:`MarkovChain` to an ONNX model.

    The graph computes: ``input_ids -> row_index -> probs -> next_id``.

    Args:
        mc: Trained Markov chain.
        path: Destination ``.onnx`` file path.
    """
    V = mc.vocab.size
    order = mc.order

    T = mc.dense_matrix()  # [V^order, V]

    powers = np.array([V ** (order - 1 - i) for i in range(order)], dtype=np.int64)

    T_init = numpy_helper.from_array(T, name="T")
    pow_init = numpy_helper.from_array(powers, name="powers")

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [order])
    probs_out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [V])
    next_out = helper.make_tensor_value_info("next_id", TensorProto.INT64, [1])

    nodes = [
        helper.make_node("Mul", ["input_ids", "powers"], ["mul_out"]),
        helper.make_node(
            "ReduceSum",
            ["mul_out"],
            ["index"],
            keepdims=0,
            noop_with_empty_axes=0,
        ),
        helper.make_node("Gather", ["T", "index"], ["probs"]),
        helper.make_node(
            "ArgMax", ["probs"], ["next_id"], axis=0, keepdims=1
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "markov_chain",
        [input_ids],
        [probs_out, next_out],
        initializer=[T_init, pow_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8

    meta = model.metadata_props.add()
    meta.key, meta.value = "model_type", "markov_chain"
    meta = model.metadata_props.add()
    meta.key, meta.value = "order", str(order)
    meta = model.metadata_props.add()
    meta.key, meta.value = "vocab", json.dumps(mc.vocab.id2tok[:500])
    meta = model.metadata_props.add()
    meta.key, meta.value = "vocab_size", str(V)

    onnx.checker.check_model(model)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"Saved: {path}  ({size_kb:.1f} KB)")


def export_markov_sparse_onnx(mc: MarkovChain, path: str) -> None:
    """Export a :class:`MarkovChain` using a sparse lookup table.

    Instead of storing the full ``V^order × V`` dense matrix, this export
    stores only the rows that have observed counts, plus a uniform fallback
    row for unseen contexts.  This dramatically reduces model size for large
    vocabularies or high orders.

    The ONNX graph hashes the context to an index, looks it up in the
    sparse table via a two-step Gather (keys → position → rows), and
    falls back to a uniform distribution for misses.

    Args:
        mc: Trained Markov chain.
        path: Destination ``.onnx`` file path.
    """
    V = mc.vocab.size
    order = mc.order

    # Build sparse table: only rows with observed counts
    smoothing = mc.smoothing
    ctx_indices: list = []
    rows: list = []

    for ci, row in mc._counts.items():
        if row.sum() > 0:
            if mc.kneser_ney:
                d = mc._kn_discount
                total = row.sum()
                n_pos = float((row > 0).sum())
                lam = d * n_pos / total
                prob_row = np.maximum(row - d, 0.0) / total + lam * (1.0 / V)
            else:
                prob_row = (row + smoothing) / (row.sum() + smoothing * V)
            ctx_indices.append(ci)
            rows.append(prob_row)

    # Uniform fallback row
    uniform_row = np.full(V, 1.0 / V, dtype=np.float32)

    n_sparse = len(ctx_indices)
    # Sparse table: [n_sparse + 1, V] (last row is uniform fallback)
    sparse_table = np.zeros((n_sparse + 1, V), dtype=np.float32)
    for i, row in enumerate(rows):
        sparse_table[i] = row
    sparse_table[n_sparse] = uniform_row

    # Context index keys for lookup
    keys = np.array(ctx_indices, dtype=np.int64)
    fallback_idx = np.array([n_sparse], dtype=np.int64)

    powers = np.array([V ** (order - 1 - i) for i in range(order)], dtype=np.int64)

    # ONNX initializers
    table_init = numpy_helper.from_array(sparse_table, name="sparse_table")
    keys_init = numpy_helper.from_array(keys, name="keys")
    pow_init = numpy_helper.from_array(powers, name="powers")
    fallback_init = numpy_helper.from_array(fallback_idx, name="fallback_idx")

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [order])
    probs_out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [V])
    next_out = helper.make_tensor_value_info("next_id", TensorProto.INT64, [1])

    # The ONNX graph:
    # 1. Compute context index: sum(input_ids * powers)
    # 2. Compare against keys to find position (Equal + Cast + ArgMax)
    # 3. If found, Gather from sparse_table; else use fallback row
    #
    # Since ONNX has no hash-table op, we use a brute-force approach:
    #   match = Equal(keys, index)  -> bool[n_sparse]
    #   has_match = ReduceMax(match) -> bool scalar
    #   pos_if_found = ArgMax(Cast(match, INT64)) -> position in keys
    #   pos = Where(has_match, pos_if_found, fallback_idx)
    #   probs = Gather(sparse_table, pos)
    nodes = [
        # 1. context index
        helper.make_node("Mul", ["input_ids", "powers"], ["mul_out"]),
        helper.make_node("ReduceSum", ["mul_out"], ["index"],
                         keepdims=0, noop_with_empty_axes=0),
        # 2. search keys for match
        helper.make_node("Equal", ["keys", "index"], ["match_bool"]),
        helper.make_node("Cast", ["match_bool"], ["match_int"],
                         to=TensorProto.INT64),
        # has_match = any match
        helper.make_node("ReduceMax", ["match_int"], ["has_match_int"],
                         keepdims=0),
        helper.make_node("Cast", ["has_match_int"], ["has_match"],
                         to=TensorProto.BOOL),
        # position of the match
        helper.make_node("ArgMax", ["match_int"], ["pos_found"],
                         axis=0, keepdims=0),
        # select position or fallback
        helper.make_node("Where", ["has_match", "pos_found", "fallback_idx"],
                         ["pos"]),
        # 3. lookup
        helper.make_node("Gather", ["sparse_table", "pos"], ["probs"]),
        # 4. argmax
        helper.make_node("ArgMax", ["probs"], ["next_id"], axis=0, keepdims=1),
    ]

    graph = helper.make_graph(
        nodes,
        "markov_chain_sparse",
        [input_ids],
        [probs_out, next_out],
        initializer=[table_init, keys_init, pow_init, fallback_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8

    # Metadata
    for k, v in [
        ("model_type", "markov_chain_sparse"),
        ("order", str(order)),
        ("vocab", json.dumps(mc.vocab.id2tok[:500])),
        ("vocab_size", str(V)),
        ("n_sparse_rows", str(n_sparse)),
    ]:
        meta = model.metadata_props.add()
        meta.key, meta.value = k, v

    onnx.checker.check_model(model)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)

    dense_kb = V ** order * V * 4 / 1024
    sparse_kb = Path(path).stat().st_size / 1024
    print(
        f"Saved sparse: {path}  ({sparse_kb:.1f} KB, "
        f"{n_sparse} rows vs {V ** order} dense, "
        f"would-be dense: {dense_kb:.1f} KB)"
    )


def export_hmm_onnx(hmm: HiddenMarkovModel, path: str) -> None:
    """Export a :class:`HiddenMarkovModel` to an ONNX model.

    The graph computes one HMM forward step:
    ``(alpha_t, obs_t) -> alpha_{t+1}, best_state``.

    Args:
        hmm: Trained HMM.
        path: Destination ``.onnx`` file path.
    """
    S = hmm.n_states
    O = hmm.obs_vocab.size

    A_init = numpy_helper.from_array(hmm.A, name="A")
    B_init = numpy_helper.from_array(hmm.B, name="B")
    pi_init = numpy_helper.from_array(hmm.pi, name="pi")

    obs_in = helper.make_tensor_value_info("obs_id", TensorProto.INT64, [1])
    alpha_in = helper.make_tensor_value_info("alpha_in", TensorProto.FLOAT, [S])
    alpha_out = helper.make_tensor_value_info("alpha_out", TensorProto.FLOAT, [S])
    best_out = helper.make_tensor_value_info("best_state", TensorProto.INT64, [1])

    nodes = [
        helper.make_node("Gather", ["B", "obs_id"], ["b_vec"], axis=1),
        helper.make_node("Squeeze", ["b_vec"], ["b_sq"]),
        helper.make_node("MatMul", ["alpha_in", "A"], ["fwd"]),
        helper.make_node("Mul", ["fwd", "b_sq"], ["alpha_raw"]),
        helper.make_node(
            "ReduceSum",
            ["alpha_raw"],
            ["norm_val"],
            keepdims=1,
            noop_with_empty_axes=0,
        ),
        helper.make_node("Div", ["alpha_raw", "norm_val"], ["alpha_out"]),
        helper.make_node(
            "ArgMax", ["alpha_out"], ["best_state"], axis=0, keepdims=1
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "hmm_step",
        [obs_in, alpha_in],
        [alpha_out, best_out],
        initializer=[A_init, B_init, pi_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8

    meta = model.metadata_props.add()
    meta.key, meta.value = "model_type", "hmm"
    meta = model.metadata_props.add()
    meta.key, meta.value = "n_states", str(S)

    onnx.checker.check_model(model)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"Saved HMM ONNX: {path}  ({size_kb:.1f} KB)")


def quantize_model(
    onnx_path: str,
    quant_path: str,
) -> Optional[str]:
    """Quantize an ONNX model to INT8 (dynamic quantization).

    Args:
        onnx_path: Path to the full-precision ONNX model.
        quant_path: Destination path for the quantized model.

    Returns:
        Path to the quantized model on success, ``None`` on failure.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=onnx_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
        )
        orig_kb = Path(onnx_path).stat().st_size / 1024
        qnt_kb = Path(quant_path).stat().st_size / 1024
        print(f"Quantized: {orig_kb:.1f} KB -> {qnt_kb:.1f} KB ({100 * qnt_kb / orig_kb:.1f}%)")
        return quant_path
    except Exception as e:
        print(f"Quantization failed ({e})")
        return None
