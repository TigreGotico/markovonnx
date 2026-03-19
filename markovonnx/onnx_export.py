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
