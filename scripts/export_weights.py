#!/usr/bin/env python3
"""Export the 16 kHz Silero VAD weights from the official ONNX model.

Writes the weights used by SileroVAD.java and the ONNX reference output used by the tests.

    pip install onnx onnxruntime numpy
    python scripts/export_weights.py silero_vad.onnx

The ONNX model is available at
https://github.com/snakers4/silero-vad/tree/<version>/src/silero_vad/data
"""
import argparse
import hashlib
import wave

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

WEIGHTS_OUT = "src/main/resources/io/github/jvoiceproject/silerovad/silero_vad_16k.bin"
TEST_WAV = "src/test/resources/speech.wav"
PROBS_OUT = "src/test/resources/speech-silero-probs.txt"
TRAILING_SILENCE = 8000


def load_weights(path):
    """Load and extract constant model weights for the 16 kHz branch.

    Inspects the ONNX computational graph, locates the conditional `If` node
    discriminating between sample rates, and extracts constant tensor attributes
    from the 16 kHz `then_branch`.

    Args:
        path (str): File system path to the input `silero_vad.onnx` model.

    Returns:
        dict[str, np.ndarray]: Mapping of parameter names to weight arrays
        represented as 32-bit floating-point numbers.
    """
    model = onnx.load(path)
    # the If node holds a 16 kHz (then) and an 8 kHz (else) branch
    branch = next(n for n in model.graph.node if n.op_type == "If")
    graph = next(a.g for a in branch.attribute if a.name == "then_branch")
    prefix = "If_0_then_branch__Inline_0__"
    weights = {}
    for node in graph.node:
        if node.op_type == "Constant":
            name = node.output[0].replace(prefix, "")
            if "." in name and not name.startswith("/"):
                weights[name] = numpy_helper.to_array(node.attribute[0].t).astype(np.float32)
    return weights


def export(weights, out):
    """Serialize extracted model weights into a contiguous raw binary blob.

    Packs the weights in the exact order and memory layout expected by
    `SileroVAD.Weights` in the Java implementation:
      - STFT forward basis buffer
      - 4x Encoder reparam convolution weights and biases
      - Decoder RNN weights (ih, hh) and fused biases (ih + hh)
      - Decoder final linear layer weight and bias

    Weights are serialized as raw 32-bit little-endian floats (`<f4`).

    Args:
        weights (dict[str, np.ndarray]): Dictionary containing the extracted weights.
        out (str): Destination path for the `.bin` output file.
    """
    # order and layout must match SileroVAD.Weights
    names = ["stft.forward_basis_buffer"]
    names += [f"encoder.{i}.reparam_conv.{p}" for i in range(4) for p in ("weight", "bias")]
    names += ["decoder.rnn.weight_ih", "decoder.rnn.weight_hh"]
    arrays = [weights[n] for n in names]
    arrays.append(weights["decoder.rnn.bias_ih"] + weights["decoder.rnn.bias_hh"])
    arrays.append(weights["decoder.decoder.2.weight"].reshape(-1))
    arrays.append(weights["decoder.decoder.2.bias"])
    blob = b"".join(a.astype("<f4").tobytes() for a in arrays)
    with open(out, "wb") as f:
        f.write(blob)
    print(f"{out}: {len(blob) // 4} floats, sha256 {hashlib.sha256(blob).hexdigest()}")


def reference(path, wav, out):
    """Run ONNX runtime inference on a test WAV file and export probability reference values.

    Reads a 16 kHz mono WAV file, appends trailing silence samples, and processes
    the audio in overlapping frames (64 samples context + 512 samples chunk).
    The resulting frame-by-frame speech probabilities are formatted to 6 decimal places
    and written to disk for regression and consistency testing.

    Args:
        path (str): File system path to the input `silero_vad.onnx` model.
        wav (str): File system path to the 16 kHz mono reference WAV file.
        out (str): Destination text file path for writing probability values.

    Raises:
        AssertionError: If the input WAV file is not 16 kHz, mono, or 16-bit PCM.
    """
    with wave.open(wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    audio = np.concatenate([audio, np.zeros(TRAILING_SILENCE, np.int16)]).astype(np.float32) / 32768

    session = ort.InferenceSession(path)
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros(64, np.float32)
    probs = []
    for i in range(0, len(audio) - 511, 512):
        frame = np.concatenate([context, audio[i : i + 512]])
        context = frame[-64:]
        prob, state = session.run(
            None, {"input": frame[None], "state": state, "sr": np.array(16000, np.int64)}
        )
        probs.append("%.6f" % prob[0, 0])
    with open(out, "w") as f:
        f.write("\n".join(probs) + "\n")
    print(f"{out}: {len(probs)} frames")


def main():
    """Parse CLI arguments, print checksums, and orchestrate export and reference generation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx", help="path to silero_vad.onnx")
    args = parser.parse_args()

    with open(args.onnx, "rb") as f:
        print(f"{args.onnx}: sha256 {hashlib.sha256(f.read()).hexdigest()}")
    export(load_weights(args.onnx), WEIGHTS_OUT)
    reference(args.onnx, TEST_WAV, PROBS_OUT)


if __name__ == "__main__":
    main()
