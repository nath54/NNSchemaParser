"""Trace the espnet_transformer failure to the None input of dropout."""

# Import Modules
import sys

sys.path.insert(0, r"C:/Users/Cerisara Nathan/Documents/GitHub/NNViewer")

import torch  # noqa: I001

import tests.test_speech_models as T
from nnschemaparser.interpreter import SchemaInterpreter, TraceRecorder
from nnschemaparser.parser import NNParser

# Build the first failing model exactly like the test
transformer = T.__dict__ if False else None

# Rebuild the model of the test (espnet_transformer)
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder  # noqa: E402

model = TransformerEncoder(
    input_size=80,
    output_size=64,
    attention_heads=4,
    linear_units=128,
    num_blocks=2,
    input_layer="conv2d"
).eval()

inputs = {
    "xs_pad": torch.randn(2, 50, 80),
    "ilens": torch.tensor([50, 40])
}

# Parse with the example values like the harness
schema = NNParser.parse_model(
    model=model,
    inputs_tensors_shapes={
        "xs_pad": (2, 50, 80),
        "ilens": (2,)
    },
    inputs_tensors_dtypes={},
    inputs_tensors_values=inputs
)

# Execute with a tracer up to the failure
tracer = TraceRecorder(cap=100000)
try:
    SchemaInterpreter(capture_values=True, listeners=[tracer]).run(
        block=schema,
        scope={key: value.clone() for key, value in inputs.items()}
    )
    print("ALL OK")
finally:
    tracer.save(r"C:/Users/Cerisara Nathan/AppData/Local/Temp/espnet_trace.txt")
    print("trace saved")
