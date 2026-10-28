"""
Tests of the NNParser on real speech models.

Tests the ESPnet transformer, conformer and branchformer encoders, and the
torchaudio wav2vec2, hubert and wavlm models.

Run with: python -m tests.test_speech_models
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor

from torchaudio.models import hubert_base, wav2vec2_base, wavlm_base
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder

from src.core.model_schema import ModelBlockSchema
from src.model_parser.parser import NNParser
from src.model_parser.interpreter import SchemaInterpreter


def check_block_instructions(block: ModelBlockSchema, path: str) -> None:
    """
    Check that the instructions of a schema block got resolved representations.

    Args:
        block (ModelBlockSchema): The block schema to check recursively.
        path (str): The display path of the block in the model.
    """

    # Count the instructions of the block that got a resolved shape
    resolved_count: int = 0
    instruction: Any
    for instruction in block.body:
        if hasattr(instruction, "resolved") and instruction.resolved is not None:
            resolved_count += 1

    # The instructions of the sub-block bodies are checked recursively
    sub_name: str
    sub_block: ModelBlockSchema
    for sub_name, sub_block in block.sub_blocks.items():
        check_block_instructions(block=sub_block, path=f"{path}.{sub_name}")


def check_model(
    name: str,
    model: Any,
    inputs_tensors_shapes: dict[str, tuple[int, ...]],
    inputs_tensors_dtypes: dict[str, str],
    expected_outputs: int
) -> None:
    """
    Parse a model, execute its schema and compare with the real forward.

    Args:
        name (str): The name of the test model.
        model (Any): The torch model to test.
        inputs_tensors_shapes (dict[str, tuple[int, ...]]): The input shapes.
        inputs_tensors_dtypes (dict[str, str]): The input dtypes.
        expected_outputs (int): The expected output count of the forward.
    """

    # Build the example input values of the model
    example_values: dict[str, Tensor] = {}
    parameter_name: str
    parameter_shape: tuple[int, ...]
    for parameter_name, parameter_shape in inputs_tensors_shapes.items():
        dtype_name: str = inputs_tensors_dtypes.get(parameter_name, "float32")
        dtype: torch.dtype = getattr(torch, dtype_name)
        if dtype.is_floating_point:
            example_values[parameter_name] = torch.randn(
                size=parameter_shape, dtype=dtype
            )
        else:
            example_values[parameter_name] = torch.ones(
                size=parameter_shape, dtype=dtype
            )

    # Fix the random input dtypes that must stay integer or boolean
    if "ilens" in inputs_tensors_shapes:
        example_values["ilens"] = torch.tensor(
            [inputs_tensors_shapes["ilens"][0]] * inputs_tensors_shapes["ilens"][0],
            dtype=torch.int64
        )[: inputs_tensors_shapes["ilens"][0]]
    if "lengths" in inputs_tensors_shapes:
        example_values["lengths"] = torch.full(
            (inputs_tensors_shapes["lengths"][0],),
            inputs_tensors_shapes["waveforms"][1],
            dtype=torch.int64
        )

    # Parse the model into a schema with the example values for the resolution
    schema: ModelBlockSchema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes=inputs_tensors_shapes,
        inputs_tensors_dtypes=inputs_tensors_dtypes,
        inputs_tensors_values=example_values
    )

    # Check the whole schema of the model
    check_block_instructions(block=schema, path=name)

    # Execute the schema with real input tensors and capture the values
    scope: dict[str, Any] = {
        name: value.clone() for name, value in example_values.items()
    }

    # Execute the schema of the model
    with torch.no_grad():
        outputs: list[Any] = SchemaInterpreter(capture_values=True).run(
            block=schema,
            scope=scope
        )

    # Execute the real forward of the model with the same input values
    reference_args: tuple = tuple(example_values.values())
    with torch.no_grad():
        reference: Any = model(*reference_args)
    # Compare the outputs of the schema and of the real forward
    if not isinstance(reference, tuple):
        reference = (reference,)
    reference_outputs: list[Any] = list(reference)

    # Check the output count of the schema execution
    if len(outputs) != expected_outputs:
        raise AssertionError(
            f"{name}: expected {expected_outputs} outputs, got {len(outputs)}"
        )

    # Check each output tensor against the reference output
    output_index: int
    for output_index, (output, reference_output) in enumerate(
        zip(outputs, reference_outputs)
    ):
        if isinstance(output, Tensor) and isinstance(reference_output, Tensor):
            if not torch.allclose(output, reference_output, atol=1e-5):
                raise AssertionError(
                    f"{name}: output {output_index} mismatch with the real forward"
                )

    # Report the success of the test
    print(f"OK {name}: outputs={len(outputs)}, sub_blocks={len(schema.sub_blocks)}")


def main() -> None:
    """
    Run all the speech model tests.
    """

    # The input tensors shared by the espnet encoders
    xs_pad: Tensor = torch.randn(2, 50, 80)
    ilens: Tensor = torch.tensor([50, 50], dtype=torch.int64)

    # The espnet transformer encoder
    transformer: TransformerEncoder = TransformerEncoder(
        input_size=80,
        output_size=64,
        attention_heads=4,
        linear_units=128,
        num_blocks=2,
        input_layer="conv2d"
    ).eval()
    check_model(
        name="espnet_transformer",
        model=transformer,
        inputs_tensors_shapes={"xs_pad": (2, 50, 80), "ilens": (2,)},
        inputs_tensors_dtypes={"ilens": "int64"},
        expected_outputs=3
    )

    # The espnet conformer encoder
    conformer: ConformerEncoder = ConformerEncoder(
        input_size=80,
        output_size=64,
        attention_heads=4,
        linear_units=128,
        num_blocks=2,
        input_layer="conv2d"
    ).eval()
    check_model(
        name="espnet_conformer",
        model=conformer,
        inputs_tensors_shapes={"xs_pad": (2, 50, 80), "ilens": (2,)},
        inputs_tensors_dtypes={"ilens": "int64"},
        expected_outputs=3
    )

    # The espnet branchformer encoder
    branchformer: BranchformerEncoder = BranchformerEncoder(
        input_size=80,
        output_size=64,
        attention_heads=4,
        cgmlp_linear_units=128,
        num_blocks=2,
        input_layer="conv2d"
    ).eval()
    check_model(
        name="espnet_branchformer",
        model=branchformer,
        inputs_tensors_shapes={"xs_pad": (2, 50, 80), "ilens": (2,)},
        inputs_tensors_dtypes={"ilens": "int64"},
        expected_outputs=3
    )

    # The torchaudio wav2vec2 model
    wav2vec2: Any = wav2vec2_base().eval()
    check_model(
        name="wav2vec2",
        model=wav2vec2,
        inputs_tensors_shapes={"waveforms": (2, 16000), "lengths": (2,)},
        inputs_tensors_dtypes={},
        expected_outputs=2
    )

    # The torchaudio hubert model
    hubert: Any = hubert_base().eval()
    check_model(
        name="hubert",
        model=hubert,
        inputs_tensors_shapes={"waveforms": (2, 16000), "lengths": (2,)},
        inputs_tensors_dtypes={},
        expected_outputs=2
    )

    # The torchaudio wavlm model
    wavlm: Any = wavlm_base().eval()
    check_model(
        name="wavlm",
        model=wavlm,
        inputs_tensors_shapes={"waveforms": (2, 16000), "lengths": (2,)},
        inputs_tensors_dtypes={},
        expected_outputs=2
    )


# Main entry of the test file
if __name__ == "__main__":
    main()
