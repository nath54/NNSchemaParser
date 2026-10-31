"""
Core schema types of NNSchemaParser.

A parsed model is a tree of ModelBlockSchema blocks, each holding its inputs,
outputs, weights, sub-blocks and the instruction list of its forward body.
"""

# Import Modules
from typing import Any

from typing import Any

from nnschemaparser.core.instructions import (
    BlockInstruction,
    ControlInstruction,
    Instruction,
    TryInstruction
)
from nnschemaparser.core.tensor_schema import TensorRepr, TensorReprWithValue


class ModelBlockSchema:
    """
    Schema of one block of a parsed model.

    A block mirrors one nn.Module of the source model: its forward parameters,
    its returned values, its direct parameters and buffers, its child modules
    as sub-blocks, and the lowered body of its forward method.
    """

    def __init__(
        self,
        inputs: dict[str, TensorRepr],
        outputs: list[TensorRepr],
        weights: dict[str, TensorReprWithValue],
        sub_blocks: dict[str, "ModelBlockSchema"],
        body: list[Instruction],
        module: Any | None = None,
        methods: dict[str, "ModelBlockSchema"] | None = None,
        method_name: str | None = None,
        input_defaults: dict[str, Any] | None = None,
        skipped: dict[str, str] | None = None
    ) -> None:
        """
        Init function.

        Args:
            inputs (dict[str, TensorRepr]): The list of all named input values of the model
                                            (often one X value)
            outputs (list[TensorRepr]): The list of all named output values of the model
                                            (often one Y value)
            weights: dict[str, TensorReprWithValue]: The weights of the current block.
            sub_blocks (dict[str, ModelBlockSchema]): The list of all the sub block of the model
                                            (all the other nn.Modules in the model class attribute)
            body (list[Instruction]): What is happening inside the current model block
                                            (the graph that will be shown in the webUI)
            module (Any | None): The source module of the block, used by the
                                            interpreter to resolve dynamic attributes.
            methods (dict[str, ModelBlockSchema] | None): The other methods of the
                                            source module called by the body, parsed
                                            as block schemas (e.g. get_image_features).
            method_name (str | None): The name of the parsed method when the block
                                            describes a method instead of the forward.
            input_defaults (dict[str, Any] | None): The default values of the
                                            optional inputs, by parameter name.
            skipped (dict[str, str] | None): The children of the module that
                                            could not be parsed, by name with
                                            their skip reasons. They are
                                            reached at run time through the
                                            source module instead.
        """

        # Store here the model input(s) representation(s)
        self.inputs: dict[str, TensorRepr] = inputs

        # Store here the model output(s) representation(s)
        self.outputs: list[TensorRepr] = outputs

        # Store here the weights of the block
        self.weights: dict[str, TensorReprWithValue] = weights

        # Store here the model sub blocks
        self.sub_blocks: dict[str, ModelBlockSchema] = sub_blocks

        # Store here the model body list of instructions / expressions
        self.body: list[Instruction] = body

        # Store here the source module of the block
        self.module: Any | None = module

        # Store here the other methods of the module called by the body
        self.methods: dict[str, ModelBlockSchema] = methods or {}

        # Store here the default values of the optional inputs
        self.input_defaults: dict[str, Any] = input_defaults or {}

        # Store here the skipped children re-dispatched at run time
        self.skipped: dict[str, str] = skipped or {}

        # Store here the name of the described method, if not the forward
        self.method_name: str | None = method_name

    def sub_block_call_sites(self) -> dict[str, list[int]]:
        """
        Get the body indices where each sub-block of the block is called.

        The keys are the block reference paths of the body call sites: the
        attribute name of a direct sub-block (e.g. "attention"), the
        container path of a ModuleList or ModuleDict iteration (e.g.
        "experts", one static call site for the whole container whatever
        the number of its runtime iterations) or "$self" for an indexed
        self call. The values are the sorted indices of the calls in the
        top level body of the block: a call nested in a control flow or
        try body is reported at the index of its outermost enclosing
        instruction, which is the anchor of the call in a linear layout of
        the body operations.

        Returns:
            dict[str, list[int]]: The call site indices of each referenced
            sub-block path, in body order.
        """

        # One entry per referenced path, filled by the recursive walk
        sites: dict[str, list[int]] = {}

        # The top level instructions are their own enclosing indices
        index: int
        instruction: Instruction
        for index, instruction in enumerate(self.body):
            _collect_call_sites(
                instructions=[instruction],
                top_index=index,
                sites=sites
            )
        return sites


def _collect_call_sites(
    instructions: list[Instruction],
    top_index: int,
    sites: dict[str, list[int]]
) -> None:
    """
    Collect the sub-block call sites of a list of instructions.

    Args:
        instructions (list[Instruction]): The instructions to walk.
        top_index (int): The top level body index reported for the call
            sites found in the instructions.
        sites (dict[str, list[int]]): The call sites to fill, by referenced
            sub-block path.
    """

    # Walk the instructions, recursing into the nested bodies
    instruction: Instruction
    for instruction in instructions:

        # A block call is a call site of its referenced sub-block
        if isinstance(instruction, BlockInstruction):
            sites.setdefault(instruction.block_ref.path, []).append(top_index)

        # The control flow bodies nest their own call sites
        elif isinstance(instruction, ControlInstruction):
            _collect_call_sites(
                instructions=instruction.body,
                top_index=top_index,
                sites=sites
            )
            _collect_call_sites(
                instructions=instruction.else_body,
                top_index=top_index,
                sites=sites
            )

        # The try bodies nest their own call sites
        elif isinstance(instruction, TryInstruction):
            _collect_call_sites(
                instructions=instruction.body,
                top_index=top_index,
                sites=sites
            )
            handler_types: tuple[type[BaseException], ...]
            handler_body: list[Instruction]
            for handler_types, handler_body in instruction.handlers:
                _collect_call_sites(
                    instructions=handler_body,
                    top_index=top_index,
                    sites=sites
                )
            _collect_call_sites(
                instructions=instruction.final_body,
                top_index=top_index,
                sites=sites
            )
