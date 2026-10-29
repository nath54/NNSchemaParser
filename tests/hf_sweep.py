"""
Robustness sweep of the NNSchemaParser parser over many pretrained architectures.

Each model of the registry is loaded from the Hugging Face hub (or built
locally), parsed into a ModelBlockSchema, executed by the schema interpreter
and compared with the real forward. The schema and the inference trace are
written to outputs/hf_sweep/<name>/ and the downloaded weights are deleted
right after each model. The failures are appended to .errors_to_fix.md.

Run with: python -m tests.hf_sweep --range 0:10 [--only substr] [--category
Text] [--force]
"""

# Import Modules
import os
from pathlib import Path

# Redirect the Hugging Face cache before importing transformers
_ROOT: Path = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_ROOT / "outputs" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gc  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any  # noqa: E402

sys.path.insert(0, str(_ROOT))  # noqa: E402

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from nnschemaparser.core.model_schema import ModelBlockSchema  # noqa: E402
from nnschemaparser.parser.children import SKIPPED_CHILDREN, collect_sub_blocks  # noqa: E402
from nnschemaparser.interpreter import SchemaInterpreter  # noqa: E402
from nnschemaparser.parser.parser import NNParser  # noqa: E402
from nnschemaparser.dump import describe_value, dump_schema  # noqa: E402
from nnschemaparser.interpreter import SchemaInterpreter, TraceRecorder  # noqa: E402
from tests.hf_registry import MODEL_SPECS, ModelSpec  # noqa: E402

# The output directories of the sweep
SWEEP_DIR: Path = _ROOT / "outputs" / "hf_sweep"
ERRORS_FILE: Path = _ROOT / ".errors_to_fix.md"


def collect_tensors(value: Any) -> list[Tensor]:
    """
    Collect recursively all the tensors of a model output.

    Args:
        value (Any): The output value, possibly nested.

    Returns:
        list[Tensor]: The tensors in encounter order.
    """

    # The tensors themselves
    if isinstance(value, Tensor):
        return [value]

    # The sequences and the model output mappings
    if isinstance(value, (tuple, list)):
        items: list[Tensor] = []
        item: Any
        for item in value:
            items.extend(collect_tensors(item))
        return items
    if isinstance(value, dict):
        items = []
        for item in value.values():
            items.extend(collect_tensors(item))
        return items

    # The dataclass like outputs of the transformers models
    if hasattr(value, "__dataclass_fields__"):
        items = []
        for field_name in value.__dataclass_fields__:
            items.extend(collect_tensors(getattr(value, field_name)))
        return items

    # Any other value carries no tensor
    return []


def build_inputs(kind: str, config: Any) -> dict[str, Tensor]:
    """
    Build the dummy inputs of a loaded model from its input kind.

    Args:
        kind (str): The input kind of the model spec.
        config (Any): The configuration of the loaded model.

    Returns:
        dict[str, Tensor]: The dummy inputs of the forward.
    """

    # The text ids bounded by the vocabulary size
    vocab: int = int(getattr(config, "vocab_size", 30000) or 30000)
    high: int = max(2, min(vocab - 1, 900))
    input_ids: Tensor = torch.randint(0, high, (2, 16))
    decoder_ids: Tensor = torch.randint(0, high, (2, 8))
    image_size: int = int(getattr(config, "image_size", 224) or 224)

    # The inputs by kind
    if kind == "text":
        return {"input_ids": input_ids, "attention_mask": torch.ones(2, 16)}
    if kind == "encdec":
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(2, 16),
            "decoder_input_ids": decoder_ids
        }
    if kind == "vision":
        return {"pixel_values": torch.randn(2, 3, image_size, image_size)}
    if kind == "audio_raw":
        return {"input_values": torch.randn(2, 16000)}
    if kind == "audio_spec":
        return {"input_values": torch.randn(2, 1024, 128)}
    if kind == "speech_spec":
        return {
            "input_features": torch.randn(2, 80, 3000),
            "decoder_input_ids": decoder_ids
        }
    if kind == "clip":
        return {
            "input_ids": input_ids,
            "pixel_values": torch.randn(2, 3, image_size, image_size)
        }
    if kind == "layout":
        return {
            "input_ids": input_ids,
            "bbox": torch.randint(0, 500, (2, 16, 4)).sort(dim=2).values,
            "attention_mask": torch.ones(2, 16)
        }
    if kind == "detr":
        return {
            "pixel_values": torch.randn(2, 3, image_size, image_size),
            "pixel_mask": torch.ones(2, image_size, image_size)
        }
    if kind == "vision_encdec":
        encoder_size: int = int(
            getattr(getattr(config, "encoder", config), "image_size", 224)
            or 224
        )
        return {
            "pixel_values": torch.randn(2, 3, encoder_size, encoder_size),
            "decoder_input_ids": decoder_ids
        }
    if kind == "vits":
        return {"input_ids": input_ids}
    if kind == "processor":
        return _processor_inputs(config=config)
    if kind == "audio_processor":
        return _audio_processor_inputs(config=config)
    if kind == "s2t_spec":
        return {
            "input_features": torch.randn(2, 3000, 80),
            "decoder_input_ids": decoder_ids
        }
    if kind == "audio_hidden":
        return {"hidden_states": torch.randn(2, 16000)}

    # The unknown kind builds no input
    raise ValueError(f"Unknown input kind: {kind}")


def _audio_processor_inputs(config: Any) -> dict[str, Tensor]:
    """Build the inputs of an audio model through its feature extractor."""

    # White noise speech processed by the feature extractor of the model
    import numpy
    from transformers import AutoFeatureExtractor

    # The repository of the extractor is the one of the config
    repo: str = getattr(config, "name_or_path", "")
    extractor: Any = AutoFeatureExtractor.from_pretrained(repo)
    sampling_rate: int = int(getattr(extractor, "sampling_rate", 16000))
    duration: int = max(1, int(getattr(extractor, "chunk_length", 0) and
        extractor.chunk_length * 0.99) or int(10.24 * sampling_rate / 1000))
    speech: numpy.ndarray = numpy.random.randn(
        2, duration * sampling_rate // 1000 if duration > 100 else 102400
    ).astype("float32")
    processed: Any = extractor(
        speech.tolist(), sampling_rate=sampling_rate,
        return_tensors="pt", padding=True
    )
    return {
        key: value for key, value in processed.items()
        if isinstance(value, Tensor)
    }


def _processor_inputs(config: Any) -> dict[str, Tensor]:
    """Build the inputs of a model through its image processor."""

    # A dummy image pair processed by the image processor of the model
    import numpy
    from PIL import Image
    from transformers import AutoImageProcessor

    # The repository of the processor is the one of the config
    repo: str = getattr(config, "name_or_path", "")
    processor: Any = AutoImageProcessor.from_pretrained(repo)
    array = (numpy.random.rand(480, 640, 3) * 255).astype("uint8")
    images: list[Image.Image] = [
        Image.fromarray(array), Image.fromarray(array)
    ]
    processed: Any = processor(images=images, return_tensors="pt")
    return {
        key: value for key, value in processed.items()
        if isinstance(value, Tensor)
    }


def load_model(spec: ModelSpec) -> tuple[nn.Module, dict[str, Tensor]]:
    """
    Load a model of the registry with its dummy inputs.

    Args:
        spec (ModelSpec): The registry entry of the model.

    Returns:
        tuple[nn.Module, dict[str, Tensor]]: The model in eval mode and its
            dummy inputs.
    """

    # The local builders return the model and the inputs directly
    if spec.loader == "local":
        assert spec.builder is not None
        model, inputs = spec.builder()
        return model.eval(), inputs

    # The Hugging Face models are loaded through their auto class
    import transformers

    auto_class: Any = getattr(transformers, spec.loader)
    model: nn.Module = auto_class.from_pretrained(spec.repo)
    model.eval()
    return model, build_inputs(kind=spec.input_kind, config=model.config)


def clear_hf_cache() -> None:
    """Delete the downloaded Hugging Face weights to free the disk."""

    # Remove every downloaded model directory of the hub cache
    hub_dir: Path = Path(os.environ["HF_HOME"]) / "hub"
    if not hub_dir.exists():
        return
    cached: Path
    for cached in hub_dir.glob("models--*"):
        shutil.rmtree(cached, ignore_errors=True)


def record_error(spec: ModelSpec, phase: str, error: BaseException) -> None:
    """
    Append or replace the error section of a model in .errors_to_fix.md.

    Args:
        spec (ModelSpec): The registry entry of the failing model.
        phase (str): The phase where the failure happened.
        error (BaseException): The raised exception.
    """

    # The section header of the model
    header: str = f"## [SWEEP] {spec.name} ({spec.category})"
    marker: str = f"## [SWEEP] "

    # Drop the previous section of the same model, if any
    if ERRORS_FILE.exists():
        content: str = ERRORS_FILE.read_text(encoding="utf-8")
        if header in content:
            before: str = content[:content.index(header)]
            rest: str = content[content.index(header):]
            next_index: int = rest.find(marker, 1)
            after: str = rest[next_index:] if next_index >= 0 else ""
            ERRORS_FILE.write_text(before + after, encoding="utf-8")

    # The new section with the phase, the message and the traceback
    section: str = (
        f"{header} — {phase}\n\n"
        f"**{type(error).__name__}**: {error}\n\n"
        "<details><summary>Traceback</summary>\n\n"
        "```py\n"
        + traceback.format_exc()
        + "\n```\n\n</details>\n\n"
    )
    with open(ERRORS_FILE, "a", encoding="utf-8") as handle:
        handle.write(section)


def run_model(spec: ModelSpec, force: bool = False) -> dict[str, Any]:
    """
    Test one model of the registry: parse, execute and compare.

    Args:
        spec (ModelSpec): The registry entry of the model.
        force (bool): Rerun the model even if a result already exists.

    Returns:
        dict[str, Any]: The result dictionary of the model.
    """

    # Skip the models already tested successfully
    model_dir: Path = SWEEP_DIR / spec.name
    result_file: Path = model_dir / "result.json"
    if result_file.exists() and not force:
        with open(result_file, encoding="utf-8") as handle:
            previous: dict[str, Any] = json.load(handle)
        if previous.get("status") == "OK":
            return previous

    # The result dictionary of the model
    result: dict[str, Any] = {
        "name": spec.name,
        "category": spec.category,
        "repo": spec.repo,
        "loader": spec.loader,
        "status": "OK",
        "maxdiffs": [],
        "durations": {}
    }
    model_dir.mkdir(parents=True, exist_ok=True)

    # The model and its inputs
    try:
        start: float = time.time()
        torch.manual_seed(0)
        model, inputs = load_model(spec=spec)
        result["durations"]["load"] = round(time.time() - start, 1)
        result["parameters"] = sum(
            p.numel() for p in model.parameters()
        )
    except Exception as error:  # noqa: BLE001 - recorded and continued
        result["status"] = f"LOAD_ERROR: {type(error).__name__}: {error}"
        record_error(spec=spec, phase="load", error=error)
        clear_hf_cache()
        _write_result(result_file, result)
        return result

    # The real forward on the dummy inputs, as the reference
    try:
        start = time.time()
        torch.manual_seed(0)
        with torch.no_grad():
            reference: Any = model(**inputs)
        result["durations"]["real_forward"] = round(time.time() - start, 1)
    except Exception as error:  # noqa: BLE001
        result["status"] = f"REAL_FORWARD_ERROR: {type(error).__name__}: {error}"
        record_error(spec=spec, phase="real forward", error=error)
        clear_hf_cache()
        _write_result(result_file, result)
        return result

    # The parse of the model into a block schema
    try:
        start = time.time()
        SKIPPED_CHILDREN.clear()
        schema = NNParser.parse_model(
            model=model,
            inputs_tensors_shapes={
                key: tuple(value.shape) for key, value in inputs.items()
            },
            inputs_tensors_dtypes={
                key: str(value.dtype) for key, value in inputs.items()
            },
            inputs_tensors_values=inputs
        )
        result["durations"]["parse"] = round(time.time() - start, 1)
        result["sub_blocks"] = len(schema.sub_blocks)
        result["instructions"] = _count_instructions(schema)
        result["skipped_children"] = list(SKIPPED_CHILDREN)
    except Exception as error:  # noqa: BLE001
        result["status"] = f"PARSE_ERROR: {type(error).__name__}: {error}"
        record_error(spec=spec, phase="parse", error=error)
        clear_hf_cache()
        _write_result(result_file, result)
        return result

    # The schema text dump, before the capture pollutes the shapes
    try:
        dump_schema(schema=schema, path=str(model_dir / "schema.txt"))
    except Exception as error:  # noqa: BLE001
        result["status"] = f"DUMP_ERROR: {type(error).__name__}: {error}"
        record_error(spec=spec, phase="schema dump", error=error)

    # The schema execution with the trace recorder and the captured values
    outputs: Any = None
    try:
        start = time.time()
        torch.manual_seed(0)
        tracer: TraceRecorder = TraceRecorder()
        # The capture runs without grad like the reference forward, as
        # some models produce slightly different values with grad
        with torch.no_grad():
            outputs = SchemaInterpreter(
                capture_values=True,
                listeners=[tracer]
            ).run(
                block=schema,
                scope={key: value.clone() for key, value in inputs.items()}
            )
        result["durations"]["execute"] = round(time.time() - start, 1)
        tracer.save(str(model_dir / "trace.txt"))
    except Exception as error:  # noqa: BLE001
        result["status"] = f"EXECUTE_ERROR: {type(error).__name__}: {error}"
        record_error(spec=spec, phase="schema execution", error=error)
        clear_hf_cache()
        _write_result(result_file, result)
        return result

    # The comparison of the schema outputs with the real forward
    try:
        reference_tensors: list[Tensor] = collect_tensors(reference)
        output_tensors: list[Tensor] = collect_tensors(outputs)
        if len(reference_tensors) != len(output_tensors):
            raise ValueError(
                f"Output count mismatch: {len(reference_tensors)} real vs "
                f"{len(output_tensors)} schema"
            )
        maxdiffs: list[float] = []
        index: int
        expected: Tensor
        actual: Tensor
        for index, (expected, actual) in enumerate(
            zip(reference_tensors, output_tensors)
        ):
            maxdiff: float = (expected - actual).abs().max().item()
            maxdiffs.append(maxdiff)
        result["maxdiffs"] = maxdiffs
        if maxdiffs and max(maxdiffs) > 1e-4:
            raise ValueError(f"Output mismatch, maxdiff: {max(maxdiffs):.6f}")
    except Exception as error:  # noqa: BLE001
        result["status"] = f"COMPARE_ERROR: {type(error).__name__}: {error}"
        record_error(spec=spec, phase="output comparison", error=error)

    # Cleanup of the model and of the downloaded weights
    del model, schema
    gc.collect()
    clear_hf_cache()
    _write_result(result_file, result)
    return result


def _count_instructions(schema: ModelBlockSchema) -> int:
    """
    Count all the instructions of a schema tree.

    Args:
        schema (ModelBlockSchema): The parsed schema.

    Returns:
        int: The total instruction count.
    """

    # The count of this block plus the recursive sub-blocks
    total: int = len(schema.body)
    sub: ModelBlockSchema
    for sub in schema.sub_blocks.values():
        total += _count_instructions(sub)
    return total


def _write_result(path: Path, result: dict[str, Any]) -> None:
    """
    Write the result dictionary of a model to its json file.

    Args:
        path (Path): The result json path.
        result (dict[str, Any]): The result to write.
    """

    # The json dump of the result
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def write_summary() -> None:
    """Write the summary markdown of all the recorded results."""

    # The result files of all the models
    lines: list[str] = [
        "# Hugging Face sweep results", "",
        "| model | category | status | maxdiff | instructions |",
        "|---|---|---|---|---|"
    ]
    spec: ModelSpec
    for spec in MODEL_SPECS:
        result_file: Path = SWEEP_DIR / spec.name / "result.json"
        if not result_file.exists():
            continue
        with open(result_file, encoding="utf-8") as handle:
            result: dict[str, Any] = json.load(handle)
        maxdiffs: list[float] = result.get("maxdiffs", [])
        maxdiff: str = f"{max(maxdiffs):.2e}" if maxdiffs else "-"
        lines.append(
            f"| {spec.name} | {spec.category} | {result['status']} "
            f"| {maxdiff} | {result.get('instructions', '-')} |"
        )

    # The summary file of the sweep
    with open(SWEEP_DIR / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    """Run the sweep over the selected registry entries."""

    # The command line selection of the models
    selection: list[ModelSpec] = list(MODEL_SPECS)
    arguments: list[str] = sys.argv[1:]
    if "--only" in arguments:
        needle: str = arguments[arguments.index("--only") + 1]
        selection = [spec for spec in selection if needle in spec.name]
    if "--category" in arguments:
        categories: set[str] = {
            arguments[index + 1]
            for index, argument in enumerate(arguments)
            if argument == "--category"
        }
        selection = [spec for spec in selection if spec.category in categories]
    if "--range" in arguments:
        start_s, end_s = arguments[arguments.index("--range") + 1].split(":")
        selection = selection[int(start_s):int(end_s)]
    force: bool = "--force" in arguments

    # The sequential test of each selected model
    spec: ModelSpec
    for spec in selection:
        print(f"=== {spec.name} ({spec.category}) {spec.repo} ===", flush=True)
        try:
            result: dict[str, Any] = run_model(spec=spec, force=force)
            print(f"    -> {result['status']}", flush=True)
        except Exception as error:  # noqa: BLE001 - never stop the sweep
            print(f"    -> CRASH: {type(error).__name__}: {error}", flush=True)
            record_error(spec=spec, phase="sweep harness", error=error)

    # The summary of all the recorded results
    write_summary()
    print("Sweep done.", flush=True)


if __name__ == "__main__":
    main()
