"""
Comparison of the tensor traces of two schema executions.

The compare_traces function aligns the steps of two TensorTrace instances —
typically the ActivationTraces or GradientTraces of a base run and a forked
run — by their container ids, with a role fallback matching the block path,
variable name and operation name when an edit changed the control path. Each
aligned pair produces a TraceDiff carrying the alignment status and, when
both sides hold captured values, the magnitude of the difference.
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor

from nnschemaparser.core.tensor_schema import TensorContainer
from nnschemaparser.core.traces import TensorTrace, TraceStep


class TraceDiff:
    """
    The comparison of one step of two tensor traces.

    A trace diff describes how one activation or gradient step of a first
    run relates to the steps of a second run: which container ids were
    aligned, by which matching strategy, whether the shapes agree and how
    much the captured values differ.
    """

    def __init__(
        self,
        container_id_a: str | None,
        container_id_b: str | None,
        status: str,
        matched_by: str,
        description: str,
        out_var: str,
        block_path: str,
        max_abs_diff: float | None = None,
        mean_abs_diff: float | None = None,
        changed: bool | None = None
    ) -> None:
        """
        Init function.

        Args:
            container_id_a (str | None): The container id of the step of
                the first trace, None for a step only in the second trace.
            container_id_b (str | None): The container id of the step of
                the second trace, None for a step only in the first trace.
            status (str): The alignment status, one of "ok" (both steps
                aligned), "shape_changed" (aligned but different shapes),
                "value_unavailable" (aligned without captured values),
                "only_in_a" or "only_in_b".
            matched_by (str): The matching strategy, "container_id" for an
                exact id match or "role" for the block path, variable name
                and operation name fallback.
            description (str): The human readable description of the step.
            out_var (str): The original variable name of the step.
            block_path (str): The attribute path of the producing block.
            max_abs_diff (float | None): The maximum absolute difference of
                the captured values, None when not computable.
            mean_abs_diff (float | None): The mean absolute difference of
                the captured values, None when not computable.
            changed (bool | None): Whether the difference exceeds the
                tolerance of the comparison, None when not computable.
        """

        # Store here the container ids of the two aligned steps
        self.container_id_a: str | None = container_id_a
        self.container_id_b: str | None = container_id_b

        # Store here the alignment status and the matching strategy
        self.status: str = status
        self.matched_by: str = matched_by

        # Store here the description of the compared step
        self.description: str = description
        self.out_var: str = out_var
        self.block_path: str = block_path

        # Store here the magnitude of the difference when computable
        self.max_abs_diff: float | None = max_abs_diff
        self.mean_abs_diff: float | None = mean_abs_diff
        self.changed: bool | None = changed


def _role_key(step: TraceStep) -> tuple[str, str, str]:
    """
    Get the role key of a trace step for the fallback matching.

    Args:
        step (TraceStep): The trace step to key.

    Returns:
        tuple[str, str, str]: The (block path, variable name, operation
        name) role of the step.
    """

    # The role of a step is where and what it computes
    return (step.block_path, step.out_var, step.op_name)


def _value_diff(
    tensor_a: TensorContainer,
    tensor_b: TensorContainer,
    atol: float
) -> tuple[float | None, float | None, bool | None]:
    """
    Compare the captured values of two aligned tensor containers.

    Args:
        tensor_a (TensorContainer): The container of the first trace.
        tensor_b (TensorContainer): The container of the second trace.
        atol (float): The tolerance above which a difference is a change.

    Returns:
        tuple[float | None, float | None, bool | None]: The maximum and
        mean absolute differences and the changed flag, or (None, None,
        None) when the values are not comparable.
    """

    # The shapes must agree for an elementwise comparison
    if tensor_a.shape != tensor_b.shape:
        return None, None, None

    # The captured values must be tensors
    value_a: Any = tensor_a.value
    value_b: Any = tensor_b.value
    if not isinstance(value_a, Tensor) or not isinstance(value_b, Tensor):
        return None, None, None

    # The elementwise difference of the two captured values
    delta: Tensor = (value_a.detach().to(torch.float32) -
                     value_b.detach().to(torch.float32)).abs()
    max_abs_diff: float = float(delta.max().item())
    mean_abs_diff: float = float(delta.mean().item())

    # The magnitude of the difference and the tolerance flag
    return max_abs_diff, mean_abs_diff, bool(max_abs_diff > atol)


def compare_traces(
    trace_a: TensorTrace,
    trace_b: TensorTrace,
    atol: float = 0.0
) -> list[TraceDiff]:
    """
    Compare the steps of two tensor traces step by step.

    The steps of the first trace are aligned with the steps of the second
    trace by their container ids; a step with no id match is aligned with
    the unmatched step of the second trace holding the same role (block
    path, variable name and operation name), which survives an edit that
    changed the control path of the model. The steps of the second trace
    matched by no step of the first are reported at the end.

    Args:
        trace_a (TensorTrace): The trace of the first run, defining the
            order of the returned diffs.
        trace_b (TensorTrace): The trace of the second run.
        atol (float): The tolerance above which a difference of captured
            values is reported as changed.

    Returns:
        list[TraceDiff]: The diffs of the aligned and unmatched steps.
    """

    # The steps of the second trace indexed by container id and by role
    steps_by_id: dict[str, TraceStep] = {
        step.container_id: step for step in trace_b.steps
    }
    steps_by_role: dict[tuple[str, str, str], list[TraceStep]] = {}
    step_b: TraceStep
    for step_b in trace_b.steps:
        steps_by_role.setdefault(_role_key(step=step_b), []).append(step_b)

    # The container ids of the second trace steps already matched
    matched_ids: set[str] = set()

    # The diffs of the first trace steps, in their order
    diffs: list[TraceDiff] = []
    step_a: TraceStep
    for step_a in trace_a.steps:

        # The exact container id match of the second trace
        matched: TraceStep | None = steps_by_id.get(step_a.container_id)
        matched_by: str = "container_id"

        # The role fallback match among the unmatched second trace steps
        if matched is None:
            role_key: tuple[str, str, str] = _role_key(step=step_a)
            role_candidates: list[TraceStep] = steps_by_role.get(role_key, [])
            role_candidate: TraceStep
            for role_candidate in role_candidates:
                if role_candidate.container_id not in matched_ids:
                    matched = role_candidate
                    matched_by = "role"
                    break

        # A step of the first trace matched by no step of the second
        if matched is None:
            diffs.append(TraceDiff(
                container_id_a=step_a.container_id,
                container_id_b=None,
                status="only_in_a",
                matched_by="none",
                description=step_a.description,
                out_var=step_a.out_var,
                block_path=step_a.block_path
            ))
            continue

        # Mark the matched step of the second trace as used
        matched_ids.add(matched.container_id)

        # The shapes of the two aligned containers
        shape_changed: bool = step_a.tensor.shape != matched.tensor.shape
        status: str = "shape_changed" if shape_changed else "ok"

        # The magnitude of the difference of the captured values
        max_abs_diff: float | None = None
        mean_abs_diff: float | None = None
        changed: bool | None = None
        if not shape_changed:
            max_abs_diff, mean_abs_diff, changed = _value_diff(
                tensor_a=step_a.tensor,
                tensor_b=matched.tensor,
                atol=atol
            )
            if changed is None:
                status = "value_unavailable"

        # The diff of the aligned pair
        diffs.append(TraceDiff(
            container_id_a=step_a.container_id,
            container_id_b=matched.container_id,
            status=status,
            matched_by=matched_by,
            description=step_a.description,
            out_var=step_a.out_var,
            block_path=step_a.block_path,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            changed=changed
        ))

    # The steps of the second trace matched by no step of the first
    step_b: TraceStep
    for step_b in trace_b.steps:
        if step_b.container_id not in matched_ids:
            diffs.append(TraceDiff(
                container_id_a=None,
                container_id_b=step_b.container_id,
                status="only_in_b",
                matched_by="none",
                description=step_b.description,
                out_var=step_b.out_var,
                block_path=step_b.block_path
            ))

    # The diffs of the two traces
    return diffs


def diff_summary(diffs: list[TraceDiff]) -> dict[str, int]:
    """
    Summarize the statuses of a trace comparison.

    Args:
        diffs (list[TraceDiff]): The diffs returned by compare_traces.

    Returns:
        dict[str, int]: The count of each alignment status.
    """

    # The count of each status of the diffs
    summary: dict[str, int] = {}
    diff: TraceDiff
    for diff in diffs:
        summary[diff.status] = summary.get(diff.status, 0) + 1
    return summary
