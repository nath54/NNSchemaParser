"""
Lowering of the expressions of a forward method into instructions.

Each expression is lowered into a variable reference of the block scope or a
literal value, emitting the intermediate operations into the body.
"""

# Import Modules
from typing import Any

import ast
import builtins
import operator

from src.core.instructions import ControlInstruction, Instruction, OpInstruction, VarRef
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.lowering.errors import UnsupportedSyntaxError
from src.model_parser.lowering.naming import RESOLVE_NAMESPACE, resolve_dotted_name
from src.model_parser.lowering.runtime_ops import make_list, make_tuple

# The prefix of the parameters of the compiled comprehension helpers
_COMP_PARAM_PREFIX: str = "__nnviewer_"




def _comprehension_target_names(target: ast.expr) -> list[str]:
    """
    Collect the bound names of a comprehension loop target.

    Args:
        target (ast.expr): The loop target expression of the comprehension.

    Returns:
        list[str]: The names bound by the target, in encounter order.
    """

    # A single name target
    if isinstance(target, ast.Name):
        return [target.id]

    # A tuple or list target binds all of its element names
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        element: ast.expr
        for element in target.elts:
            names.extend(_comprehension_target_names(target=element))
        return names

    # Other target kinds bind no plain name
    return []


class _ComprehensionNameRewriter(ast.NodeTransformer):
    """
    Rewrites the free names of a comprehension element or condition into
    parameters of the compiled comprehension helper.
    """

    def __init__(self, free_names: dict[str, str], target_names: set[str]) -> None:
        """
        Init function.

        Args:
            free_names (dict[str, str]): The mapping of the source names to
                the helper parameter names.
            target_names (set[str]): The names bound by the loop targets.
        """

        # Init parent
        super().__init__()

        # Keep the name mappings and the loop bound names
        self.free_names: dict[str, str] = free_names
        self.target_names: set[str] = target_names

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        """
        Rewrite a name access into the matching helper parameter.

        Args:
            node (ast.Name): The name node to rewrite.

        Returns:
            ast.Name: The rewritten name node.
        """

        # The loop bound names are kept as they are
        if node.id in self.target_names:
            return node

        # A free name is rewritten into its helper parameter
        if node.id in self.free_names:
            return ast.copy_location(
                ast.Name(id=self.free_names[node.id], ctx=node.ctx), node
            )

        # Any other name (a builtin) is kept as it is
        return node


def _compile_comprehension(
    kind: str,
    element: ast.expr,
    generators: list[ast.comprehension],
    free_names: dict[str, str],
    target_names: set[str],
    env: dict[str, Any]
) -> Any:
    """
    Compile a comprehension into a callable taking the iterable and the free
    names as positional arguments.

    Args:
        kind (str): The constructor of the comprehension result ("list",
            "set" or "dict").
        element (ast.expr): The element expression of the comprehension.
        generators (list[ast.comprehension]): The generators of the
            comprehension (only the first is supported).
        free_names (dict[str, str]): The mapping of the source names to the
            helper parameter names.
        target_names (set[str]): The names bound by the loop targets.
        env (dict[str, Any]): The globals of the compiled helper.

    Returns:
        Any: The compiled helper function.
    """

    # The parameter names of the helper
    iterable_param: str = _COMP_PARAM_PREFIX + "it0"
    param_names: list[str] = [
        _COMP_PARAM_PREFIX + name for name in free_names
    ]

    # The rewritten element and conditions of the comprehension
    rewriter: _ComprehensionNameRewriter = _ComprehensionNameRewriter(
        free_names={name: _COMP_PARAM_PREFIX + name for name in free_names},
        target_names=target_names
    )
    # A dict comprehension element is the key value pair of the mapping
    element_source: str
    if isinstance(element, ast.DictComp):
        key: ast.expr = rewriter.visit(element.key)
        value: ast.expr = rewriter.visit(element.value)
        element_source = ast.unparse(
            ast.copy_location(
                ast.Tuple(elts=[key, value], ctx=ast.Load()), element
            )
        )
    else:
        element_source = ast.unparse(rewriter.visit(element))
    rewritten_conditions: list[ast.expr] = [
        rewriter.visit(condition) for condition in generators[0].ifs
    ]

    # The source of the generator part of the comprehension
    generator_source: str = element_source
    generator_source += " for " + ast.unparse(generators[0].target)
    generator_source += " in " + iterable_param
    condition_source: ast.expr
    for condition_source in rewritten_conditions:
        generator_source += " if " + ast.unparse(condition_source)

    # The dict kind needs a key value pair syntax
    if kind == "dict":
        helper_source: str = (
            "def " + _COMP_PARAM_PREFIX + "comp(" + ", ".join(
                [iterable_param] + param_names
            ) + "):\n"
            "    return dict([" + generator_source + "])"
        )
    else:
        helper_source = (
            "def " + _COMP_PARAM_PREFIX + "comp(" + ", ".join(
                [iterable_param] + param_names
            ) + "):\n"
            "    return " + kind + "([" + generator_source + "])"
        )

    # Compile the helper with the globals of the forward method
    namespace: dict[str, Any] = dict(env)
    exec(compile(helper_source, "<nnviewer_comprehension>", "exec"), namespace)  # noqa: S102
    return namespace[_COMP_PARAM_PREFIX + "comp"]


class ExpressionLowering(LoweringContext):
    """
    Lowers the expressions of a forward method into instructions.
    """

    def _lower_expr(self, expr: ast.expr, body: list[Instruction]) -> VarRef | Any:
        """
        Lower an expression, returning a variable reference or a literal value.

        Args:
            expr (ast.expr): The expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # A variable name is a direct reference to the scope
        if isinstance(expr, ast.Name):

            # The self name is a reference to the source module at run time
            if expr.id == "self":
                return VarRef(name="$self")

            # The forward parameters are always references to the scope
            if expr.id in self._param_names:
                return VarRef(name=expr.id)

            # A global class or function is embedded as a constant value
            if (
                expr.id in self.forward_globals
                or expr.id in RESOLVE_NAMESPACE
                or hasattr(builtins, expr.id)
            ):
                return resolve_dotted_name(
                    dotted=expr.id,
                    node=expr,
                    forward_globals=self.forward_globals
                )

            # Else the name is a reference to the scope
            return VarRef(name=expr.id)

        # A constant is a literal value
        if isinstance(expr, ast.Constant):
            return expr.value

        # A call is lowered to an instruction bound to a temporary variable
        if isinstance(expr, ast.Call):
            out_var: str = self._new_temp()
            self._lower_call(call=expr, body=body, out_var=out_var)
            return VarRef(name=out_var)

        # An attribute access is handled depending on its object
        if isinstance(expr, ast.Attribute):
            return self._lower_attribute(expr=expr, body=body)

        # A subscript is lowered to a getitem operation
        if isinstance(expr, ast.Subscript):
            return self._lower_subscript(expr=expr, body=body)

        # A binary, unary, comparison or boolean operation
        if isinstance(expr, ast.BinOp):
            return self._lower_bin_op(expr=expr, body=body)
        if isinstance(expr, ast.UnaryOp):
            return self._lower_unary_op(expr=expr, body=body)
        if isinstance(expr, ast.Compare):
            return self._lower_compare(expr=expr, body=body)
        if isinstance(expr, ast.BoolOp):
            return self._lower_bool_op(expr=expr, body=body)

        # A tuple or list display is lowered to its constructor
        if isinstance(expr, (ast.Tuple, ast.List)):
            return self._lower_sequence(expr=expr, body=body)

        # A conditional expression is lowered to an if instruction binding
        if isinstance(expr, ast.IfExp):
            return self._lower_ifexp(expr=expr, body=body)

        # An f-string is lowered to its formatted parts joined together
        if isinstance(expr, ast.JoinedStr):
            return self._lower_joined_str(expr=expr, body=body)

        # A formatted value is lowered to its formatted value
        if isinstance(expr, ast.FormattedValue):
            return self._lower_formatted_value(expr=expr, body=body)

        # A comprehension is compiled into a helper called at run time
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return self._lower_comprehension(expr=expr, body=body)

        # Any other expression is not supported
        raise UnsupportedSyntaxError(
            f"Unsupported expression: {type(expr).__name__}", expr
        )

    def _lower_comprehension(self, expr: ast.expr, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a comprehension into a call of a compiled helper.

        The helper is compiled from the source of the comprehension at parse
        time, taking the iterable and the free names of the element as
        positional arguments resolved from the scope at run time.

        Args:
            expr (ast.expr): The comprehension expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result of the helper.
        """

        # The kind of the comprehension
        kind: str
        if isinstance(expr, (ast.ListComp, ast.GeneratorExp)):
            kind = "list"
        elif isinstance(expr, ast.SetComp):
            kind = "set"
        else:
            kind = "dict"

        # Only a single generator is supported
        if len(expr.generators) != 1:
            raise UnsupportedSyntaxError(
                "Comprehensions with several generators are not supported", expr
            )

        # The names bound by the loop target of the generator
        target_names: set[str] = set(
            _comprehension_target_names(target=expr.generators[0].target)
        )

        # The element nodes of the comprehension
        element_nodes: list[ast.expr] = (
            [expr.key, expr.value] if isinstance(expr, ast.DictComp) else [expr.elt]
        )

        # The free names of the element and of the conditions
        free_name_nodes: list[ast.Name] = []
        element_node: ast.expr
        for element_node in element_nodes:
            free_name_nodes.extend([
                node for node in ast.walk(element_node)
                if isinstance(node, ast.Name) and node.id not in target_names
            ])
        condition: ast.expr
        for condition in expr.generators[0].ifs:
            free_name_nodes.extend([
                node for node in ast.walk(condition)
                if isinstance(node, ast.Name) and node.id not in target_names
            ])
        free_names: list[str] = []
        node: ast.Name
        for node in free_name_nodes:
            if node.id in target_names or node.id in free_names:
                continue
            # The builtins are resolved from the globals of the helper
            if hasattr(builtins, node.id):
                continue
            free_names.append(node.id)

        # The iterable of the generator lowered in the body
        iterable: VarRef | Any = self._lower_expr(
            expr=expr.generators[0].iter, body=body
        )

        # The scope reference of each free name
        arg_refs: list[VarRef | Any] = [iterable]
        arg_names: list[str] = []
        free_name: str
        for free_name in free_names:
            if free_name == "self":
                arg_refs.append(VarRef(name="$self"))
            else:
                arg_refs.append(VarRef(name=free_name))
            arg_names.append(free_name)

        # The compiled helper of the comprehension
        helper: Any = _compile_comprehension(
            kind=kind,
            element=expr if isinstance(expr, ast.DictComp) else expr.elt,
            generators=expr.generators,
            free_names={name: name for name in arg_names},
            target_names=target_names,
            env=self.forward_globals
        )

        # The call of the helper binds the result of the comprehension
        out_var: str = self._new_temp()
        self._emit_op(
            target=helper,
            qualified_name=type(expr).__name__,
            args=arg_refs,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the result of the comprehension
        return VarRef(name=out_var)

    def _lower_attribute(self, expr: ast.Attribute, body: list[Instruction]) -> VarRef | Any:
        """
        Lower an attribute access expression.

        Args:
            expr (ast.Attribute): The attribute access expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # An attribute of self is resolved in the symbol table of the module
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            return self._resolve_self_attr(
                attr=expr.attr,
                node=expr,
                body=body
            )

        # Else the object is lowered then accessed with a getattr operation
        obj: VarRef | Any = self._lower_expr(expr=expr.value, body=body)
        out_var: str = self._new_temp()
        self._emit_op(
            target=getattr,
            qualified_name="getattr",
            args=[obj, expr.attr],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the attribute value
        return VarRef(name=out_var)

    def _resolve_self_attr(
        self,
        attr: str,
        node: ast.AST,
        body: list[Instruction]
    ) -> VarRef | Any:
        """
        Resolve an attribute of self in the symbol table of the module.

        Args:
            attr (str): The name of the attribute of self.
            node (ast.AST): The node of the access, for error reporting.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the attribute value, or a
                literal.
        """

        # A weight or buffer is referenced by its name in the block scope,
        # unless the name is shadowed by a parameter or a local variable of
        # the forward, in which case the runtime attribute is used instead
        if attr in self.weight_names and attr not in self._param_names                 and attr not in self._local_names:
            return VarRef(name=attr)

        # A literal attribute is embedded as a constant value
        if attr in self.literals:
            return self.literals[attr]

        # Else the attribute is resolved on the source module at run time
        out_var: str = self._new_temp()
        self._emit_op(
            target=getattr,
            qualified_name="getattr",
            args=[VarRef(name="$self"), attr],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the attribute value
        return VarRef(name=out_var)

    def _lower_subscript(self, expr: ast.Subscript, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a subscript expression to a getitem operation.

        Args:
            expr (ast.Subscript): The subscript expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # A subscript of a container of self is only valid as a called function
        if (
            isinstance(expr.value, ast.Attribute)
            and isinstance(expr.value.value, ast.Name)
            and expr.value.value.id == "self"
            and expr.value.attr in self.container_names
        ):
            raise UnsupportedSyntaxError(
                "A subscript of a container is only valid as a called function", expr
            )

        # Lower the object and the index of the subscript
        obj: VarRef | Any = self._lower_expr(expr=expr.value, body=body)
        index: VarRef | Any = self._lower_index(index=expr.slice, body=body)

        # Emit the getitem operation of the subscript
        out_var: str = self._new_temp()
        self._emit_op(
            target=operator.getitem,
            qualified_name="getitem",
            args=[obj, index],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the subscript value
        return VarRef(name=out_var)

    def _lower_index(self, index: ast.expr, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a subscript index expression, including slices.

        Args:
            index (ast.expr): The index expression of the subscript.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the index, or a literal.
        """

        # A slice is lowered to the slice constructor
        if isinstance(index, ast.Slice):
            return self._lower_slice(index=index, body=body)

        # A tuple index may contain slices (e.g. x[:, 0:10])
        if isinstance(index, ast.Tuple):

            # Lower all the elements of the tuple index
            elements: list[VarRef | Any] = []
            element: ast.expr
            for element in index.elts:
                elements.append(self._lower_index(index=element, body=body))

            # Emit the tuple constructor with the lowered elements
            out_var: str = self._new_temp()
            self._emit_op(
                target=make_tuple,
                qualified_name="tuple",
                args=elements,
                kwargs={},
                out_var=out_var,
                body=body
            )

            # Return the reference to the tuple index
            return VarRef(name=out_var)

        # Else the index is a regular expression
        return self._lower_expr(expr=index, body=body)

    def _lower_slice(self, index: ast.Slice, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a slice expression to the slice constructor.

        Args:
            index (ast.Slice): The slice expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the built slice.
        """

        # Lower the bounds of the slice, the missing bounds are None
        bounds: list[VarRef | Any] = []
        bound: ast.expr | None
        for bound in (index.lower, index.upper, index.step):
            if bound is None:
                bounds.append(None)
            else:
                bounds.append(self._lower_expr(expr=bound, body=body))

        # Emit the slice constructor with the bounds of the slice
        out_var: str = self._new_temp()
        self._emit_op(
            target=slice,
            qualified_name="slice",
            args=bounds,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the built slice
        return VarRef(name=out_var)

    def _lower_sequence(self, expr: ast.Tuple | ast.List, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a tuple or list display to its constructor.

        Args:
            expr (ast.Tuple | ast.List): The sequence display expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the constructed sequence.
        """

        # Lower all the elements of the sequence
        elements: list[VarRef | Any] = []
        element: ast.expr
        for element in expr.elts:
            elements.append(self._lower_expr(expr=element, body=body))

        # Emit the constructor of the sequence with its elements
        target: Any = make_tuple if isinstance(expr, ast.Tuple) else make_list
        out_var: str = self._new_temp()
        self._emit_op(
            target=target,
            qualified_name="tuple" if isinstance(expr, ast.Tuple) else "list",
            args=elements,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the constructed sequence
        return VarRef(name=out_var)

    def _lower_ifexp(self, expr: ast.IfExp, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a conditional expression to an if instruction binding its output.

        Args:
            expr (ast.IfExp): The conditional expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the selected value.
        """

        # Lower the condition of the conditional expression
        condition: OpInstruction = self._lower_condition(
            test=expr.test,
            body=body
        )

        # Lower the two branches binding the same output variable
        out_var: str = self._new_temp()
        if_body: list[Instruction] = []
        else_body: list[Instruction] = []
        self._lower_assign_value(value=expr.body, out_var=out_var, body=if_body)
        self._lower_assign_value(value=expr.orelse, out_var=out_var, body=else_body)

        # Append the control instruction of the conditional expression
        body.append(ControlInstruction(
            kind="if",
            condition=condition,
            body=if_body,
            else_body=else_body,
            bind_var=None,
            out_var=out_var
        ))

        # Return the reference to the selected value
        return VarRef(name=out_var)
