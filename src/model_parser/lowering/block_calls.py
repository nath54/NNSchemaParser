"""
Lowering of the sub-block and sub-block container calls of a forward method.

The block calls are lowered to BlockInstruction objects: the direct sub-block
calls, the indexed container entry calls and the chained container calls.
"""

# Import Modules
from typing import Any

import ast

from src.core.instructions import BlockInstruction, BlockRef, Instruction, VarRef
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.container_keys import container_entry_keys
from src.model_parser.lowering.errors import UnsupportedSyntaxError


class BlockCallLowering(LoweringContext):
    """
    Lowers the sub-block calls of a forward method into instructions.
    """

    def _lower_container_call(
        self,
        container_name: str,
        call: ast.Call,
        body: list[Instruction],
        out_var: str
    ) -> str:
        """
        Lower a call to a sub-block container into chained block instructions.

        Args:
            container_name (str): The name of the called container.
            call (ast.Call): The call expression of the container.
            body (list[Instruction]): The body to append the instructions to.
            out_var (str): The name of the variable the result is bound to.

        Returns:
            str: The name of the variable the result is bound to.
        """

        # The call must not use keyword arguments
        if call.keywords:
            raise UnsupportedSyntaxError(
                "Keyword arguments are not supported for container calls", call
            )

        # Lower the arguments of the call for the first entry of the chain
        lowered_args: list[VarRef | Any]
        lowered_kwargs: dict[str, VarRef | Any]
        lowered_args, lowered_kwargs = self._lower_call_args(
            call=call,
            body=body
        )
        chained_args: list[VarRef | Any] = lowered_args

        # Get the ordered entry keys of the called container
        entry_keys: list[str] = container_entry_keys(
            sub_block_names=self.sub_block_names,
            container_name=container_name
        )

        # Chain the entries of the container from the first to the last
        entry_index: int
        entry_key: str
        for entry_index, entry_key in enumerate(entry_keys):

            # The last entry of the chain binds the output variable
            entry_out_var: str = (
                out_var if entry_index == len(entry_keys) - 1 else self._new_temp()
            )

            # The first entry gets the lowered arguments of the call
            if entry_index == 0:
                self._emit_entry_call(
                    entry_key=entry_key,
                    args=chained_args,
                    out_var=entry_out_var,
                    body=body
                )

            # The next entries unpack the tuple result of the previous entry
            else:
                self._emit_splat_entry_call(
                    entry_key=entry_key,
                    previous=chained_args[0],
                    out_var=entry_out_var,
                    body=body
                )

            # The result of the entry is the chained argument tuple
            chained_args = [VarRef(name=entry_out_var)]

        # Return the output variable name of the call
        return out_var

    def _emit_entry_call(
        self,
        entry_key: str,
        args: list[VarRef | Any],
        out_var: str,
        body: list[Instruction]
    ) -> None:
        """
        Append the block instruction of the first entry of a container chain.

        Args:
            entry_key (str): The sub-block key of the chained entry.
            args (list[VarRef | Any]): The positional arguments of the call.
            out_var (str): The name of the variable the result is bound to.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Append the block instruction of the container entry call
        body.append(BlockInstruction(
            block_ref=BlockRef(path=entry_key),
            args=args,
            out_var=out_var
        ))

    def _emit_splat_entry_call(
        self,
        entry_key: str,
        previous: VarRef,
        out_var: str,
        body: list[Instruction]
    ) -> None:
        """
        Append the block instruction of a chained container entry unpacking.

        Args:
            entry_key (str): The sub-block key of the chained entry.
            previous (VarRef): The reference to the previous entry result.
            out_var (str): The name of the variable the result is bound to.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Append the splat block instruction of the container entry call
        body.append(BlockInstruction(
            block_ref=BlockRef(path=entry_key),
            args=[previous],
            out_var=out_var,
            splat_args=True
        ))

    def _lower_block_call(
        self,
        block_ref: BlockRef,
        call: ast.Call,
        body: list[Instruction],
        out_var: str
    ) -> str:
        """
        Lower a call to a sub-block into a block instruction.

        Args:
            block_ref (BlockRef): The reference to the called sub-block.
            call (ast.Call): The call expression of the sub-block.
            body (list[Instruction]): The body to append the instruction to.
            out_var (str): The name of the variable the result is bound to.

        Returns:
            str: The name of the variable the result is bound to.
        """

        # Lower all the positional and the keyword arguments of the call
        args: list[VarRef | Any]
        kwargs: dict[str, VarRef | Any]
        args, kwargs = self._lower_call_args(call=call, body=body)

        # Append the block instruction of the call
        body.append(BlockInstruction(
            block_ref=block_ref,
            args=args,
            out_var=out_var,
            kwargs=kwargs
        ))

        # Return the output variable name of the call
        return out_var

    def _lower_container_index(self, index_expr: ast.expr, body: list[Instruction]) -> str:
        """
        Lower the index expression of a container sub-block call.

        Args:
            index_expr (ast.expr): The index expression of the container entry.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            str: The name of the scope variable holding the index.
        """

        # A variable index is used directly
        if isinstance(index_expr, ast.Name):
            return index_expr.id

        # Else the index expression is lowered to a temporary variable
        index_value: VarRef | Any = self._lower_expr(expr=index_expr, body=body)
        index_var: str = self._new_temp()
        self._emit_op(
            target=identity,
            qualified_name="index",
            args=[index_value],
            kwargs={},
            out_var=index_var,
            body=body
        )

        # Return the temporary variable name of the index
        return index_var

    def _lower_call_args(
        self,
        call: ast.Call,
        body: list[Instruction]
    ) -> tuple[list[VarRef | Any], dict[str, VarRef | Any]]:
        """
        Lower the positional and the keyword arguments of a call.

        Args:
            call (ast.Call): The call expression whose arguments are lowered.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            tuple[list[VarRef | Any], dict[str, VarRef | Any]]: The lowered
                positional arguments and the lowered keyword arguments.
        """

        # Lower all the positional arguments of the call
        args: list[VarRef | Any] = []
        arg: ast.expr
        for arg in call.args:
            args.append(self._lower_expr(expr=arg, body=body))

        # Lower all the keyword arguments of the call
        kwargs: dict[str, VarRef | Any] = {}
        kw: ast.keyword
        for kw in call.keywords:
            kwargs[kw.arg] = self._lower_expr(expr=kw.value, body=body)

        # Return the lowered arguments of the call
        return args, kwargs
