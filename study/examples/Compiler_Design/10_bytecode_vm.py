"""
10_bytecode_vm.py - Bytecode Compiler and Stack-Based Virtual Machine

Demonstrates the final phases of a compiler:
  1. Bytecode compiler: translates an AST into a sequence of bytecode instructions
  2. Virtual machine: executes the bytecode on a stack-based architecture

Architecture overview:
  - Stack machine: operands pushed, instructions consume/produce stack values
  - Operand stack: holds intermediate values during expression evaluation
  - Call stack (frames): local variables, return addresses, function arguments
  - Constant pool: stores literal values referenced by instructions

Instruction set:
  PUSH <val>      Push a constant value onto the stack
  POP             Discard the top of stack
  LOAD <name>     Load a local variable onto the stack
  STORE <name>    Pop and store to a local variable
  ADD, SUB, MUL, DIV, MOD    Binary arithmetic
  NEG             Negate top of stack
  EQ, NE, LT, GT, LE, GE     Comparison (pushes 0 or 1)
  AND, OR, NOT    Logical operators
  JUMP <offset>   Unconditional jump (relative offset)
  JUMP_IF_FALSE <offset>   Pop and jump if zero/false
  CALL <name> <argc>  Call a function
  RETURN          Return from function
  PRINT           Pop and print (for demo output)

Topics covered:
  - Bytecode instruction encoding
  - Stack-based expression evaluation
  - Function frames with local variables
  - Jump offsets for control flow
  - Bytecode disassembler
  - Recursive functions via call stack
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Instruction set
# ---------------------------------------------------------------------------

class Op(Enum):
    PUSH           = auto()   # PUSH <value>
    POP            = auto()   # POP
    LOAD           = auto()   # LOAD <name>
    STORE          = auto()   # STORE <name>
    ADD            = auto()
    SUB            = auto()
    MUL            = auto()
    DIV            = auto()
    MOD            = auto()
    NEG            = auto()
    EQ             = auto()
    NE             = auto()
    LT             = auto()
    GT             = auto()
    LE             = auto()
    GE             = auto()
    AND            = auto()
    OR             = auto()
    NOT            = auto()
    JUMP           = auto()   # JUMP <target_ip>  (absolute)
    JUMP_IF_FALSE  = auto()   # JUMP_IF_FALSE <target_ip>
    CALL           = auto()   # CALL <func_name> <argc>
    RETURN         = auto()   # RETURN (uses top of stack as return value)
    RETURN_NONE    = auto()   # RETURN with no value (void)
    PRINT          = auto()   # Pop and print
    HALT           = auto()   # Stop execution
    DUP            = auto()   # Duplicate top of stack


@dataclass
class Instruction:
    op: Op
    arg1: Any = None   # First operand (name, value, or jump target)
    arg2: Any = None   # Second operand (e.g., argc for CALL)

    def __repr__(self) -> str:
        if self.arg1 is not None and self.arg2 is not None:
            return f"{self.op.name:<18} {self.arg1!r:<16} {self.arg2!r}"
        elif self.arg1 is not None:
            return f"{self.op.name:<18} {self.arg1!r}"
        else:
            return f"{self.op.name}"


# ---------------------------------------------------------------------------
# Compiled function object
# ---------------------------------------------------------------------------

@dataclass
class Function:
    name: str
    params: list[str]
    code: list[Instruction] = field(default_factory=list)

    def __repr__(self):
        return f"<Function {self.name}({', '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Call frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """A single activation record (stack frame) for a function call."""
    func: Function
    ip: int = 0                         # instruction pointer
    locals: dict[str, Any] = field(default_factory=dict)
    return_value: Any = None


# ---------------------------------------------------------------------------
# AST node types (mini-language: same as 04_recursive_descent_parser)
# ---------------------------------------------------------------------------

@dataclass
class NumLit:   value: Any
@dataclass
class StrLit:   value: str
@dataclass
class BoolLit:  value: bool
@dataclass
class Var:      name: str
@dataclass
class BinOp:    op: str; left: Any; right: Any
@dataclass
class UnaryOp:  op: str; operand: Any
@dataclass
class Assign:   name: str; value: Any
@dataclass
class IfStmt:   condition: Any; then_branch: Any; else_branch: Optional[Any] = None
@dataclass
class WhileStmt:condition: Any; body: Any
@dataclass
class PrintStmt:value: Any
@dataclass
class ReturnStmt: value: Optional[Any] = None
@dataclass
class Block:    stmts: list = field(default_factory=list)
@dataclass
class Program:  stmts: list = field(default_factory=list)
@dataclass
class FuncDef:  name: str; params: list; body: Any
@dataclass
class CallExpr: name: str; args: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bytecode Compiler
# ---------------------------------------------------------------------------

class Compiler:
    """
    Compiles an AST into bytecode (a list of Instructions).
    Handles top-level statements and function definitions.
    """

    def __init__(self):
        self.functions: dict[str, Function] = {}
        self._current: Optional[Function] = None

    def _emit(self, op: Op, arg1=None, arg2=None) -> int:
        """Emit an instruction and return its index."""
        instr = Instruction(op, arg1, arg2)
        self._current.code.append(instr)
        return len(self._current.code) - 1

    def _patch(self, idx: int, target: int) -> None:
        """Patch a jump instruction's target address."""
        self._current.code[idx].arg1 = target

    def _ip(self) -> int:
        """Current next instruction index."""
        return len(self._current.code)

    def compile_program(self, program: Program) -> None:
        """
        Two-pass compilation:
          Pass 1: collect function definitions (so forward calls work)
          Pass 2: compile the main body and all functions
        """
        # Pass 1: Register function signatures
        func_defs = []
        main_stmts = []
        for stmt in program.stmts:
            if isinstance(stmt, FuncDef):
                func_defs.append(stmt)
            else:
                main_stmts.append(stmt)

        # Create main function
        main_func = Function(name='__main__', params=[])
        self.functions['__main__'] = main_func

        # Register user-defined functions
        for fd in func_defs:
            f = Function(name=fd.name, params=fd.params)
            self.functions[fd.name] = f

        # Compile function bodies
        for fd in func_defs:
            self._current = self.functions[fd.name]
            self.compile_stmt(fd.body)
            # Implicit void return
            self._emit(Op.PUSH, None)
            self._emit(Op.RETURN)

        # Compile main body
        self._current = main_func
        for stmt in main_stmts:
            self.compile_stmt(stmt)
        self._emit(Op.HALT)

    def compile_stmt(self, node) -> None:
        match node:
            case Program(stmts=stmts) | Block(stmts=stmts):
                for s in stmts:
                    self.compile_stmt(s)

            case Assign(name=name, value=val):
                self.compile_expr(val)
                self._emit(Op.STORE, name)

            case PrintStmt(value=val):
                self.compile_expr(val)
                self._emit(Op.PRINT)

            case ReturnStmt(value=val):
                if val is not None:
                    self.compile_expr(val)
                    self._emit(Op.RETURN)
                else:
                    self._emit(Op.PUSH, None)
                    self._emit(Op.RETURN)

            case IfStmt(condition=cond, then_branch=then_br, else_branch=else_br):
                # Compile condition
                self.compile_expr(cond)
                # Emit JUMP_IF_FALSE (target patched later)
                jif = self._emit(Op.JUMP_IF_FALSE, None)
                # Compile then branch
                self.compile_stmt(then_br)
                if else_br:
                    # Jump over else branch
                    jmp = self._emit(Op.JUMP, None)
                    # Patch jif to here (else branch start)
                    self._patch(jif, self._ip())
                    self.compile_stmt(else_br)
                    # Patch jmp to here (after else)
                    self._patch(jmp, self._ip())
                else:
                    self._patch(jif, self._ip())

            case WhileStmt(condition=cond, body=body):
                loop_start = self._ip()
                self.compile_expr(cond)
                jif = self._emit(Op.JUMP_IF_FALSE, None)   # exit loop
                self.compile_stmt(body)
                self._emit(Op.JUMP, loop_start)             # loop back
                self._patch(jif, self._ip())               # patch exit

            case FuncDef():
                pass   # handled in compile_program

            case _:
                # Treat as expression statement
                self.compile_expr(node)
                self._emit(Op.POP)

    def compile_expr(self, node) -> None:
        match node:
            case NumLit(value=v):
                self._emit(Op.PUSH, v)

            case StrLit(value=v):
                self._emit(Op.PUSH, v)

            case BoolLit(value=v):
                self._emit(Op.PUSH, 1 if v else 0)

            case Var(name=n):
                self._emit(Op.LOAD, n)

            case BinOp(op=op, left=left, right=right):
                # Short-circuit for && and ||
                if op == '&&':
                    self.compile_expr(left)
                    self._emit(Op.DUP)
                    jif = self._emit(Op.JUMP_IF_FALSE, None)
                    self._emit(Op.POP)
                    self.compile_expr(right)
                    self._patch(jif, self._ip())
                    return
                if op == '||':
                    self.compile_expr(left)
                    self._emit(Op.DUP)
                    jif_true = self._emit(Op.JUMP_IF_FALSE, None)
                    # left is truthy: jump past right
                    jmp_end = self._emit(Op.JUMP, None)
                    self._patch(jif_true, self._ip())
                    self._emit(Op.POP)
                    self.compile_expr(right)
                    self._patch(jmp_end, self._ip())
                    return

                self.compile_expr(left)
                self.compile_expr(right)
                op_map = {
                    '+': Op.ADD, '-': Op.SUB, '*': Op.MUL, '/': Op.DIV, '%': Op.MOD,
                    '==': Op.EQ, '!=': Op.NE, '<': Op.LT, '>': Op.GT,
                    '<=': Op.LE, '>=': Op.GE,
                    '&&': Op.AND, '||': Op.OR,
                }
                self._emit(op_map[op])

            case UnaryOp(op=op, operand=operand):
                self.compile_expr(operand)
                if op == '-':  self._emit(Op.NEG)
                elif op == '!': self._emit(Op.NOT)

            case CallExpr(name=name, args=args):
                for arg in args:
                    self.compile_expr(arg)
                self._emit(Op.CALL, name, len(args))

            case _:
                raise ValueError(f"Unknown expression: {node!r}")


# ---------------------------------------------------------------------------
# Virtual Machine
# ---------------------------------------------------------------------------

class VMError(Exception):
    pass


class VM:
    """
    Stack-based virtual machine that executes bytecode.

    Architecture:
      - operand_stack: stack of values during expression evaluation
      - call_stack: stack of Frame objects for function calls
    """

    def __init__(self, functions: dict[str, Function]):
        self.functions = functions
        self.operand_stack: list[Any] = []
        self.call_stack: list[Frame] = []

    def push(self, val: Any) -> None:
        self.operand_stack.append(val)

    def pop(self) -> Any:
        if not self.operand_stack:
            raise VMError("Stack underflow")
        return self.operand_stack.pop()

    def peek(self) -> Any:
        if not self.operand_stack:
            raise VMError("Stack empty")
        return self.operand_stack[-1]

    def run(self, func_name: str = '__main__') -> Any:
        func = self.functions.get(func_name)
        if func is None:
            raise VMError(f"Function not found: {func_name!r}")
        frame = Frame(func=func)
        self.call_stack.append(frame)
        return self._execute()

    def _execute(self) -> Any:
        MAX_INSTRUCTIONS = 100_000
        count = 0

        while self.call_stack:
            frame = self.call_stack[-1]
            if frame.ip >= len(frame.func.code):
                # Implicit return
                self.call_stack.pop()
                self.push(None)
                continue

            instr = frame.func.code[frame.ip]
            frame.ip += 1
            count += 1
            if count > MAX_INSTRUCTIONS:
                raise VMError("Execution limit exceeded (infinite loop?)")

            match instr.op:
                case Op.PUSH:
                    self.push(instr.arg1)

                case Op.POP:
                    self.pop()

                case Op.DUP:
                    self.push(self.peek())

                case Op.LOAD:
                    name = instr.arg1
                    # Walk up call stack to find variable
                    for f in reversed(self.call_stack):
                        if name in f.locals:
                            self.push(f.locals[name])
                            break
                    else:
                        raise VMError(f"Undefined variable: {name!r}")

                case Op.STORE:
                    val = self.pop()
                    frame.locals[instr.arg1] = val

                case Op.ADD:
                    b, a = self.pop(), self.pop()
                    self.push(a + b)

                case Op.SUB:
                    b, a = self.pop(), self.pop()
                    self.push(a - b)

                case Op.MUL:
                    b, a = self.pop(), self.pop()
                    self.push(a * b)

                case Op.DIV:
                    b, a = self.pop(), self.pop()
                    if b == 0: raise VMError("Division by zero")
                    self.push(a // b if isinstance(a, int) and isinstance(b, int) else a / b)

                case Op.MOD:
                    b, a = self.pop(), self.pop()
                    self.push(a % b)

                case Op.NEG:
                    self.push(-self.pop())

                case Op.NOT:
                    self.push(0 if self.pop() else 1)

                case Op.EQ:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a == b else 0)

                case Op.NE:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a != b else 0)

                case Op.LT:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a < b else 0)

                case Op.GT:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a > b else 0)

                case Op.LE:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a <= b else 0)

                case Op.GE:
                    b, a = self.pop(), self.pop()
                    self.push(1 if a >= b else 0)

                case Op.AND:
                    b, a = self.pop(), self.pop()
                    self.push(1 if (a and b) else 0)

                case Op.OR:
                    b, a = self.pop(), self.pop()
                    self.push(1 if (a or b) else 0)

                case Op.JUMP:
                    frame.ip = instr.arg1

                case Op.JUMP_IF_FALSE:
                    cond = self.pop()
                    if not cond:
                        frame.ip = instr.arg1

                case Op.CALL:
                    func_name = instr.arg1
                    argc = instr.arg2
                    # Pop arguments in reverse order
                    args = []
                    for _ in range(argc):
                        args.insert(0, self.pop())

                    if func_name in self.functions:
                        callee = self.functions[func_name]
                        new_frame = Frame(func=callee)
                        # Bind parameters
                        for pname, pval in zip(callee.params, args):
                            new_frame.locals[pname] = pval
                        self.call_stack.append(new_frame)
                    else:
                        # Built-in functions
                        result = self._call_builtin(func_name, args)
                        self.push(result)

                case Op.RETURN:
                    retval = self.pop()
                    self.call_stack.pop()
                    self.push(retval)

                case Op.RETURN_NONE:
                    self.call_stack.pop()
                    self.push(None)

                case Op.PRINT:
                    val = self.pop()
                    print(f"  [vm output] {val}")

                case Op.HALT:
                    return self.operand_stack[-1] if self.operand_stack else None

        return self.operand_stack[-1] if self.operand_stack else None

    def _call_builtin(self, name: str, args: list) -> Any:
        builtins = {
            'abs':   lambda xs: abs(xs[0]),
            'max':   lambda xs: max(xs),
            'min':   lambda xs: min(xs),
            'str':   lambda xs: str(xs[0]),
            'int':   lambda xs: int(xs[0]),
            'float': lambda xs: float(xs[0]),
        }
        if name not in builtins:
            raise VMError(f"Unknown function: {name!r}")
        return builtins[name](args)


def disassemble(func: Function) -> None:
    """Print a human-readable disassembly of a function's bytecode."""
    print(f"Function: {func.name}({', '.join(func.params)})")
    for i, instr in enumerate(func.code):
        print(f"  {i:>4}  {instr}")
    print()


# ---------------------------------------------------------------------------
# Demo programs
# ---------------------------------------------------------------------------

def demo1_arithmetic():
    """Compile and run: x = 5; y = 3; print(x * y + 2)"""
    print("--- Demo 1: Arithmetic ---")
    prog = Program([
        Assign('x', NumLit(5)),
        Assign('y', NumLit(3)),
        PrintStmt(BinOp('+', BinOp('*', Var('x'), Var('y')), NumLit(2))),
    ])
    compiler = Compiler()
    compiler.compile_program(prog)
    print("Bytecode:")
    disassemble(compiler.functions['__main__'])
    vm = VM(compiler.functions)
    vm.run()


def demo2_if_else():
    """Compile and run: if/else statement"""
    print("--- Demo 2: If/Else ---")
    prog = Program([
        Assign('x', NumLit(7)),
        IfStmt(
            BinOp('>', Var('x'), NumLit(5)),
            Block([PrintStmt(StrLit("x > 5"))]),
            Block([PrintStmt(StrLit("x <= 5"))]),
        ),
    ])
    compiler = Compiler()
    compiler.compile_program(prog)
    print("Bytecode:")
    disassemble(compiler.functions['__main__'])
    vm = VM(compiler.functions)
    vm.run()


def demo3_while_loop():
    """Compile and run: while loop computing sum 1..10"""
    print("--- Demo 3: While Loop (sum 1..10) ---")
    prog = Program([
        Assign('i', NumLit(1)),
        Assign('total', NumLit(0)),
        WhileStmt(
            BinOp('<=', Var('i'), NumLit(10)),
            Block([
                Assign('total', BinOp('+', Var('total'), Var('i'))),
                Assign('i', BinOp('+', Var('i'), NumLit(1))),
            ])
        ),
        PrintStmt(Var('total')),
    ])
    compiler = Compiler()
    compiler.compile_program(prog)
    print("Bytecode:")
    disassemble(compiler.functions['__main__'])
    vm = VM(compiler.functions)
    vm.run()


def demo4_function_call():
    """Compile and run: recursive factorial function"""
    print("--- Demo 4: Recursive Factorial ---")
    # factorial(n) = if n <= 1 then 1 else n * factorial(n-1)
    factorial_def = FuncDef(
        name='factorial',
        params=['n'],
        body=Block([
            IfStmt(
                BinOp('<=', Var('n'), NumLit(1)),
                Block([ReturnStmt(NumLit(1))]),
                Block([ReturnStmt(
                    BinOp('*', Var('n'),
                          CallExpr('factorial', [BinOp('-', Var('n'), NumLit(1))]))
                )]),
            )
        ])
    )
    prog = Program([
        factorial_def,
        Assign('result', CallExpr('factorial', [NumLit(6)])),
        PrintStmt(Var('result')),
    ])
    compiler = Compiler()
    compiler.compile_program(prog)
    print("Bytecode for 'factorial':")
    disassemble(compiler.functions['factorial'])
    print("Bytecode for '__main__':")
    disassemble(compiler.functions['__main__'])
    vm = VM(compiler.functions)
    vm.run()


def main():
    print("=" * 60)
    print("Bytecode Compiler and Stack-Based VM Demo")
    print("=" * 60)
    print()
    demo1_arithmetic()
    print()
    demo2_if_else()
    print()
    demo3_while_loop()
    print()
    demo4_function_call()


if __name__ == "__main__":
    main()
