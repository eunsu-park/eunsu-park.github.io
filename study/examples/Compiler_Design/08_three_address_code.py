"""
08_three_address_code.py - Three-Address Code (TAC) Generation

Demonstrates intermediate code generation: converting an AST into
Three-Address Code (TAC) and building a Control Flow Graph (CFG).

Three-Address Code uses at most three operands per instruction:
  t1 = a + b
  t2 = t1 * c
  if t2 > 0 goto L1
  ...

TAC instruction types:
  ASSIGN    t = a                 (copy)
  BINOP     t = a op b            (binary operation)
  UNARY     t = op a              (unary operation)
  COPY      t = a                 (same as ASSIGN, alias)
  LABEL     L:                    (branch target)
  JUMP      goto L                (unconditional jump)
  CJUMP     if a goto L           (conditional jump)
  PARAM     param a               (function call parameter)
  CALL      t = call f, n         (function call, n args)
  RETURN    return a              (function return)
  LOAD      t = a[i]              (array read)
  STORE     a[i] = t              (array write)

Control Flow Graph:
  - Partition TAC into basic blocks (maximal straight-line sequences)
  - Add edges: sequential, conditional, jump edges

Topics covered:
  - Temporary variable generation
  - Label generation
  - AST to TAC translation
  - Short-circuit evaluation for boolean expressions
  - CFG construction from TAC
  - Basic block identification
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# TAC Instructions
# ---------------------------------------------------------------------------

@dataclass
class TACInstr:
    """Base class for all TAC instructions."""

    def __str__(self) -> str:
        return repr(self)


@dataclass
class Assign(TACInstr):
    dest: str
    src: Any    # variable name or literal value

    def __str__(self):
        return f"    {self.dest} = {self.src}"


@dataclass
class BinOp(TACInstr):
    dest: str
    left: Any
    op: str
    right: Any

    def __str__(self):
        return f"    {self.dest} = {self.left} {self.op} {self.right}"


@dataclass
class UnaryOp(TACInstr):
    dest: str
    op: str
    src: Any

    def __str__(self):
        return f"    {self.dest} = {self.op}{self.src}"


@dataclass
class Label(TACInstr):
    name: str

    def __str__(self):
        return f"{self.name}:"


@dataclass
class Jump(TACInstr):
    target: str

    def __str__(self):
        return f"    goto {self.target}"


@dataclass
class CondJump(TACInstr):
    condition: Any
    true_target: str
    false_target: Optional[str] = None

    def __str__(self):
        s = f"    if {self.condition} goto {self.true_target}"
        if self.false_target:
            s += f" else goto {self.false_target}"
        return s


@dataclass
class Param(TACInstr):
    value: Any

    def __str__(self):
        return f"    param {self.value}"


@dataclass
class Call(TACInstr):
    dest: Optional[str]
    func: str
    num_args: int

    def __str__(self):
        if self.dest:
            return f"    {self.dest} = call {self.func}, {self.num_args}"
        return f"    call {self.func}, {self.num_args}"


@dataclass
class Return(TACInstr):
    value: Optional[Any] = None

    def __str__(self):
        if self.value is not None:
            return f"    return {self.value}"
        return f"    return"


# ---------------------------------------------------------------------------
# TAC Generator
# ---------------------------------------------------------------------------

class TACGenerator:
    """
    Translates an expression/statement AST into a flat list of TAC instructions.
    Generates fresh temporary variables (t0, t1, ...) and labels (L0, L1, ...).
    """

    def __init__(self):
        self._temp_count = 0
        self._label_count = 0
        self.instrs: list[TACInstr] = []

    def new_temp(self) -> str:
        name = f"t{self._temp_count}"
        self._temp_count += 1
        return name

    def new_label(self) -> str:
        name = f"L{self._label_count}"
        self._label_count += 1
        return name

    def emit(self, instr: TACInstr) -> None:
        self.instrs.append(instr)

    # --- Expression translation ---
    # Returns the "address" (temp name or literal) holding the result.

    def gen_expr(self, node) -> Any:
        match node:
            case {'kind': 'num', 'value': v}:
                return v
            case {'kind': 'str', 'value': v}:
                return repr(v)
            case {'kind': 'bool', 'value': v}:
                return 1 if v else 0
            case {'kind': 'var', 'name': n}:
                return n
            case {'kind': 'unary', 'op': op, 'operand': operand}:
                src = self.gen_expr(operand)
                t = self.new_temp()
                self.emit(UnaryOp(t, op, src))
                return t
            case {'kind': 'binop', 'op': op, 'left': left, 'right': right}:
                # Short-circuit for && and ||
                if op == '&&':
                    return self.gen_and(left, right)
                if op == '||':
                    return self.gen_or(left, right)
                l = self.gen_expr(left)
                r = self.gen_expr(right)
                t = self.new_temp()
                self.emit(BinOp(t, l, op, r))
                return t
            case {'kind': 'call', 'func': f, 'args': args}:
                for arg in args:
                    a = self.gen_expr(arg)
                    self.emit(Param(a))
                t = self.new_temp()
                self.emit(Call(t, f, len(args)))
                return t
            case {'kind': 'index', 'array': arr, 'index': idx}:
                a = self.gen_expr(arr)
                i = self.gen_expr(idx)
                t = self.new_temp()
                self.emit(Assign(t, f"{a}[{i}]"))
                return t
            case _:
                raise ValueError(f"Unknown expression node: {node}")

    def gen_and(self, left, right) -> str:
        """Short-circuit &&: if left is false, skip right."""
        result = self.new_temp()
        false_lbl = self.new_label()
        end_lbl   = self.new_label()

        l = self.gen_expr(left)
        # if NOT l, jump to false_lbl
        not_l = self.new_temp()
        self.emit(UnaryOp(not_l, '!', l))
        self.emit(CondJump(not_l, false_lbl))
        # evaluate right
        r = self.gen_expr(right)
        self.emit(Assign(result, r))
        self.emit(Jump(end_lbl))
        self.emit(Label(false_lbl))
        self.emit(Assign(result, 0))
        self.emit(Label(end_lbl))
        return result

    def gen_or(self, left, right) -> str:
        """Short-circuit ||: if left is true, skip right."""
        result = self.new_temp()
        true_lbl = self.new_label()
        end_lbl  = self.new_label()

        l = self.gen_expr(left)
        self.emit(CondJump(l, true_lbl))
        r = self.gen_expr(right)
        self.emit(Assign(result, r))
        self.emit(Jump(end_lbl))
        self.emit(Label(true_lbl))
        self.emit(Assign(result, 1))
        self.emit(Label(end_lbl))
        return result

    # --- Statement translation ---

    def gen_stmt(self, node) -> None:
        match node:
            case {'kind': 'assign', 'target': target, 'value': value}:
                v = self.gen_expr(value)
                self.emit(Assign(target, v))

            case {'kind': 'index_assign', 'array': arr, 'index': idx, 'value': value}:
                i = self.gen_expr(idx)
                v = self.gen_expr(value)
                # Represent as: arr[i] = v  (STORE)
                t = self.new_temp()
                self.emit(BinOp(t, arr, '[]', i))   # compute address
                self.emit(Assign(f"{arr}[{i}]", v))

            case {'kind': 'if', 'cond': cond, 'then': then_stmts,
                  'else': else_stmts}:
                self.gen_if(cond, then_stmts, else_stmts)

            case {'kind': 'while', 'cond': cond, 'body': body}:
                self.gen_while(cond, body)

            case {'kind': 'return', 'value': value}:
                if value is not None:
                    v = self.gen_expr(value)
                    self.emit(Return(v))
                else:
                    self.emit(Return())

            case {'kind': 'expr_stmt', 'expr': expr}:
                self.gen_expr(expr)

            case {'kind': 'block', 'stmts': stmts}:
                for s in stmts:
                    self.gen_stmt(s)

            case _:
                raise ValueError(f"Unknown statement node: {node}")

    def gen_if(self, cond, then_stmts, else_stmts) -> None:
        then_lbl = self.new_label()
        else_lbl = self.new_label()
        end_lbl  = self.new_label()

        c = self.gen_expr(cond)
        self.emit(CondJump(c, then_lbl, else_lbl if else_stmts else end_lbl))

        self.emit(Label(then_lbl))
        for s in then_stmts:
            self.gen_stmt(s)
        self.emit(Jump(end_lbl))

        if else_stmts:
            self.emit(Label(else_lbl))
            for s in else_stmts:
                self.gen_stmt(s)
            self.emit(Jump(end_lbl))

        self.emit(Label(end_lbl))

    def gen_while(self, cond, body) -> None:
        test_lbl = self.new_label()
        body_lbl = self.new_label()
        end_lbl  = self.new_label()

        self.emit(Label(test_lbl))
        c = self.gen_expr(cond)
        self.emit(CondJump(c, body_lbl, end_lbl))

        self.emit(Label(body_lbl))
        for s in body:
            self.gen_stmt(s)
        self.emit(Jump(test_lbl))

        self.emit(Label(end_lbl))


# ---------------------------------------------------------------------------
# Control Flow Graph
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    id: int
    label: Optional[str]
    instrs: list[TACInstr] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)

    def __repr__(self):
        return f"BB{self.id}({self.label or 'entry' if self.id == 0 else ''})"


def build_cfg(instrs: list[TACInstr]) -> list[BasicBlock]:
    """
    Partition TAC into basic blocks and build control flow edges.

    A basic block starts at:
      - The first instruction
      - Any labeled instruction (branch target)
      - Any instruction immediately after a jump/conditional jump
    """
    if not instrs:
        return []

    # Find leaders (first instruction of each basic block)
    leaders: set[int] = {0}
    for i, instr in enumerate(instrs):
        if isinstance(instr, (Jump, CondJump, Return)):
            if i + 1 < len(instrs):
                leaders.add(i + 1)
        if isinstance(instr, Label):
            leaders.add(i)

    sorted_leaders = sorted(leaders)

    # Build blocks
    blocks: list[BasicBlock] = []
    label_to_block: dict[str, int] = {}

    for bi, start in enumerate(sorted_leaders):
        end = sorted_leaders[bi + 1] if bi + 1 < len(sorted_leaders) else len(instrs)
        block_instrs = instrs[start:end]
        lbl = block_instrs[0].name if isinstance(block_instrs[0], Label) else None
        bb = BasicBlock(id=bi, label=lbl, instrs=block_instrs)
        blocks.append(bb)
        if lbl:
            label_to_block[lbl] = bi

    # Build edges
    for bi, bb in enumerate(blocks):
        last = bb.instrs[-1] if bb.instrs else None
        if isinstance(last, Jump):
            target_bi = label_to_block.get(last.target)
            if target_bi is not None:
                bb.successors.append(target_bi)
        elif isinstance(last, CondJump):
            t_bi = label_to_block.get(last.true_target)
            if t_bi is not None:
                bb.successors.append(t_bi)
            if last.false_target:
                f_bi = label_to_block.get(last.false_target)
                if f_bi is not None:
                    bb.successors.append(f_bi)
            # fall-through
            elif bi + 1 < len(blocks):
                bb.successors.append(bi + 1)
        elif not isinstance(last, Return) and bi + 1 < len(blocks):
            bb.successors.append(bi + 1)

    return blocks


def print_cfg(blocks: list[BasicBlock]) -> None:
    print("\nControl Flow Graph:")
    for bb in blocks:
        name = f"BB{bb.id}" + (f" [{bb.label}]" if bb.label else "")
        succs = [f"BB{s}" for s in bb.successors]
        print(f"\n  {name}  (successors: {succs or ['none']})")
        for instr in bb.instrs:
            print(f"  {instr}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Three-Address Code Generation Demo")
    print("=" * 60)

    gen = TACGenerator()

    # Example 1: Arithmetic expression
    # t = (a + b) * (c - 2)
    print("\n--- Example 1: (a + b) * (c - 2) ---")
    expr1 = {'kind': 'binop', 'op': '*',
              'left':  {'kind': 'binop', 'op': '+',
                        'left':  {'kind': 'var', 'name': 'a'},
                        'right': {'kind': 'var', 'name': 'b'}},
              'right': {'kind': 'binop', 'op': '-',
                        'left':  {'kind': 'var', 'name': 'c'},
                        'right': {'kind': 'num', 'value': 2}}}
    gen.instrs.clear()
    result = gen.gen_expr(expr1)
    print(f"Result in: {result}")
    for i in gen.instrs:
        print(i)

    # Example 2: If statement with CFG
    print("\n--- Example 2: if (x > 0) { y = x; } else { y = -x; } ---")
    gen.instrs.clear()
    gen._temp_count = gen._label_count = 0
    if_stmt = {
        'kind': 'if',
        'cond': {'kind': 'binop', 'op': '>',
                 'left': {'kind': 'var', 'name': 'x'},
                 'right': {'kind': 'num', 'value': 0}},
        'then': [{'kind': 'assign', 'target': 'y',
                  'value': {'kind': 'var', 'name': 'x'}}],
        'else': [{'kind': 'assign', 'target': 'y',
                  'value': {'kind': 'unary', 'op': '-',
                            'operand': {'kind': 'var', 'name': 'x'}}}],
    }
    gen.gen_stmt(if_stmt)
    print("TAC:")
    for i in gen.instrs:
        print(i)
    print_cfg(build_cfg(gen.instrs))

    # Example 3: While loop with CFG
    print("\n--- Example 3: sum = 0; i = 1; while (i <= n) { sum += i; i++; } ---")
    gen.instrs.clear()
    gen._temp_count = gen._label_count = 0
    stmts = [
        {'kind': 'assign', 'target': 'sum', 'value': {'kind': 'num', 'value': 0}},
        {'kind': 'assign', 'target': 'i',   'value': {'kind': 'num', 'value': 1}},
        {'kind': 'while',
         'cond': {'kind': 'binop', 'op': '<=',
                  'left': {'kind': 'var', 'name': 'i'},
                  'right': {'kind': 'var', 'name': 'n'}},
         'body': [
             {'kind': 'assign', 'target': 'sum',
              'value': {'kind': 'binop', 'op': '+',
                        'left': {'kind': 'var', 'name': 'sum'},
                        'right': {'kind': 'var', 'name': 'i'}}},
             {'kind': 'assign', 'target': 'i',
              'value': {'kind': 'binop', 'op': '+',
                        'left': {'kind': 'var', 'name': 'i'},
                        'right': {'kind': 'num', 'value': 1}}},
         ]},
        {'kind': 'return', 'value': {'kind': 'var', 'name': 'sum'}},
    ]
    for s in stmts:
        gen.gen_stmt(s)
    print("TAC:")
    for i in gen.instrs:
        print(i)
    print_cfg(build_cfg(gen.instrs))


if __name__ == "__main__":
    main()
