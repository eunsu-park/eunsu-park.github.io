"""
09_optimizer.py - Local Optimizations on Three-Address Code

Demonstrates classic local (basic-block-level) optimizations:

  1. Constant Folding
     Evaluate expressions with all-constant operands at compile time.
     e.g.,  t1 = 3 + 4  -->  t1 = 7

  2. Constant Propagation
     Replace uses of a variable with its known constant value.
     e.g.,  x = 5; t1 = x + 2  -->  x = 5; t1 = 5 + 2 --> t1 = 7

  3. Algebraic Simplification
     Apply algebraic identities to simplify expressions.
     e.g.,  t = x * 1  -->  t = x
            t = x + 0  -->  t = x
            t = x * 0  -->  t = 0
            t = x - x  -->  t = 0

  4. Common Subexpression Elimination (CSE)
     Avoid recomputing the same expression twice within a block.
     e.g.,  t1 = a + b; t2 = a + b  -->  t1 = a + b; t2 = t1

  5. Dead Code Elimination
     Remove assignments to temporaries that are never used.
     e.g.,  t1 = 5 (never read)  -->  (removed)

Topics covered:
  - Value numbering for CSE
  - Dataflow information within a basic block
  - Iterative optimization (apply passes until no changes)
  - Counting how many optimizations each pass performed
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# TAC instruction types (simplified from 08_three_address_code.py)
# ---------------------------------------------------------------------------

@dataclass
class Assign:
    dest: str
    src: Any        # literal value or variable name

    def __str__(self):
        return f"    {self.dest} = {self.src}"

    def copy(self):
        return Assign(self.dest, self.src)


@dataclass
class BinOp:
    dest: str
    left: Any
    op: str
    right: Any

    def __str__(self):
        return f"    {self.dest} = {self.left} {self.op} {self.right}"

    def copy(self):
        return BinOp(self.dest, self.left, self.op, self.right)


@dataclass
class UnaryOp:
    dest: str
    op: str
    src: Any

    def __str__(self):
        return f"    {self.dest} = {self.op}{self.src}"

    def copy(self):
        return UnaryOp(self.dest, self.op, self.src)


@dataclass
class Label:
    name: str

    def __str__(self):
        return f"{self.name}:"

    def copy(self):
        return Label(self.name)


@dataclass
class Jump:
    target: str

    def __str__(self):
        return f"    goto {self.target}"

    def copy(self):
        return Jump(self.target)


@dataclass
class CondJump:
    condition: Any
    true_target: str
    false_target: Optional[str] = None

    def __str__(self):
        s = f"    if {self.condition} goto {self.true_target}"
        if self.false_target:
            s += f" else goto {self.false_target}"
        return s

    def copy(self):
        return CondJump(self.condition, self.true_target, self.false_target)


@dataclass
class Return:
    value: Optional[Any] = None

    def __str__(self):
        return f"    return {self.value}" if self.value is not None else "    return"

    def copy(self):
        return Return(self.value)


TACInstr = Union[Assign, BinOp, UnaryOp, Label, Jump, CondJump, Return]


# ---------------------------------------------------------------------------
# Utility: is a value a compile-time constant?
# ---------------------------------------------------------------------------

def is_const(v: Any) -> bool:
    return isinstance(v, (int, float, bool)) or (isinstance(v, str) and v.lstrip('-').replace('.', '', 1).isdigit())


def to_num(v: Any) -> Union[int, float]:
    """Convert a value to a number for constant folding."""
    if isinstance(v, (int, float)):
        return v
    s = str(v)
    try:
        return int(s)
    except ValueError:
        return float(s)


def is_number(v: Any) -> bool:
    try:
        to_num(v)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Optimization Pass Base
# ---------------------------------------------------------------------------

class OptPass:
    """Base class for optimization passes."""
    name: str = "unnamed"

    def run(self, instrs: list[TACInstr]) -> tuple[list[TACInstr], int]:
        """
        Run the pass on a list of instructions.
        Returns (new_instrs, num_changes).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pass 1: Constant Folding
# ---------------------------------------------------------------------------

def fold_binop(op: str, l: Any, r: Any) -> Optional[Any]:
    """
    Try to fold a binary operation on two constant operands.
    Returns the folded value or None if not foldable.
    """
    if not (is_number(l) and is_number(r)):
        return None
    lv, rv = to_num(l), to_num(r)
    try:
        match op:
            case '+':  return int(lv + rv) if isinstance(lv, int) and isinstance(rv, int) else lv + rv
            case '-':  return int(lv - rv) if isinstance(lv, int) and isinstance(rv, int) else lv - rv
            case '*':  return int(lv * rv) if isinstance(lv, int) and isinstance(rv, int) else lv * rv
            case '/':
                if rv == 0: return None
                return lv // rv if isinstance(lv, int) and isinstance(rv, int) else lv / rv
            case '%':  return int(lv) % int(rv) if rv != 0 else None
            case '<':  return int(lv < rv)
            case '>':  return int(lv > rv)
            case '<=': return int(lv <= rv)
            case '>=': return int(lv >= rv)
            case '==': return int(lv == rv)
            case '!=': return int(lv != rv)
    except Exception:
        pass
    return None


class ConstantFolding(OptPass):
    name = "Constant Folding"

    def run(self, instrs: list[TACInstr]) -> tuple[list[TACInstr], int]:
        new_instrs = []
        changes = 0
        for instr in instrs:
            if isinstance(instr, BinOp):
                folded = fold_binop(instr.op, instr.left, instr.right)
                if folded is not None:
                    new_instrs.append(Assign(instr.dest, folded))
                    changes += 1
                    continue
            elif isinstance(instr, UnaryOp):
                if instr.op == '-' and is_number(instr.src):
                    new_instrs.append(Assign(instr.dest, -to_num(instr.src)))
                    changes += 1
                    continue
                if instr.op == '!' and is_number(instr.src):
                    new_instrs.append(Assign(instr.dest, int(not to_num(instr.src))))
                    changes += 1
                    continue
            new_instrs.append(instr)
        return new_instrs, changes


# ---------------------------------------------------------------------------
# Pass 2: Constant Propagation
# ---------------------------------------------------------------------------

class ConstantPropagation(OptPass):
    name = "Constant Propagation"

    def run(self, instrs: list[TACInstr]) -> tuple[list[TACInstr], int]:
        """
        Forward analysis: maintain a map {var -> constant_value}.
        Replace variable references with their known constant values.
        Invalidate a variable's constant when it is reassigned to a non-constant.
        """
        const_map: dict[str, Any] = {}
        new_instrs = []
        changes = 0

        def subst(v: Any) -> Any:
            if isinstance(v, str) and v in const_map:
                return const_map[v]
            return v

        for instr in instrs:
            if isinstance(instr, Assign):
                new_src = subst(instr.src)
                if new_src != instr.src:
                    changes += 1
                new_instrs.append(Assign(instr.dest, new_src))
                # Update const map
                if is_number(new_src):
                    const_map[instr.dest] = new_src
                else:
                    const_map.pop(instr.dest, None)

            elif isinstance(instr, BinOp):
                nl = subst(instr.left)
                nr = subst(instr.right)
                if nl != instr.left or nr != instr.right:
                    changes += 1
                new_instrs.append(BinOp(instr.dest, nl, instr.op, nr))
                const_map.pop(instr.dest, None)

            elif isinstance(instr, UnaryOp):
                ns = subst(instr.src)
                if ns != instr.src:
                    changes += 1
                new_instrs.append(UnaryOp(instr.dest, instr.op, ns))
                const_map.pop(instr.dest, None)

            elif isinstance(instr, CondJump):
                nc = subst(instr.condition)
                if nc != instr.condition:
                    changes += 1
                new_instrs.append(CondJump(nc, instr.true_target, instr.false_target))

            elif isinstance(instr, Return):
                nv = subst(instr.value) if instr.value is not None else None
                if nv != instr.value:
                    changes += 1
                new_instrs.append(Return(nv))

            else:
                # Label, Jump: don't invalidate anything
                new_instrs.append(instr)

        return new_instrs, changes


# ---------------------------------------------------------------------------
# Pass 3: Algebraic Simplification
# ---------------------------------------------------------------------------

class AlgebraicSimplification(OptPass):
    name = "Algebraic Simplification"

    def run(self, instrs: list[TACInstr]) -> tuple[list[TACInstr], int]:
        new_instrs = []
        changes = 0
        for instr in instrs:
            if isinstance(instr, BinOp):
                simplified = self._simplify(instr)
                if simplified is not instr:
                    new_instrs.append(simplified)
                    changes += 1
                    continue
            new_instrs.append(instr)
        return new_instrs, changes

    def _simplify(self, b: BinOp) -> TACInstr:
        l, op, r = b.left, b.op, b.right
        # x + 0 = x,  0 + x = x
        if op == '+' and r == 0: return Assign(b.dest, l)
        if op == '+' and l == 0: return Assign(b.dest, r)
        # x - 0 = x
        if op == '-' and r == 0: return Assign(b.dest, l)
        # x * 1 = x,  1 * x = x
        if op == '*' and r == 1: return Assign(b.dest, l)
        if op == '*' and l == 1: return Assign(b.dest, r)
        # x * 0 = 0,  0 * x = 0
        if op == '*' and (r == 0 or l == 0): return Assign(b.dest, 0)
        # x / 1 = x
        if op == '/' and r == 1: return Assign(b.dest, l)
        # x - x = 0 (only if same variable)
        if op == '-' and l == r and isinstance(l, str): return Assign(b.dest, 0)
        # x / x = 1 (only if same variable, ignore division by zero)
        if op == '/' and l == r and isinstance(l, str): return Assign(b.dest, 1)
        return b


# ---------------------------------------------------------------------------
# Pass 4: Common Subexpression Elimination (CSE)
# ---------------------------------------------------------------------------

class CSE(OptPass):
    name = "Common Subexpression Elimination"

    def run(self, instrs: list[TACInstr]) -> tuple[list[TACInstr], int]:
        """
        For each BinOp/UnaryOp, check if the same expression was computed before.
        If so, replace with the earlier result.
        Invalidate when any operand variable is redefined.
        """
        # Maps (left, op, right) -> existing_temp
        expr_map: dict[tuple, str] = {}
        # Maps variable -> set of expression keys using it
        var_to_exprs: dict[str, set[tuple]] = {}

        new_instrs = []
        changes = 0

        def invalidate(var: str):
            """Remove all cached expressions that use 'var'."""
            for key in list(var_to_exprs.get(var, set())):
                expr_map.pop(key, None)
            var_to_exprs.pop(var, None)

        def record(key: tuple, dest: str, operands: list[str]):
            expr_map[key] = dest
            for op in operands:
                if isinstance(op, str):
                    var_to_exprs.setdefault(op, set()).add(key)

        for instr in instrs:
            if isinstance(instr, BinOp):
                key = (instr.left, instr.op, instr.right)
                if key in expr_map:
                    # Replace with copy from earlier result
                    new_instrs.append(Assign(instr.dest, expr_map[key]))
                    changes += 1
                    # The dest is still being defined; invalidate it
                    invalidate(instr.dest)
                    # Record new alias: dest -> same expr
                    record(key, expr_map[key], [instr.left, instr.right])
                else:
                    new_instrs.append(instr)
                    invalidate(instr.dest)
                    record(key, instr.dest, [instr.left, instr.right])

            elif isinstance(instr, UnaryOp):
                key = (instr.op, instr.src)
                if key in expr_map:
                    new_instrs.append(Assign(instr.dest, expr_map[key]))
                    changes += 1
                    invalidate(instr.dest)
                else:
                    new_instrs.append(instr)
                    invalidate(instr.dest)
                    record(key, instr.dest, [instr.src])

            elif isinstance(instr, Assign):
                # Redefining dest: invalidate cached expressions that use dest
                invalidate(instr.dest)
                new_instrs.append(instr)

            elif isinstance(instr, Label):
                # At a label (block boundary), clear all cached info
                expr_map.clear()
                var_to_exprs.clear()
                new_instrs.append(instr)

            else:
                new_instrs.append(instr)

        return new_instrs, changes


# ---------------------------------------------------------------------------
# Pass 5: Dead Code Elimination
# ---------------------------------------------------------------------------

class DeadCodeElimination(OptPass):
    name = "Dead Code Elimination"

    def run(self, instrs: list[TACInstr]) -> tuple[list[TACInstr], int]:
        """
        Remove instructions that assign to variables/temps that are never
        subsequently used. Uses a backward liveness analysis.

        A variable is 'live' at a point if it may be used after that point.
        An assignment 'dest = ...' is dead if dest is not live after the assignment.
        """
        # Collect all 'uses' in the instruction list
        def uses_of(instr: TACInstr) -> set[str]:
            u: set[str] = set()
            if isinstance(instr, Assign):
                if isinstance(instr.src, str): u.add(instr.src)
            elif isinstance(instr, BinOp):
                if isinstance(instr.left, str):  u.add(instr.left)
                if isinstance(instr.right, str): u.add(instr.right)
            elif isinstance(instr, UnaryOp):
                if isinstance(instr.src, str): u.add(instr.src)
            elif isinstance(instr, CondJump):
                if isinstance(instr.condition, str): u.add(instr.condition)
            elif isinstance(instr, Return):
                if isinstance(instr.value, str): u.add(instr.value)
            return u

        def def_of(instr: TACInstr) -> Optional[str]:
            if isinstance(instr, (Assign, BinOp, UnaryOp)):
                return instr.dest
            return None

        # Backward pass: compute live variables at each point
        # live[i] = set of variables live AFTER instruction i
        n = len(instrs)
        live: list[set[str]] = [set() for _ in range(n + 1)]

        for i in range(n - 1, -1, -1):
            live[i] = set(live[i + 1])
            live[i].update(uses_of(instrs[i]))
            d = def_of(instrs[i])
            if d is not None:
                live[i].discard(d)

        # Eliminate dead assignments
        new_instrs = []
        changes = 0
        for i, instr in enumerate(instrs):
            d = def_of(instr)
            # Only eliminate temporaries (t0, t1, ...), not user variables
            if d is not None and d.startswith('t') and d[1:].isdigit():
                if d not in live[i + 1]:
                    changes += 1
                    continue   # skip this dead instruction
            new_instrs.append(instr)

        return new_instrs, changes


# ---------------------------------------------------------------------------
# Optimization pipeline
# ---------------------------------------------------------------------------

def run_optimizer(instrs: list[TACInstr], max_passes: int = 10) -> list[TACInstr]:
    """
    Run all optimization passes iteratively until no changes occur.
    """
    passes = [
        ConstantPropagation(),
        ConstantFolding(),
        AlgebraicSimplification(),
        CSE(),
        DeadCodeElimination(),
    ]

    print("\nOptimization log:")
    for iteration in range(max_passes):
        total_changes = 0
        for p in passes:
            instrs, n = p.run(instrs)
            if n:
                print(f"  Pass {iteration+1} [{p.name}]: {n} change(s)")
            total_changes += n
        if total_changes == 0:
            print(f"  Fixed point reached after {iteration+1} iteration(s).")
            break

    return instrs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def make_sample_tac() -> list[TACInstr]:
    """
    TAC for: t0 = (3 + 4) * (x * 1 - 0); t1 = (3 + 4); t2 = t1 * (x * 1 - 0)
    Contains several optimization opportunities:
      - 3+4 -> 7 (constant folding)
      - x*1 -> x (algebraic simplification)
      - x-0 -> x (algebraic simplification)
      - second 3+4 -> t0 copy is dead (CSE + DCE)
    """
    return [
        BinOp('t0', 3, '+', 4),           # t0 = 3 + 4  (constant fold -> 7)
        BinOp('t1', 'x', '*', 1),          # t1 = x * 1  (algebraic -> x)
        BinOp('t2', 't1', '-', 0),         # t2 = t1 - 0 (algebraic -> t1 -> x)
        BinOp('t3', 't0', '*', 't2'),      # t3 = t0 * t2 -> 7 * x
        # CSE: same as first computation of 3+4
        BinOp('t4', 3, '+', 4),            # t4 = 3 + 4  (CSE -> t4 = t0)
        BinOp('t5', 't4', '*', 't2'),      # t5 = t4 * t2 (t4 is dead if t5 == t3 by CSE)
        # Dead code: t6 assigned but never used
        BinOp('t6', 'a', '+', 'b'),        # dead if t6 never read
        Return('t3'),
    ]


def print_tac(label: str, instrs: list[TACInstr]) -> None:
    print(f"\n{label}:")
    for i in instrs:
        print(i)


def main():
    print("=" * 60)
    print("TAC Optimizer Demo")
    print("=" * 60)

    original = make_sample_tac()
    print_tac("Original TAC", original)

    optimized = run_optimizer(list(original))
    print_tac("Optimized TAC", optimized)

    print(f"\nReduction: {len(original)} -> {len(optimized)} instructions")

    # Another example: constant propagation chain
    print("\n" + "=" * 60)
    print("Example 2: Constant propagation chain")
    tac2 = [
        Assign('a', 5),
        Assign('b', 3),
        BinOp('t0', 'a', '+', 'b'),       # -> 5+3=8
        BinOp('t1', 't0', '*', 2),         # -> 8*2=16
        BinOp('t2', 't1', '-', 1),         # -> 16-1=15
        CondJump('t2', 'L_true', 'L_false'),
        Label('L_true'),
        Return('t2'),
        Label('L_false'),
        Return(0),
    ]
    print_tac("Original", tac2)
    opt2 = run_optimizer(list(tac2))
    print_tac("Optimized", opt2)


if __name__ == "__main__":
    main()
