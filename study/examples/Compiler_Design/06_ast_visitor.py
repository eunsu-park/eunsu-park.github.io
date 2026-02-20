"""
06_ast_visitor.py - AST Node Classes and Visitor Pattern

Demonstrates the Visitor design pattern applied to an Abstract Syntax Tree.
The visitor pattern separates tree traversal logic from node definitions,
making it easy to add new operations without modifying node classes.

AST nodes are defined using Python dataclasses.
Three visitors are implemented:
  1. EvalVisitor    - evaluates the expression tree to a numeric value
  2. TypeCheckVisitor - infers and checks types (int, float, bool, str)
  3. PrettyPrintVisitor - produces a formatted source code string

The language supports:
  - Numeric literals (int and float)
  - Boolean literals
  - String literals
  - Arithmetic: +, -, *, /, % (unary -)
  - Comparison: ==, !=, <, >, <=, >=
  - Logical: and, or, not
  - String concatenation with +
  - Variables and let bindings
  - If expressions (ternary style)

Topics covered:
  - Dataclass-based AST nodes
  - Abstract base class for visitors
  - Double dispatch via visit_<NodeType> method naming
  - Type inference without annotations
  - Environment (symbol table) as a dict
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# AST Node definitions
# ---------------------------------------------------------------------------

class Node:
    """Base class for all AST nodes."""
    pass


@dataclass
class IntLit(Node):
    value: int

@dataclass
class FloatLit(Node):
    value: float

@dataclass
class BoolLit(Node):
    value: bool

@dataclass
class StrLit(Node):
    value: str

@dataclass
class Var(Node):
    name: str

@dataclass
class BinOp(Node):
    op: str     # '+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', 'and', 'or'
    left: Node
    right: Node

@dataclass
class UnaryOp(Node):
    op: str     # '-', 'not'
    operand: Node

@dataclass
class IfExpr(Node):
    """Ternary if: condition ? then_expr : else_expr"""
    condition: Node
    then_expr: Node
    else_expr: Node

@dataclass
class LetExpr(Node):
    """let name = value in body"""
    name: str
    value: Node
    body: Node

@dataclass
class FuncCall(Node):
    name: str
    args: list[Node] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class Visitor(ABC):
    """
    Abstract visitor. Each concrete visitor implements visit_<NodeType>
    methods. The dispatch method calls the appropriate visit method.
    """

    def visit(self, node: Node) -> Any:
        """Dispatch to the appropriate visit_* method."""
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)

    def generic_visit(self, node: Node) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} has no handler for {type(node).__name__}"
        )


# ---------------------------------------------------------------------------
# Visitor 1: Evaluator
# ---------------------------------------------------------------------------

class EvalError(Exception):
    pass


class EvalVisitor(Visitor):
    """
    Evaluates an AST expression to a Python value.
    Maintains an environment mapping variable names to values.
    """

    def __init__(self, env: Optional[dict[str, Any]] = None):
        self.env: dict[str, Any] = env or {}

    def visit_IntLit(self, node: IntLit) -> int:
        return node.value

    def visit_FloatLit(self, node: FloatLit) -> float:
        return node.value

    def visit_BoolLit(self, node: BoolLit) -> bool:
        return node.value

    def visit_StrLit(self, node: StrLit) -> str:
        return node.value

    def visit_Var(self, node: Var) -> Any:
        if node.name not in self.env:
            raise EvalError(f"Undefined variable: {node.name!r}")
        return self.env[node.name]

    def visit_BinOp(self, node: BinOp) -> Any:
        l = self.visit(node.left)
        # Short-circuit evaluation for 'and' and 'or'
        if node.op == 'and':
            return l and self.visit(node.right)
        if node.op == 'or':
            return l or self.visit(node.right)
        r = self.visit(node.right)
        ops = {
            '+':  lambda a, b: a + b,
            '-':  lambda a, b: a - b,
            '*':  lambda a, b: a * b,
            '/':  lambda a, b: a / b,
            '%':  lambda a, b: a % b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<':  lambda a, b: a < b,
            '>':  lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
        }
        if node.op not in ops:
            raise EvalError(f"Unknown operator: {node.op!r}")
        return ops[node.op](l, r)

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        v = self.visit(node.operand)
        if node.op == '-':    return -v
        if node.op == 'not':  return not v
        raise EvalError(f"Unknown unary operator: {node.op!r}")

    def visit_IfExpr(self, node: IfExpr) -> Any:
        cond = self.visit(node.condition)
        return self.visit(node.then_expr if cond else node.else_expr)

    def visit_LetExpr(self, node: LetExpr) -> Any:
        val = self.visit(node.value)
        old = self.env.get(node.name)
        self.env[node.name] = val
        result = self.visit(node.body)
        # Restore previous binding (lexical scoping)
        if old is None:
            self.env.pop(node.name, None)
        else:
            self.env[node.name] = old
        return result

    def visit_FuncCall(self, node: FuncCall) -> Any:
        args = [self.visit(a) for a in node.args]
        builtins = {
            'abs':   lambda x: abs(x),
            'min':   lambda *xs: min(xs),
            'max':   lambda *xs: max(xs),
            'sqrt':  lambda x: x ** 0.5,
            'len':   lambda s: len(s),
            'str':   lambda x: str(x),
            'int':   lambda x: int(x),
            'float': lambda x: float(x),
        }
        if node.name in builtins:
            return builtins[node.name](*args)
        raise EvalError(f"Unknown function: {node.name!r}")


# ---------------------------------------------------------------------------
# Visitor 2: Type Checker
# ---------------------------------------------------------------------------

class TypeError_(Exception):
    pass


# Simple type tags
INT   = 'int'
FLOAT = 'float'
BOOL  = 'bool'
STR   = 'str'
NUM   = 'num'    # int or float


class TypeCheckVisitor(Visitor):
    """
    Infers the type of an expression.
    Reports TypeError_ for type mismatches.
    Uses a type environment mapping variable names to types.
    """

    def __init__(self, type_env: Optional[dict[str, str]] = None):
        self.type_env: dict[str, str] = type_env or {}
        self.errors: list[str] = []

    def _error(self, msg: str) -> str:
        self.errors.append(msg)
        return 'error'

    def visit_IntLit(self, node: IntLit) -> str:       return INT
    def visit_FloatLit(self, node: FloatLit) -> str:   return FLOAT
    def visit_BoolLit(self, node: BoolLit) -> str:     return BOOL
    def visit_StrLit(self, node: StrLit) -> str:       return STR

    def visit_Var(self, node: Var) -> str:
        if node.name not in self.type_env:
            return self._error(f"Undefined variable: {node.name!r}")
        return self.type_env[node.name]

    def _is_numeric(self, t: str) -> bool:
        return t in (INT, FLOAT)

    def _numeric_result(self, t1: str, t2: str) -> str:
        """int op int -> int; float op float -> float; int op float -> float."""
        if t1 == FLOAT or t2 == FLOAT:
            return FLOAT
        return INT

    def visit_BinOp(self, node: BinOp) -> str:
        lt = self.visit(node.left)
        rt = self.visit(node.right)

        if node.op in ('+', '-', '*', '/', '%'):
            if node.op == '+' and lt == STR and rt == STR:
                return STR   # string concatenation
            if self._is_numeric(lt) and self._is_numeric(rt):
                return self._numeric_result(lt, rt)
            return self._error(
                f"Operator {node.op!r} not applicable to {lt} and {rt}"
            )

        if node.op in ('<', '>', '<=', '>='):
            if self._is_numeric(lt) and self._is_numeric(rt):
                return BOOL
            return self._error(
                f"Comparison {node.op!r} not applicable to {lt} and {rt}"
            )

        if node.op in ('==', '!='):
            if lt == rt or (self._is_numeric(lt) and self._is_numeric(rt)):
                return BOOL
            return self._error(f"Cannot compare {lt} with {rt}")

        if node.op in ('and', 'or'):
            if lt == BOOL and rt == BOOL:
                return BOOL
            return self._error(
                f"Logical {node.op!r} requires bool operands, got {lt} and {rt}"
            )

        return self._error(f"Unknown operator: {node.op!r}")

    def visit_UnaryOp(self, node: UnaryOp) -> str:
        t = self.visit(node.operand)
        if node.op == '-':
            if self._is_numeric(t): return t
            return self._error(f"Unary '-' not applicable to {t}")
        if node.op == 'not':
            if t == BOOL: return BOOL
            return self._error(f"'not' requires bool, got {t}")
        return self._error(f"Unknown unary operator: {node.op!r}")

    def visit_IfExpr(self, node: IfExpr) -> str:
        ct = self.visit(node.condition)
        if ct != BOOL:
            self._error(f"If condition must be bool, got {ct}")
        tt = self.visit(node.then_expr)
        et = self.visit(node.else_expr)
        if tt != et and not (self._is_numeric(tt) and self._is_numeric(et)):
            self._error(f"If branches have different types: {tt} vs {et}")
        return tt

    def visit_LetExpr(self, node: LetExpr) -> str:
        val_type = self.visit(node.value)
        old = self.type_env.get(node.name)
        self.type_env[node.name] = val_type
        body_type = self.visit(node.body)
        if old is None:
            self.type_env.pop(node.name, None)
        else:
            self.type_env[node.name] = old
        return body_type

    def visit_FuncCall(self, node: FuncCall) -> str:
        arg_types = [self.visit(a) for a in node.args]
        sigs: dict[str, str] = {
            'abs': INT, 'sqrt': FLOAT, 'len': INT,
            'str': STR, 'int': INT, 'float': FLOAT,
            'min': FLOAT, 'max': FLOAT,
        }
        return sigs.get(node.name, 'unknown')


# ---------------------------------------------------------------------------
# Visitor 3: Pretty Printer
# ---------------------------------------------------------------------------

class PrettyPrintVisitor(Visitor):
    """
    Produces a readable string representation of the AST.
    Adds parentheses based on operator precedence.
    """

    PREC = {
        'or': 1, 'and': 2,
        '==': 3, '!=': 3,
        '<': 4, '>': 4, '<=': 4, '>=': 4,
        '+': 5, '-': 5,
        '*': 6, '/': 6, '%': 6,
    }

    def _prec(self, op: str) -> int:
        return self.PREC.get(op, 99)

    def visit_IntLit(self, node: IntLit) -> str:      return str(node.value)
    def visit_FloatLit(self, node: FloatLit) -> str:  return str(node.value)
    def visit_BoolLit(self, node: BoolLit) -> str:    return 'true' if node.value else 'false'
    def visit_StrLit(self, node: StrLit) -> str:      return f'"{node.value}"'
    def visit_Var(self, node: Var) -> str:             return node.name

    def visit_BinOp(self, node: BinOp) -> str:
        l = self.visit(node.left)
        r = self.visit(node.right)
        # Add parens for sub-expressions with lower precedence
        if isinstance(node.left, BinOp) and self._prec(node.left.op) < self._prec(node.op):
            l = f"({l})"
        if isinstance(node.right, BinOp) and self._prec(node.right.op) <= self._prec(node.op):
            r = f"({r})"
        return f"{l} {node.op} {r}"

    def visit_UnaryOp(self, node: UnaryOp) -> str:
        inner = self.visit(node.operand)
        if isinstance(node.operand, BinOp):
            inner = f"({inner})"
        return f"{node.op}{inner}"

    def visit_IfExpr(self, node: IfExpr) -> str:
        cond = self.visit(node.condition)
        then = self.visit(node.then_expr)
        els  = self.visit(node.else_expr)
        return f"({cond} ? {then} : {els})"

    def visit_LetExpr(self, node: LetExpr) -> str:
        val  = self.visit(node.value)
        body = self.visit(node.body)
        return f"let {node.name} = {val} in {body}"

    def visit_FuncCall(self, node: FuncCall) -> str:
        args = ', '.join(self.visit(a) for a in node.args)
        return f"{node.name}({args})"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo(label: str, ast: Node) -> None:
    print(f"\n{'─'*52}")
    print(f"Expression: {label}")

    pp = PrettyPrintVisitor()
    printed = pp.visit(ast)
    print(f"  Pretty:    {printed}")

    tc = TypeCheckVisitor(type_env={'x': INT, 'y': FLOAT, 's': STR, 'flag': BOOL})
    inferred = tc.visit(ast)
    if tc.errors:
        print(f"  Type:      ERROR")
        for e in tc.errors:
            print(f"             {e}")
    else:
        print(f"  Type:      {inferred}")

    ev = EvalVisitor(env={'x': 5, 'y': 2.0, 's': 'hello', 'flag': True})
    try:
        result = ev.visit(ast)
        print(f"  Value:     {result!r}")
    except EvalError as e:
        print(f"  Eval err:  {e}")


def main():
    print("=" * 60)
    print("AST Visitor Pattern Demo")
    print("=" * 60)

    # 1. Simple arithmetic: x * 2 + y
    demo("x * 2 + y",
         BinOp('+', BinOp('*', Var('x'), IntLit(2)), Var('y')))

    # 2. Comparison: x > 3 and flag
    demo("x > 3 and flag",
         BinOp('and', BinOp('>', Var('x'), IntLit(3)), Var('flag')))

    # 3. Ternary if: (flag ? x : -x)
    demo("flag ? x : -x",
         IfExpr(Var('flag'), Var('x'), UnaryOp('-', Var('x'))))

    # 4. Let expression: let z = x + 1 in z * z
    demo("let z = x + 1 in z * z",
         LetExpr('z', BinOp('+', Var('x'), IntLit(1)),
                 BinOp('*', Var('z'), Var('z'))))

    # 5. Function call: sqrt(x * x + y * y)
    demo("sqrt(x * x + y * y)",
         FuncCall('sqrt', [
             BinOp('+',
                   BinOp('*', Var('x'), Var('x')),
                   BinOp('*', Var('y'), Var('y')))
         ]))

    # 6. String concatenation: s + " world"
    demo('s + " world"',
         BinOp('+', Var('s'), StrLit(' world')))

    # 7. Type error: x + flag (int + bool)
    demo("x + flag (type error)",
         BinOp('+', Var('x'), Var('flag')))

    # 8. Nested let: let a = 3 in let b = a + 2 in a * b
    demo("let a = 3 in let b = a + 2 in a * b",
         LetExpr('a', IntLit(3),
                 LetExpr('b', BinOp('+', Var('a'), IntLit(2)),
                         BinOp('*', Var('a'), Var('b')))))


if __name__ == "__main__":
    main()
