"""
07_type_checker.py - Symbol Table and Type Checker

Demonstrates semantic analysis: symbol table management with nested
scopes, and type checking for a mini-language.

Mini-language features:
  - Types: int, float, bool, string, void
  - Variables: declaration and use
  - Functions: declaration with typed parameters and return type
  - Control flow: if/else, while
  - Expressions: arithmetic, comparison, logical
  - Return statements

The type checker:
  1. Builds a symbol table with nested scopes
  2. Checks that variables are declared before use
  3. Checks that assignments are type-compatible
  4. Checks function call arity and argument types
  5. Checks that return types match function signatures
  6. Reports all errors (not just the first)

Topics covered:
  - Scope chain (linked symbol tables)
  - Type compatibility (int/float coercion)
  - Function type signatures
  - Two-pass: collect function declarations, then type-check bodies
  - Accumulating multiple type errors
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Type:
    name: str

    def __repr__(self) -> str:
        return self.name


INT    = Type('int')
FLOAT  = Type('float')
BOOL   = Type('bool')
STRING = Type('string')
VOID   = Type('void')
ERROR  = Type('error')    # sentinel for type errors

TYPE_MAP: dict[str, Type] = {
    'int': INT, 'float': FLOAT, 'bool': BOOL,
    'string': STRING, 'void': VOID,
}


def is_numeric(t: Type) -> bool:
    return t in (INT, FLOAT)


def numeric_result(t1: Type, t2: Type) -> Type:
    """int op int -> int; anything with float -> float."""
    if t1 == ERROR or t2 == ERROR:
        return ERROR
    if t1 == FLOAT or t2 == FLOAT:
        return FLOAT
    return INT


def compatible(expected: Type, actual: Type) -> bool:
    """Is 'actual' assignable to 'expected'?"""
    if expected == actual:
        return True
    # int is implicitly convertible to float
    if expected == FLOAT and actual == INT:
        return True
    return False


# ---------------------------------------------------------------------------
# Symbol Table
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    name: str
    type: Type
    is_function: bool = False
    param_types: list[Type] = field(default_factory=list)
    return_type: Optional[Type] = None
    defined_at: int = 0    # line number


class Scope:
    """A single scope level (function body, block, etc.)."""

    def __init__(self, name: str, parent: Optional[Scope] = None):
        self.name = name
        self.parent = parent
        self._symbols: dict[str, Symbol] = {}

    def define(self, sym: Symbol) -> bool:
        """Define a symbol. Returns False if already defined in this scope."""
        if sym.name in self._symbols:
            return False
        self._symbols[sym.name] = sym
        return True

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a name in this scope and all parent scopes."""
        if name in self._symbols:
            return self._symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up only in this scope (not parents)."""
        return self._symbols.get(name)

    def depth(self) -> int:
        d = 0
        s = self
        while s.parent:
            d += 1
            s = s.parent
        return d

    def __repr__(self) -> str:
        syms = list(self._symbols.keys())
        return f"Scope({self.name!r}, symbols={syms})"


class SymbolTable:
    """
    The symbol table manages the scope stack.
    Supports entering/exiting scopes and looking up symbols.
    """

    def __init__(self):
        self.global_scope = Scope("global")
        self.current: Scope = self.global_scope

    def enter_scope(self, name: str) -> Scope:
        new_scope = Scope(name, parent=self.current)
        self.current = new_scope
        return new_scope

    def exit_scope(self) -> Scope:
        exited = self.current
        if self.current.parent:
            self.current = self.current.parent
        return exited

    def define(self, sym: Symbol) -> bool:
        return self.current.define(sym)

    def lookup(self, name: str) -> Optional[Symbol]:
        return self.current.lookup(name)

    def depth(self) -> int:
        return self.current.depth()


# ---------------------------------------------------------------------------
# AST Nodes (minimal, focus on type checking)
# ---------------------------------------------------------------------------

@dataclass
class Program:
    declarations: list

@dataclass
class FuncDecl:
    name: str
    return_type: str
    params: list[tuple[str, str]]   # [(name, type_str), ...]
    body: list                       # list of statements
    line: int = 0

@dataclass
class VarDecl:
    name: str
    type_str: str
    init: Optional[Any] = None
    line: int = 0

@dataclass
class Assign:
    name: str
    value: Any
    line: int = 0

@dataclass
class ReturnStmt:
    value: Optional[Any] = None
    line: int = 0

@dataclass
class IfStmt:
    condition: Any
    then_block: list
    else_block: Optional[list] = None
    line: int = 0

@dataclass
class WhileStmt:
    condition: Any
    body: list
    line: int = 0

@dataclass
class ExprStmt:
    expr: Any
    line: int = 0

@dataclass
class IntLit:
    value: int

@dataclass
class FloatLit:
    value: float

@dataclass
class BoolLit:
    value: bool

@dataclass
class StrLit:
    value: str

@dataclass
class VarRef:
    name: str
    line: int = 0

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    line: int = 0

@dataclass
class UnaryOp:
    op: str
    operand: Any
    line: int = 0

@dataclass
class CallExpr:
    name: str
    args: list
    line: int = 0


# ---------------------------------------------------------------------------
# Type Checker
# ---------------------------------------------------------------------------

class TypeChecker:
    """
    Walks the AST and performs type checking.
    Errors are accumulated in self.errors (list of strings).
    """

    def __init__(self):
        self.table = SymbolTable()
        self.errors: list[str] = []
        self._current_func_return: Optional[Type] = None

    def error(self, msg: str, line: int = 0) -> Type:
        loc = f" [line {line}]" if line else ""
        self.errors.append(f"TypeError{loc}: {msg}")
        return ERROR

    # --- Top-level ---

    def check_program(self, program: Program) -> None:
        # First pass: register all function signatures (forward declarations)
        for decl in program.declarations:
            if isinstance(decl, FuncDecl):
                self._register_func(decl)

        # Second pass: type-check all declarations
        for decl in program.declarations:
            self.check_decl(decl)

    def _register_func(self, decl: FuncDecl) -> None:
        ret_type = TYPE_MAP.get(decl.return_type, ERROR)
        param_types = [TYPE_MAP.get(pt, ERROR) for _, pt in decl.params]
        sym = Symbol(
            name=decl.name,
            type=ret_type,
            is_function=True,
            param_types=param_types,
            return_type=ret_type,
            defined_at=decl.line,
        )
        if not self.table.define(sym):
            self.error(f"Function {decl.name!r} already defined", decl.line)

    def check_decl(self, decl) -> None:
        if isinstance(decl, FuncDecl):
            self.check_func(decl)
        elif isinstance(decl, VarDecl):
            self.check_var_decl(decl, global_scope=True)

    def check_func(self, decl: FuncDecl) -> None:
        ret_type = TYPE_MAP.get(decl.return_type, ERROR)
        self._current_func_return = ret_type

        self.table.enter_scope(f"func:{decl.name}")

        # Define parameters in function scope
        for pname, ptype_str in decl.params:
            ptype = TYPE_MAP.get(ptype_str, ERROR)
            if ptype == ERROR:
                self.error(f"Unknown parameter type {ptype_str!r}", decl.line)
            sym = Symbol(name=pname, type=ptype, defined_at=decl.line)
            if not self.table.define(sym):
                self.error(f"Duplicate parameter {pname!r}", decl.line)

        for stmt in decl.body:
            self.check_stmt(stmt)

        self.table.exit_scope()
        self._current_func_return = None

    def check_var_decl(self, decl: VarDecl, global_scope: bool = False) -> None:
        declared_type = TYPE_MAP.get(decl.type_str)
        if declared_type is None:
            self.error(f"Unknown type {decl.type_str!r}", decl.line)
            declared_type = ERROR

        if decl.init is not None:
            init_type = self.check_expr(decl.init)
            if init_type != ERROR and declared_type != ERROR:
                if not compatible(declared_type, init_type):
                    self.error(
                        f"Cannot initialize {declared_type} variable with {init_type} value",
                        decl.line
                    )

        sym = Symbol(name=decl.name, type=declared_type, defined_at=decl.line)
        if not self.table.define(sym):
            self.error(f"Variable {decl.name!r} already declared in this scope", decl.line)

    def check_stmt(self, stmt) -> None:
        if isinstance(stmt, VarDecl):
            self.check_var_decl(stmt)
        elif isinstance(stmt, Assign):
            self.check_assign(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.check_return(stmt)
        elif isinstance(stmt, IfStmt):
            self.check_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self.check_while(stmt)
        elif isinstance(stmt, ExprStmt):
            self.check_expr(stmt.expr)

    def check_assign(self, stmt: Assign) -> None:
        sym = self.table.lookup(stmt.name)
        if sym is None:
            self.error(f"Undefined variable {stmt.name!r}", stmt.line)
            return
        val_type = self.check_expr(stmt.value)
        if val_type != ERROR and sym.type != ERROR:
            if not compatible(sym.type, val_type):
                self.error(
                    f"Cannot assign {val_type} to variable {stmt.name!r} of type {sym.type}",
                    stmt.line
                )

    def check_return(self, stmt: ReturnStmt) -> None:
        if self._current_func_return is None:
            self.error("'return' outside function", stmt.line)
            return
        if stmt.value is None:
            if self._current_func_return != VOID:
                self.error(
                    f"Function must return {self._current_func_return}, got void",
                    stmt.line
                )
        else:
            val_type = self.check_expr(stmt.value)
            if val_type != ERROR and self._current_func_return != ERROR:
                if not compatible(self._current_func_return, val_type):
                    self.error(
                        f"Return type mismatch: expected {self._current_func_return}, got {val_type}",
                        stmt.line
                    )

    def check_if(self, stmt: IfStmt) -> None:
        cond_type = self.check_expr(stmt.condition)
        if cond_type not in (BOOL, ERROR):
            self.error(f"If condition must be bool, got {cond_type}", stmt.line)
        self.table.enter_scope("if-then")
        for s in stmt.then_block:
            self.check_stmt(s)
        self.table.exit_scope()
        if stmt.else_block is not None:
            self.table.enter_scope("if-else")
            for s in stmt.else_block:
                self.check_stmt(s)
            self.table.exit_scope()

    def check_while(self, stmt: WhileStmt) -> None:
        cond_type = self.check_expr(stmt.condition)
        if cond_type not in (BOOL, ERROR):
            self.error(f"While condition must be bool, got {cond_type}", stmt.line)
        self.table.enter_scope("while-body")
        for s in stmt.body:
            self.check_stmt(s)
        self.table.exit_scope()

    def check_expr(self, expr) -> Type:
        if isinstance(expr, IntLit):   return INT
        if isinstance(expr, FloatLit): return FLOAT
        if isinstance(expr, BoolLit):  return BOOL
        if isinstance(expr, StrLit):   return STRING
        if isinstance(expr, VarRef):
            sym = self.table.lookup(expr.name)
            if sym is None:
                return self.error(f"Undefined variable {expr.name!r}", expr.line)
            return sym.type
        if isinstance(expr, BinOp):
            return self.check_binop(expr)
        if isinstance(expr, UnaryOp):
            return self.check_unaryop(expr)
        if isinstance(expr, CallExpr):
            return self.check_call(expr)
        self.error(f"Unknown expression type: {type(expr).__name__}")
        return ERROR

    def check_binop(self, expr: BinOp) -> Type:
        lt = self.check_expr(expr.left)
        rt = self.check_expr(expr.right)
        op = expr.op
        if op in ('+', '-', '*', '/'):
            if op == '+' and lt == STRING and rt == STRING:
                return STRING
            if is_numeric(lt) and is_numeric(rt):
                return numeric_result(lt, rt)
            if lt != ERROR and rt != ERROR:
                self.error(f"Operator {op!r} not valid for {lt} and {rt}", expr.line)
            return ERROR
        if op in ('<', '>', '<=', '>='):
            if is_numeric(lt) and is_numeric(rt):
                return BOOL
            if lt != ERROR and rt != ERROR:
                self.error(f"Comparison {op!r} not valid for {lt} and {rt}", expr.line)
            return ERROR
        if op in ('==', '!='):
            if lt == rt or (is_numeric(lt) and is_numeric(rt)):
                return BOOL
            if lt != ERROR and rt != ERROR:
                self.error(f"Cannot compare {lt} with {rt}", expr.line)
            return ERROR
        if op in ('&&', '||', 'and', 'or'):
            if lt == BOOL and rt == BOOL:
                return BOOL
            if lt != ERROR and rt != ERROR:
                self.error(f"Logical {op!r} requires bool operands", expr.line)
            return ERROR
        self.error(f"Unknown operator {op!r}", expr.line)
        return ERROR

    def check_unaryop(self, expr: UnaryOp) -> Type:
        t = self.check_expr(expr.operand)
        if expr.op == '-':
            if is_numeric(t): return t
            if t != ERROR: self.error(f"Unary '-' not valid for {t}", expr.line)
            return ERROR
        if expr.op in ('!', 'not'):
            if t == BOOL: return BOOL
            if t != ERROR: self.error(f"'not' requires bool, got {t}", expr.line)
            return ERROR
        self.error(f"Unknown unary operator {expr.op!r}", expr.line)
        return ERROR

    def check_call(self, expr: CallExpr) -> Type:
        sym = self.table.lookup(expr.name)
        if sym is None:
            return self.error(f"Undefined function {expr.name!r}", expr.line)
        if not sym.is_function:
            return self.error(f"{expr.name!r} is not a function", expr.line)
        if len(expr.args) != len(sym.param_types):
            self.error(
                f"Function {expr.name!r} expects {len(sym.param_types)} args, got {len(expr.args)}",
                expr.line
            )
        for i, (arg, expected) in enumerate(zip(expr.args, sym.param_types)):
            at = self.check_expr(arg)
            if at != ERROR and expected != ERROR and not compatible(expected, at):
                self.error(
                    f"Argument {i+1} of {expr.name!r}: expected {expected}, got {at}",
                    expr.line
                )
        return sym.return_type or VOID


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def build_program() -> Program:
    """
    Build an AST representing:

    int add(int x, int y) { return x + y; }
    float average(int a, int b) { return (a + b) / 2.0; }
    void main() {
        int result = add(3, 4);
        float avg = average(10, 20);
        bool flag = result > 5;
        string msg = "done";
        if (flag) {
            int local = result * 2;
        } else {
            int local = 0;  // same name, different scope: ok
        }
        // Error: assigning float to int
        result = avg;
        // Error: wrong arg count
        int bad = add(1, 2, 3);
    }
    """
    add_func = FuncDecl(
        name='add', return_type='int',
        params=[('x', 'int'), ('y', 'int')],
        body=[ReturnStmt(BinOp('+', VarRef('x'), VarRef('y')), line=2)],
        line=1
    )
    avg_func = FuncDecl(
        name='average', return_type='float',
        params=[('a', 'int'), ('b', 'int')],
        body=[ReturnStmt(
            BinOp('/', BinOp('+', VarRef('a'), VarRef('b')), FloatLit(2.0)),
            line=5
        )],
        line=4
    )
    main_func = FuncDecl(
        name='main', return_type='void',
        params=[],
        body=[
            VarDecl('result', 'int',
                    CallExpr('add', [IntLit(3), IntLit(4)], line=8), line=8),
            VarDecl('avg', 'float',
                    CallExpr('average', [IntLit(10), IntLit(20)], line=9), line=9),
            VarDecl('flag', 'bool',
                    BinOp('>', VarRef('result'), IntLit(5), line=10), line=10),
            VarDecl('msg', 'string', StrLit('done'), line=11),
            IfStmt(
                condition=VarRef('flag'),
                then_block=[VarDecl('local', 'int',
                                    BinOp('*', VarRef('result'), IntLit(2)), line=13)],
                else_block=[VarDecl('local', 'int', IntLit(0), line=15)],
                line=12
            ),
            # Intentional type error: assigning float to int
            Assign('result', VarRef('avg'), line=18),
            # Intentional arity error
            VarDecl('bad', 'int',
                    CallExpr('add', [IntLit(1), IntLit(2), IntLit(3)], line=20), line=20),
        ],
        line=7
    )
    return Program(declarations=[add_func, avg_func, main_func])


def main():
    print("=" * 60)
    print("Symbol Table and Type Checker Demo")
    print("=" * 60)

    program = build_program()
    tc = TypeChecker()
    tc.check_program(program)

    print("\nGlobal scope symbols:")
    for name, sym in tc.table.global_scope._symbols.items():
        if sym.is_function:
            params = ', '.join(str(t) for t in sym.param_types)
            print(f"  function {name}({params}) -> {sym.return_type}")
        else:
            print(f"  variable {name}: {sym.type}")

    print(f"\nType checking complete.")
    print(f"Errors found: {len(tc.errors)}")
    if tc.errors:
        print("\nError list:")
        for err in tc.errors:
            print(f"  {err}")
    else:
        print("  No errors.")

    # Demonstrate correct program
    print("\n--- Correct program (no errors) ---")
    correct = Program([
        FuncDecl('square', 'int', [('n', 'int')],
                 [ReturnStmt(BinOp('*', VarRef('n'), VarRef('n')), line=1)], line=1),
        FuncDecl('main', 'void', [],
                 [VarDecl('x', 'int', IntLit(5), line=3),
                  VarDecl('sq', 'int', CallExpr('square', [VarRef('x')], line=4), line=4),
                  VarDecl('ok', 'bool', BinOp('>', VarRef('sq'), IntLit(10), line=5), line=5)],
                 line=2)
    ])
    tc2 = TypeChecker()
    tc2.check_program(correct)
    print(f"Errors: {len(tc2.errors)}")
    if tc2.errors:
        for e in tc2.errors: print(f"  {e}")
    else:
        print("  No errors. Program is type-correct.")


if __name__ == "__main__":
    main()
