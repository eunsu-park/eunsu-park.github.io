"""
04_recursive_descent_parser.py - Recursive Descent Parser

Implements a hand-written recursive descent parser for a simple
imperative language. Builds an AST and provides evaluation.

Grammar (simplified, left-recursion removed):
  program     ::= stmt*
  stmt        ::= if_stmt | while_stmt | print_stmt | assign_stmt | block
  if_stmt     ::= 'if' '(' expr ')' stmt ('else' stmt)?
  while_stmt  ::= 'while' '(' expr ')' stmt
  print_stmt  ::= 'print' '(' expr ')' ';'
  assign_stmt ::= IDENT '=' expr ';'
  block       ::= '{' stmt* '}'
  expr        ::= or_expr
  or_expr     ::= and_expr ('||' and_expr)*
  and_expr    ::= eq_expr ('&&' eq_expr)*
  eq_expr     ::= rel_expr (('=='|'!=') rel_expr)*
  rel_expr    ::= add_expr (('<'|'>'|'<='|'>=') add_expr)*
  add_expr    ::= mul_expr (('+'|'-') mul_expr)*
  mul_expr    ::= unary (('*'|'/') unary)*
  unary       ::= ('-'|'!') unary | primary
  primary     ::= INT | FLOAT | STRING | BOOL | IDENT | '(' expr ')'

Topics covered:
  - Recursive descent parsing
  - Abstract Syntax Tree (AST) construction
  - Operator precedence via grammar layers
  - Tree evaluation (interpreter)
  - AST pretty-printing
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Lexer (minimal, reused from 01_lexer concepts)
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: str
    value: str
    line: int

PATTERNS = [
    ('FLOAT',   r'\d+\.\d*'),
    ('INT',     r'\d+'),
    ('STRING',  r'"[^"]*"'),
    ('BOOL',    r'\b(true|false)\b'),
    ('KW',      r'\b(if|else|while|print)\b'),
    ('IDENT',   r'[A-Za-z_]\w*'),
    ('OP',      r'==|!=|<=|>=|&&|\|\||[+\-*/<>=!]'),
    ('PUNCT',   r'[(){};,]'),
    ('WS',      r'\s+'),
]
_LEX_RE = re.compile('|'.join(f'(?P<{name}>{pat})' for name, pat in PATTERNS))


def tokenize(source: str) -> list[Token]:
    tokens = []
    line = 1
    for m in _LEX_RE.finditer(source):
        kind = m.lastgroup
        val = m.group()
        if kind == 'WS':
            line += val.count('\n')
            continue
        if kind == 'KW':
            tokens.append(Token(val, val, line))  # keyword type IS the word
        elif kind == 'BOOL':
            tokens.append(Token('BOOL', val, line))
        else:
            tokens.append(Token(kind, val, line))
    tokens.append(Token('EOF', '', line))
    return tokens


# ---------------------------------------------------------------------------
# AST Node definitions
# ---------------------------------------------------------------------------

@dataclass
class NumLit:
    value: float
    is_int: bool = True

@dataclass
class StrLit:
    value: str

@dataclass
class BoolLit:
    value: bool

@dataclass
class Var:
    name: str

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    op: str
    operand: Any

@dataclass
class Assign:
    name: str
    value: Any

@dataclass
class IfStmt:
    condition: Any
    then_branch: Any
    else_branch: Optional[Any] = None

@dataclass
class WhileStmt:
    condition: Any
    body: Any

@dataclass
class PrintStmt:
    value: Any

@dataclass
class Block:
    stmts: list = field(default_factory=list)

@dataclass
class Program:
    stmts: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, msg: str, token: Token):
        super().__init__(f"Parse error at line {token.line}: {msg} (got {token.type!r} = {token.value!r})")


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type_: str, value: str = None) -> Token:
        tok = self.peek()
        if tok.type != type_:
            raise ParseError(f"Expected {type_!r}", tok)
        if value is not None and tok.value != value:
            raise ParseError(f"Expected {value!r}", tok)
        return self.advance()

    def match(self, type_: str, value: str = None) -> bool:
        tok = self.peek()
        if tok.type != type_:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    # --- Statements ---

    def parse_program(self) -> Program:
        stmts = []
        while not self.match('EOF'):
            stmts.append(self.parse_stmt())
        return Program(stmts)

    def parse_stmt(self):
        tok = self.peek()
        if tok.type == 'if':
            return self.parse_if()
        elif tok.type == 'while':
            return self.parse_while()
        elif tok.type == 'print':
            return self.parse_print()
        elif tok.type == 'PUNCT' and tok.value == '{':
            return self.parse_block()
        elif tok.type == 'IDENT':
            return self.parse_assign()
        else:
            raise ParseError("Expected statement", tok)

    def parse_if(self) -> IfStmt:
        self.expect('if')
        self.expect('PUNCT', '(')
        cond = self.parse_expr()
        self.expect('PUNCT', ')')
        then_br = self.parse_stmt()
        else_br = None
        if self.match('else'):
            self.advance()
            else_br = self.parse_stmt()
        return IfStmt(cond, then_br, else_br)

    def parse_while(self) -> WhileStmt:
        self.expect('while')
        self.expect('PUNCT', '(')
        cond = self.parse_expr()
        self.expect('PUNCT', ')')
        body = self.parse_stmt()
        return WhileStmt(cond, body)

    def parse_print(self) -> PrintStmt:
        self.expect('print')
        self.expect('PUNCT', '(')
        val = self.parse_expr()
        self.expect('PUNCT', ')')
        self.expect('PUNCT', ';')
        return PrintStmt(val)

    def parse_assign(self) -> Assign:
        name = self.expect('IDENT').value
        self.expect('OP', '=')
        val = self.parse_expr()
        self.expect('PUNCT', ';')
        return Assign(name, val)

    def parse_block(self) -> Block:
        self.expect('PUNCT', '{')
        stmts = []
        while not (self.match('PUNCT', '}') or self.match('EOF')):
            stmts.append(self.parse_stmt())
        self.expect('PUNCT', '}')
        return Block(stmts)

    # --- Expressions (precedence climbing via recursive functions) ---

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.match('OP', '||'):
            op = self.advance().value
            left = BinOp(op, left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_eq()
        while self.match('OP', '&&'):
            op = self.advance().value
            left = BinOp(op, left, self.parse_eq())
        return left

    def parse_eq(self):
        left = self.parse_rel()
        while self.peek().type == 'OP' and self.peek().value in ('==', '!='):
            op = self.advance().value
            left = BinOp(op, left, self.parse_rel())
        return left

    def parse_rel(self):
        left = self.parse_add()
        while self.peek().type == 'OP' and self.peek().value in ('<', '>', '<=', '>='):
            op = self.advance().value
            left = BinOp(op, left, self.parse_add())
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek().type == 'OP' and self.peek().value in ('+', '-'):
            op = self.advance().value
            left = BinOp(op, left, self.parse_mul())
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek().type == 'OP' and self.peek().value in ('*', '/'):
            op = self.advance().value
            left = BinOp(op, left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.peek().type == 'OP' and self.peek().value in ('-', '!'):
            op = self.advance().value
            return UnaryOp(op, self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok.type == 'INT':
            self.advance()
            return NumLit(int(tok.value), is_int=True)
        elif tok.type == 'FLOAT':
            self.advance()
            return NumLit(float(tok.value), is_int=False)
        elif tok.type == 'STRING':
            self.advance()
            return StrLit(tok.value[1:-1])   # strip quotes
        elif tok.type == 'BOOL':
            self.advance()
            return BoolLit(tok.value == 'true')
        elif tok.type == 'IDENT':
            self.advance()
            return Var(tok.value)
        elif tok.type == 'PUNCT' and tok.value == '(':
            self.advance()
            expr = self.parse_expr()
            self.expect('PUNCT', ')')
            return expr
        else:
            raise ParseError("Expected primary expression", tok)


# ---------------------------------------------------------------------------
# Evaluator (tree-walk interpreter)
# ---------------------------------------------------------------------------

class EvalError(Exception):
    pass


class Evaluator:
    def __init__(self):
        self.env: dict[str, Any] = {}

    def eval(self, node) -> Any:
        match node:
            case Program(stmts=stmts) | Block(stmts=stmts):
                result = None
                for s in stmts:
                    result = self.eval(s)
                return result
            case Assign(name=name, value=val):
                self.env[name] = self.eval(val)
                return self.env[name]
            case IfStmt(condition=cond, then_branch=then_br, else_branch=else_br):
                if self.eval(cond):
                    return self.eval(then_br)
                elif else_br:
                    return self.eval(else_br)
            case WhileStmt(condition=cond, body=body):
                while self.eval(cond):
                    self.eval(body)
            case PrintStmt(value=val):
                v = self.eval(val)
                print(f"  [output] {v}")
                return v
            case NumLit(value=v):
                return v
            case StrLit(value=v):
                return v
            case BoolLit(value=v):
                return v
            case Var(name=name):
                if name not in self.env:
                    raise EvalError(f"Undefined variable: {name!r}")
                return self.env[name]
            case BinOp(op=op, left=left, right=right):
                l, r = self.eval(left), self.eval(right)
                match op:
                    case '+':  return l + r
                    case '-':  return l - r
                    case '*':  return l * r
                    case '/':  return l / r if r != 0 else (raise_(EvalError("Division by zero")))
                    case '<':  return l < r
                    case '>':  return l > r
                    case '<=': return l <= r
                    case '>=': return l >= r
                    case '==': return l == r
                    case '!=': return l != r
                    case '&&': return bool(l) and bool(r)
                    case '||': return bool(l) or bool(r)
            case UnaryOp(op=op, operand=operand):
                v = self.eval(operand)
                if op == '-': return -v
                if op == '!': return not v
        return None


def raise_(exc):
    raise exc


# ---------------------------------------------------------------------------
# AST Pretty Printer
# ---------------------------------------------------------------------------

def pprint(node, indent: int = 0) -> None:
    pad = "  " * indent
    match node:
        case Program(stmts=stmts):
            print(f"{pad}Program")
            for s in stmts: pprint(s, indent+1)
        case Block(stmts=stmts):
            print(f"{pad}Block")
            for s in stmts: pprint(s, indent+1)
        case Assign(name=n, value=v):
            print(f"{pad}Assign({n!r})")
            pprint(v, indent+1)
        case IfStmt(condition=c, then_branch=t, else_branch=e):
            print(f"{pad}If")
            print(f"{pad}  cond:"); pprint(c, indent+2)
            print(f"{pad}  then:"); pprint(t, indent+2)
            if e: print(f"{pad}  else:"); pprint(e, indent+2)
        case WhileStmt(condition=c, body=b):
            print(f"{pad}While")
            print(f"{pad}  cond:"); pprint(c, indent+2)
            print(f"{pad}  body:"); pprint(b, indent+2)
        case PrintStmt(value=v):
            print(f"{pad}Print"); pprint(v, indent+1)
        case BinOp(op=op, left=l, right=r):
            print(f"{pad}BinOp({op!r})")
            pprint(l, indent+1); pprint(r, indent+1)
        case UnaryOp(op=op, operand=o):
            print(f"{pad}Unary({op!r})"); pprint(o, indent+1)
        case NumLit(value=v): print(f"{pad}Num({v})")
        case StrLit(value=v): print(f"{pad}Str({v!r})")
        case BoolLit(value=v): print(f"{pad}Bool({v})")
        case Var(name=n): print(f"{pad}Var({n!r})")
        case _: print(f"{pad}{node!r}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

PROGRAM_1 = """\
x = 10;
y = 3;
z = x * y + 2;
print(z);
"""

PROGRAM_2 = """\
n = 10;
fib_a = 0;
fib_b = 1;
i = 2;
while (i <= n) {
    tmp = fib_a + fib_b;
    fib_a = fib_b;
    fib_b = tmp;
    i = i + 1;
}
print(fib_b);
"""

PROGRAM_3 = """\
x = 7;
if (x > 5) {
    print("x is greater than 5");
} else {
    print("x is not greater than 5");
}
"""


def demo(label: str, source: str) -> None:
    print(f"\n{'─'*56}")
    print(f"Demo: {label}")
    print(f"Source:\n{source}")
    tokens = tokenize(source)
    parser = Parser(tokens)
    ast = parser.parse_program()

    print("AST:")
    pprint(ast)

    print("\nEvaluation:")
    ev = Evaluator()
    ev.eval(ast)
    print(f"Final env: {ev.env}")


def main():
    print("=" * 60)
    print("Recursive Descent Parser Demo")
    print("=" * 60)
    demo("Arithmetic", PROGRAM_1)
    demo("Fibonacci (while loop)", PROGRAM_2)
    demo("If/Else with strings", PROGRAM_3)


if __name__ == "__main__":
    main()
