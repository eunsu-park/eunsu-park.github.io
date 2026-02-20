"""
05_ll1_parser.py - LL(1) Table-Driven Parser

Demonstrates the construction and use of an LL(1) parser:
  1. Represent a context-free grammar
  2. Compute FIRST sets for all symbols
  3. Compute FOLLOW sets for all non-terminals
  4. Build the LL(1) parsing table
  5. Table-driven parsing using an explicit stack

Grammar (unambiguous arithmetic expressions):
  E  -> T E'
  E' -> '+' T E' | '-' T E' | ε
  T  -> F T'
  T' -> '*' F T' | '/' F T' | ε
  F  -> '(' E ')' | num | id

This grammar is already left-recursion-free and suitable for LL(1).

Topics covered:
  - Grammar representation using Python dicts
  - FIRST set computation (handles ε productions)
  - FOLLOW set computation
  - LL(1) parse table construction
  - Detecting grammar conflicts (not LL(1) if conflicts exist)
  - Stack-based parsing with parse trace
"""

from __future__ import annotations
from collections import defaultdict
from typing import Optional

EPSILON = 'ε'
EOF_SYM = '$'


# ---------------------------------------------------------------------------
# Grammar Representation
# ---------------------------------------------------------------------------

class Grammar:
    """
    A context-free grammar.
    Productions are stored as:
      { NonTerminal: [ [sym1, sym2, ...], [sym1, ...], ... ] }
    Terminals are single-quoted strings like 'id', 'num', '+', etc.
    Non-terminals are unquoted strings: 'E', 'T', 'F', etc.
    """

    def __init__(self, start: str, productions: dict[str, list[list[str]]]):
        self.start = start
        self.productions = productions

        # Identify all non-terminals and terminals
        self.non_terminals: set[str] = set(productions.keys())
        self.terminals: set[str] = set()
        for rhs_list in productions.values():
            for rhs in rhs_list:
                for sym in rhs:
                    if sym not in self.non_terminals and sym != EPSILON:
                        self.terminals.add(sym)

    def is_terminal(self, sym: str) -> bool:
        return sym in self.terminals or sym == EOF_SYM

    def is_nonterminal(self, sym: str) -> bool:
        return sym in self.non_terminals

    def __repr__(self) -> str:
        lines = [f"Grammar (start={self.start!r}):"]
        for nt, prods in self.productions.items():
            for rhs in prods:
                lines.append(f"  {nt} -> {' '.join(rhs)}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# FIRST Set Computation
# ---------------------------------------------------------------------------

def compute_first(grammar: Grammar) -> dict[str, set[str]]:
    """
    Compute FIRST(X) for every symbol X in the grammar.

    FIRST(X) = set of terminals that can appear as the first symbol
               of any string derived from X. Includes ε if X =>* ε.

    Rules:
      1. If X is terminal: FIRST(X) = {X}
      2. If X -> ε: ε ∈ FIRST(X)
      3. If X -> Y1 Y2 ... Yk:
           add FIRST(Y1) - {ε} to FIRST(X)
           if ε ∈ FIRST(Y1), add FIRST(Y2) - {ε}, etc.
           if ε ∈ FIRST(Yi) for all i, add ε to FIRST(X)
    """
    first: dict[str, set[str]] = defaultdict(set)

    # Terminals: FIRST(t) = {t}
    for t in grammar.terminals:
        first[t].add(t)
    first[EOF_SYM].add(EOF_SYM)

    # Iteratively compute FIRST for non-terminals
    changed = True
    while changed:
        changed = False
        for nt, prods in grammar.productions.items():
            for rhs in prods:
                # Handle epsilon production
                if rhs == [EPSILON]:
                    if EPSILON not in first[nt]:
                        first[nt].add(EPSILON)
                        changed = True
                    continue

                # Compute FIRST of the RHS
                all_have_epsilon = True
                for sym in rhs:
                    sym_first = first[sym] - {EPSILON}
                    before = len(first[nt])
                    first[nt].update(sym_first)
                    if len(first[nt]) > before:
                        changed = True
                    if EPSILON not in first[sym]:
                        all_have_epsilon = False
                        break
                else:
                    pass

                if all_have_epsilon:
                    if EPSILON not in first[nt]:
                        first[nt].add(EPSILON)
                        changed = True

    return dict(first)


def first_of_string(symbols: list[str], first: dict[str, set[str]]) -> set[str]:
    """
    Compute FIRST of a sequence of symbols (e.g., the RHS of a production).
    """
    result: set[str] = set()
    all_nullable = True
    for sym in symbols:
        if sym == EPSILON:
            continue
        result.update(first.get(sym, set()) - {EPSILON})
        if EPSILON not in first.get(sym, set()):
            all_nullable = False
            break
    if all_nullable:
        result.add(EPSILON)
    return result


# ---------------------------------------------------------------------------
# FOLLOW Set Computation
# ---------------------------------------------------------------------------

def compute_follow(grammar: Grammar, first: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Compute FOLLOW(A) for every non-terminal A.

    FOLLOW(A) = set of terminals that can appear immediately to the
                right of A in some sentential form.

    Rules:
      1. EOF_SYM ($) ∈ FOLLOW(start)
      2. If B -> α A β: add FIRST(β) - {ε} to FOLLOW(A)
      3. If B -> α A or B -> α A β where ε ∈ FIRST(β):
           add FOLLOW(B) to FOLLOW(A)
    """
    follow: dict[str, set[str]] = defaultdict(set)
    follow[grammar.start].add(EOF_SYM)

    changed = True
    while changed:
        changed = False
        for nt, prods in grammar.productions.items():
            for rhs in prods:
                for i, sym in enumerate(rhs):
                    if not grammar.is_nonterminal(sym):
                        continue
                    # Compute FIRST(β) where β = rhs[i+1:]
                    beta = rhs[i+1:]
                    beta_first = first_of_string(beta, first) if beta else {EPSILON}

                    # Rule 2: add FIRST(β) - {ε}
                    before = len(follow[sym])
                    follow[sym].update(beta_first - {EPSILON})
                    if len(follow[sym]) > before:
                        changed = True

                    # Rule 3: if ε ∈ FIRST(β), add FOLLOW(B)
                    if EPSILON in beta_first:
                        before = len(follow[sym])
                        follow[sym].update(follow[nt])
                        if len(follow[sym]) > before:
                            changed = True

    return dict(follow)


# ---------------------------------------------------------------------------
# LL(1) Parse Table Construction
# ---------------------------------------------------------------------------

class GrammarConflictError(Exception):
    pass


def build_ll1_table(
    grammar: Grammar,
    first: dict[str, set[str]],
    follow: dict[str, set[str]],
) -> dict[tuple[str, str], list[str]]:
    """
    Build the LL(1) parse table M[A, a].

    For each production A -> α:
      1. For each terminal a in FIRST(α): set M[A, a] = α
      2. If ε ∈ FIRST(α):
           for each terminal b in FOLLOW(A): set M[A, b] = α
           if $ ∈ FOLLOW(A): set M[A, $] = α

    Raises GrammarConflictError if a cell would be filled twice
    (indicating the grammar is not LL(1)).
    """
    table: dict[tuple[str, str], list[str]] = {}
    conflicts: list[str] = []

    for nt, prods in grammar.productions.items():
        for rhs in prods:
            f = first_of_string(rhs, first)

            for terminal in f - {EPSILON}:
                key = (nt, terminal)
                if key in table:
                    conflicts.append(f"Conflict at M[{nt}, {terminal!r}]")
                table[key] = rhs

            if EPSILON in f:
                for terminal in follow.get(nt, set()):
                    key = (nt, terminal)
                    if key in table:
                        conflicts.append(f"Conflict at M[{nt}, {terminal!r}]")
                    table[key] = rhs

    if conflicts:
        raise GrammarConflictError(
            "Grammar is NOT LL(1):\n" + "\n".join(f"  {c}" for c in conflicts)
        )
    return table


# ---------------------------------------------------------------------------
# Table-Driven LL(1) Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def ll1_parse(tokens: list[str], table: dict, grammar: Grammar) -> list[str]:
    """
    Parse a token list using the LL(1) table.
    Returns a list of strings describing each parse step (the parse trace).

    Stack-based algorithm:
      Initialize: stack = [$, start]
      Loop:
        Let X = top of stack, a = current input token
        if X == a == $: success
        if X is terminal: if X == a, pop and advance; else error
        if X is non-terminal: look up M[X, a]; push RHS in reverse
    """
    trace: list[str] = []
    input_tokens = tokens + [EOF_SYM]
    stack = [EOF_SYM, grammar.start]
    ip = 0   # input pointer

    while stack:
        X = stack[-1]
        a = input_tokens[ip]

        if X == EOF_SYM and a == EOF_SYM:
            trace.append(f"  ACCEPT")
            break

        if X == a:   # terminal match
            trace.append(f"  match {X!r}")
            stack.pop()
            ip += 1
        elif grammar.is_nonterminal(X):
            key = (X, a)
            if key not in table:
                raise ParseError(
                    f"No entry in table for M[{X!r}, {a!r}]. Input: {input_tokens}"
                )
            rhs = table[key]
            stack.pop()
            display_rhs = ' '.join(rhs) if rhs != [EPSILON] else 'ε'
            trace.append(f"  {X} -> {display_rhs}")
            if rhs != [EPSILON]:
                for sym in reversed(rhs):
                    stack.append(sym)
        else:
            raise ParseError(f"Unexpected token {a!r}, expected {X!r}")

    return trace


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def print_sets(label: str, sets: dict[str, set[str]], symbols: list[str]) -> None:
    print(f"\n{label}:")
    for sym in symbols:
        s = sets.get(sym, set())
        items = ', '.join(sorted(s))
        print(f"  {sym:<8} = {{ {items} }}")


def print_parse_table(
    table: dict[tuple[str, str], list[str]],
    grammar: Grammar,
    terminals: list[str],
) -> None:
    nts = list(grammar.non_terminals)
    terminals_all = sorted(terminals) + [EOF_SYM]
    col_w = 18
    print("\nLL(1) Parse Table:")
    header = f"  {'':12}" + ''.join(f"  {t!r:^{col_w}}" for t in terminals_all)
    print(header)
    print("  " + "-" * (12 + len(terminals_all) * (col_w + 2)))
    for nt in sorted(nts):
        row = f"  {nt:<12}"
        for t in terminals_all:
            rhs = table.get((nt, t))
            if rhs:
                cell = nt + '->' + ' '.join(rhs)
            else:
                cell = ''
            row += f"  {cell:^{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# Demo Grammar and Tests
# ---------------------------------------------------------------------------

# Standard unambiguous expression grammar (left-recursion eliminated)
EXPR_GRAMMAR = Grammar(
    start='E',
    productions={
        'E':  [['T', "E'"]],
        "E'": [['+', 'T', "E'"], ['-', 'T', "E'"], [EPSILON]],
        'T':  [['F', "T'"]],
        "T'": [['*', 'F', "T'"], ['/', 'F', "T'"], [EPSILON]],
        'F':  [['(', 'E', ')'], ['num'], ['id']],
    }
)

# Test token sequences (already tokenized)
TEST_INPUTS = [
    ['num', '+', 'num', '*', 'num'],        # num + num * num
    ['id', '*', '(', 'num', '+', 'id', ')'],# id * (num + id)
    ['num'],                                 # just a number
    ['(', 'id', '+', 'num', ')', '*', 'id'],# (id + num) * id
]


def main():
    print("=" * 60)
    print("LL(1) Parser Demo")
    print("=" * 60)

    grammar = EXPR_GRAMMAR
    print(f"\n{grammar}")

    # Compute FIRST and FOLLOW sets
    first = compute_first(grammar)
    follow = compute_follow(grammar, first)

    nt_order = ['E', "E'", 'T', "T'", 'F']
    print_sets("FIRST sets", first, nt_order)
    print_sets("FOLLOW sets", follow, nt_order)

    # Build parse table
    table = build_ll1_table(grammar, first, follow)
    print_parse_table(table, grammar, sorted(grammar.terminals))

    # Parse test inputs
    print("\n--- Parse Traces ---")
    for tokens in TEST_INPUTS:
        expr_display = ' '.join(tokens)
        print(f"\nInput: {expr_display}")
        try:
            trace = ll1_parse(tokens, table, grammar)
            for step in trace:
                print(step)
        except ParseError as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
