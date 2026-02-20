"""
02_thompson_nfa.py - Thompson's Construction: Regex to NFA

Demonstrates converting a regular expression to a Non-deterministic
Finite Automaton (NFA) using Thompson's construction algorithm.

Algorithm overview:
  1. Parse the regex into postfix notation (handles |, *, +, ?, concatenation)
  2. Use a stack of NFA fragments
  3. For each operator, pop fragment(s), apply construction, push result
  4. Final stack item is the complete NFA

Topics covered:
  - Infix-to-postfix conversion (Shunting Yard)
  - NFA state and transition representation
  - Thompson's construction rules for each operator
  - Epsilon-closure
  - NFA simulation (accepts/rejects input strings)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

EPSILON = ''   # Represents an epsilon (empty) transition


# ---------------------------------------------------------------------------
# NFA State and Fragment
# ---------------------------------------------------------------------------

class State:
    """A single NFA state. Uses an auto-incrementing ID for display."""
    _counter = 0

    def __init__(self):
        State._counter += 1
        self.id: int = State._counter
        # transitions: dict mapping symbol -> list[State]
        self.transitions: dict[str, list[State]] = {}
        self.is_accept: bool = False

    def add_transition(self, symbol: str, target: State) -> None:
        self.transitions.setdefault(symbol, []).append(target)

    def __repr__(self) -> str:
        return f"S{self.id}"


@dataclass
class NFA:
    """
    An NFA fragment used during Thompson's construction.
    Each fragment has exactly one start state and one accept state.
    After construction, the final NFA's accept state's is_accept is set to True.
    """
    start: State
    accept: State


# ---------------------------------------------------------------------------
# Thompson's Construction
# ---------------------------------------------------------------------------

def make_literal(ch: str) -> NFA:
    """NFA that accepts exactly one character 'ch'."""
    s = State()
    a = State()
    s.add_transition(ch, a)
    return NFA(s, a)


def make_epsilon() -> NFA:
    """NFA that accepts the empty string."""
    s = State()
    a = State()
    s.add_transition(EPSILON, a)
    return NFA(s, a)


def make_concat(nfa1: NFA, nfa2: NFA) -> NFA:
    """
    NFA for (nfa1)(nfa2): connect nfa1's accept to nfa2's start via epsilon.
    """
    nfa1.accept.add_transition(EPSILON, nfa2.start)
    return NFA(nfa1.start, nfa2.accept)


def make_union(nfa1: NFA, nfa2: NFA) -> NFA:
    """
    NFA for (nfa1)|(nfa2):
      new_start --eps--> nfa1.start
      new_start --eps--> nfa2.start
      nfa1.accept --eps--> new_accept
      nfa2.accept --eps--> new_accept
    """
    new_start = State()
    new_accept = State()
    new_start.add_transition(EPSILON, nfa1.start)
    new_start.add_transition(EPSILON, nfa2.start)
    nfa1.accept.add_transition(EPSILON, new_accept)
    nfa2.accept.add_transition(EPSILON, new_accept)
    return NFA(new_start, new_accept)


def make_star(nfa: NFA) -> NFA:
    """
    NFA for (nfa)*:
      new_start --eps--> nfa.start
      new_start --eps--> new_accept  (zero repetitions)
      nfa.accept --eps--> nfa.start  (repeat)
      nfa.accept --eps--> new_accept (exit loop)
    """
    new_start = State()
    new_accept = State()
    new_start.add_transition(EPSILON, nfa.start)
    new_start.add_transition(EPSILON, new_accept)
    nfa.accept.add_transition(EPSILON, nfa.start)
    nfa.accept.add_transition(EPSILON, new_accept)
    return NFA(new_start, new_accept)


def make_plus(nfa: NFA) -> NFA:
    """NFA for (nfa)+: one or more. Equivalent to nfa·nfa*."""
    star = make_star(NFA(nfa.start, nfa.accept))
    # We need a fresh copy concept; instead use: nfa · (nfa)*
    # For simplicity in Thompson's, plus(nfa) = concat(nfa, star(nfa_copy))
    # We reuse existing states: after nfa.accept, loop back
    # Simpler: make_concat(nfa, make_star(nfa_copy))
    # Since we can't easily copy states, implement directly:
    new_start = State()
    new_accept = State()
    new_start.add_transition(EPSILON, nfa.start)
    nfa.accept.add_transition(EPSILON, nfa.start)   # repeat
    nfa.accept.add_transition(EPSILON, new_accept)   # exit
    return NFA(new_start, new_accept)


def make_question(nfa: NFA) -> NFA:
    """NFA for (nfa)?: zero or one. Adds epsilon bypass."""
    new_start = State()
    new_accept = State()
    new_start.add_transition(EPSILON, nfa.start)
    new_start.add_transition(EPSILON, new_accept)   # skip
    nfa.accept.add_transition(EPSILON, new_accept)
    return NFA(new_start, new_accept)


# ---------------------------------------------------------------------------
# Regex to NFA: Shunting Yard + Thompson
# ---------------------------------------------------------------------------

# Operator precedence (higher = tighter binding)
PRECEDENCE = {'|': 1, '.': 2, '*': 3, '+': 3, '?': 3}
BINARY_OPS = {'|', '.'}
UNARY_OPS = {'*', '+', '?'}


def add_concat_operator(regex: str) -> str:
    """
    Insert explicit concatenation operator '.' between adjacent tokens.
    E.g., "ab" -> "a.b",  "a*b" -> "a*.b",  "(a)(b)" -> "(a).(b)"
    """
    result = []
    for i, ch in enumerate(regex):
        result.append(ch)
        if i + 1 < len(regex):
            left = ch
            right = regex[i + 1]
            # Insert '.' if left is not an open paren or binary operator,
            # and right is not a close paren, unary op, or binary op.
            if (left not in ('(', '|', '.') and
                    right not in (')', '|', '.', '*', '+', '?')):
                result.append('.')
    return ''.join(result)


def to_postfix(regex: str) -> str:
    """Convert infix regex (with explicit '.') to postfix using Shunting Yard."""
    output = []
    stack = []
    for ch in regex:
        if ch == '(':
            stack.append(ch)
        elif ch == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack:
                stack.pop()   # pop '('
        elif ch in PRECEDENCE:
            while (stack and stack[-1] != '(' and
                   stack[-1] in PRECEDENCE and
                   PRECEDENCE[stack[-1]] >= PRECEDENCE[ch] and
                   ch not in UNARY_OPS):   # right-associative unary
                output.append(stack.pop())
            stack.append(ch)
        else:
            output.append(ch)   # literal character
    while stack:
        output.append(stack.pop())
    return ''.join(output)


def regex_to_nfa(regex: str) -> NFA:
    """Full pipeline: regex string -> NFA."""
    State._counter = 0   # reset state IDs for clean output
    explicit = add_concat_operator(regex)
    postfix = to_postfix(explicit)

    stack: list[NFA] = []

    for ch in postfix:
        if ch == '.':
            b = stack.pop()
            a = stack.pop()
            stack.append(make_concat(a, b))
        elif ch == '|':
            b = stack.pop()
            a = stack.pop()
            stack.append(make_union(a, b))
        elif ch == '*':
            a = stack.pop()
            stack.append(make_star(a))
        elif ch == '+':
            a = stack.pop()
            stack.append(make_plus(a))
        elif ch == '?':
            a = stack.pop()
            stack.append(make_question(a))
        else:
            stack.append(make_literal(ch))

    nfa = stack[0]
    nfa.accept.is_accept = True
    return nfa


# ---------------------------------------------------------------------------
# NFA Simulation
# ---------------------------------------------------------------------------

def epsilon_closure(states: set[State]) -> set[State]:
    """Compute the set of states reachable via epsilon transitions."""
    closure = set(states)
    worklist = list(states)
    while worklist:
        s = worklist.pop()
        for target in s.transitions.get(EPSILON, []):
            if target not in closure:
                closure.add(target)
                worklist.append(target)
    return closure


def nfa_move(states: set[State], symbol: str) -> set[State]:
    """Compute states reachable from 'states' via 'symbol' (not epsilon)."""
    result: set[State] = set()
    for s in states:
        result.update(s.transitions.get(symbol, []))
    return result


def nfa_accepts(nfa: NFA, text: str) -> bool:
    """Simulate the NFA on 'text'. Returns True if any accepting state is reached."""
    current = epsilon_closure({nfa.start})
    for ch in text:
        current = epsilon_closure(nfa_move(current, ch))
        if not current:
            return False
    return any(s.is_accept for s in current)


def collect_states(nfa: NFA) -> list[State]:
    """BFS to collect all reachable states from nfa.start."""
    visited: set[int] = set()
    states: list[State] = []
    queue = [nfa.start]
    while queue:
        s = queue.pop(0)
        if s.id in visited:
            continue
        visited.add(s.id)
        states.append(s)
        for targets in s.transitions.values():
            queue.extend(targets)
    return states


def print_nfa(nfa: NFA, regex: str) -> None:
    """Print a human-readable representation of the NFA."""
    states = collect_states(nfa)
    print(f"  Regex  : {regex}")
    print(f"  Start  : S{nfa.start.id}")
    print(f"  Accept : S{nfa.accept.id}")
    print(f"  States : {len(states)}")
    print(f"  Transitions:")
    for s in sorted(states, key=lambda x: x.id):
        for sym, targets in sorted(s.transitions.items()):
            sym_display = 'ε' if sym == EPSILON else repr(sym)
            for t in targets:
                marker = ' (accept)' if t.is_accept else ''
                print(f"    S{s.id} --{sym_display}--> S{t.id}{marker}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Thompson's NFA Construction")
    print("=" * 60)

    test_cases = [
        # (regex, strings_to_test)
        ("a",           ["a", "b", ""]),
        ("ab",          ["ab", "a", "abc"]),
        ("a|b",         ["a", "b", "c", ""]),
        ("a*",          ["", "a", "aa", "aaa", "b"]),
        ("a+",          ["", "a", "aa", "b"]),
        ("a?b",         ["b", "ab", "aab"]),
        ("(a|b)*",      ["", "a", "b", "ab", "ba", "aabb", "c"]),
        ("(a|b)*abb",   ["abb", "aabb", "babb", "ab", "abba"]),
        ("a(b|c)*d",    ["ad", "abd", "acd", "abcd", "abbd", "x"]),
    ]

    for regex, tests in test_cases:
        print(f"\n{'─'*56}")
        nfa = regex_to_nfa(regex)
        print_nfa(nfa, regex)
        print(f"  Simulation:")
        for text in tests:
            result = nfa_accepts(nfa, text)
            symbol = "ACCEPT" if result else "reject"
            display = repr(text) if text else "''"
            print(f"    {display:>12}  ->  {symbol}")


if __name__ == "__main__":
    main()
