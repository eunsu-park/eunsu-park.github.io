"""
03_subset_construction.py - NFA to DFA via Subset Construction

Demonstrates the subset construction (powerset construction) algorithm,
which converts a Non-deterministic Finite Automaton (NFA) into an
equivalent Deterministic Finite Automaton (DFA).

Key idea:
  Each DFA state corresponds to a *set* of NFA states.
  The DFA's transition on symbol 'a' from state {s1,s2,...} is:
    epsilon_closure( move({s1,s2,...}, 'a') )

Topics covered:
  - Subset construction algorithm
  - DFA minimization concept (identifying accept states)
  - Printing transition tables
  - Comparing NFA vs DFA simulation results
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# Reuse NFA machinery from 02_thompson_nfa
# (copied here to keep this file standalone)

EPSILON = ''


class State:
    _counter = 0

    def __init__(self):
        State._counter += 1
        self.id = State._counter
        self.transitions: dict[str, list[State]] = {}
        self.is_accept: bool = False

    def add_transition(self, symbol: str, target: State) -> None:
        self.transitions.setdefault(symbol, []).append(target)

    def __repr__(self):
        return f"S{self.id}"


@dataclass
class NFA:
    start: State
    accept: State


# --- Thompson's construction (abbreviated) -----------------------------------

def _lit(ch):
    s, a = State(), State()
    s.add_transition(ch, a)
    return NFA(s, a)

def _concat(n1, n2):
    n1.accept.add_transition(EPSILON, n2.start)
    return NFA(n1.start, n2.accept)

def _union(n1, n2):
    s, a = State(), State()
    s.add_transition(EPSILON, n1.start)
    s.add_transition(EPSILON, n2.start)
    n1.accept.add_transition(EPSILON, a)
    n2.accept.add_transition(EPSILON, a)
    return NFA(s, a)

def _star(n):
    s, a = State(), State()
    s.add_transition(EPSILON, n.start)
    s.add_transition(EPSILON, a)
    n.accept.add_transition(EPSILON, n.start)
    n.accept.add_transition(EPSILON, a)
    return NFA(s, a)

PREC = {'|': 1, '.': 2, '*': 3, '+': 3, '?': 3}
BINOPS = {'|', '.'}
UNOPS  = {'*', '+', '?'}

def _add_concat(r):
    res = []
    for i, c in enumerate(r):
        res.append(c)
        if i + 1 < len(r):
            l, ri = c, r[i+1]
            if l not in ('(', '|', '.') and ri not in (')', '|', '.', '*', '+', '?'):
                res.append('.')
    return ''.join(res)

def _postfix(r):
    out, stk = [], []
    for c in r:
        if c == '(':
            stk.append(c)
        elif c == ')':
            while stk and stk[-1] != '(':
                out.append(stk.pop())
            if stk: stk.pop()
        elif c in PREC:
            while (stk and stk[-1] != '(' and stk[-1] in PREC and
                   PREC[stk[-1]] >= PREC[c] and c not in UNOPS):
                out.append(stk.pop())
            stk.append(c)
        else:
            out.append(c)
    while stk: out.append(stk.pop())
    return ''.join(out)

def regex_to_nfa(regex: str) -> NFA:
    State._counter = 0
    postfix = _postfix(_add_concat(regex))
    stk: list[NFA] = []
    for c in postfix:
        if c == '.':
            b, a = stk.pop(), stk.pop(); stk.append(_concat(a, b))
        elif c == '|':
            b, a = stk.pop(), stk.pop(); stk.append(_union(a, b))
        elif c == '*':
            stk.append(_star(stk.pop()))
        elif c == '+':
            n = stk.pop()
            # plus: concat(n, star(n)) -- reuse same states
            s2, a2 = State(), State()
            s2.add_transition(EPSILON, n.start)
            n.accept.add_transition(EPSILON, n.start)
            n.accept.add_transition(EPSILON, a2)
            stk.append(NFA(s2, a2))
        elif c == '?':
            n = stk.pop()
            s2, a2 = State(), State()
            s2.add_transition(EPSILON, n.start)
            s2.add_transition(EPSILON, a2)
            n.accept.add_transition(EPSILON, a2)
            stk.append(NFA(s2, a2))
        else:
            stk.append(_lit(c))
    nfa = stk[0]
    nfa.accept.is_accept = True
    return nfa


# ---------------------------------------------------------------------------
# NFA simulation helpers
# ---------------------------------------------------------------------------

def epsilon_closure(states: frozenset[State]) -> frozenset[State]:
    closure = set(states)
    worklist = list(states)
    while worklist:
        s = worklist.pop()
        for t in s.transitions.get(EPSILON, []):
            if t not in closure:
                closure.add(t)
                worklist.append(t)
    return frozenset(closure)


def nfa_move(states: frozenset[State], sym: str) -> frozenset[State]:
    result: set[State] = set()
    for s in states:
        result.update(s.transitions.get(sym, []))
    return frozenset(result)


def alphabet(nfa: NFA) -> set[str]:
    """Collect all non-epsilon symbols used in the NFA."""
    syms: set[str] = set()
    visited: set[int] = set()
    queue = [nfa.start]
    while queue:
        s = queue.pop()
        if s.id in visited:
            continue
        visited.add(s.id)
        for sym, targets in s.transitions.items():
            if sym != EPSILON:
                syms.add(sym)
            queue.extend(targets)
    return syms


# ---------------------------------------------------------------------------
# DFA representation
# ---------------------------------------------------------------------------

@dataclass
class DFAState:
    """A DFA state representing a frozenset of NFA states."""
    id: int
    nfa_states: frozenset[State]
    is_accept: bool
    transitions: dict[str, int] = field(default_factory=dict)  # symbol -> DFA state id

    def name(self) -> str:
        ids = sorted(s.id for s in self.nfa_states)
        return '{' + ','.join(f'S{i}' for i in ids) + '}'


@dataclass
class DFA:
    states: list[DFAState]
    start_id: int
    alphabet: set[str]

    def start(self) -> DFAState:
        return self.states[self.start_id]

    def accepts(self, text: str) -> bool:
        current = self.start_id
        for ch in text:
            state = self.states[current]
            if ch not in state.transitions:
                return False
            current = state.transitions[ch]
        return self.states[current].is_accept


# ---------------------------------------------------------------------------
# Subset Construction Algorithm
# ---------------------------------------------------------------------------

def subset_construction(nfa: NFA) -> DFA:
    """
    Convert an NFA to a DFA using the subset construction algorithm.

    Steps:
    1. Start with epsilon_closure({nfa.start}) as the initial DFA state.
    2. For each unmarked DFA state D and each symbol a in the alphabet:
       a. Compute T = epsilon_closure(move(D, a))
       b. If T is not already a DFA state, add it (unmarked)
       c. Record the transition D --a--> T
    3. A DFA state is accepting if any of its NFA states is an accept state.
    """
    syms = sorted(alphabet(nfa))
    start_set = epsilon_closure(frozenset({nfa.start}))

    # Map from frozenset[State] -> DFA state id
    state_map: dict[frozenset[State], int] = {}
    dfa_states: list[DFAState] = []

    def get_or_create(nfa_set: frozenset[State]) -> int:
        if nfa_set in state_map:
            return state_map[nfa_set]
        sid = len(dfa_states)
        is_acc = any(s.is_accept for s in nfa_set)
        ds = DFAState(id=sid, nfa_states=nfa_set, is_accept=is_acc)
        dfa_states.append(ds)
        state_map[nfa_set] = sid
        return sid

    start_id = get_or_create(start_set)
    worklist = [start_id]

    while worklist:
        did = worklist.pop()
        ds = dfa_states[did]
        for sym in syms:
            moved = nfa_move(ds.nfa_states, sym)
            if not moved:
                continue   # dead state (no transition)
            closed = epsilon_closure(moved)
            target_id = get_or_create(closed)
            ds.transitions[sym] = target_id
            if target_id == len(dfa_states) - 1:   # newly created
                worklist.append(target_id)

    return DFA(states=dfa_states, start_id=start_id, alphabet=set(syms))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_dfa_table(dfa: DFA, regex: str) -> None:
    """Print the DFA transition table."""
    syms = sorted(dfa.alphabet)
    print(f"\nDFA for regex: {regex!r}")
    print(f"  States: {len(dfa.states)}")
    print(f"  Alphabet: {{{', '.join(repr(s) for s in syms)}}}")
    print(f"  Start state: {dfa.states[dfa.start_id].name()}")
    accept_names = [dfa.states[i].name() for i, s in enumerate(dfa.states) if s.is_accept]
    print(f"  Accept states: {accept_names}")

    # Header
    col_w = max(20, max(len(dfa.states[i].name()) for i in range(len(dfa.states))) + 2)
    sym_w = 12
    header = f"  {'STATE':<{col_w}}"
    for sym in syms:
        header += f"  {repr(sym):^{sym_w}}"
    header += f"  {'ACCEPT':>6}"
    print()
    print(header)
    print("  " + "-" * (col_w + len(syms) * (sym_w + 2) + 10))

    for ds in dfa.states:
        marker = " <-- start" if ds.id == dfa.start_id else ""
        row = f"  {ds.name():<{col_w}}"
        for sym in syms:
            if sym in ds.transitions:
                target_name = dfa.states[ds.transitions[sym]].name()
                row += f"  {target_name:^{sym_w}}"
            else:
                row += f"  {'(dead)':^{sym_w}}"
        row += f"  {'YES' if ds.is_accept else 'no':>6}"
        row += marker
        print(row)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo(regex: str, test_strings: list[str]) -> None:
    nfa = regex_to_nfa(regex)
    dfa = subset_construction(nfa)
    print_dfa_table(dfa, regex)

    print(f"\n  Simulation results:")
    for text in test_strings:
        result = dfa.accepts(text)
        display = repr(text) if text else "''"
        print(f"    {display:>14}  ->  {'ACCEPT' if result else 'reject'}")


def main():
    print("=" * 60)
    print("Subset Construction: NFA -> DFA")
    print("=" * 60)

    demos = [
        # Simple examples to show the state explosion clearly
        ("a|b",         ["a", "b", "c", "ab"]),
        ("a*b",         ["b", "ab", "aab", "aaab", "a", ""]),
        ("(a|b)*abb",   ["abb", "aabb", "babb", "ab", "abba", ""]),
        ("ab?c",        ["ac", "abc", "abbc", "bc"]),
    ]

    for regex, tests in demos:
        print(f"\n{'='*60}")
        run_demo(regex, tests)

    # Show NFA state count vs DFA state count for (a|b)*abb
    print(f"\n{'='*60}")
    print("NFA vs DFA state count comparison:")
    print(f"{'REGEX':<20} {'NFA states':>12} {'DFA states':>12}")
    print("-" * 46)
    for regex, _ in demos:
        nfa = regex_to_nfa(regex)
        # count NFA states
        visited: set[int] = set()
        queue = [nfa.start]
        while queue:
            s = queue.pop()
            if s.id in visited: continue
            visited.add(s.id)
            for tgts in s.transitions.values():
                queue.extend(tgts)
        nfa_count = len(visited)
        dfa = subset_construction(nfa)
        print(f"  {regex!r:<18} {nfa_count:>12} {len(dfa.states):>12}")


if __name__ == "__main__":
    main()
