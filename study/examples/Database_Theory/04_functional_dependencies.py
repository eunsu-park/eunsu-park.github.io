"""
Functional Dependencies and Armstrong's Axioms

Demonstrates key concepts in functional dependency theory:
- Functional dependencies (FDs): X → Y
- Armstrong's axioms: reflexivity, augmentation, transitivity
- Derived rules: union, decomposition, pseudotransitivity
- Attribute closure: F+(X) - all attributes determined by X
- Closure of FD set: F+ - all FDs implied by F
- Candidate key finding
- Minimal/Canonical cover of FDs

Theory:
- FD X → Y holds if whenever two tuples agree on X, they agree on Y
- Armstrong's axioms are sound (only derive valid FDs) and complete (can derive all valid FDs)
- Attribute closure is used to find keys and test FD membership
- Minimal cover removes redundant FDs and extraneous attributes
"""

import sqlite3
from typing import Set, List, Tuple, FrozenSet


class FunctionalDependency:
    """Represents a functional dependency X → Y."""

    def __init__(self, lhs: Set[str], rhs: Set[str]):
        """
        Args:
            lhs: Left-hand side (determinant) attributes
            rhs: Right-hand side (dependent) attributes
        """
        self.lhs = frozenset(lhs)
        self.rhs = frozenset(rhs)

    def __repr__(self):
        lhs_str = ''.join(sorted(self.lhs))
        rhs_str = ''.join(sorted(self.rhs))
        return f"{lhs_str} → {rhs_str}"

    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash((self.lhs, self.rhs))


def demonstrate_armstrong_axioms():
    """Demonstrate Armstrong's axioms."""
    print("=" * 60)
    print("ARMSTRONG'S AXIOMS")
    print("=" * 60)
    print()

    # Example: R(A, B, C, D) with FD: AB → C
    print("Given: R(A, B, C, D) and FD: AB → C")
    print()

    # Axiom 1: Reflexivity
    print("1. REFLEXIVITY: If Y ⊆ X, then X → Y")
    print("-" * 60)
    print("   AB → A (trivial)")
    print("   AB → B (trivial)")
    print("   AB → AB (trivial)")
    print("   (These hold by definition)\n")

    # Axiom 2: Augmentation
    print("2. AUGMENTATION: If X → Y, then XZ → YZ")
    print("-" * 60)
    print("   Given: AB → C")
    print("   Add D to both sides: ABD → CD")
    print("   (If AB determines C, then AB with D also determines C and D)\n")

    # Axiom 3: Transitivity
    print("3. TRANSITIVITY: If X → Y and Y → Z, then X → Z")
    print("-" * 60)
    print("   Example chain:")
    print("   If AB → C and C → D, then AB → D")
    print("   (Dependencies chain together)\n")

    print("=" * 60)
    print("DERIVED RULES")
    print("=" * 60)
    print()

    # Union rule
    print("4. UNION: If X → Y and X → Z, then X → YZ")
    print("-" * 60)
    print("   Derived from augmentation + transitivity")
    print("   If AB → C and AB → D, then AB → CD\n")

    # Decomposition rule
    print("5. DECOMPOSITION: If X → YZ, then X → Y and X → Z")
    print("-" * 60)
    print("   Derived from reflexivity + transitivity")
    print("   If AB → CD, then AB → C and AB → D\n")

    # Pseudotransitivity
    print("6. PSEUDOTRANSITIVITY: If X → Y and YW → Z, then XW → Z")
    print("-" * 60)
    print("   Derived from augmentation + transitivity")
    print("   If A → B and BC → D, then AC → D\n")


def compute_closure(attributes: Set[str], fds: Set[FunctionalDependency]) -> Set[str]:
    """
    Compute attribute closure: X+ under F.

    Algorithm:
    1. Start with result = X
    2. Repeat until no change:
       - For each FD Y → Z in F:
         - If Y ⊆ result, add Z to result
    3. Return result

    Args:
        attributes: Set of attributes to close
        fds: Set of functional dependencies

    Returns:
        Closure of attributes (all attributes determined by input)
    """
    closure = set(attributes)
    changed = True

    while changed:
        changed = False
        for fd in fds:
            # If LHS is subset of current closure, add RHS
            if fd.lhs.issubset(closure):
                before_size = len(closure)
                closure.update(fd.rhs)
                if len(closure) > before_size:
                    changed = True

    return closure


def demonstrate_attribute_closure():
    """Demonstrate attribute closure computation."""
    print("=" * 60)
    print("ATTRIBUTE CLOSURE")
    print("=" * 60)
    print()

    # Example from database theory textbook
    print("Given: R(A, B, C, D, E, F)")
    print("FDs: {A → BC, B → E, CD → F}")
    print()

    fds = {
        FunctionalDependency({'A'}, {'B', 'C'}),
        FunctionalDependency({'B'}, {'E'}),
        FunctionalDependency({'C', 'D'}, {'F'})
    }

    # Compute {A}+
    print("Computing {A}+:")
    print("-" * 60)
    closure = compute_closure({'A'}, fds)
    print(f"  Start with: {{A}}")
    print(f"  A → BC applies: {{A, B, C}}")
    print(f"  B → E applies: {{A, B, C, E}}")
    print(f"  CD → F doesn't apply (no D)")
    print(f"  Final: {A}+ = {{{', '.join(sorted(closure))}}}\n")

    # Compute {A, D}+
    print("Computing {AD}+:")
    print("-" * 60)
    closure = compute_closure({'A', 'D'}, fds)
    print(f"  Start with: {{A, D}}")
    print(f"  A → BC applies: {{A, B, C, D}}")
    print(f"  B → E applies: {{A, B, C, D, E}}")
    print(f"  CD → F applies: {{A, B, C, D, E, F}}")
    print(f"  Final: {{AD}}+ = {{{', '.join(sorted(closure))}}}")
    print(f"  ✓ AD is a superkey (determines all attributes)\n")

    # Compute {B}+
    print("Computing {B}+:")
    print("-" * 60)
    closure = compute_closure({'B'}, fds)
    print(f"  Start with: {{B}}")
    print(f"  B → E applies: {{B, E}}")
    print(f"  No other FDs apply")
    print(f"  Final: {{B}}+ = {{{', '.join(sorted(closure))}}}")
    print(f"  ✗ B is not a superkey\n")


def find_candidate_keys(all_attrs: Set[str], fds: Set[FunctionalDependency]) -> List[FrozenSet[str]]:
    """
    Find all candidate keys for a relation.

    Algorithm:
    1. Find attributes that never appear on RHS (must be in every key)
    2. Check if these alone form a key
    3. If not, systematically add other attributes
    4. Remove supersets (keep only minimal keys)

    Args:
        all_attrs: All attributes in the relation
        fds: Set of functional dependencies

    Returns:
        List of candidate keys (minimal superkeys)
    """
    # Attributes that never appear on RHS must be in every key
    rhs_attrs = set()
    for fd in fds:
        rhs_attrs.update(fd.rhs)
    must_include = all_attrs - rhs_attrs

    # Check if must_include alone is a key
    if compute_closure(must_include, fds) == all_attrs:
        return [frozenset(must_include)]

    # Try adding other attributes
    candidate_keys = []
    other_attrs = list(all_attrs - must_include)

    # Generate all possible combinations
    from itertools import combinations
    for size in range(1, len(other_attrs) + 1):
        for combo in combinations(other_attrs, size):
            candidate = must_include | set(combo)
            if compute_closure(candidate, fds) == all_attrs:
                # Check if it's minimal (no proper subset is also a key)
                is_minimal = True
                for attr in combo:
                    subset = candidate - {attr}
                    if compute_closure(subset, fds) == all_attrs:
                        is_minimal = False
                        break
                if is_minimal:
                    candidate_keys.append(frozenset(candidate))

    return candidate_keys if candidate_keys else [frozenset(must_include)]


def demonstrate_candidate_keys():
    """Demonstrate finding candidate keys."""
    print("=" * 60)
    print("FINDING CANDIDATE KEYS")
    print("=" * 60)
    print()

    print("Example 1: R(A, B, C, D)")
    print("FDs: {A → B, B → C, C → D}")
    print("-" * 60)

    fds1 = {
        FunctionalDependency({'A'}, {'B'}),
        FunctionalDependency({'B'}, {'C'}),
        FunctionalDependency({'C'}, {'D'})
    }

    keys1 = find_candidate_keys({'A', 'B', 'C', 'D'}, fds1)
    print(f"Candidate keys: {['{' + ''.join(sorted(k)) + '}' for k in keys1]}")
    print(f"Explanation: A doesn't appear on RHS, so it must be in every key")
    print(f"             A+ = {{A,B,C,D}}, so {{A}} is the only candidate key\n")

    print("Example 2: R(A, B, C, D)")
    print("FDs: {AB → C, C → D, D → A}")
    print("-" * 60)

    fds2 = {
        FunctionalDependency({'A', 'B'}, {'C'}),
        FunctionalDependency({'C'}, {'D'}),
        FunctionalDependency({'D'}, {'A'})
    }

    keys2 = find_candidate_keys({'A', 'B', 'C', 'D'}, fds2)
    print(f"Candidate keys: {['{' + ''.join(sorted(k)) + '}' for k in keys2]}")
    print(f"Explanation: B doesn't appear on RHS, so it must be in every key")
    print(f"             {AB}+ = {{A,B,C,D}}, so {{AB}} is a candidate key")
    print(f"             {BC}+ = {{A,B,C,D}}, so {{BC}} is a candidate key")
    print(f"             {BD}+ = {{A,B,C,D}}, so {{BD}} is a candidate key\n")


def compute_minimal_cover(fds: Set[FunctionalDependency]) -> Set[FunctionalDependency]:
    """
    Compute minimal (canonical) cover of FDs.

    Algorithm:
    1. Decompose RHS to single attributes (F → {A→B, A→C} instead of A→BC)
    2. Remove extraneous attributes from LHS
    3. Remove redundant FDs

    Args:
        fds: Set of functional dependencies

    Returns:
        Minimal cover of FDs
    """
    # Step 1: Decompose RHS to single attributes
    decomposed = set()
    for fd in fds:
        for attr in fd.rhs:
            decomposed.add(FunctionalDependency(fd.lhs, {attr}))

    # Step 2: Remove extraneous attributes from LHS
    result = set(decomposed)
    changed = True
    while changed:
        changed = False
        for fd in list(result):
            if len(fd.lhs) > 1:
                # Try removing each attribute from LHS
                for attr in fd.lhs:
                    reduced_lhs = fd.lhs - {attr}
                    # Check if fd is still implied
                    other_fds = result - {fd}
                    closure = compute_closure(reduced_lhs, other_fds)
                    if fd.rhs.issubset(closure):
                        # attr is extraneous
                        result.remove(fd)
                        result.add(FunctionalDependency(reduced_lhs, fd.rhs))
                        changed = True
                        break
            if changed:
                break

    # Step 3: Remove redundant FDs
    final = set()
    for fd in result:
        # Check if fd is implied by others
        other_fds = result - {fd}
        closure = compute_closure(fd.lhs, other_fds)
        if not fd.rhs.issubset(closure):
            # fd is not redundant
            final.add(fd)

    return final


def demonstrate_minimal_cover():
    """Demonstrate computing minimal cover."""
    print("=" * 60)
    print("MINIMAL (CANONICAL) COVER")
    print("=" * 60)
    print()

    print("Given FDs:")
    print("-" * 60)
    fds = {
        FunctionalDependency({'A', 'B'}, {'C'}),
        FunctionalDependency({'A'}, {'B', 'C'}),
        FunctionalDependency({'B'}, {'C'}),
        FunctionalDependency({'A'}, {'B'})
    }
    for fd in sorted(fds, key=str):
        print(f"  {fd}")

    print("\nComputing minimal cover:")
    print("-" * 60)

    # Step 1: Decompose
    print("Step 1: Decompose RHS to single attributes")
    decomposed = set()
    for fd in fds:
        for attr in fd.rhs:
            decomposed.add(FunctionalDependency(fd.lhs, {attr}))
    for fd in sorted(decomposed, key=str):
        print(f"  {fd}")

    # Compute minimal cover
    minimal = compute_minimal_cover(fds)

    print("\nStep 2-3: Remove extraneous attributes and redundant FDs")
    print("Final minimal cover:")
    for fd in sorted(minimal, key=str):
        print(f"  {fd}")

    print("\nExplanation:")
    print("  - A → BC decomposed to A → B and A → C")
    print("  - AB → C is redundant (A → C already exists)")
    print("  - B → C is redundant (implied by A → B and A → C)")
    print("  - A → C is redundant (implied by A → B and B → C)")
    print("  - Result: {A → B, B → C}\n")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         FUNCTIONAL DEPENDENCIES THEORY                       ║
║  Armstrong's Axioms, Closure, Keys, Minimal Cover            ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_armstrong_axioms()
    demonstrate_attribute_closure()
    demonstrate_candidate_keys()
    demonstrate_minimal_cover()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Armstrong's Axioms (sound and complete):")
    print("  - Reflexivity: Y ⊆ X ⟹ X → Y")
    print("  - Augmentation: X → Y ⟹ XZ → YZ")
    print("  - Transitivity: X → Y, Y → Z ⟹ X → Z")
    print()
    print("Key algorithms:")
    print("  - Attribute closure: Find X+ to test superkeys")
    print("  - Candidate keys: Minimal superkeys")
    print("  - Minimal cover: Remove redundancy in FD set")
    print("=" * 60)
