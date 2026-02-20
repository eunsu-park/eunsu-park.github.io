"""
BAD CODE EXAMPLE - Before Refactoring

This code DELIBERATELY violates clean code principles:
1. Long, complex functions (God functions)
2. Magic numbers everywhere
3. Poor naming (abbreviations, unclear names)
4. God class (does too much)
5. No separation of concerns
6. Tight coupling
7. No error handling
8. Deep nesting
9. Code duplication

DO NOT use this code as a reference!
This is the "before" example for refactoring.
"""


# ❌ God class - does everything!
class UsrMgr:
    """Terrible class name and does way too much"""

    def __init__(self):
        self.usrs = []  # Bad name
        self.cnt = 0  # What is cnt?
        self.x = 100  # What is x?

    # ❌ Long function with magic numbers and deep nesting
    def p(self, n, e, a, s):  # Terrible parameter names!
        """What does 'p' mean? Process? Print? ???"""

        # Magic number! What is 18?
        if a < 18:
            print("too young")
            return False

        # What is 150000?
        if s < 150000:
            # Deep nesting starts here
            if len(n) < 2:
                print("bad name")
                return False
            else:
                if "@" not in e:
                    print("bad email")
                    return False
                else:
                    # More nesting!
                    if len(self.usrs) > 0:
                        for u in self.usrs:
                            if u["e"] == e:  # Duplicate email check
                                print("dup email")
                                return False
                    # Finally add user (buried deep in nesting!)
                    self.usrs.append({"n": n, "e": e, "a": a, "s": s})
                    self.cnt += 1
                    return True
        else:
            # Code duplication! Same validation as above
            if len(n) < 2:
                print("bad name")
                return False
            else:
                if "@" not in e:
                    print("bad email")
                    return False
                else:
                    if len(self.usrs) > 0:
                        for u in self.usrs:
                            if u["e"] == e:
                                print("dup email")
                                return False
                    # Different logic based on salary - but duplicated validation!
                    self.usrs.append({"n": n, "e": e, "a": a, "s": s, "pr": True})
                    self.cnt += 1
                    # Magic number 1.2!
                    self.x = self.x * 1.2
                    return True

    # ❌ Does too much, magic numbers, no separation of concerns
    def calc(self):
        """Calculate what? Naming is unclear"""
        t = 0  # What is t?
        for u in self.usrs:
            # Magic numbers: 0.1, 0.15, 0.2
            if u["s"] < 50000:
                t += u["s"] * 0.1
            elif u["s"] < 100000:
                t += u["s"] * 0.15
            elif u["s"] < 150000:
                t += u["s"] * 0.2
            else:
                # More magic numbers!
                t += u["s"] * 0.25 + 5000

        # What are these conditions?
        if len(self.usrs) > 10 and self.cnt > 5:
            t *= 1.1

        # Magic number 0.3
        if t > 100000:
            t -= t * 0.3

        return t

    # ❌ Another long function with unclear purpose
    def g(self, e):
        """Get what? User? Data? Unclear name"""
        for u in self.usrs:
            if u["e"] == e:
                # Direct data access - no encapsulation
                return u
        return None

    # ❌ Function that modifies global state unpredictably
    def upd(self, e, **kwargs):
        """Update - but what can be updated? No validation!"""
        u = self.g(e)
        if u:
            # Direct modification without validation!
            for k, v in kwargs.items():
                u[k] = v
            return True
        return False

    # ❌ Complex logic with no error handling
    def del_usr(self, e):
        """Only good name in this class!"""
        for i, u in enumerate(self.usrs):
            if u["e"] == e:
                # What if user doesn't exist? No error handling!
                self.usrs.pop(i)
                self.cnt -= 1
                # Forgot to update self.x! Bug!
                return True
        return False


# ❌ Function with magic numbers and unclear purpose
def process_data(d):
    """What data? What processing? Unclear name and purpose"""
    r = []
    for item in d:
        # Magic number 5
        if item > 5:
            # Magic numbers 2 and 10
            r.append(item * 2 + 10)
        else:
            # Magic number 3
            r.append(item * 3)
    return r


# ❌ Global variables! Very bad practice
total = 0
count = 0
flag = False


def calculate_something(nums):
    """What is something? Uses global state!"""
    global total, count, flag

    for n in nums:
        # Magic number 100
        if n > 100:
            total += n
            count += 1
            flag = True


# ❌ Long parameter list, unclear purpose
def do_stuff(a, b, c, d, e, f, g, h):
    """Too many parameters! What does this do?"""
    # Magic numbers everywhere!
    return (a * 2 + b * 3 - c * 0.5 + d / 2 + e * 1.5 - f * 0.8 + g * 2.3 + h / 1.7)


# ❌ No error handling, assumes everything works
def read_and_process(filename):
    """No error handling for file operations!"""
    # What if file doesn't exist?
    f = open(filename, 'r')
    data = f.read()
    # What if file isn't properly formatted?
    lines = data.split('\n')
    result = []
    for line in lines:
        # What if line is empty or malformed?
        parts = line.split(',')
        # What if there aren't enough parts?
        result.append({
            'name': parts[0],
            'value': int(parts[1]),  # What if not a number?
            'category': parts[2]
        })
    # File never closed!
    return result


# ❌ Boolean parameter (flag argument) - code smell!
def get_users(include_inactive):
    """Boolean parameter makes function do two different things"""
    users = []
    # ... get users from somewhere ...
    if include_inactive:
        # One behavior
        return users
    else:
        # Different behavior
        return [u for u in users if u.get('active')]


# ❌ Comments explaining bad code instead of fixing it
def complex_calculation(x, y, z):
    """Bad code with comments trying to explain it"""
    # Add x and y
    temp1 = x + y
    # Multiply by z
    temp2 = temp1 * z
    # Divide by 2 (why? magic number!)
    temp3 = temp2 / 2
    # Subtract 10 (why? magic number!)
    temp4 = temp3 - 10
    # Multiply by 1.5 (why? magic number!)
    result = temp4 * 1.5
    return result


# ❌ Demonstration of bad code
if __name__ == "__main__":
    print("=" * 70)
    print("BAD CODE DEMONSTRATION")
    print("=" * 70)
    print("\nThis code has many problems:")
    print("  ✗ Magic numbers everywhere")
    print("  ✗ Poor naming (abbreviations, single letters)")
    print("  ✗ God class doing too much")
    print("  ✗ Deep nesting")
    print("  ✗ Code duplication")
    print("  ✗ No error handling")
    print("  ✗ Global state")
    print("  ✗ Long parameter lists")
    print("  ✗ Unclear purpose")
    print("  ✗ No separation of concerns")
    print("\nSee refactoring_after.py for the cleaned-up version!")

    # Using the bad code
    print("\n" + "-" * 70)
    print("Running bad code example:")
    print("-" * 70)

    mgr = UsrMgr()

    # These function calls are unclear without looking at implementation!
    mgr.p("Alice", "alice@example.com", 25, 60000)
    mgr.p("Bob", "bob@example.com", 30, 180000)
    mgr.p("Charlie", "charlie@example.com", 15, 0)  # Too young

    print(f"\nTotal users: {mgr.cnt}")
    print(f"Calculation result: {mgr.calc():.2f}")

    # What does 'g' return? Need to check code!
    user = mgr.g("alice@example.com")
    if user:
        print(f"Found user: {user}")

    # Global state makes this unpredictable
    calculate_something([50, 150, 200, 30])
    print(f"\nGlobal total: {total}, count: {count}, flag: {flag}")

    print("\n" + "=" * 70)
    print("PROBLEMS WITH THIS CODE:")
    print("=" * 70)
    print("""
1. READABILITY
   • Impossible to understand without deep inspection
   • Abbreviations hide meaning
   • Magic numbers have no context

2. MAINTAINABILITY
   • Hard to modify without breaking things
   • Code duplication means bugs multiply
   • Deep nesting is hard to follow

3. TESTABILITY
   • God class is hard to test
   • Global state makes tests unreliable
   • No error handling means tests can't verify error cases

4. EXTENSIBILITY
   • Tight coupling prevents easy extension
   • No interfaces or abstractions
   • Violates SOLID principles

5. RELIABILITY
   • No error handling
   • No input validation
   • Assumes everything works perfectly

NEXT: See refactoring_after.py for the solution!
""")
