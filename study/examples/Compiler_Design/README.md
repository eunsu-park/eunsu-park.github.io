# Compiler Design Examples

This directory contains 10 Python examples demonstrating key compiler design concepts, from lexical analysis through bytecode generation and execution. All examples use only Python's standard library (no external dependencies).

## Files Overview

### 1. `01_lexer.py` - Lexical Analysis (Scanner)
**Concepts:**
- Token types and Token representation (dataclass)
- Regular expressions for token pattern matching
- Keyword vs identifier recognition
- String and character literal scanning
- Line/column tracking for error reporting
- Whitespace and comment handling

**Run:** `python 01_lexer.py`

---

### 2. `02_thompson_nfa.py` - Thompson's Construction (Regex to NFA)
**Concepts:**
- Converting regular expressions to Non-deterministic Finite Automata (NFA)
- Infix-to-postfix conversion using Shunting Yard algorithm
- Thompson's construction rules for operators: |, *, +, ?, concatenation
- NFA state and transition representation
- Epsilon-closure computation
- NFA simulation and string acceptance

**Run:** `python 02_thompson_nfa.py`

---

### 3. `03_subset_construction.py` - NFA to DFA Conversion
**Concepts:**
- Subset construction (powerset construction) algorithm
- Converting NFA to equivalent Deterministic Finite Automaton (DFA)
- Epsilon-closure and move operations
- DFA state minimization concepts
- Transition table representation
- Comparing NFA vs DFA simulation performance

**Run:** `python 03_subset_construction.py`

---

### 4. `04_recursive_descent_parser.py` - Recursive Descent Parser
**Concepts:**
- Hand-written recursive descent parsing
- Abstract Syntax Tree (AST) construction
- Grammar representation (left-recursion removed)
- Operator precedence and associativity
- Control flow statements: if/else, while loops
- Expression evaluation with operator precedence
- Error handling and recovery

**Run:** `python 04_recursive_descent_parser.py`

---

### 5. `05_ll1_parser.py` - LL(1) Table-Driven Parser
**Concepts:**
- Context-free grammar (CFG) representation
- FIRST set computation
- FOLLOW set computation
- LL(1) parsing table construction
- Table-driven parsing with explicit stack
- Conflict detection in parsing tables
- Parsing ambiguous vs unambiguous grammars

**Run:** `python 05_ll1_parser.py`

---

### 6. `06_ast_visitor.py` - AST and Visitor Pattern
**Concepts:**
- Abstract Syntax Tree node classes (dataclass-based)
- Visitor design pattern for tree traversal
- Multiple operations on same AST: evaluation, type checking, pretty-printing
- Type inference for literals and operators
- Expression evaluation with different data types
- Separation of structure from operations

**Run:** `python 06_ast_visitor.py`

---

### 7. `07_type_checker.py` - Semantic Analysis and Type Checking
**Concepts:**
- Symbol table management with nested scopes
- Variable declaration and scope resolution
- Function definitions with typed parameters and return types
- Type compatibility checking for assignments
- Function call validation (arity and argument types)
- Return type matching
- Error detection and reporting

**Run:** `python 07_type_checker.py`

---

### 8. `08_three_address_code.py` - Intermediate Code Generation
**Concepts:**
- Three-Address Code (TAC) representation
- Control Flow Graph (CFG) construction
- TAC instruction types: ASSIGN, BINOP, UNARY, LABEL, JUMP, CJUMP
- Converting AST to intermediate representation
- Basic block identification
- Lowering high-level constructs to TAC

**Run:** `python 08_three_address_code.py`

---

### 9. `09_optimizer.py` - Local Optimizations
**Concepts:**
- Constant folding at compile time
- Constant propagation through basic blocks
- Algebraic simplification (identities like x*1=x, x+0=x)
- Dead code elimination
- Copy propagation
- Redundant computation elimination
- Data flow analysis and reaching definitions

**Run:** `python 09_optimizer.py`

---

### 10. `10_bytecode_vm.py` - Bytecode Compiler and Virtual Machine
**Concepts:**
- Bytecode instruction set design
- Bytecode compilation from AST
- Stack-based virtual machine architecture
- Operand stack and call stack (frames)
- Local variable storage and function calls
- Constant pool for literals
- Instruction execution and stack machine simulation
- Function parameters and return values

**Run:** `python 10_bytecode_vm.py`

---

## Requirements

- Python 3.8 or higher
- Standard library only (no external dependencies)

## Usage

Each file is self-contained and can be run independently:

```bash
# Run a specific example
python 01_lexer.py

# Or run all examples sequentially
for f in *.py; do echo "=== $f ==="; python "$f"; echo; done
```

## Learning Path

Recommended order for learning the compilation process:

1. **Frontend (Analysis):**
   - 01_lexer.py (lexical analysis)
   - 02_thompson_nfa.py → 03_subset_construction.py (automata theory)
   - 04_recursive_descent_parser.py → 05_ll1_parser.py (parsing)
   - 06_ast_visitor.py (AST representation)
   - 07_type_checker.py (semantic analysis)

2. **Middle-end (Intermediate Code):**
   - 08_three_address_code.py (TAC generation)
   - 09_optimizer.py (code optimization)

3. **Backend (Code Generation):**
   - 10_bytecode_vm.py (bytecode and execution)

## Key Takeaways

- **Lexical Analysis** tokenizes source code and handles the lexicon of a language
- **Parsing** builds a hierarchical tree structure (AST) from a flat token stream
- **Two approaches:** Recursive descent (top-down) and table-driven (bottom-up via LR/LL)
- **Automata theory** (NFA/DFA) provides theoretical foundation for lexer design
- **Semantic analysis** ensures type safety and correct variable scoping
- **Intermediate representation** (TAC) is language and machine-independent
- **Optimization** improves performance by eliminating redundant/unnecessary code
- **Code generation** translates optimized IR into executable bytecode or native code
- **Virtual machines** provide a portable execution platform independent of host CPU

## Additional Resources

- *Compilers: Principles, Techniques, and Tools* (Dragon Book) by Aho, Lam, Sethi, Ullman
- *Crafting Interpreters* by Robert Nystrom (free online: https://craftinginterpreters.com/)
- *Engineering a Compiler* by Cooper and Torczon
- *Essentials of Compilation* by Siek (free lectures: https://www.youtube.com/playlist?list=PLOPRGayY_TV-cxQSuqklRqYfWgAbQv25x)
