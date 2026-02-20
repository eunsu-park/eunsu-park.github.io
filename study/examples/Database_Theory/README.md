# Database Theory Examples

This directory contains 10 Python examples demonstrating key database theory concepts using SQLite3 (built-in, no external dependencies).

## Files Overview

### 1. `01_relational_model.py` - Relational Model Fundamentals
**Concepts:**
- Relations (tables), tuples (rows), attributes (columns), domains
- Keys: superkey, candidate key, primary key, foreign key
- Integrity constraints: entity integrity, referential integrity
- NULL handling and 3-valued logic (TRUE, FALSE, UNKNOWN)

**Run:** `python 01_relational_model.py`

---

### 2. `02_relational_algebra.py` - Relational Algebra Operations
**Concepts:**
- Selection (σ), Projection (π), Cartesian Product (×)
- Join operations (⋈): natural join, theta join, outer join
- Set operations: Union (∪), Intersection (∩), Difference (−)
- Division (÷), Rename (ρ)

**Run:** `python 02_relational_algebra.py`

---

### 3. `03_er_to_relational.py` - ER-to-Relational Mapping
**Concepts:**
- 7-step ER-to-relational mapping algorithm
- Regular entities → tables
- Weak entities → tables with composite keys
- 1:1, 1:N, M:N relationships
- Multivalued attributes → separate tables

**Run:** `python 03_er_to_relational.py`

---

### 4. `04_functional_dependencies.py` - Functional Dependencies
**Concepts:**
- Armstrong's axioms: reflexivity, augmentation, transitivity
- Derived rules: union, decomposition, pseudotransitivity
- Attribute closure (X+)
- Finding candidate keys
- Minimal/canonical cover of FDs

**Run:** `python 04_functional_dependencies.py`

---

### 5. `05_normalization.py` - Normalization and Normal Forms
**Concepts:**
- First Normal Form (1NF): atomic values, no repeating groups
- Second Normal Form (2NF): no partial dependencies
- Third Normal Form (3NF): no transitive dependencies
- Boyce-Codd Normal Form (BCNF): every determinant is a superkey
- Lossless-join decomposition
- Dependency-preserving decomposition

**Run:** `python 05_normalization.py`

---

### 6. `06_indexing_btree.py` - B+Tree Indexing
**Concepts:**
- B+Tree data structure: insert, search, range queries
- Performance comparison: with vs without index
- Composite (multi-column) indexes
- Leftmost prefix rule
- When to use indexes

**Run:** `python 06_indexing_btree.py`

---

### 7. `07_transactions_acid.py` - Transactions and ACID
**Concepts:**
- **Atomicity:** All-or-nothing execution (COMMIT/ROLLBACK)
- **Consistency:** Maintaining database invariants (constraints, triggers)
- **Isolation:** Isolation levels (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE)
- **Durability:** Persistence via Write-Ahead Logging (WAL)

**Run:** `python 07_transactions_acid.py`

---

### 8. `08_concurrency_mvcc.py` - Concurrency Control
**Concepts:**
- Multi-Version Concurrency Control (MVCC)
- Two-Phase Locking (2PL): growing phase, shrinking phase
- Lock types: shared (S), exclusive (X)
- Deadlock detection and prevention
- Read/write conflicts

**Run:** `python 08_concurrency_mvcc.py`

---

### 9. `09_nosql_document_store.py` - NoSQL Document Store
**Concepts:**
- Schema-less storage with JSON documents
- CRUD operations on JSON data
- Nested queries and array operations
- Indexing JSON fields
- Relational vs document approach trade-offs

**Run:** `python 09_nosql_document_store.py`

---

### 10. `10_query_optimizer.py` - Query Optimization
**Concepts:**
- Query execution plans (EXPLAIN QUERY PLAN)
- Join ordering optimization
- Index selection strategies
- Query equivalence transformations
- Statistics and cost estimation
- Manual optimization techniques

**Run:** `python 10_query_optimizer.py`

---

## Requirements

- Python 3.10 or higher
- SQLite3 (built-in with Python, no installation needed)

## Usage

Each file is self-contained and can be run independently:

```bash
# Run a specific example
python 01_relational_model.py

# Or run all examples sequentially
for f in *.py; do echo "=== $f ==="; python "$f"; echo; done
```

## Learning Path

Recommended order for learning:

1. **Fundamentals:** 01 → 02 → 03
2. **Design Theory:** 04 → 05
3. **Performance:** 06 → 10
4. **Transactions:** 07 → 08
5. **Alternative Models:** 09

## Key Takeaways

- **Relational model** provides strong theoretical foundation (algebra, normal forms)
- **Normalization** eliminates redundancy but requires more joins
- **Indexes** dramatically improve read performance (B+Tree)
- **ACID properties** ensure data integrity in multi-user environments
- **Concurrency control** (MVCC, 2PL) enables safe concurrent access
- **NoSQL** trades consistency/integrity for flexibility and performance
- **Query optimization** is critical for large-scale databases

## Additional Resources

- *Database System Concepts* by Silberschatz, Korth, Sudarshan
- *Fundamentals of Database Systems* by Elmasri, Navathe
- SQLite documentation: https://www.sqlite.org/docs.html
- PostgreSQL documentation (for production systems): https://www.postgresql.org/docs/
