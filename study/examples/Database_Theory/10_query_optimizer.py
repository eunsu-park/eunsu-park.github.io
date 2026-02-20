"""
Query Optimization and Execution Plans

Demonstrates query optimization concepts:
- Query execution plans (EXPLAIN QUERY PLAN)
- Cost estimation for different strategies
- Join ordering and optimization
- Query equivalence transformations
- Index selection
- Statistics and cardinality estimation

Theory:
- Query optimizer translates SQL to efficient execution plan
- Cost-based optimization: estimate cost of different plans, choose cheapest
- Relational algebra equivalences allow rewriting queries
- Join ordering affects performance (e.g., smaller result sets first)
- Selectivity and cardinality estimation guide optimizer decisions
- EXPLAIN shows the chosen execution plan

Optimization techniques:
- Predicate pushdown: apply filters early
- Join reordering: minimize intermediate results
- Index selection: choose appropriate indexes
- Common subexpression elimination
"""

import sqlite3
import time
import random
from typing import List, Tuple


def setup_sample_database() -> sqlite3.Connection:
    """Create sample database for optimization demonstrations."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT,
            country TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            total REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            price REAL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )
    ''')

    # Insert sample data
    cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney']
    countries = ['USA', 'UK', 'France', 'Japan', 'Australia']

    # Customers
    for i in range(1, 1001):
        city = random.choice(cities)
        country = random.choice(countries)
        cursor.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            (i, f'Customer_{i}', city, country)
        )

    # Orders
    order_id = 1
    for customer_id in range(1, 1001):
        num_orders = random.randint(0, 5)
        for _ in range(num_orders):
            cursor.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?)",
                (order_id, customer_id, '2024-01-01', random.uniform(10, 1000))
            )
            order_id += 1

    # Order items
    cursor.execute("SELECT order_id FROM orders")
    order_ids = [row[0] for row in cursor.fetchall()]
    item_id = 1
    for order_id in order_ids:
        num_items = random.randint(1, 5)
        for _ in range(num_items):
            cursor.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
                (item_id, order_id, random.randint(1, 100), random.randint(1, 10), random.uniform(5, 500))
            )
            item_id += 1

    conn.commit()
    return conn


def demonstrate_explain_query_plan():
    """Demonstrate EXPLAIN QUERY PLAN."""
    print("=" * 60)
    print("QUERY EXECUTION PLANS")
    print("=" * 60)
    print()

    conn = setup_sample_database()
    cursor = conn.cursor()

    # Simple query
    print("1. Simple SELECT with WHERE clause")
    print("-" * 60)
    query = "SELECT * FROM customers WHERE city = 'London'"
    print(f"Query: {query}")
    print("\nExecution plan:")

    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nInterpretation: SCAN TABLE (full table scan, no index)")

    # Query with JOIN
    print("\n\n2. JOIN query")
    print("-" * 60)
    query = """
        SELECT c.name, COUNT(o.order_id) as num_orders
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id
    """
    print(f"Query: Multi-line JOIN with GROUP BY")
    print("\nExecution plan:")

    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


def demonstrate_join_ordering():
    """Demonstrate impact of join ordering."""
    print("=" * 60)
    print("JOIN ORDERING OPTIMIZATION")
    print("=" * 60)
    print()

    conn = setup_sample_database()
    cursor = conn.cursor()

    # Count rows in each table
    cursor.execute("SELECT COUNT(*) FROM customers")
    num_customers = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM orders")
    num_orders = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM order_items")
    num_items = cursor.fetchone()[0]

    print("Table sizes:")
    print("-" * 60)
    print(f"  customers:    {num_customers:5} rows")
    print(f"  orders:       {num_orders:5} rows")
    print(f"  order_items:  {num_items:5} rows")

    # Three-way join
    print("\n\n1. Three-way JOIN (customers → orders → order_items)")
    print("-" * 60)
    query = """
        SELECT c.name, SUM(oi.price * oi.quantity) as total
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE c.country = 'USA'
        GROUP BY c.customer_id
    """

    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    print("Execution plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    start = time.time()
    cursor.execute(query)
    results = cursor.fetchall()
    elapsed = time.time() - start
    print(f"\nExecuted in {elapsed*1000:.2f} ms, {len(results)} results")

    # Same query, different table order in FROM clause
    print("\n\n2. Same query, reordered tables (optimizer handles this)")
    print("-" * 60)
    query2 = """
        SELECT c.name, SUM(oi.price * oi.quantity) as total
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE c.country = 'USA'
        GROUP BY c.customer_id
    """

    cursor.execute(f"EXPLAIN QUERY PLAN {query2}")
    print("Execution plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nNote: SQLite's optimizer reorders joins for efficiency")
    print("      (predicate pushdown: filters applied early)")

    conn.close()
    print()


def demonstrate_index_selection():
    """Demonstrate how indexes affect query plans."""
    print("=" * 60)
    print("INDEX SELECTION")
    print("=" * 60)
    print()

    conn = setup_sample_database()
    cursor = conn.cursor()

    # Query without index
    print("1. Query WITHOUT index on country")
    print("-" * 60)
    query = "SELECT * FROM customers WHERE country = 'USA'"

    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    print("Execution plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    start = time.time()
    cursor.execute(query)
    results = cursor.fetchall()
    elapsed_no_idx = time.time() - start
    print(f"\nExecuted in {elapsed_no_idx*1000:.2f} ms, {len(results)} results")

    # Create index
    print("\n\n2. Creating index on country")
    print("-" * 60)
    cursor.execute("CREATE INDEX idx_country ON customers(country)")
    print("✓ Index created")

    # Query with index
    print("\n\n3. Query WITH index on country")
    print("-" * 60)
    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    print("Execution plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    start = time.time()
    cursor.execute(query)
    results = cursor.fetchall()
    elapsed_with_idx = time.time() - start
    print(f"\nExecuted in {elapsed_with_idx*1000:.2f} ms, {len(results)} results")

    # Covering index
    print("\n\n4. Covering index (includes all needed columns)")
    print("-" * 60)
    cursor.execute("CREATE INDEX idx_country_city ON customers(country, city)")
    print("✓ Created composite index on (country, city)")

    query_covering = "SELECT country, city FROM customers WHERE country = 'USA'"
    cursor.execute(f"EXPLAIN QUERY PLAN {query_covering}")
    print("\nExecution plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nNote: If plan shows 'USING COVERING INDEX', all data comes from index")
    print("      (no need to access table rows - faster)")

    conn.close()
    print()


def demonstrate_query_transformations():
    """Demonstrate query equivalence transformations."""
    print("=" * 60)
    print("QUERY EQUIVALENCE TRANSFORMATIONS")
    print("=" * 60)
    print()

    conn = setup_sample_database()
    cursor = conn.cursor()

    # Create index for demonstration
    cursor.execute("CREATE INDEX idx_customer ON orders(customer_id)")

    print("Transformation 1: Subquery → JOIN")
    print("-" * 60)

    # Original: subquery
    query1 = """
        SELECT name FROM customers
        WHERE customer_id IN (
            SELECT customer_id FROM orders WHERE total > 500
        )
    """
    print("Original (subquery):")
    print(query1)
    cursor.execute(f"EXPLAIN QUERY PLAN {query1}")
    print("\nPlan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Transformed: JOIN
    query2 = """
        SELECT DISTINCT c.name
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.total > 500
    """
    print("\n\nTransformed (JOIN):")
    print(query2)
    cursor.execute(f"EXPLAIN QUERY PLAN {query2}")
    print("\nPlan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nNote: Optimizer may automatically transform queries")

    # Predicate pushdown
    print("\n\nTransformation 2: Predicate Pushdown")
    print("-" * 60)

    query3 = """
        SELECT * FROM (
            SELECT c.*, o.total
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
        ) AS subq
        WHERE total > 500
    """
    print("Query with filter outside subquery:")
    print(query3)

    cursor.execute(f"EXPLAIN QUERY PLAN {query3}")
    print("\nPlan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nNote: Optimizer pushes 'total > 500' predicate into subquery")
    print("      (filters applied early to reduce intermediate results)")

    conn.close()
    print()


def demonstrate_statistics():
    """Demonstrate role of statistics in optimization."""
    print("=" * 60)
    print("STATISTICS AND COST ESTIMATION")
    print("=" * 60)
    print()

    conn = setup_sample_database()
    cursor = conn.cursor()

    # Analyze tables to gather statistics
    print("1. Gathering statistics with ANALYZE")
    print("-" * 60)
    cursor.execute("ANALYZE")
    print("✓ Statistics gathered")

    # Check statistics
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sqlite_stat%'")
    stat_tables = cursor.fetchall()
    print(f"\nStatistics stored in tables: {[t[0] for t in stat_tables]}")

    # View statistics for a table
    if stat_tables:
        cursor.execute("SELECT * FROM sqlite_stat1 WHERE tbl = 'customers' LIMIT 5")
        print("\nSample statistics for customers table:")
        for row in cursor.fetchall():
            print(f"  {row}")

    print("\n\n2. How optimizer uses statistics")
    print("-" * 60)
    cursor.execute("CREATE INDEX idx_city ON customers(city)")

    # Selective query (few results)
    selective_query = "SELECT * FROM customers WHERE city = 'Tokyo'"
    cursor.execute(f"SELECT COUNT(*) FROM customers WHERE city = 'Tokyo'")
    count = cursor.fetchone()[0]
    print(f"\nSelective query (city='Tokyo'): {count} rows match")

    cursor.execute(f"EXPLAIN QUERY PLAN {selective_query}")
    print("Plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Non-selective query (many results)
    cursor.execute("SELECT COUNT(DISTINCT city) FROM customers")
    num_cities = cursor.fetchone()[0]
    print(f"\n\nNon-selective query: Only {num_cities} distinct cities")
    print("If most rows match, full scan may be cheaper than index")

    print("\n✓ Optimizer uses statistics to estimate:")
    print("  - Table cardinality (number of rows)")
    print("  - Index selectivity (how many rows match)")
    print("  - Join cardinality (size of join result)")

    conn.close()
    print()


def demonstrate_optimization_hints():
    """Demonstrate when manual optimization is needed."""
    print("=" * 60)
    print("MANUAL OPTIMIZATION TECHNIQUES")
    print("=" * 60)
    print()

    print("1. Rewrite correlated subquery as JOIN")
    print("-" * 60)
    print("Slow (correlated subquery - runs for each row):")
    print("""
    SELECT c.name,
           (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id)
    FROM customers c
    """)

    print("\nFast (JOIN with GROUP BY):")
    print("""
    SELECT c.name, COUNT(o.order_id)
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id
    """)

    print("\n\n2. Use LIMIT for pagination")
    print("-" * 60)
    print("Bad (fetch all, discard most):")
    print("  SELECT * FROM large_table ORDER BY id")
    print("\nGood (only fetch needed rows):")
    print("  SELECT * FROM large_table ORDER BY id LIMIT 20 OFFSET 100")

    print("\n\n3. Avoid SELECT *")
    print("-" * 60)
    print("Bad (fetches unnecessary data):")
    print("  SELECT * FROM customers WHERE country = 'USA'")
    print("\nGood (only needed columns):")
    print("  SELECT customer_id, name FROM customers WHERE country = 'USA'")
    print("  (Enables covering index optimization)")

    print("\n\n4. Use EXISTS instead of COUNT for existence check")
    print("-" * 60)
    print("Slow (counts all matches):")
    print("  SELECT COUNT(*) FROM orders WHERE customer_id = 123")
    print("\nFast (stops after finding first match):")
    print("  SELECT EXISTS(SELECT 1 FROM orders WHERE customer_id = 123)")

    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║            QUERY OPTIMIZATION                                ║
║  Execution Plans, Join Ordering, Index Selection             ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_explain_query_plan()
    demonstrate_join_ordering()
    demonstrate_index_selection()
    demonstrate_query_transformations()
    demonstrate_statistics()
    demonstrate_optimization_hints()

    print("=" * 60)
    print("SUMMARY: QUERY OPTIMIZATION")
    print("=" * 60)
    print("Optimizer's job:")
    print("  1. Parse SQL into relational algebra")
    print("  2. Apply equivalence transformations")
    print("  3. Estimate cost of alternative plans")
    print("  4. Choose plan with lowest estimated cost")
    print()
    print("Key optimization techniques:")
    print("  - Predicate pushdown (filter early)")
    print("  - Join reordering (minimize intermediates)")
    print("  - Index selection (avoid full scans)")
    print("  - Subquery flattening (convert to JOIN)")
    print()
    print("Tools:")
    print("  - EXPLAIN QUERY PLAN: see execution plan")
    print("  - ANALYZE: gather statistics")
    print("  - CREATE INDEX: speed up queries")
    print("=" * 60)
