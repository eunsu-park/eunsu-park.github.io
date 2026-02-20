"""
B+Tree Indexing

Demonstrates B+Tree data structure for database indexing:
- B+Tree properties: balanced, sorted, multi-way tree
- Insert, search, range query operations
- Performance comparison: with vs without index
- Leaf node chaining for sequential access

Theory:
- B+Tree is optimized for disk I/O (high branching factor)
- All data stored in leaves; internal nodes only store keys
- Leaves are linked for efficient range queries
- Height is O(log_m N) where m is order, N is number of keys
- Guarantees balanced tree (all paths to leaves have same length)
- Insert/delete maintain balance through splits and merges

Simplified implementation focuses on concepts, not production-ready code.
"""

import sqlite3
import time
from typing import Optional, List, Tuple, Any


class BPlusTreeNode:
    """Node in a B+Tree."""

    def __init__(self, is_leaf: bool = False):
        self.is_leaf = is_leaf
        self.keys: List[int] = []
        self.children: List[BPlusTreeNode] = []  # For internal nodes
        self.values: List[Any] = []  # For leaf nodes
        self.next: Optional[BPlusTreeNode] = None  # Link to next leaf


class BPlusTree:
    """Simplified B+Tree implementation for demonstration."""

    def __init__(self, order: int = 4):
        """
        Args:
            order: Maximum number of children per node (minimum is ceil(order/2))
        """
        self.order = order
        self.root = BPlusTreeNode(is_leaf=True)

    def search(self, key: int) -> Optional[Any]:
        """Search for a key in the tree."""
        node = self.root

        # Traverse to leaf
        while not node.is_leaf:
            # Find appropriate child
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]

        # Search in leaf
        if key in node.keys:
            idx = node.keys.index(key)
            return node.values[idx]
        return None

    def range_query(self, start_key: int, end_key: int) -> List[Tuple[int, Any]]:
        """Return all key-value pairs in range [start_key, end_key]."""
        results = []
        node = self.root

        # Find leaf containing start_key
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and start_key >= node.keys[i]:
                i += 1
            node = node.children[i]

        # Traverse leaves using next pointers
        while node is not None:
            for i, key in enumerate(node.keys):
                if start_key <= key <= end_key:
                    results.append((key, node.values[i]))
                elif key > end_key:
                    return results
            node = node.next

        return results

    def insert(self, key: int, value: Any):
        """Insert a key-value pair."""
        root = self.root

        # If root is full, split it
        if len(root.keys) >= self.order:
            new_root = BPlusTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, key, value)

    def _insert_non_full(self, node: BPlusTreeNode, key: int, value: Any):
        """Insert into a node that is not full."""
        if node.is_leaf:
            # Insert into sorted position
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            node.keys.insert(i, key)
            node.values.insert(i, value)
        else:
            # Find child to insert into
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1

            # Split child if full
            if len(node.children[i].keys) >= self.order:
                self._split_child(node, i)
                if key >= node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key, value)

    def _split_child(self, parent: BPlusTreeNode, index: int):
        """Split a full child node."""
        full_child = parent.children[index]
        mid = len(full_child.keys) // 2

        # Create new sibling
        new_child = BPlusTreeNode(is_leaf=full_child.is_leaf)

        if full_child.is_leaf:
            # Copy upper half to new node
            new_child.keys = full_child.keys[mid:]
            new_child.values = full_child.values[mid:]
            full_child.keys = full_child.keys[:mid]
            full_child.values = full_child.values[:mid]

            # Link leaves
            new_child.next = full_child.next
            full_child.next = new_child

            # Promote copy of first key in new node
            parent.keys.insert(index, new_child.keys[0])
        else:
            # Internal node split
            new_child.keys = full_child.keys[mid+1:]
            new_child.children = full_child.children[mid+1:]
            promote_key = full_child.keys[mid]
            full_child.keys = full_child.keys[:mid]
            full_child.children = full_child.children[:mid+1]

            parent.keys.insert(index, promote_key)

        parent.children.insert(index + 1, new_child)

    def print_tree(self, node: Optional[BPlusTreeNode] = None, level: int = 0):
        """Print tree structure for visualization."""
        if node is None:
            node = self.root

        indent = "  " * level
        if node.is_leaf:
            print(f"{indent}LEAF: {node.keys}")
        else:
            print(f"{indent}INTERNAL: {node.keys}")
            for child in node.children:
                self.print_tree(child, level + 1)


def demonstrate_btree_operations():
    """Demonstrate B+Tree operations."""
    print("=" * 60)
    print("B+TREE OPERATIONS")
    print("=" * 60)
    print()

    tree = BPlusTree(order=4)  # Max 4 keys per node

    # Insert keys
    print("Inserting keys: 10, 20, 5, 6, 12, 30, 7, 17")
    print("-" * 60)
    for key in [10, 20, 5, 6, 12, 30, 7, 17]:
        tree.insert(key, f"value_{key}")
        print(f"Inserted {key}")

    print("\nTree structure:")
    print("-" * 60)
    tree.print_tree()

    # Search
    print("\n\nSearch operations:")
    print("-" * 60)
    for key in [12, 15, 30]:
        result = tree.search(key)
        print(f"Search({key}): {result if result else 'Not found'}")

    # Range query
    print("\n\nRange query [6, 17]:")
    print("-" * 60)
    results = tree.range_query(6, 17)
    for key, value in results:
        print(f"  {key}: {value}")
    print()


def demonstrate_index_performance():
    """Compare query performance with and without index."""
    print("=" * 60)
    print("INDEX PERFORMANCE COMPARISON")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create table with many rows
    print("Creating table with 100,000 rows...")
    cursor.execute('''
        CREATE TABLE Products (
            product_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        )
    ''')

    # Insert data
    import random
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys']
    data = [
        (i, f'Product_{i}', random.choice(categories), random.uniform(10, 1000))
        for i in range(100000)
    ]
    cursor.executemany("INSERT INTO Products VALUES (?, ?, ?, ?)", data)
    conn.commit()

    print("✓ Inserted 100,000 products\n")

    # Query without index
    print("1. Query WITHOUT index on category:")
    print("-" * 60)
    start = time.time()
    cursor.execute("SELECT COUNT(*) FROM Products WHERE category = 'Electronics'")
    count = cursor.fetchone()[0]
    elapsed = time.time() - start
    print(f"Found {count} electronics in {elapsed*1000:.2f} ms")

    # Show query plan
    cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM Products WHERE category = 'Electronics'")
    print("\nQuery plan:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Create index
    print("\n\n2. Creating index on category...")
    print("-" * 60)
    start = time.time()
    cursor.execute("CREATE INDEX idx_category ON Products(category)")
    elapsed = time.time() - start
    print(f"✓ Index created in {elapsed*1000:.2f} ms\n")

    # Query with index
    print("3. Query WITH index on category:")
    print("-" * 60)
    start = time.time()
    cursor.execute("SELECT COUNT(*) FROM Products WHERE category = 'Electronics'")
    count = cursor.fetchone()[0]
    elapsed = time.time() - start
    print(f"Found {count} electronics in {elapsed*1000:.2f} ms")

    # Show query plan
    cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM Products WHERE category = 'Electronics'")
    print("\nQuery plan (now uses index):")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Range query
    print("\n\n4. Range query (price between 100 and 200):")
    print("-" * 60)

    # Without index
    start = time.time()
    cursor.execute("SELECT COUNT(*) FROM Products WHERE price BETWEEN 100 AND 200")
    count = cursor.fetchone()[0]
    elapsed_no_idx = time.time() - start
    print(f"Without index: {count} products in {elapsed_no_idx*1000:.2f} ms")

    # With index
    cursor.execute("CREATE INDEX idx_price ON Products(price)")
    start = time.time()
    cursor.execute("SELECT COUNT(*) FROM Products WHERE price BETWEEN 100 AND 200")
    count = cursor.fetchone()[0]
    elapsed_with_idx = time.time() - start
    print(f"With index:    {count} products in {elapsed_with_idx*1000:.2f} ms")

    if elapsed_no_idx > elapsed_with_idx:
        speedup = elapsed_no_idx / elapsed_with_idx
        print(f"\n✓ Speedup: {speedup:.1f}x faster with index")

    conn.close()
    print()


def demonstrate_composite_index():
    """Demonstrate composite (multi-column) index."""
    print("=" * 60)
    print("COMPOSITE INDEX")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE Orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            total REAL
        )
    ''')

    # Insert sample data
    import random
    from datetime import date, timedelta
    start_date = date(2023, 1, 1)
    data = [
        (i, random.randint(1, 100), (start_date + timedelta(days=random.randint(0, 365))).isoformat(), random.uniform(10, 500))
        for i in range(10000)
    ]
    cursor.executemany("INSERT INTO Orders VALUES (?, ?, ?, ?)", data)

    print("Inserted 10,000 orders\n")

    # Query on customer_id and order_date
    print("Query: SELECT * FROM Orders WHERE customer_id = 50 AND order_date >= '2023-06-01'")
    print("-" * 60)

    # Without index
    cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM Orders WHERE customer_id = 50 AND order_date >= '2023-06-01'")
    print("Without index:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Create composite index
    print("\nCreating composite index on (customer_id, order_date)...")
    cursor.execute("CREATE INDEX idx_cust_date ON Orders(customer_id, order_date)")

    cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM Orders WHERE customer_id = 50 AND order_date >= '2023-06-01'")
    print("\nWith composite index:")
    for row in cursor.fetchall():
        print(f"  {row}")

    print("\nNote: Composite index can be used for:")
    print("  ✓ WHERE customer_id = X")
    print("  ✓ WHERE customer_id = X AND order_date = Y")
    print("  ✗ WHERE order_date = Y (only) - cannot use index efficiently")
    print("    (Leftmost prefix rule)")

    conn.close()
    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                B+TREE INDEXING                               ║
║  Data Structure, Operations, Performance                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_btree_operations()
    demonstrate_index_performance()
    demonstrate_composite_index()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("B+Tree properties:")
    print("  - Balanced: all leaves at same depth")
    print("  - Sorted: supports range queries efficiently")
    print("  - High branching factor: minimizes disk I/O")
    print("  - Leaves linked: sequential access")
    print()
    print("When to use indexes:")
    print("  ✓ Columns in WHERE clauses")
    print("  ✓ Columns in JOIN conditions")
    print("  ✓ Columns in ORDER BY")
    print("  ✗ Small tables (overhead not worth it)")
    print("  ✗ Frequently updated columns (index maintenance cost)")
    print("=" * 60)
