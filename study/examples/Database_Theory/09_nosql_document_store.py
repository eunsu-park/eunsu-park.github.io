"""
NoSQL Document Store with SQLite JSON

Demonstrates document-oriented database concepts using SQLite's JSON functions:
- Schema-less storage (flexible documents)
- CRUD operations on JSON documents
- Nested queries and array operations
- Indexing JSON fields
- Comparison with relational approach

Theory:
- Document stores organize data as semi-structured documents (JSON, BSON, XML)
- Schema flexibility: different documents can have different structures
- Trade-offs: flexibility vs. data integrity
- SQLite supports JSON1 extension for JSON operations
- Useful for: hierarchical data, variable schemas, rapid development

CAP theorem consideration:
- Most NoSQL systems sacrifice strong consistency for availability/partition tolerance
- SQLite is single-writer, so it maintains strong consistency
"""

import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime


def demonstrate_document_storage():
    """Demonstrate storing and querying JSON documents."""
    print("=" * 60)
    print("DOCUMENT STORAGE (Schema-less)")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create simple document store
    cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            data TEXT NOT NULL,  -- JSON document
            created_at REAL DEFAULT (julianday('now'))
        )
    ''')

    print("1. Inserting documents with different schemas")
    print("-" * 60)

    # User documents with varying fields
    users = [
        {
            "name": "Alice",
            "email": "alice@example.com",
            "age": 30,
            "address": {
                "city": "New York",
                "country": "USA"
            }
        },
        {
            "name": "Bob",
            "email": "bob@example.com",
            "phone": "555-1234",  # Different field!
            "preferences": {
                "theme": "dark",
                "notifications": True
            }
        },
        {
            "name": "Charlie",
            "age": 25,
            "skills": ["Python", "SQL", "Docker"],  # Array field!
            "address": {
                "city": "London",
                "country": "UK",
                "postal_code": "SW1A 1AA"  # Extra nested field
            }
        }
    ]

    for user in users:
        cursor.execute(
            "INSERT INTO documents (collection, data) VALUES (?, ?)",
            ("users", json.dumps(user))
        )
    print(f"✓ Inserted {len(users)} user documents with different schemas")

    # Retrieve documents
    print("\n2. Retrieving all user documents")
    print("-" * 60)
    cursor.execute("SELECT id, data FROM documents WHERE collection = 'users'")
    for doc_id, data in cursor.fetchall():
        doc = json.loads(data)
        print(f"  Doc {doc_id}: {doc.get('name')} - {json.dumps(doc, indent=4)}")

    print("\n✓ Schema flexibility: each document can have different fields")
    print()


def demonstrate_json_queries():
    """Demonstrate JSON query operations."""
    print("=" * 60)
    print("JSON QUERY OPERATIONS")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL
        )
    ''')

    # Insert product documents
    products = [
        {
            "name": "Laptop",
            "price": 999.99,
            "specs": {"cpu": "i7", "ram": 16, "storage": 512},
            "tags": ["electronics", "computers"]
        },
        {
            "name": "Mouse",
            "price": 29.99,
            "specs": {"type": "wireless", "dpi": 1600},
            "tags": ["electronics", "accessories"]
        },
        {
            "name": "Book",
            "price": 19.99,
            "author": "John Doe",
            "tags": ["books", "fiction"]
        }
    ]

    for product in products:
        cursor.execute("INSERT INTO products (data) VALUES (?)", (json.dumps(product),))

    # Extract specific fields
    print("1. Extract specific JSON fields (json_extract)")
    print("-" * 60)
    cursor.execute("""
        SELECT
            json_extract(data, '$.name') as name,
            json_extract(data, '$.price') as price
        FROM products
        ORDER BY price DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:15} ${row[1]:.2f}")

    # Filter by nested field
    print("\n2. Filter by nested field (specs.ram >= 16)")
    print("-" * 60)
    cursor.execute("""
        SELECT json_extract(data, '$.name') as name
        FROM products
        WHERE json_extract(data, '$.specs.ram') >= 16
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    # Array operations
    print("\n3. Find products with 'electronics' tag (array search)")
    print("-" * 60)
    cursor.execute("""
        SELECT json_extract(data, '$.name') as name
        FROM products, json_each(json_extract(data, '$.tags'))
        WHERE json_each.value = 'electronics'
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    # Full-text search in JSON
    print("\n4. Search for 'wireless' anywhere in document")
    print("-" * 60)
    cursor.execute("""
        SELECT json_extract(data, '$.name') as name
        FROM products
        WHERE json_extract(data, '$.specs.type') = 'wireless'
           OR data LIKE '%wireless%'
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    conn.close()
    print()


def demonstrate_json_indexing():
    """Demonstrate indexing JSON fields for performance."""
    print("=" * 60)
    print("INDEXING JSON FIELDS")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL
        )
    ''')

    # Insert many events
    print("Inserting 10,000 event documents...")
    import random
    event_types = ["login", "logout", "purchase", "view", "click"]
    for i in range(10000):
        event = {
            "type": random.choice(event_types),
            "user_id": random.randint(1, 1000),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "ip": f"192.168.1.{random.randint(1, 255)}",
                "session_id": f"sess_{random.randint(1000, 9999)}"
            }
        }
        cursor.execute("INSERT INTO events (data) VALUES (?)", (json.dumps(event),))
    conn.commit()
    print("✓ Inserted 10,000 events\n")

    # Query without index
    print("1. Query without index (count by event type)")
    print("-" * 60)
    import time
    start = time.time()
    cursor.execute("""
        SELECT
            json_extract(data, '$.type') as event_type,
            COUNT(*) as count
        FROM events
        WHERE json_extract(data, '$.type') = 'purchase'
    """)
    result = cursor.fetchone()
    elapsed_no_idx = time.time() - start
    print(f"  Found {result[1]} purchases in {elapsed_no_idx*1000:.2f} ms")

    # Create index on extracted JSON field
    print("\n2. Creating index on json_extract(data, '$.type')")
    print("-" * 60)
    cursor.execute("""
        CREATE INDEX idx_event_type
        ON events(json_extract(data, '$.type'))
    """)
    print("✓ Index created")

    # Query with index
    print("\n3. Query with index (same query)")
    print("-" * 60)
    start = time.time()
    cursor.execute("""
        SELECT
            json_extract(data, '$.type') as event_type,
            COUNT(*) as count
        FROM events
        WHERE json_extract(data, '$.type') = 'purchase'
    """)
    result = cursor.fetchone()
    elapsed_with_idx = time.time() - start
    print(f"  Found {result[1]} purchases in {elapsed_with_idx*1000:.2f} ms")

    if elapsed_no_idx > elapsed_with_idx:
        speedup = elapsed_no_idx / elapsed_with_idx
        print(f"\n✓ Speedup: {speedup:.1f}x faster with index")

    # Show query plan
    print("\n4. Query plan analysis")
    print("-" * 60)
    cursor.execute("""
        EXPLAIN QUERY PLAN
        SELECT * FROM events
        WHERE json_extract(data, '$.type') = 'purchase'
    """)
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


def demonstrate_relational_vs_document():
    """Compare relational and document approaches."""
    print("=" * 60)
    print("RELATIONAL VS DOCUMENT APPROACH")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Relational approach: normalized tables
    print("RELATIONAL APPROACH (Normalized)")
    print("-" * 60)
    cursor.execute('''
        CREATE TABLE blog_posts (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            content TEXT,
            created_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            post_id INTEGER,
            author TEXT,
            text TEXT,
            created_at TEXT,
            FOREIGN KEY (post_id) REFERENCES blog_posts(id)
        )
    ''')

    cursor.execute("""
        INSERT INTO blog_posts VALUES
        (1, 'First Post', 'Alice', 'Hello world!', '2024-01-01')
    """)
    cursor.execute("""
        INSERT INTO comments VALUES
        (1, 1, 'Bob', 'Great post!', '2024-01-02'),
        (2, 1, 'Charlie', 'Thanks for sharing', '2024-01-03')
    """)

    print("Query: Get post with all comments (requires JOIN)")
    cursor.execute("""
        SELECT
            p.title,
            p.author as post_author,
            c.author as comment_author,
            c.text as comment_text
        FROM blog_posts p
        LEFT JOIN comments c ON p.id = c.post_id
        WHERE p.id = 1
    """)
    print("Results:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # Document approach: embedded documents
    print("\n\nDOCUMENT APPROACH (Denormalized)")
    print("-" * 60)
    cursor.execute('''
        CREATE TABLE blog_docs (
            id INTEGER PRIMARY KEY,
            data TEXT NOT NULL
        )
    ''')

    blog_post = {
        "title": "First Post",
        "author": "Alice",
        "content": "Hello world!",
        "created_at": "2024-01-01",
        "comments": [
            {"author": "Bob", "text": "Great post!", "created_at": "2024-01-02"},
            {"author": "Charlie", "text": "Thanks for sharing", "created_at": "2024-01-03"}
        ]
    }

    cursor.execute("INSERT INTO blog_docs VALUES (1, ?)", (json.dumps(blog_post),))

    print("Query: Get post with all comments (single document, no JOIN)")
    cursor.execute("SELECT data FROM blog_docs WHERE id = 1")
    doc = json.loads(cursor.fetchone()[0])
    print(f"Post: {doc['title']} by {doc['author']}")
    print("Comments:")
    for comment in doc['comments']:
        print(f"  - {comment['author']}: {comment['text']}")

    print("\n\nTRADE-OFFS:")
    print("-" * 60)
    print("Relational:")
    print("  ✓ No data duplication")
    print("  ✓ Easy to update (update in one place)")
    print("  ✓ Strong schema validation")
    print("  ✗ Requires JOINs (slower for reads)")
    print()
    print("Document:")
    print("  ✓ Fast reads (no JOINs needed)")
    print("  ✓ Schema flexibility")
    print("  ✓ Natural hierarchical data representation")
    print("  ✗ Data duplication possible")
    print("  ✗ Updates may require updating multiple documents")
    print("  ✗ Weaker data integrity")

    conn.close()
    print()


def demonstrate_update_operations():
    """Demonstrate updating JSON documents."""
    print("=" * 60)
    print("UPDATING JSON DOCUMENTS")
    print("=" * 60)
    print()

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE user_profiles (
            id INTEGER PRIMARY KEY,
            data TEXT NOT NULL
        )
    ''')

    user = {
        "name": "Alice",
        "email": "alice@example.com",
        "settings": {
            "theme": "light",
            "notifications": True
        }
    }
    cursor.execute("INSERT INTO user_profiles VALUES (1, ?)", (json.dumps(user),))

    print("Original document:")
    print("-" * 60)
    cursor.execute("SELECT data FROM user_profiles WHERE id = 1")
    print(json.dumps(json.loads(cursor.fetchone()[0]), indent=2))

    # Update nested field
    print("\n1. Update nested field (theme: light → dark)")
    print("-" * 60)
    cursor.execute("""
        UPDATE user_profiles
        SET data = json_set(data, '$.settings.theme', 'dark')
        WHERE id = 1
    """)

    cursor.execute("SELECT data FROM user_profiles WHERE id = 1")
    print(json.dumps(json.loads(cursor.fetchone()[0]), indent=2))

    # Add new field
    print("\n2. Add new field (age: 30)")
    print("-" * 60)
    cursor.execute("""
        UPDATE user_profiles
        SET data = json_set(data, '$.age', 30)
        WHERE id = 1
    """)

    cursor.execute("SELECT data FROM user_profiles WHERE id = 1")
    print(json.dumps(json.loads(cursor.fetchone()[0]), indent=2))

    # Remove field
    print("\n3. Remove field (email)")
    print("-" * 60)
    cursor.execute("""
        UPDATE user_profiles
        SET data = json_remove(data, '$.email')
        WHERE id = 1
    """)

    cursor.execute("SELECT data FROM user_profiles WHERE id = 1")
    print(json.dumps(json.loads(cursor.fetchone()[0]), indent=2))

    conn.close()
    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          NoSQL DOCUMENT STORE (SQLite JSON)                  ║
║  Schema-less Storage, JSON Queries, Indexing                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_document_storage()
    demonstrate_json_queries()
    demonstrate_json_indexing()
    demonstrate_relational_vs_document()
    demonstrate_update_operations()

    print("=" * 60)
    print("SUMMARY: DOCUMENT STORES")
    print("=" * 60)
    print("When to use:")
    print("  ✓ Hierarchical/nested data")
    print("  ✓ Schema flexibility needed")
    print("  ✓ Rapid prototyping")
    print("  ✓ Read-heavy workloads")
    print()
    print("When NOT to use:")
    print("  ✗ Complex relationships (many-to-many)")
    print("  ✗ Strong ACID guarantees required")
    print("  ✗ Complex queries across documents")
    print("  ✗ Frequent updates to shared data")
    print("=" * 60)
