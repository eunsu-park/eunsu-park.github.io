"""
OWASP Top 10 Security Patterns Demo
====================================

Educational demonstration of secure vs vulnerable code patterns
based on the OWASP Top 10 Web Application Security Risks.

Topics covered:
- SQL Injection (A03:2021 - Injection)
- Input validation
- Secure error handling
- Secure session configuration concepts
- Broken Access Control patterns (A01:2021)
- Security Misconfiguration (A05:2021)

All examples are DEFENSIVE - showing how to identify and FIX
vulnerabilities. Uses Python standard library + sqlite3.
"""

import sqlite3
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import traceback
from urllib.parse import urlparse

print("=" * 65)
print("  OWASP Top 10 Security Patterns Demo")
print("=" * 65)
print()


# ============================================================
# Section 1: A03:2021 - Injection (SQL Injection)
# ============================================================

print("-" * 65)
print("  Section 1: A03 - SQL Injection")
print("-" * 65)

# Setup in-memory database
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        email TEXT,
        password_hash TEXT,
        is_admin INTEGER DEFAULT 0
    )
""")
cursor.executemany(
    "INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)",
    [
        ("alice", "alice@example.com", hashlib.sha256(b"hashed_pw").hexdigest(), 0),
        ("bob", "bob@example.com", hashlib.sha256(b"hashed_pw2").hexdigest(), 0),
        ("admin", "admin@example.com", hashlib.sha256(b"admin_pw").hexdigest(), 1),
    ],
)
conn.commit()

print("""
  VULNERABLE: String formatting in SQL queries
  SECURE:     Parameterized queries (prepared statements)
""")


# VULNERABLE: String concatenation
def get_user_vulnerable(username: str) -> list:
    """VULNERABLE: SQL injection via string formatting."""
    query = f"SELECT * FROM users WHERE username = '{username}'"
    print(f"    Query: {query}")
    return cursor.execute(query).fetchall()


# SECURE: Parameterized query
def get_user_secure(username: str) -> list:
    """SECURE: Uses parameterized query."""
    query = "SELECT * FROM users WHERE username = ?"
    print(f"    Query: {query}  params=[{username}]")
    return cursor.execute(query, (username,)).fetchall()


# Normal usage
print("  Normal input: 'alice'")
print("  -- Vulnerable --")
result_v = get_user_vulnerable("alice")
print(f"    Result: {result_v}")
print("  -- Secure --")
result_s = get_user_secure("alice")
print(f"    Result: {result_s}")
print()

# SQL Injection attack
malicious_input = "' OR '1'='1"
print(f"  Malicious input: {malicious_input!r}")
print("  -- Vulnerable --")
result_v = get_user_vulnerable(malicious_input)
print(f"    Result: {len(result_v)} rows returned (ALL users leaked!)")
for row in result_v:
    print(f"      {row}")
print("  -- Secure --")
result_s = get_user_secure(malicious_input)
print(f"    Result: {len(result_s)} rows (correctly returns nothing)")
print()


# ============================================================
# Section 2: Input Validation
# ============================================================

print("-" * 65)
print("  Section 2: Input Validation")
print("-" * 65)

print("""
  Validate ALL user input. Reject invalid data early.
  Use allowlists (not blocklists) when possible.
""")


class InputValidator:
    """Collection of input validation methods."""

    @staticmethod
    def validate_email(email: str) -> tuple[bool, str]:
        """Validate email format."""
        if not email or len(email) > 254:
            return False, "Email too long or empty"
        # Basic RFC 5322 pattern (simplified)
        pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, email):
            return False, "Invalid email format"
        return True, "Valid"

    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        """Validate username: alphanumeric + underscore, 3-32 chars."""
        if not username:
            return False, "Username is required"
        if len(username) < 3 or len(username) > 32:
            return False, "Username must be 3-32 characters"
        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            return False, "Username can only contain letters, numbers, underscore"
        return True, "Valid"

    @staticmethod
    def validate_url(url: str) -> tuple[bool, str]:
        """Validate URL - prevent open redirect / SSRF."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False, "Invalid URL format"

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False, "URL must include scheme and host"

        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            return False, f"Scheme '{parsed.scheme}' not allowed (http/https only)"

        # Block internal IPs (SSRF prevention)
        blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"]
        if parsed.hostname in blocked_hosts:
            return False, f"Host '{parsed.hostname}' is blocked (SSRF prevention)"

        # Block internal network ranges
        if parsed.hostname and parsed.hostname.startswith(("10.", "192.168.", "172.")):
            return False, "Internal network addresses are blocked"

        return True, "Valid"

    @staticmethod
    def validate_integer(value: str, min_val: int = None,
                         max_val: int = None) -> tuple[bool, str]:
        """Validate integer input with range check."""
        try:
            num = int(value)
        except (ValueError, TypeError):
            return False, "Not a valid integer"
        if min_val is not None and num < min_val:
            return False, f"Value must be >= {min_val}"
        if max_val is not None and num > max_val:
            return False, f"Value must be <= {max_val}"
        return True, "Valid"


validator = InputValidator()

# Test email validation
print("\n  Email Validation:")
test_emails = [
    "user@example.com",
    "invalid-email",
    "user@.com",
    "<script>@evil.com",
    "a" * 300 + "@example.com",
]
for email in test_emails:
    valid, msg = validator.validate_email(email)
    status = "PASS" if valid else "FAIL"
    print(f"    [{status}] {email[:35]:<35} -> {msg}")

# Test URL validation (SSRF prevention)
print("\n  URL Validation (SSRF Prevention):")
test_urls = [
    "https://example.com/page",
    "http://localhost/admin",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
    "https://192.168.1.1/internal",
]
for url in test_urls:
    valid, msg = validator.validate_url(url)
    status = "PASS" if valid else "BLOCK"
    print(f"    [{status}] {url[:40]:<40} -> {msg}")
print()


# ============================================================
# Section 3: Secure Error Handling
# ============================================================

print("-" * 65)
print("  Section 3: Secure Error Handling")
print("-" * 65)

print("""
  VULNERABLE: Exposing stack traces, SQL errors, internal paths
  SECURE:     Generic messages to users, detailed logs internally
""")


class SecureErrorHandler:
    """Demonstrates secure vs insecure error handling."""

    def __init__(self):
        self.error_log: list[dict] = []  # Simulates server-side log

    def handle_error_vulnerable(self, error: Exception) -> str:
        """VULNERABLE: Exposes internal details to user."""
        return f"Error: {type(error).__name__}: {error}\n{traceback.format_exc()}"

    def handle_error_secure(self, error: Exception, request_id: str = None) -> str:
        """SECURE: Generic message to user, details to internal log."""
        if request_id is None:
            request_id = secrets.token_hex(8)

        # Log full details internally
        self.error_log.append({
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        })

        # Return sanitized message to user
        return (
            f"An unexpected error occurred. "
            f"Reference ID: {request_id}. "
            f"Please contact support if this persists."
        )


handler = SecureErrorHandler()

# Simulate an error
try:
    result = 1 / 0
except Exception as e:
    print("\n  -- VULNERABLE error response (exposes internals) --")
    vuln_response = handler.handle_error_vulnerable(e)
    for line in vuln_response.strip().split("\n")[:4]:
        print(f"    {line}")
    print("    ...")

    print("\n  -- SECURE error response (generic to user) --")
    secure_response = handler.handle_error_secure(e)
    print(f"    {secure_response}")

    print("\n  -- Internal log (server-side only) --")
    log_entry = handler.error_log[-1]
    print(f"    Request ID:  {log_entry['request_id']}")
    print(f"    Error Type:  {log_entry['error_type']}")
    print(f"    Message:     {log_entry['error_message']}")
print()


# ============================================================
# Section 4: Secure Session Configuration
# ============================================================

print("-" * 65)
print("  Section 4: Secure Session Configuration")
print("-" * 65)

print("""
  Session security checklist for web applications:
""")

secure_config = {
    "session_cookie_name": "__Host-session",  # __Host- prefix for extra security
    "session_cookie_httponly": True,   # No JavaScript access
    "session_cookie_secure": True,    # HTTPS only
    "session_cookie_samesite": "Lax", # CSRF protection
    "session_cookie_path": "/",
    "session_lifetime_seconds": 3600,    # 1 hour
    "session_idle_timeout_seconds": 900, # 15 minutes
    "session_regenerate_on_login": True,
    "session_regenerate_on_privilege_change": True,
    "max_concurrent_sessions": 3,
}

insecure_config = {
    "session_cookie_name": "session",     # No prefix
    "session_cookie_httponly": False,      # JS can steal cookie!
    "session_cookie_secure": False,       # Sent over HTTP!
    "session_cookie_samesite": "None",    # No CSRF protection!
    "session_cookie_path": "/",
    "session_lifetime_seconds": 86400 * 30,  # 30 days!
    "session_idle_timeout_seconds": None,    # Never expires!
    "session_regenerate_on_login": False,
    "session_regenerate_on_privilege_change": False,
    "max_concurrent_sessions": None,         # Unlimited!
}

print(f"  {'Setting':<42} {'INSECURE':<14} {'SECURE':<14}")
print(f"  {'-'*42} {'-'*14} {'-'*14}")
for key in secure_config:
    insecure_val = str(insecure_config.get(key, "N/A"))[:12]
    secure_val = str(secure_config[key])[:12]
    clean_key = key.replace("session_cookie_", "cookie.").replace("session_", "")
    print(f"  {clean_key:<42} {insecure_val:<14} {secure_val:<14}")
print()


# ============================================================
# Section 5: A01:2021 - Broken Access Control
# ============================================================

print("-" * 65)
print("  Section 5: A01 - Broken Access Control")
print("-" * 65)

print("""
  Broken access control allows users to act outside
  their intended permissions.
""")


# VULNERABLE: Direct object reference without authorization check
class VulnerableAPI:
    """VULNERABLE: No authorization checks on resource access."""

    def __init__(self):
        self.documents = {
            1: {"owner": "alice", "title": "Alice's Report", "content": "Secret data"},
            2: {"owner": "bob", "title": "Bob's Notes", "content": "Private notes"},
        }

    def get_document(self, doc_id: int, requesting_user: str) -> dict:
        """VULNERABLE: Any user can access any document by ID."""
        # No authorization check!
        doc = self.documents.get(doc_id)
        if doc:
            return {"status": "ok", "document": doc}
        return {"status": "error", "message": "Not found"}


# SECURE: Authorization check before access
class SecureAPI:
    """SECURE: Checks authorization before returning resources."""

    def __init__(self):
        self.documents = {
            1: {"owner": "alice", "title": "Alice's Report", "content": "Secret data"},
            2: {"owner": "bob", "title": "Bob's Notes", "content": "Private notes"},
        }
        self.admin_users = {"admin"}

    def get_document(self, doc_id: int, requesting_user: str) -> dict:
        """SECURE: Checks ownership or admin status before access."""
        doc = self.documents.get(doc_id)
        if not doc:
            return {"status": "error", "message": "Not found"}

        # Authorization check
        if doc["owner"] != requesting_user and requesting_user not in self.admin_users:
            return {"status": "error", "message": "Access denied"}

        return {"status": "ok", "document": doc}


vuln_api = VulnerableAPI()
secure_api = SecureAPI()

print("\n  Scenario: Bob tries to access Alice's document (doc_id=1)")
print()
print("  -- VULNERABLE API --")
result = vuln_api.get_document(1, "bob")
print(f"    Result: {result['status']} - {result.get('document', {}).get('title', 'N/A')}")
print(f"    Content exposed: {result.get('document', {}).get('content', 'N/A')}")

print("\n  -- SECURE API --")
result = secure_api.get_document(1, "bob")
print(f"    Result: {result['status']} - {result.get('message', 'N/A')}")

result = secure_api.get_document(1, "alice")
print(f"    Alice's own request: {result['status']} - {result.get('document', {}).get('title', 'N/A')}")
print()


# ============================================================
# Section 6: A05:2021 - Security Misconfiguration
# ============================================================

print("-" * 65)
print("  Section 6: A05 - Security Misconfiguration")
print("-" * 65)


def security_headers_check(headers: dict) -> list[dict]:
    """Check HTTP response headers for security issues."""
    findings = []

    recommended_headers = {
        "Strict-Transport-Security": {
            "expected": "max-age=31536000; includeSubDomains",
            "severity": "HIGH",
            "description": "HSTS forces HTTPS, prevents downgrade attacks",
        },
        "Content-Security-Policy": {
            "expected": "default-src 'self'",
            "severity": "HIGH",
            "description": "CSP prevents XSS and data injection",
        },
        "X-Content-Type-Options": {
            "expected": "nosniff",
            "severity": "MEDIUM",
            "description": "Prevents MIME type sniffing",
        },
        "X-Frame-Options": {
            "expected": "DENY",
            "severity": "MEDIUM",
            "description": "Prevents clickjacking via iframes",
        },
        "Referrer-Policy": {
            "expected": "strict-origin-when-cross-origin",
            "severity": "LOW",
            "description": "Controls referrer information leakage",
        },
        "Permissions-Policy": {
            "expected": "camera=(), microphone=(), geolocation=()",
            "severity": "LOW",
            "description": "Restricts browser feature access",
        },
    }

    # Headers that should NOT be present
    dangerous_headers = {
        "Server": "Reveals server software version",
        "X-Powered-By": "Reveals framework/language",
    }

    for header, info in recommended_headers.items():
        if header not in headers:
            findings.append({
                "type": "MISSING",
                "header": header,
                "severity": info["severity"],
                "description": info["description"],
                "recommendation": f"Add: {header}: {info['expected']}",
            })

    for header, desc in dangerous_headers.items():
        if header in headers:
            findings.append({
                "type": "REMOVE",
                "header": header,
                "severity": "LOW",
                "description": desc,
                "recommendation": f"Remove {header} header",
            })

    return findings


# Example: Check insecure headers
insecure_headers = {
    "Server": "Apache/2.4.41 (Ubuntu)",
    "X-Powered-By": "Express",
    "Content-Type": "text/html",
}

print("\n  Security Headers Audit:")
print(f"  Checking response headers: {json.dumps(insecure_headers, indent=2)}")
print()

findings = security_headers_check(insecure_headers)
for f in findings:
    icon = "!" if f["severity"] == "HIGH" else "-"
    print(f"    [{f['severity']:<6}] {icon} {f['type']}: {f['header']}")
    print(f"              {f['description']}")
    print(f"              Fix: {f['recommendation']}")
print()


# ============================================================
# Section 7: Summary
# ============================================================

print("=" * 65)
print("  OWASP Top 10 (2021) Summary")
print("=" * 65)
print("""
  Rank | Category                         | Key Defense
  -----+----------------------------------+---------------------------
  A01  | Broken Access Control            | Authz checks, deny default
  A02  | Cryptographic Failures           | TLS, strong hashing, KMS
  A03  | Injection                        | Parameterized queries
  A04  | Insecure Design                  | Threat modeling, secure arch
  A05  | Security Misconfiguration        | Hardening, security headers
  A06  | Vulnerable Components            | Dependency scanning, SCA
  A07  | Auth & Identity Failures         | MFA, rate limiting
  A08  | Software & Data Integrity        | CI/CD security, signatures
  A09  | Logging & Monitoring Failures    | Centralized logging, alerts
  A10  | Server-Side Request Forgery      | Input validation, allowlists

  Remember: Security is a process, not a product.
  Apply defense in depth - no single control is sufficient.
""")

# Cleanup
conn.close()
