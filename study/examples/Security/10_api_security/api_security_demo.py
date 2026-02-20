"""
API 보안 기법 데모
API Security Techniques Demo

토큰 버킷 레이트 리미터, 입력 검증, CORS 헤더, API 키 해싱,
HMAC 기반 요청 서명 등 API 보안 핵심 기법을 구현합니다.

Demonstrates token bucket rate limiting, input validation, CORS headers,
API key hashing/verification, and HMAC-based request signing.
"""

import hashlib
import hmac
import re
import time
import secrets
import json
import base64
from datetime import datetime, timedelta, timezone


# =============================================================================
# 1. Token Bucket Rate Limiter (토큰 버킷 레이트 리미터)
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket algorithm for rate limiting.
    Tokens are added at a fixed rate. Each request consumes one token.
    If no tokens are available, the request is rejected.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def allow_request(self) -> bool:
        """Check if a request is allowed and consume a token."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def wait_time(self) -> float:
        """Estimated seconds until a token is available."""
        self._refill()
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.refill_rate


def demo_rate_limiter():
    print("=" * 60)
    print("1. Token Bucket Rate Limiter")
    print("=" * 60)

    limiter = TokenBucketRateLimiter(capacity=5, refill_rate=2.0)

    print(f"  Bucket capacity: 5 tokens, refill rate: 2 tokens/sec\n")

    # Burst: send 7 requests rapidly
    for i in range(1, 8):
        allowed = limiter.allow_request()
        status = "ALLOWED" if allowed else "REJECTED"
        print(f"  Request {i}: {status}  (tokens left: {limiter.tokens:.1f})")

    # Wait for refill
    print("\n  Waiting 1.5 seconds for token refill...")
    time.sleep(1.5)

    for i in range(8, 12):
        allowed = limiter.allow_request()
        status = "ALLOWED" if allowed else "REJECTED"
        print(f"  Request {i}: {status}  (tokens left: {limiter.tokens:.1f})")

    print()


# =============================================================================
# 2. Input Validation Functions (입력 검증 함수)
# =============================================================================

class InputValidator:
    """Collection of input validation methods for API endpoints."""

    # Email: simplified RFC 5322 pattern
    EMAIL_REGEX = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    # Username: alphanumeric + underscore, 3-30 chars
    USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_]{3,30}$')

    @staticmethod
    def validate_email(email: str) -> tuple[bool, str]:
        if not email or len(email) > 254:
            return False, "Email is empty or exceeds 254 characters"
        if not InputValidator.EMAIL_REGEX.match(email):
            return False, "Invalid email format"
        return True, "Valid"

    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        if not username:
            return False, "Username is empty"
        if not InputValidator.USERNAME_REGEX.match(username):
            return False, "Username must be 3-30 alphanumeric/underscore characters"
        return True, "Valid"

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        issues = []
        if len(password) < 8:
            issues.append("at least 8 characters")
        if not re.search(r'[A-Z]', password):
            issues.append("an uppercase letter")
        if not re.search(r'[a-z]', password):
            issues.append("a lowercase letter")
        if not re.search(r'[0-9]', password):
            issues.append("a digit")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("a special character")
        if issues:
            return False, "Password needs: " + ", ".join(issues)
        return True, "Strong password"

    @staticmethod
    def sanitize_string(value: str, max_length: int = 200) -> str:
        """Remove potentially dangerous characters and truncate."""
        value = value[:max_length]
        # Remove null bytes and control characters
        value = re.sub(r'[\x00-\x1f\x7f]', '', value)
        # Escape HTML special characters
        value = (value.replace('&', '&amp;').replace('<', '&lt;')
                 .replace('>', '&gt;').replace('"', '&quot;')
                 .replace("'", '&#x27;'))
        return value


def demo_input_validation():
    print("=" * 60)
    print("2. Input Validation")
    print("=" * 60)

    validator = InputValidator()

    # Email validation
    emails = ["user@example.com", "bad-email", "a@b.c", "test@domain.co.uk"]
    print("\n  Email Validation:")
    for email in emails:
        valid, msg = validator.validate_email(email)
        symbol = "OK" if valid else "FAIL"
        print(f"    [{symbol}] {email:30s} -> {msg}")

    # Username validation
    usernames = ["alice_01", "ab", "valid_user", "no spaces!", "x" * 31]
    print("\n  Username Validation:")
    for uname in usernames:
        display = uname if len(uname) <= 20 else uname[:17] + "..."
        valid, msg = validator.validate_username(uname)
        symbol = "OK" if valid else "FAIL"
        print(f"    [{symbol}] {display:25s} -> {msg}")

    # Password strength
    passwords = ["short", "alllowercase1!", "NoDigits!here", "Str0ng!Pass"]
    print("\n  Password Strength:")
    for pw in passwords:
        valid, msg = validator.validate_password_strength(pw)
        symbol = "OK" if valid else "FAIL"
        print(f"    [{symbol}] {pw:25s} -> {msg}")

    # Sanitization
    dangerous = '<script>alert("xss")</script>'
    sanitized = validator.sanitize_string(dangerous)
    print(f"\n  Sanitization:")
    print(f"    Input:     {dangerous}")
    print(f"    Sanitized: {sanitized}")
    print()


# =============================================================================
# 3. CORS Header Generation (CORS 헤더 생성)
# =============================================================================

class CORSPolicy:
    """Generate CORS (Cross-Origin Resource Sharing) response headers."""

    def __init__(self, allowed_origins: list[str], allowed_methods: list[str],
                 allowed_headers: list[str], max_age: int = 3600):
        self.allowed_origins = set(allowed_origins)
        self.allowed_methods = allowed_methods
        self.allowed_headers = allowed_headers
        self.max_age = max_age

    def get_headers(self, request_origin: str) -> dict[str, str]:
        """Generate CORS headers for a given request origin."""
        headers = {}

        if request_origin in self.allowed_origins or "*" in self.allowed_origins:
            headers["Access-Control-Allow-Origin"] = request_origin
            headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            headers["Access-Control-Max-Age"] = str(self.max_age)
            headers["Vary"] = "Origin"
        # If origin not allowed, return empty headers (browser will block)

        return headers


def demo_cors():
    print("=" * 60)
    print("3. CORS Header Generation")
    print("=" * 60)

    cors = CORSPolicy(
        allowed_origins=["https://myapp.com", "https://admin.myapp.com"],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
        allowed_headers=["Content-Type", "Authorization"],
        max_age=7200,
    )

    origins = ["https://myapp.com", "https://evil.com", "https://admin.myapp.com"]
    for origin in origins:
        headers = cors.get_headers(origin)
        status = "ALLOWED" if headers else "BLOCKED"
        print(f"\n  Origin: {origin}  [{status}]")
        if headers:
            for k, v in headers.items():
                print(f"    {k}: {v}")
        else:
            print(f"    (no CORS headers -> browser will block request)")
    print()


# =============================================================================
# 4. API Key Hashing and Verification (API 키 해싱 및 검증)
# =============================================================================

class APIKeyManager:
    """
    Generate and verify API keys using salted SHA-256 hashing.
    Only the hash is stored; the raw key is shown once at creation.
    """

    def __init__(self):
        self.key_store: dict[str, dict] = {}  # key_id -> {hash, salt, created}

    def generate_key(self, owner: str) -> tuple[str, str]:
        """Generate a new API key. Returns (key_id, raw_key)."""
        key_id = f"key_{secrets.token_hex(4)}"
        raw_key = secrets.token_urlsafe(32)
        salt = secrets.token_bytes(16)

        key_hash = hashlib.sha256(salt + raw_key.encode()).hexdigest()

        self.key_store[key_id] = {
            "hash": key_hash,
            "salt": salt,
            "owner": owner,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        return key_id, raw_key

    def verify_key(self, key_id: str, raw_key: str) -> bool:
        """Verify a raw API key against stored hash."""
        if key_id not in self.key_store:
            return False
        entry = self.key_store[key_id]
        computed = hashlib.sha256(entry["salt"] + raw_key.encode()).hexdigest()
        return hmac.compare_digest(computed, entry["hash"])


def demo_api_key():
    print("=" * 60)
    print("4. API Key Hashing & Verification")
    print("=" * 60)

    mgr = APIKeyManager()

    key_id, raw_key = mgr.generate_key("alice")
    print(f"\n  Generated key for 'alice':")
    print(f"    Key ID:  {key_id}")
    print(f"    Raw Key: {raw_key[:12]}... (show once, then discard)")
    print(f"    Stored hash: {mgr.key_store[key_id]['hash'][:24]}...")

    # Verify correct key
    result = mgr.verify_key(key_id, raw_key)
    print(f"\n  Verify with correct key:  {'PASS' if result else 'FAIL'}")

    # Verify wrong key
    result = mgr.verify_key(key_id, "wrong-key-value")
    print(f"  Verify with wrong key:    {'PASS' if result else 'FAIL'}")

    # Verify non-existent key ID
    result = mgr.verify_key("key_nonexistent", raw_key)
    print(f"  Verify non-existent ID:   {'PASS' if result else 'FAIL'}")
    print()


# =============================================================================
# 5. HMAC-Based Request Signing (HMAC 기반 요청 서명)
# =============================================================================

class RequestSigner:
    """
    Sign and verify API requests using HMAC-SHA256.
    Prevents request tampering and replay attacks.
    """

    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.replay_window = 300  # 5 minutes

    def sign_request(self, method: str, path: str, body: str,
                     timestamp: str | None = None) -> dict[str, str]:
        """Create signature headers for an outgoing request."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Canonical string: method + path + body + timestamp
        canonical = f"{method}\n{path}\n{body}\n{timestamp}"

        signature = hmac.new(
            self.secret_key, canonical.encode(), hashlib.sha256
        ).hexdigest()

        return {
            "X-Timestamp": timestamp,
            "X-Signature": signature,
        }

    def verify_request(self, method: str, path: str, body: str,
                       timestamp: str, signature: str) -> tuple[bool, str]:
        """Verify an incoming signed request."""
        # Check replay window
        try:
            req_time = datetime.fromisoformat(timestamp)
            now = datetime.now(timezone.utc)
            age = abs((now - req_time).total_seconds())
            if age > self.replay_window:
                return False, f"Request expired ({age:.0f}s old, max {self.replay_window}s)"
        except ValueError:
            return False, "Invalid timestamp format"

        # Recompute signature
        canonical = f"{method}\n{path}\n{body}\n{timestamp}"
        expected = hmac.new(
            self.secret_key, canonical.encode(), hashlib.sha256
        ).hexdigest()

        if hmac.compare_digest(signature, expected):
            return True, "Signature valid"
        return False, "Signature mismatch"


def demo_request_signing():
    print("=" * 60)
    print("5. HMAC-Based Request Signing")
    print("=" * 60)

    secret = secrets.token_bytes(32)
    signer = RequestSigner(secret)

    method, path = "POST", "/api/v1/orders"
    body = json.dumps({"item": "widget", "qty": 3})

    # Sign request
    headers = signer.sign_request(method, path, body)
    print(f"\n  Request: {method} {path}")
    print(f"  Body: {body}")
    print(f"  Signature headers:")
    for k, v in headers.items():
        display = v if len(v) <= 40 else v[:37] + "..."
        print(f"    {k}: {display}")

    # Verify valid request
    ok, msg = signer.verify_request(
        method, path, body, headers["X-Timestamp"], headers["X-Signature"]
    )
    print(f"\n  Verify original request:  [{msg}]")

    # Tampered body
    tampered_body = json.dumps({"item": "widget", "qty": 999})
    ok, msg = signer.verify_request(
        method, path, tampered_body, headers["X-Timestamp"], headers["X-Signature"]
    )
    print(f"  Verify tampered body:     [{msg}]")

    # Expired timestamp
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    old_headers = signer.sign_request(method, path, body, timestamp=old_time)
    ok, msg = signer.verify_request(
        method, path, body, old_time, old_headers["X-Signature"]
    )
    print(f"  Verify expired request:   [{msg}]")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  API Security Techniques Demo")
    print("  API 보안 기법 데모")
    print("=" * 60 + "\n")

    demo_rate_limiter()
    demo_input_validation()
    demo_cors()
    demo_api_key()
    demo_request_signing()

    print("=" * 60)
    print("  Demo complete. All examples use stdlib only.")
    print("=" * 60)
