"""
Authentication Mechanisms Demo
==============================

Educational demonstration of authentication concepts:
- JWT (JSON Web Token) creation and verification (manual, no PyJWT)
- TOTP (Time-based One-Time Password) from scratch
- Password strength checker
- Secure token generation for password resets
- Session ID generation

Uses only Python standard library (base64, hmac, hashlib, secrets, time).
No external dependencies required.
"""

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import struct
import time
import string
from datetime import datetime, timezone, timedelta

print("=" * 65)
print("  Authentication Mechanisms Demo")
print("=" * 65)
print()


# ============================================================
# Section 1: JWT (JSON Web Token) - Manual Implementation
# ============================================================

print("-" * 65)
print("  Section 1: JWT (JSON Web Token) Implementation")
print("-" * 65)

print("""
  JWT Structure: header.payload.signature
  - Header:    {"alg": "HS256", "typ": "JWT"}  (base64url)
  - Payload:   Claims (iss, sub, exp, iat, ...)  (base64url)
  - Signature: HMAC-SHA256(header.payload, secret)
""")


def base64url_encode(data: bytes) -> str:
    """Base64url encoding without padding (JWT standard)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(s: str) -> bytes:
    """Base64url decoding with padding restoration."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def jwt_create(payload: dict, secret: str, exp_minutes: int = 60) -> str:
    """Create a JWT token with HS256 signature."""
    # Header
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64url_encode(json.dumps(header, separators=(",", ":")).encode())

    # Add standard claims
    now = int(time.time())
    payload = {
        **payload,
        "iat": now,
        "exp": now + exp_minutes * 60,
    }
    payload_b64 = base64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    # Signature
    signing_input = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    signature_b64 = base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def jwt_verify(token: str, secret: str) -> dict:
    """Verify and decode a JWT token."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format: expected 3 parts")

    header_b64, payload_b64, signature_b64 = parts

    # Verify signature
    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(
        secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    actual_sig = base64url_decode(signature_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid signature")

    # Decode header and verify algorithm
    header = json.loads(base64url_decode(header_b64))
    if header.get("alg") != "HS256":
        raise ValueError(f"Unsupported algorithm: {header.get('alg')}")

    # Decode payload
    payload = json.loads(base64url_decode(payload_b64))

    # Check expiration
    if "exp" in payload and payload["exp"] < int(time.time()):
        raise ValueError("Token has expired")

    return payload


# Demo
jwt_secret = "super-secret-key-change-in-production"
claims = {
    "sub": "user_12345",
    "name": "Alice",
    "role": "admin",
    "iss": "auth.example.com",
}

token = jwt_create(claims, jwt_secret, exp_minutes=30)
print(f"\n  JWT Token:")
parts = token.split(".")
print(f"    Header:    {parts[0][:40]}...")
print(f"    Payload:   {parts[1][:40]}...")
print(f"    Signature: {parts[2]}")
print(f"    Full ({len(token)} chars): {token[:50]}...")
print()

# Decode and display
decoded_header = json.loads(base64url_decode(parts[0]))
decoded_payload = json.loads(base64url_decode(parts[1]))
print(f"  Decoded Header:  {json.dumps(decoded_header)}")
print(f"  Decoded Payload: {json.dumps(decoded_payload, indent=2)}")
print()

# Verify valid token
try:
    verified = jwt_verify(token, jwt_secret)
    print(f"  Verification:    VALID")
    print(f"  User:            {verified['name']} ({verified['sub']})")
    print(f"  Role:            {verified['role']}")
except ValueError as e:
    print(f"  Verification:    FAILED - {e}")

# Verify with wrong secret
try:
    jwt_verify(token, "wrong-secret")
    print(f"  Wrong secret:    VALID (should not happen!)")
except ValueError as e:
    print(f"  Wrong secret:    FAILED - {e}")

# Verify expired token
expired_token = jwt_create(claims, jwt_secret, exp_minutes=-1)
try:
    jwt_verify(expired_token, jwt_secret)
    print(f"  Expired token:   VALID (should not happen!)")
except ValueError as e:
    print(f"  Expired token:   FAILED - {e}")

print()


# ============================================================
# Section 2: TOTP (Time-based One-Time Password)
# ============================================================

print("-" * 65)
print("  Section 2: TOTP (Time-based One-Time Password)")
print("-" * 65)

print("""
  TOTP (RFC 6238) generates 6-digit codes that change every 30s.
  Used by: Google Authenticator, Authy, 1Password, etc.

  Algorithm:
  1. shared_secret = random 20+ bytes (base32 encoded)
  2. counter = floor(current_time / 30)
  3. hmac = HMAC-SHA1(secret, counter_as_8_bytes)
  4. offset = hmac[-1] & 0x0F
  5. code = (hmac[offset:offset+4] & 0x7FFFFFFF) % 10^digits
""")


def generate_totp_secret(length: int = 20) -> bytes:
    """Generate a random TOTP secret."""
    return os.urandom(length)


def totp_generate(secret: bytes, time_step: int = 30, digits: int = 6,
                  timestamp: float = None) -> str:
    """Generate a TOTP code (RFC 6238)."""
    if timestamp is None:
        timestamp = time.time()

    # Step 1: Calculate time counter
    counter = int(timestamp) // time_step

    # Step 2: HMAC-SHA1 of counter
    counter_bytes = struct.pack(">Q", counter)  # 8-byte big-endian
    hmac_digest = hmac.new(secret, counter_bytes, hashlib.sha1).digest()

    # Step 3: Dynamic truncation
    offset = hmac_digest[-1] & 0x0F
    binary_code = struct.unpack(">I", hmac_digest[offset:offset + 4])[0]
    binary_code &= 0x7FFFFFFF  # Remove sign bit

    # Step 4: Modulo to get desired digits
    otp = binary_code % (10 ** digits)
    return str(otp).zfill(digits)


def totp_verify(secret: bytes, code: str, time_step: int = 30,
                window: int = 1) -> bool:
    """Verify a TOTP code with a time window for clock skew."""
    now = time.time()
    for offset in range(-window, window + 1):
        check_time = now + offset * time_step
        expected = totp_generate(secret, time_step, len(code), check_time)
        if hmac.compare_digest(code, expected):
            return True
    return False


# Demo
totp_secret = generate_totp_secret()
secret_b32 = base64.b32encode(totp_secret).decode()

print(f"\n  Secret (base32): {secret_b32}")
print(f"  Secret (hex):    {totp_secret.hex()}")
print()

# Generate current code
current_time = time.time()
current_code = totp_generate(totp_secret, timestamp=current_time)
time_step = 30
remaining = time_step - (int(current_time) % time_step)

print(f"  Current TOTP:    {current_code}")
print(f"  Valid for:       {remaining}s (of {time_step}s window)")
print()

# Show codes across time windows
print("  TOTP codes across time windows:")
for offset in range(-2, 3):
    t = current_time + offset * 30
    code = totp_generate(totp_secret, timestamp=t)
    marker = " <-- current" if offset == 0 else ""
    label = f"T{offset:+d}" if offset != 0 else "T=0"
    print(f"    {label}: {code}{marker}")
print()

# Verify
is_valid = totp_verify(totp_secret, current_code)
is_invalid = totp_verify(totp_secret, "000000")
print(f"  Verify '{current_code}': {is_valid}")
print(f"  Verify '000000': {is_invalid}")
print()

# QR code URL format (for authenticator apps)
account = "user@example.com"
issuer = "MyApp"
otpauth_url = (
    f"otpauth://totp/{issuer}:{account}"
    f"?secret={secret_b32}&issuer={issuer}&algorithm=SHA1&digits=6&period=30"
)
print(f"  QR Code URL (for authenticator apps):")
print(f"    {otpauth_url}")
print()


# ============================================================
# Section 3: Password Strength Checker
# ============================================================

print("-" * 65)
print("  Section 3: Password Strength Checker")
print("-" * 65)


def check_password_strength(password: str) -> dict:
    """Evaluate password strength with detailed feedback."""
    score = 0
    feedback = []
    checks = {}

    # Length check
    length = len(password)
    checks["length"] = length
    if length >= 16:
        score += 3
    elif length >= 12:
        score += 2
    elif length >= 8:
        score += 1
    else:
        feedback.append("Use at least 8 characters (12+ recommended)")

    # Character class checks
    has_lower = bool(re.search(r"[a-z]", password))
    has_upper = bool(re.search(r"[A-Z]", password))
    has_digit = bool(re.search(r"\d", password))
    has_special = bool(re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password))

    checks["lowercase"] = has_lower
    checks["uppercase"] = has_upper
    checks["digits"] = has_digit
    checks["special"] = has_special

    char_classes = sum([has_lower, has_upper, has_digit, has_special])
    score += char_classes
    if not has_lower:
        feedback.append("Add lowercase letters")
    if not has_upper:
        feedback.append("Add uppercase letters")
    if not has_digit:
        feedback.append("Add numbers")
    if not has_special:
        feedback.append("Add special characters (!@#$%^&*...)")

    # Common pattern checks
    common_patterns = [
        (r"(.)\1{2,}", "Avoid repeated characters (aaa, 111)"),
        (r"(012|123|234|345|456|567|678|789)", "Avoid sequential numbers"),
        (r"(abc|bcd|cde|def|efg|fgh|ghi)", "Avoid sequential letters"),
        (r"(?i)(password|qwerty|admin|login|welcome)", "Avoid common words"),
    ]

    for pattern, message in common_patterns:
        if re.search(pattern, password):
            score -= 1
            feedback.append(message)

    # Entropy estimation
    pool_size = 0
    if has_lower:
        pool_size += 26
    if has_upper:
        pool_size += 26
    if has_digit:
        pool_size += 10
    if has_special:
        pool_size += 32
    entropy = length * math.log2(pool_size) if pool_size > 0 else 0
    checks["entropy_bits"] = round(entropy, 1)

    # Determine strength level
    score = max(0, min(score, 7))
    if score >= 6:
        strength = "STRONG"
    elif score >= 4:
        strength = "MODERATE"
    elif score >= 2:
        strength = "WEAK"
    else:
        strength = "VERY WEAK"

    return {
        "password": password[:2] + "*" * (len(password) - 2),
        "score": score,
        "max_score": 7,
        "strength": strength,
        "entropy_bits": checks["entropy_bits"],
        "checks": checks,
        "feedback": feedback if feedback else ["Good password!"],
    }


# Test various passwords
test_passwords = [
    "password",
    "Admin123",
    "MyD0g$N@me!",
    "correct-horse-battery-staple",
    "Tr0ub4dor&3",
    "j&Hx9#mK2$pL@nQ!",
]

print()
for pwd in test_passwords:
    result = check_password_strength(pwd)
    bar = "#" * result["score"] + "." * (result["max_score"] - result["score"])
    print(f"  Password: {result['password']:<22} [{bar}] {result['strength']}")
    print(f"    Entropy: {result['entropy_bits']} bits, Score: {result['score']}/{result['max_score']}")
    if result["feedback"] and result["feedback"][0] != "Good password!":
        for fb in result["feedback"][:2]:
            print(f"    - {fb}")
    print()

print("  Entropy benchmarks:")
print("    < 28 bits:  Trivially crackable")
print("    28-35 bits: Crackable with effort")
print("    36-59 bits: Reasonable for online attacks")
print("    60-127 bits: Strong against offline attacks")
print("    128+ bits:  Computationally infeasible")
print()


# ============================================================
# Section 4: Secure Token Generation
# ============================================================

print("-" * 65)
print("  Section 4: Secure Token Generation")
print("-" * 65)

print("""
  Tokens must be cryptographically random (not predictable).
  Python's `secrets` module uses OS-level CSPRNG.
""")

# Password reset token
reset_token = secrets.token_urlsafe(32)
print(f"  Password Reset Token:  {reset_token}")
print(f"    Length:              {len(reset_token)} chars")
print(f"    Entropy:             256 bits")
print()

# Email verification token
email_token = secrets.token_hex(16)
print(f"  Email Verify Token:    {email_token}")
print(f"    Length:              {len(email_token)} chars")
print()

# API key generation
def generate_api_key(prefix: str = "sk") -> str:
    """Generate an API key with prefix (like Stripe sk_live_...)."""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"

api_key = generate_api_key("sk_live")
print(f"  API Key:               {api_key}")
print()

# Token with expiration (stored in DB)
def create_token_record(purpose: str, ttl_hours: int = 24) -> dict:
    """Create a token record for database storage."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    return {
        "token_plaintext": token,  # Send to user (email, etc.)
        "token_hash": token_hash,  # Store in DB (never store plaintext!)
        "purpose": purpose,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=ttl_hours)).isoformat(),
    }

record = create_token_record("password_reset", ttl_hours=1)
print(f"  Token Record (for DB storage):")
print(f"    Plaintext (sent):  {record['token_plaintext'][:30]}...")
print(f"    Hash (stored):     {record['token_hash'][:32]}...")
print(f"    Purpose:           {record['purpose']}")
print(f"    Expires:           {record['expires_at']}")
print()
print("  IMPORTANT: Store only the HASH in the database.")
print("  When user presents token, hash it and compare to stored hash.")
print()


# ============================================================
# Section 5: Session ID Generation
# ============================================================

print("-" * 65)
print("  Section 5: Session ID Generation")
print("-" * 65)

print("""
  Session IDs must be:
  - Cryptographically random (unpredictable)
  - Sufficiently long (128+ bits of entropy)
  - Unique across all active sessions
  - Regenerated after authentication changes
""")


def generate_session_id() -> str:
    """Generate a secure session ID (128 bits of entropy)."""
    return secrets.token_hex(16)


def generate_session_with_metadata() -> dict:
    """Create a session with metadata for server-side storage."""
    session_id = secrets.token_hex(32)
    return {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_accessed": datetime.now(timezone.utc).isoformat(),
        "ip_address": "192.168.1.100",  # From request
        "user_agent": "Mozilla/5.0...",  # From request headers
        "user_id": None,  # Set after login
        "is_authenticated": False,
    }


session = generate_session_with_metadata()
print(f"\n  Session ID:      {session['session_id'][:32]}...")
print(f"  Created:         {session['created_at']}")
print(f"  Authenticated:   {session['is_authenticated']}")
print()

# Show multiple unique session IDs
print("  Sample session IDs (all unique):")
for i in range(5):
    sid = generate_session_id()
    print(f"    {i+1}. {sid}")
print()

# Session cookie attributes
print("  Secure session cookie attributes:")
print("    Set-Cookie: session_id=<token>;")
print("                HttpOnly;    -- no JavaScript access")
print("                Secure;      -- HTTPS only")
print("                SameSite=Lax; -- CSRF protection")
print("                Path=/;")
print("                Max-Age=3600; -- 1 hour expiry")
print()


# ============================================================
# Section 6: Summary
# ============================================================

print("=" * 65)
print("  Summary")
print("=" * 65)
print("""
  Mechanism        | Use Case              | Key Points
  -----------------+-----------------------+---------------------------
  JWT (HS256)      | Stateless auth        | Short-lived, secret key
  TOTP             | 2FA / MFA             | Time-based, 30s window
  Password check   | Registration/update   | Entropy, patterns, length
  Reset tokens     | Password recovery     | Hash before storing
  Session IDs      | Stateful auth         | Random, HttpOnly, Secure

  Authentication Best Practices:
  - Always use MFA/2FA (TOTP or WebAuthn)
  - Hash password reset tokens before DB storage
  - Set short expiration for sensitive tokens
  - Regenerate session IDs after login/logout
  - Use HttpOnly + Secure + SameSite cookies
  - Rate-limit authentication endpoints
  - Log all authentication events
""")
