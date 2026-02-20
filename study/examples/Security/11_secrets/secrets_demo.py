"""
시크릿 관리 데모
Secrets Management Demo

환경 변수 로딩, 시크릿 강도 검증, Git secrets 패턴 스캐너,
설정 파일 암호화, .env 파일 파서 등 시크릿 관리 기법을 구현합니다.

Demonstrates environment variable loading, secret strength validation,
git secrets pattern scanning, config file encryption, and .env file parsing.
"""

import os
import re
import string
import math
import hashlib
import hmac
import base64
import json
import tempfile
from pathlib import Path


# =============================================================================
# 1. Environment Variable Loading Simulation (환경 변수 로딩 시뮬레이션)
# =============================================================================

class EnvLoader:
    """
    Load configuration from environment variables with defaults and validation.
    Simulates best practices for 12-factor app config management.
    """

    def __init__(self):
        self.loaded: dict[str, str] = {}
        self.missing: list[str] = []

    def get(self, key: str, default: str | None = None,
            required: bool = False) -> str | None:
        """Retrieve an environment variable with optional default."""
        value = os.environ.get(key, default)
        if value is not None:
            self.loaded[key] = "(set)" if "SECRET" in key or "KEY" in key else value
        elif required:
            self.missing.append(key)
        return value

    def report(self) -> str:
        lines = ["  Environment Variable Report:"]
        for k, v in self.loaded.items():
            lines.append(f"    {k:30s} = {v}")
        if self.missing:
            lines.append(f"\n    MISSING (required): {', '.join(self.missing)}")
        return "\n".join(lines)


def demo_env_loading():
    print("=" * 60)
    print("1. Environment Variable Loading")
    print("=" * 60)

    # Set some simulated env vars for demo
    os.environ["APP_DB_HOST"] = "localhost"
    os.environ["APP_DB_PORT"] = "5432"
    os.environ["APP_SECRET_KEY"] = "demo-secret-abc123"

    loader = EnvLoader()
    loader.get("APP_DB_HOST", required=True)
    loader.get("APP_DB_PORT", default="5432")
    loader.get("APP_SECRET_KEY", required=True)
    loader.get("APP_DB_PASSWORD", required=True)  # Will be missing
    loader.get("APP_DEBUG", default="false")

    print(f"\n{loader.report()}")

    # Cleanup
    for key in ["APP_DB_HOST", "APP_DB_PORT", "APP_SECRET_KEY"]:
        os.environ.pop(key, None)
    print()


# =============================================================================
# 2. Secret Strength Validation (시크릿 강도 검증)
# =============================================================================

def calculate_entropy(secret: str) -> float:
    """Calculate Shannon entropy (bits) of a string."""
    if not secret:
        return 0.0
    freq: dict[str, int] = {}
    for ch in secret:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(secret)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )
    return entropy * length  # Total bits


def validate_secret_strength(secret: str) -> dict:
    """Evaluate the strength of a secret/password/API key."""
    checks = {
        "length >= 16": len(secret) >= 16,
        "has uppercase": bool(re.search(r'[A-Z]', secret)),
        "has lowercase": bool(re.search(r'[a-z]', secret)),
        "has digits": bool(re.search(r'[0-9]', secret)),
        "has special chars": bool(re.search(r'[^a-zA-Z0-9]', secret)),
        "no common patterns": not re.search(
            r'(password|secret|admin|12345|qwerty)', secret, re.IGNORECASE
        ),
    }
    entropy = calculate_entropy(secret)

    passed = sum(checks.values())
    total = len(checks)

    if passed == total and entropy >= 60:
        grade = "STRONG"
    elif passed >= 4 and entropy >= 40:
        grade = "MODERATE"
    else:
        grade = "WEAK"

    return {"checks": checks, "entropy_bits": entropy, "grade": grade}


def demo_secret_strength():
    print("=" * 60)
    print("2. Secret Strength Validation")
    print("=" * 60)

    test_secrets = [
        "password123",
        "MyS3cret!",
        "xK9#mPq2$vL7nR4@wB6j",
        "aaaaaaaaaaaaaaaa",
        "Tr0ub4dor&3",
    ]

    for secret in test_secrets:
        result = validate_secret_strength(secret)
        display = secret if len(secret) <= 25 else secret[:22] + "..."
        print(f"\n  Secret: {display}")
        print(f"    Entropy: {result['entropy_bits']:.1f} bits | Grade: {result['grade']}")
        for check, passed in result["checks"].items():
            symbol = "OK" if passed else "--"
            print(f"      [{symbol}] {check}")
    print()


# =============================================================================
# 3. Git Secrets Pattern Scanner (Git 시크릿 패턴 스캐너)
# =============================================================================

# Patterns that indicate leaked secrets in code
SECRET_PATTERNS = [
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{16,}',
     "API Key"),
    (r'(?i)(secret[_-]?key|secret)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{8,}',
     "Secret Key"),
    (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']{4,}',
     "Password"),
    (r'(?i)aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*["\']?AKIA[A-Z0-9]{16}',
     "AWS Access Key"),
    (r'(?i)ghp_[A-Za-z0-9]{36}',
     "GitHub Personal Access Token"),
    (r'(?i)sk-[A-Za-z0-9]{32,}',
     "OpenAI API Key Pattern"),
    (r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----',
     "Private Key"),
    (r'(?i)(jdbc|mysql|postgres)://[^\s]+:[^\s]+@',
     "Database Connection String with Credentials"),
]


def scan_text_for_secrets(text: str, filename: str = "<input>") -> list[dict]:
    """Scan text content for potential secret patterns."""
    findings = []
    for line_num, line in enumerate(text.splitlines(), 1):
        for pattern, label in SECRET_PATTERNS:
            if re.search(pattern, line):
                findings.append({
                    "file": filename,
                    "line": line_num,
                    "type": label,
                    "content": line.strip()[:80],
                })
    return findings


def demo_git_secrets_scanner():
    print("=" * 60)
    print("3. Git Secrets Pattern Scanner")
    print("=" * 60)

    sample_code = '''
# config.py
DATABASE_URL = "postgres://admin:SuperSecret123@db.example.com/mydb"
API_KEY = "sk-abc123def456ghi789jkl012mno345pqr678"
DEBUG = True

# Safe reference (no actual secret)
password = os.environ.get("DB_PASSWORD")

# Dangerous: hardcoded AWS key
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
ghp_EXAMPLE_TOKEN_NOT_REAL_REPLACE_ME_1234

-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF...
    '''

    print("\n  Scanning sample code for secret patterns...\n")
    findings = scan_text_for_secrets(sample_code, "config.py")

    if findings:
        for f in findings:
            print(f"    [{f['type']}]")
            print(f"      File: {f['file']}  Line: {f['line']}")
            content = f['content'] if len(f['content']) <= 60 else f['content'][:57] + "..."
            print(f"      Content: {content}\n")
        print(f"  Total findings: {len(findings)}")
    else:
        print("  No secrets detected.")
    print()


# =============================================================================
# 4. Config File Encryption (설정 파일 암호화)
# =============================================================================
# Uses a simple XOR-based approach as a stdlib-only fallback.
# In production, use the 'cryptography' library with Fernet.

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from password using PBKDF2."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)


def encrypt_config(config_data: dict, password: str) -> bytes:
    """Encrypt a config dictionary using PBKDF2 + HMAC-authenticated XOR stream."""
    salt = os.urandom(16)
    key = derive_key(password, salt)

    plaintext = json.dumps(config_data).encode()

    # Generate keystream via SHA-256 counter mode
    ciphertext = bytearray()
    for i in range(0, len(plaintext), 32):
        block_key = hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        chunk = plaintext[i:i + 32]
        ciphertext.extend(b ^ k for b, k in zip(chunk, block_key))

    # HMAC for integrity
    mac = hmac.new(key, bytes(ciphertext), hashlib.sha256).digest()

    # Format: salt (16) + mac (32) + ciphertext
    return salt + mac + bytes(ciphertext)


def decrypt_config(encrypted: bytes, password: str) -> dict | None:
    """Decrypt an encrypted config. Returns None if integrity check fails."""
    salt = encrypted[:16]
    stored_mac = encrypted[16:48]
    ciphertext = encrypted[48:]

    key = derive_key(password, salt)

    # Verify HMAC first
    computed_mac = hmac.new(key, ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(stored_mac, computed_mac):
        return None  # Integrity check failed

    # Decrypt
    plaintext = bytearray()
    for i in range(0, len(ciphertext), 32):
        block_key = hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        chunk = ciphertext[i:i + 32]
        plaintext.extend(b ^ k for b, k in zip(chunk, block_key))

    return json.loads(bytes(plaintext))


def demo_config_encryption():
    print("=" * 60)
    print("4. Config File Encryption (stdlib-only)")
    print("=" * 60)

    config = {
        "database_url": "postgres://user:pass@host/db",
        "api_secret": "my-super-secret-key-12345",
        "smtp_password": "mail_pass_789",
    }

    password = "master-encryption-password"

    print(f"\n  Original config keys: {list(config.keys())}")
    encrypted = encrypt_config(config, password)
    print(f"  Encrypted size: {len(encrypted)} bytes")
    print(f"  Encrypted (b64, first 60 chars): {base64.b64encode(encrypted).decode()[:60]}...")

    # Decrypt with correct password
    decrypted = decrypt_config(encrypted, password)
    print(f"\n  Decrypted with correct password: {decrypted is not None}")
    if decrypted:
        print(f"    Keys recovered: {list(decrypted.keys())}")
        print(f"    Values match: {decrypted == config}")

    # Decrypt with wrong password
    bad_result = decrypt_config(encrypted, "wrong-password")
    print(f"  Decrypted with wrong password:  {bad_result is not None} (integrity check)")
    print()


# =============================================================================
# 5. .env File Parser (.env 파일 파서)
# =============================================================================

def parse_env_file(content: str) -> dict[str, str]:
    """
    Parse a .env file content into a dictionary.
    Supports comments, quoted values, and export prefix.
    """
    env_vars: dict[str, str] = {}

    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Remove optional 'export ' prefix
        if line.startswith("export "):
            line = line[7:]

        # Split on first '='
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Remove surrounding quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        # Validate key format
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
            env_vars[key] = value

    return env_vars


def demo_env_parser():
    print("=" * 60)
    print("5. .env File Parser")
    print("=" * 60)

    sample_env = """
# Application Configuration
APP_NAME="My Secure App"
APP_ENV=production
APP_DEBUG=false

# Database
export DB_HOST=localhost
DB_PORT=5432
DB_USER='admin'
DB_PASSWORD="s3cur3_p@ss!"

# API Keys
API_KEY=abc123def456
SECRET_KEY="my-secret-key-value"

# Invalid lines (skipped)
not a valid line
123INVALID=starts_with_digit
"""

    print(f"\n  Parsing sample .env file...\n")
    parsed = parse_env_file(sample_env)

    for key, value in parsed.items():
        # Mask sensitive values
        if any(s in key.upper() for s in ["PASSWORD", "SECRET", "KEY"]):
            display = value[:3] + "*" * (len(value) - 3)
        else:
            display = value
        print(f"    {key:20s} = {display}")

    print(f"\n  Total variables parsed: {len(parsed)}")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Secrets Management Demo")
    print("  시크릿 관리 데모")
    print("=" * 60 + "\n")

    demo_env_loading()
    demo_secret_strength()
    demo_git_secrets_scanner()
    demo_config_encryption()
    demo_env_parser()

    print("=" * 60)
    print("  Demo complete. All examples use stdlib only.")
    print("=" * 60)
