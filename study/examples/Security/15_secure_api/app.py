"""
보안 API 서버 데모 (stdlib only)
Secure API Server Demo (stdlib only)

패스워드 해싱(PBKDF2), JWT 인증, RBAC, 입력 검증, 보안 헤더,
레이트 리미팅, 구조화 로깅을 포함하는 완전한 보안 API 서버를 구현합니다.
http.server 기반으로 외부 의존성 없이 동작합니다.

Implements a complete secure API server with password hashing (PBKDF2),
JWT authentication, RBAC, input validation, security headers, rate limiting,
and structured logging. Uses only Python stdlib (http.server).
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import time
import base64
import logging
import sys
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from functools import wraps


# =============================================================================
# Configuration (설정)
# =============================================================================

JWT_SECRET = secrets.token_bytes(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 30
BCRYPT_ITERATIONS = 100_000
SERVER_PORT = 0  # Assigned dynamically for demo

# Structured logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger("secure_api")


# =============================================================================
# 1. Password Hashing (패스워드 해싱 - PBKDF2)
# =============================================================================

class PasswordHasher:
    """Hash and verify passwords using PBKDF2-HMAC-SHA256."""

    ITERATIONS = BCRYPT_ITERATIONS
    SALT_LENGTH = 16
    KEY_LENGTH = 32

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password, returning 'salt$hash' as hex strings."""
        salt = os.urandom(PasswordHasher.SALT_LENGTH)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt,
            PasswordHasher.ITERATIONS, dklen=PasswordHasher.KEY_LENGTH,
        )
        return salt.hex() + "$" + dk.hex()

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        """Verify a password against a stored hash."""
        try:
            salt_hex, hash_hex = stored.split("$")
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(hash_hex)
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt,
                PasswordHasher.ITERATIONS, dklen=PasswordHasher.KEY_LENGTH,
            )
            return hmac.compare_digest(dk, expected)
        except (ValueError, AttributeError):
            return False


# =============================================================================
# 2. JWT Implementation (JWT 구현)
# =============================================================================

class JWT:
    """Minimal JWT (HS256) implementation using stdlib only."""

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)

    @staticmethod
    def encode(payload: dict, secret: bytes) -> str:
        """Create a JWT token."""
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = JWT._b64url_encode(json.dumps(header).encode())
        payload_b64 = JWT._b64url_encode(json.dumps(payload).encode())

        message = f"{header_b64}.{payload_b64}".encode()
        signature = hmac.new(secret, message, hashlib.sha256).digest()
        sig_b64 = JWT._b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{sig_b64}"

    @staticmethod
    def decode(token: str, secret: bytes) -> dict | None:
        """Decode and verify a JWT token. Returns None if invalid."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, sig_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}".encode()
            expected_sig = hmac.new(secret, message, hashlib.sha256).digest()
            actual_sig = JWT._b64url_decode(sig_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            # Decode payload
            payload = json.loads(JWT._b64url_decode(payload_b64))

            # Check expiration
            if "exp" in payload:
                exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_time:
                    return None

            return payload
        except Exception:
            return None


# =============================================================================
# 3. RBAC - Role-Based Access Control (역할 기반 접근 제어)
# =============================================================================

class RBAC:
    """Simple role-based access control system."""

    # Role hierarchy: admin > editor > viewer
    PERMISSIONS = {
        "admin":  {"read", "write", "delete", "admin"},
        "editor": {"read", "write"},
        "viewer": {"read"},
    }

    @staticmethod
    def has_permission(role: str, required_permission: str) -> bool:
        perms = RBAC.PERMISSIONS.get(role, set())
        return required_permission in perms

    @staticmethod
    def require_permission(role: str, permission: str) -> tuple[bool, str]:
        if RBAC.has_permission(role, permission):
            return True, f"Access granted: role '{role}' has '{permission}'"
        return False, f"Access denied: role '{role}' lacks '{permission}'"


# =============================================================================
# 4. Input Validation (입력 검증)
# =============================================================================

class Validator:
    """Input validation for API requests."""

    @staticmethod
    def validate_username(value: str) -> tuple[bool, str]:
        if not value or not re.match(r'^[a-zA-Z0-9_]{3,30}$', value):
            return False, "Username must be 3-30 alphanumeric/underscore characters"
        return True, ""

    @staticmethod
    def validate_password(value: str) -> tuple[bool, str]:
        if len(value) < 8:
            return False, "Password must be at least 8 characters"
        if not re.search(r'[A-Z]', value):
            return False, "Password must contain an uppercase letter"
        if not re.search(r'[0-9]', value):
            return False, "Password must contain a digit"
        return True, ""

    @staticmethod
    def validate_email(value: str) -> tuple[bool, str]:
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return False, "Invalid email format"
        return True, ""

    @staticmethod
    def sanitize(value: str, max_len: int = 200) -> str:
        value = value[:max_len]
        value = re.sub(r'[\x00-\x1f\x7f]', '', value)
        return value


# =============================================================================
# 5. Rate Limiter (레이트 리미터)
# =============================================================================

class RateLimiter:
    """Per-IP sliding window rate limiter."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.monotonic()
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Remove expired entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < self.window
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        self.requests[client_ip].append(now)
        return True

    def remaining(self, client_ip: str) -> int:
        now = time.monotonic()
        recent = [t for t in self.requests.get(client_ip, []) if now - t < self.window]
        return max(0, self.max_requests - len(recent))


# =============================================================================
# 6. Security Headers (보안 헤더)
# =============================================================================

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",  # Modern browsers: use CSP instead
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


# =============================================================================
# 7. In-Memory User Store (메모리 기반 사용자 저장소)
# =============================================================================

class UserStore:
    """Simple in-memory user database for demo purposes."""

    def __init__(self):
        self.users: dict[str, dict] = {}
        self.hasher = PasswordHasher()

    def register(self, username: str, password: str, email: str,
                 role: str = "viewer") -> tuple[bool, str]:
        if username in self.users:
            return False, "Username already exists"

        ok, msg = Validator.validate_username(username)
        if not ok:
            return False, msg
        ok, msg = Validator.validate_password(password)
        if not ok:
            return False, msg
        ok, msg = Validator.validate_email(email)
        if not ok:
            return False, msg

        self.users[username] = {
            "password_hash": self.hasher.hash_password(password),
            "email": email,
            "role": role,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        return True, "User registered successfully"

    def authenticate(self, username: str, password: str) -> dict | None:
        user = self.users.get(username)
        if not user:
            # Constant-time comparison to prevent timing attacks
            self.hasher.verify_password(password, "0" * 32 + "$" + "0" * 64)
            return None
        if self.hasher.verify_password(password, user["password_hash"]):
            return {"username": username, "role": user["role"], "email": user["email"]}
        return None

    def get_user(self, username: str) -> dict | None:
        user = self.users.get(username)
        if user:
            return {"username": username, "role": user["role"], "email": user["email"]}
        return None


# =============================================================================
# 8. Structured Logging (구조화 로깅)
# =============================================================================

class SecurityLogger:
    """Structured logging for security events."""

    @staticmethod
    def log_auth_attempt(username: str, success: bool, ip: str):
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"AUTH | {status} | user={username} ip={ip}")

    @staticmethod
    def log_access(method: str, path: str, ip: str, status: int, user: str = "anonymous"):
        logger.info(f"ACCESS | {method} {path} | status={status} user={user} ip={ip}")

    @staticmethod
    def log_rate_limit(ip: str, path: str):
        logger.warning(f"RATE_LIMIT | ip={ip} path={path}")

    @staticmethod
    def log_validation_error(field: str, error: str, ip: str):
        logger.warning(f"VALIDATION | field={field} error={error} ip={ip}")


# =============================================================================
# 9. API Request Handler (API 요청 핸들러)
# =============================================================================

# Global state for the handler
user_store = UserStore()
rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
sec_logger = SecurityLogger()


class SecureAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler with security features."""

    def _get_client_ip(self) -> str:
        return self.client_address[0] if self.client_address else "unknown"

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        remaining = rate_limiter.remaining(self._get_client_ip())
        self.send_header("X-RateLimit-Remaining", str(remaining))
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > 10_000:  # Max 10KB body
            return {}
        try:
            return json.loads(self.rfile.read(length)) if length > 0 else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _get_current_user(self) -> dict | None:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth[7:]
        payload = JWT.decode(token, JWT_SECRET)
        if payload and "sub" in payload:
            return user_store.get_user(payload["sub"])
        return None

    def _check_rate_limit(self) -> bool:
        ip = self._get_client_ip()
        if not rate_limiter.is_allowed(ip):
            sec_logger.log_rate_limit(ip, self.path)
            self._send_json(429, {"error": "Too many requests", "retry_after": 60})
            return False
        return True

    # --- Routes ---

    def do_GET(self):
        if not self._check_rate_limit():
            return
        ip = self._get_client_ip()

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._handle_health()
        elif path == "/api/profile":
            self._handle_profile()
        elif path == "/api/users":
            self._handle_list_users()
        else:
            self._send_json(404, {"error": "Not found"})
            sec_logger.log_access("GET", path, ip, 404)

    def do_POST(self):
        if not self._check_rate_limit():
            return
        ip = self._get_client_ip()

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/register":
            self._handle_register()
        elif path == "/api/login":
            self._handle_login()
        elif path == "/api/data":
            self._handle_create_data()
        else:
            self._send_json(404, {"error": "Not found"})
            sec_logger.log_access("POST", path, ip, 404)

    def do_DELETE(self):
        if not self._check_rate_limit():
            return
        ip = self._get_client_ip()

        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/users/"):
            self._handle_delete_user(path)
        else:
            self._send_json(404, {"error": "Not found"})
            sec_logger.log_access("DELETE", path, ip, 404)

    # --- Handlers ---

    def _handle_health(self):
        self._send_json(200, {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})

    def _handle_register(self):
        ip = self._get_client_ip()
        body = self._read_body()

        username = Validator.sanitize(body.get("username", ""))
        password = body.get("password", "")
        email = Validator.sanitize(body.get("email", ""))

        ok, msg = user_store.register(username, password, email)
        if ok:
            sec_logger.log_access("POST", "/api/register", ip, 201, username)
            self._send_json(201, {"message": msg, "username": username})
        else:
            sec_logger.log_validation_error("registration", msg, ip)
            self._send_json(400, {"error": msg})

    def _handle_login(self):
        ip = self._get_client_ip()
        body = self._read_body()

        username = body.get("username", "")
        password = body.get("password", "")

        user = user_store.authenticate(username, password)
        if user:
            payload = {
                "sub": username,
                "role": user["role"],
                "exp": int((datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES)).timestamp()),
                "iat": int(datetime.now(timezone.utc).timestamp()),
            }
            token = JWT.encode(payload, JWT_SECRET)
            sec_logger.log_auth_attempt(username, True, ip)
            self._send_json(200, {"token": token, "expires_in": JWT_EXPIRY_MINUTES * 60})
        else:
            sec_logger.log_auth_attempt(username, False, ip)
            self._send_json(401, {"error": "Invalid credentials"})

    def _handle_profile(self):
        ip = self._get_client_ip()
        user = self._get_current_user()
        if not user:
            self._send_json(401, {"error": "Authentication required"})
            return
        sec_logger.log_access("GET", "/api/profile", ip, 200, user["username"])
        self._send_json(200, {"user": user})

    def _handle_list_users(self):
        ip = self._get_client_ip()
        user = self._get_current_user()
        if not user:
            self._send_json(401, {"error": "Authentication required"})
            return

        ok, msg = RBAC.require_permission(user["role"], "admin")
        if not ok:
            sec_logger.log_access("GET", "/api/users", ip, 403, user["username"])
            self._send_json(403, {"error": msg})
            return

        users = [user_store.get_user(u) for u in user_store.users]
        sec_logger.log_access("GET", "/api/users", ip, 200, user["username"])
        self._send_json(200, {"users": users})

    def _handle_create_data(self):
        ip = self._get_client_ip()
        user = self._get_current_user()
        if not user:
            self._send_json(401, {"error": "Authentication required"})
            return

        ok, msg = RBAC.require_permission(user["role"], "write")
        if not ok:
            sec_logger.log_access("POST", "/api/data", ip, 403, user["username"])
            self._send_json(403, {"error": msg})
            return

        body = self._read_body()
        sec_logger.log_access("POST", "/api/data", ip, 201, user["username"])
        self._send_json(201, {"message": "Data created", "data": body})

    def _handle_delete_user(self, path: str):
        ip = self._get_client_ip()
        user = self._get_current_user()
        if not user:
            self._send_json(401, {"error": "Authentication required"})
            return

        ok, msg = RBAC.require_permission(user["role"], "admin")
        if not ok:
            sec_logger.log_access("DELETE", path, ip, 403, user["username"])
            self._send_json(403, {"error": msg})
            return

        target = path.split("/")[-1]
        if target in user_store.users:
            del user_store.users[target]
            sec_logger.log_access("DELETE", path, ip, 200, user["username"])
            self._send_json(200, {"message": f"User '{target}' deleted"})
        else:
            self._send_json(404, {"error": "User not found"})

    def log_message(self, format, *args):
        """Suppress default HTTP server logging (we use our own)."""
        pass


# =============================================================================
# 10. Demo: Simulate API Interactions (데모: API 상호작용 시뮬레이션)
# =============================================================================

def run_demo():
    """Demonstrate all security features without starting a real server."""
    print("=" * 60)
    print("  Secure API Server Demo (Simulation)")
    print("  보안 API 서버 데모 (시뮬레이션)")
    print("=" * 60)

    # --- Password Hashing ---
    print("\n" + "=" * 60)
    print("1. Password Hashing (PBKDF2-HMAC-SHA256)")
    print("=" * 60)

    hasher = PasswordHasher()
    pw = "MyStr0ng!Pass"
    hashed = hasher.hash_password(pw)
    print(f"\n  Password: {pw}")
    print(f"  Hash:     {hashed[:20]}...{hashed[-20:]}")
    print(f"  Verify correct:  {hasher.verify_password(pw, hashed)}")
    print(f"  Verify wrong:    {hasher.verify_password('wrong', hashed)}")

    # --- JWT ---
    print("\n" + "=" * 60)
    print("2. JWT Token (HS256)")
    print("=" * 60)

    payload = {
        "sub": "alice",
        "role": "admin",
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    token = JWT.encode(payload, JWT_SECRET)
    print(f"\n  Payload: {json.dumps(payload, indent=2)}")
    print(f"  Token:   {token[:40]}...{token[-20:]}")

    decoded = JWT.decode(token, JWT_SECRET)
    print(f"  Decode valid token:  sub={decoded['sub']}, role={decoded['role']}")

    bad_decode = JWT.decode(token + "tampered", JWT_SECRET)
    print(f"  Decode tampered:     {bad_decode}")

    expired_payload = {**payload, "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())}
    expired_token = JWT.encode(expired_payload, JWT_SECRET)
    exp_decode = JWT.decode(expired_token, JWT_SECRET)
    print(f"  Decode expired:      {exp_decode}")

    # --- RBAC ---
    print("\n" + "=" * 60)
    print("3. Role-Based Access Control")
    print("=" * 60)

    roles = ["admin", "editor", "viewer"]
    perms = ["read", "write", "delete", "admin"]
    print(f"\n  {'Role':<10} | {'read':>5} | {'write':>5} | {'delete':>6} | {'admin':>5}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}")
    for role in roles:
        row = f"  {role:<10}"
        for perm in perms:
            allowed = RBAC.has_permission(role, perm)
            row += f" | {'YES':>{len(perm)}}" if allowed else f" | {'-':>{len(perm)}}"
        print(row)

    # --- User Registration & Auth Flow ---
    print("\n" + "=" * 60)
    print("4. Registration & Authentication Flow")
    print("=" * 60)

    store = UserStore()

    # Register users
    users_to_register = [
        ("alice", "Str0ng!Admin", "alice@example.com", "admin"),
        ("bob", "B0bEdit0r!", "bob@example.com", "editor"),
        ("charlie", "Ch@rlie99", "charlie@example.com", "viewer"),
    ]

    print("\n  Registration:")
    for uname, pw, email, role in users_to_register:
        ok, msg = store.register(uname, pw, email, role)
        print(f"    {uname:10s} ({role:6s}): {msg}")

    # Duplicate registration
    ok, msg = store.register("alice", "Another!1", "a@b.com")
    print(f"    {'alice':10s} (dup)  : {msg}")

    # Weak password
    ok, msg = store.register("weak_user", "short", "w@b.com")
    print(f"    {'weak_user':10s} (weak) : {msg}")

    # Authentication
    print("\n  Authentication:")
    for uname, pw in [("alice", "Str0ng!Admin"), ("bob", "WrongPass!"), ("nonexist", "any")]:
        user = store.authenticate(uname, pw)
        status = f"role={user['role']}" if user else "FAILED"
        print(f"    {uname:10s}: {status}")

    # --- Input Validation ---
    print("\n" + "=" * 60)
    print("5. Input Validation")
    print("=" * 60)

    test_inputs = [
        ("username", "valid_user", Validator.validate_username),
        ("username", "ab", Validator.validate_username),
        ("password", "Str0ng!Pass", Validator.validate_password),
        ("password", "weak", Validator.validate_password),
        ("email", "user@example.com", Validator.validate_email),
        ("email", "not-an-email", Validator.validate_email),
    ]

    print()
    for field, value, fn in test_inputs:
        ok, msg = fn(value)
        status = "OK" if ok else f"FAIL: {msg}"
        print(f"    {field:10s} = {value:20s} -> {status}")

    # Sanitization
    dangerous = '<script>alert("xss")</script>'
    print(f"\n    Sanitize: '{dangerous[:30]}...' -> '{Validator.sanitize(dangerous)[:40]}...'")

    # --- Security Headers ---
    print("\n" + "=" * 60)
    print("6. Security Response Headers")
    print("=" * 60)

    print()
    for header, value in SECURITY_HEADERS.items():
        print(f"    {header}: {value}")

    # --- Rate Limiting ---
    print("\n" + "=" * 60)
    print("7. Rate Limiting")
    print("=" * 60)

    rl = RateLimiter(max_requests=5, window_seconds=60)
    print(f"\n  Config: max=5 requests per 60 seconds\n")
    for i in range(1, 8):
        allowed = rl.is_allowed("192.168.1.1")
        remaining = rl.remaining("192.168.1.1")
        status = "ALLOWED" if allowed else "RATE LIMITED"
        print(f"    Request {i}: {status} (remaining: {remaining})")

    # --- Structured Logging ---
    print("\n" + "=" * 60)
    print("8. Structured Security Logging")
    print("=" * 60)
    print()

    sec_logger.log_auth_attempt("alice", True, "10.0.0.1")
    sec_logger.log_auth_attempt("hacker", False, "10.0.0.99")
    sec_logger.log_access("GET", "/api/profile", "10.0.0.1", 200, "alice")
    sec_logger.log_access("DELETE", "/api/users/bob", "10.0.0.99", 403, "hacker")
    sec_logger.log_rate_limit("10.0.0.99", "/api/login")
    sec_logger.log_validation_error("password", "too short", "10.0.0.50")

    # --- Full Flow Summary ---
    print("\n" + "=" * 60)
    print("9. Complete API Flow Summary")
    print("=" * 60)

    print("""
  Typical secure API request lifecycle:

    1. Client sends request
       -> Rate limiter checks per-IP request count
    2. If POST /api/register:
       -> Validate username, password strength, email format
       -> Hash password with PBKDF2 (100k iterations)
       -> Store user with role
    3. If POST /api/login:
       -> Authenticate with constant-time comparison
       -> Issue JWT token (HS256, 30 min expiry)
       -> Log authentication attempt
    4. For protected endpoints:
       -> Extract & verify JWT from Authorization header
       -> Check RBAC permissions for user's role
       -> Process request if authorized
    5. Response:
       -> Add security headers (CSP, HSTS, X-Frame-Options, etc.)
       -> Include rate limit remaining count
       -> Return JSON response
    6. Logging:
       -> Structured logs for all auth events
       -> Access logs with user, IP, status code
       -> Rate limit violation warnings
""")

    print("=" * 60)
    print("  Server code ready at: examples/Security/15_secure_api/app.py")
    print("  To run as actual server: python app.py --serve")
    print("=" * 60)


# =============================================================================
# 11. Optional: Run as Actual HTTP Server
# =============================================================================

def run_server(port: int = 8443):
    """Start the actual HTTP server (for manual testing)."""
    # Seed with demo users
    user_store.register("admin", "Adm1n!Pass", "admin@example.com", "admin")
    user_store.register("editor", "Ed1tor!Pass", "editor@example.com", "editor")
    user_store.register("viewer", "V1ewer!Pass", "viewer@example.com", "viewer")

    server = HTTPServer(("127.0.0.1", port), SecureAPIHandler)
    print(f"\n  Secure API server running on http://127.0.0.1:{port}")
    print(f"  Demo users: admin/Adm1n!Pass, editor/Ed1tor!Pass, viewer/V1ewer!Pass")
    print(f"  Press Ctrl+C to stop.\n")
    print(f"  Try: curl -X POST http://127.0.0.1:{port}/api/login \\")
    print(f'        -H "Content-Type: application/json" \\')
    print(f"        -d '{{\"username\":\"admin\",\"password\":\"Adm1n!Pass\"}}'\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if "--serve" in sys.argv:
        port = 8443
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        run_server(port)
    else:
        run_demo()
