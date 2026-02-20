"""
보안 테스트 도구 데모
Security Testing Tools Demo

정적 분석(위험 함수 탐지), 의존성 검사, 코드 패턴 스캐너(하드코딩 패스워드,
SQL 인젝션 패턴), 보안 체크리스트 검증 등을 구현합니다.

Demonstrates simple static analysis for dangerous function calls,
dependency checking, regex-based code pattern scanning (hardcoded passwords,
SQL injection patterns), and security checklist validation.
"""

import re
import os
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field


# =============================================================================
# 1. Static Analysis: Dangerous Function Detector (위험 함수 탐지)
# =============================================================================

@dataclass
class Finding:
    """Represents a single security finding."""
    severity: str       # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str
    message: str
    file: str
    line: int
    code_snippet: str


# Dangerous patterns in Python code
DANGEROUS_FUNCTIONS = [
    # (pattern, severity, message)
    (r'\beval\s*\(', "CRITICAL",
     "eval() executes arbitrary code - use ast.literal_eval() instead"),
    (r'\bexec\s*\(', "CRITICAL",
     "exec() executes arbitrary code - avoid or sandbox carefully"),
    (r'\b__import__\s*\(', "HIGH",
     "__import__() can load arbitrary modules - use importlib instead"),
    (r'\bos\.system\s*\(', "HIGH",
     "os.system() is vulnerable to shell injection - use subprocess.run()"),
    (r'\bos\.popen\s*\(', "HIGH",
     "os.popen() is vulnerable to shell injection - use subprocess.run()"),
    (r'\bsubprocess\.\w+\(.*shell\s*=\s*True', "HIGH",
     "subprocess with shell=True is vulnerable to injection"),
    (r'\bpickle\.loads?\s*\(', "HIGH",
     "pickle.load() can execute arbitrary code during deserialization"),
    (r'\byaml\.load\s*\([^)]*\)(?!.*Loader)', "MEDIUM",
     "yaml.load() without SafeLoader can execute arbitrary code"),
    (r'\brandom\.(random|randint|choice|shuffle)\s*\(', "LOW",
     "random module is not cryptographically secure - use secrets module"),
    (r'\bmd5\s*\(', "MEDIUM",
     "MD5 is cryptographically broken - use SHA-256 or better"),
    (r'\bsha1\s*\(', "MEDIUM",
     "SHA-1 is deprecated for security - use SHA-256 or better"),
    (r'\bDESede|DES\b', "HIGH",
     "DES/3DES is deprecated - use AES-256"),
    (r'\.execute\s*\(\s*["\'].*%s', "HIGH",
     "String formatting in SQL query - use parameterized queries"),
    (r'\.execute\s*\(\s*f["\']', "HIGH",
     "f-string in SQL query - use parameterized queries"),
    (r'\bassert\s+', "INFO",
     "assert statements are removed with -O flag - don't use for security checks"),
]


def scan_python_file(content: str, filename: str = "<input>") -> list[Finding]:
    """Scan Python source code for dangerous function calls."""
    findings = []
    in_comment = False

    for line_num, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()

        # Skip comments and docstrings (simplified)
        if stripped.startswith("#"):
            continue
        if '"""' in stripped or "'''" in stripped:
            in_comment = not in_comment
            continue
        if in_comment:
            continue

        for pattern, severity, message in DANGEROUS_FUNCTIONS:
            if re.search(pattern, line):
                findings.append(Finding(
                    severity=severity,
                    category="Dangerous Function",
                    message=message,
                    file=filename,
                    line=line_num,
                    code_snippet=stripped[:80],
                ))

    return findings


def demo_static_analysis():
    print("=" * 60)
    print("1. Static Analysis: Dangerous Function Detector")
    print("=" * 60)

    sample_code = '''
import os
import pickle
import subprocess
import random
import sqlite3

# Dangerous: eval with user input
user_input = "2 + 3"
result = eval(user_input)

# Dangerous: shell injection
os.system("ls " + user_input)

# Dangerous: pickle deserialization
data = pickle.loads(some_bytes)

# Dangerous: shell=True in subprocess
subprocess.run(f"echo {user_input}", shell=True)

# Dangerous: SQL injection via f-string
db.execute(f"SELECT * FROM users WHERE name = '{name}'")

# Insecure random for tokens
token = random.randint(100000, 999999)

# Safe alternative examples
import secrets
token = secrets.token_hex(16)
subprocess.run(["ls", "-la"], shell=False)
'''

    print("\n  Scanning sample Python code...\n")
    findings = scan_python_file(sample_code, "sample_app.py")

    severity_colors = {
        "CRITICAL": "!!!",
        "HIGH": "!! ",
        "MEDIUM": "!  ",
        "LOW": ".  ",
        "INFO": "   ",
    }

    for f in sorted(findings, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"].index(x.severity)):
        prefix = severity_colors.get(f.severity, "   ")
        print(f"  {prefix} [{f.severity:8s}] Line {f.line:3d}: {f.message}")
        print(f"          Code: {f.code_snippet}")
        print()

    # Summary
    by_severity = {}
    for f in findings:
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
    print(f"  Summary: {dict(sorted(by_severity.items()))}")
    print(f"  Total findings: {len(findings)}")
    print()


# =============================================================================
# 2. Dependency Checker (의존성 검사)
# =============================================================================

# Simulated known vulnerability database (pattern-based)
KNOWN_VULNERABLE_PATTERNS = {
    "django": {
        "pattern": r"django\s*[<>=]*\s*(1\.\d|2\.[01])",
        "advisory": "Django < 2.2 has known security vulnerabilities",
        "severity": "HIGH",
    },
    "flask": {
        "pattern": r"flask\s*[<>=]*\s*(0\.)",
        "advisory": "Flask 0.x has known security issues, upgrade to 2.x+",
        "severity": "MEDIUM",
    },
    "requests": {
        "pattern": r"requests\s*[<>=]*\s*2\.(0|1|2|3|4|5)\.",
        "advisory": "requests < 2.6.0 vulnerable to CVE-2014-1829",
        "severity": "HIGH",
    },
    "pyyaml": {
        "pattern": r"pyyaml\s*[<>=]*\s*(3\.|4\.)",
        "advisory": "PyYAML < 5.1 vulnerable to arbitrary code execution",
        "severity": "CRITICAL",
    },
    "cryptography": {
        "pattern": r"cryptography\s*[<>=]*\s*(1\.|2\.[0-4])",
        "advisory": "cryptography < 2.5 has known vulnerabilities",
        "severity": "HIGH",
    },
    "jinja2": {
        "pattern": r"jinja2\s*[<>=]*\s*2\.\d\b",
        "advisory": "Jinja2 2.x has known sandbox escape vulnerabilities",
        "severity": "MEDIUM",
    },
}


def check_dependencies(requirements_text: str) -> list[dict]:
    """Check requirements.txt content against known vulnerable patterns."""
    issues = []

    for line in requirements_text.splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        for pkg_name, info in KNOWN_VULNERABLE_PATTERNS.items():
            if re.match(info["pattern"], line, re.IGNORECASE):
                issues.append({
                    "package": line,
                    "severity": info["severity"],
                    "advisory": info["advisory"],
                })

    return issues


def demo_dependency_checker():
    print("=" * 60)
    print("2. Dependency Checker")
    print("=" * 60)

    sample_requirements = """
# requirements.txt
django==2.0.13
flask==2.3.1
requests==2.28.0
pyyaml==3.13
cryptography==2.3
numpy==1.24.0
jinja2==2.11.3
pandas>=1.5.0
"""

    print(f"\n  Checking sample requirements.txt...\n")
    issues = check_dependencies(sample_requirements)

    if issues:
        for issue in sorted(issues, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM"].index(x["severity"])):
            print(f"  [{issue['severity']:8s}] {issue['package']}")
            print(f"            {issue['advisory']}\n")
        print(f"  Total vulnerable packages: {len(issues)}")
    else:
        print("  No known vulnerabilities detected.")

    # Check safe requirements
    safe_req = "flask==3.0.0\ndjango==4.2.0\nrequests==2.31.0\n"
    safe_issues = check_dependencies(safe_req)
    print(f"\n  Check modern versions: {len(safe_issues)} issues found (expected 0)")
    print()


# =============================================================================
# 3. Code Pattern Scanner (코드 패턴 스캐너)
# =============================================================================

SECURITY_PATTERNS = {
    "Hardcoded Password": [
        r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']',
        r'(?i)(password|passwd|pwd)\s*:\s*["\'][^"\']{4,}["\']',
    ],
    "Hardcoded Secret/Key": [
        r'(?i)(secret|api_key|apikey|access_key)\s*=\s*["\'][^"\']{8,}["\']',
        r'(?i)(token|auth_token)\s*=\s*["\'][^"\']{8,}["\']',
    ],
    "SQL Injection Risk": [
        r'(?i)(execute|query)\s*\(\s*["\'].*\+',
        r'(?i)(execute|query)\s*\(\s*f["\']',
        r'(?i)(execute|query)\s*\(\s*["\'].*%\s',
        r'(?i)\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)',
    ],
    "Command Injection Risk": [
        r'os\.system\s*\(.*\+',
        r'os\.popen\s*\(.*\+',
        r'subprocess\.\w+\(.*\+.*shell\s*=\s*True',
    ],
    "Insecure HTTP": [
        r'http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)',
    ],
    "Debug Mode in Production": [
        r'(?i)debug\s*=\s*True',
        r'(?i)DEBUG\s*=\s*True',
    ],
    "Hardcoded IP Address": [
        r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b',
    ],
}


def scan_code_patterns(content: str, filename: str = "<input>") -> list[dict]:
    """Scan code for security anti-patterns using regex."""
    results = []

    for line_num, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for category, patterns in SECURITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line):
                    results.append({
                        "category": category,
                        "file": filename,
                        "line": line_num,
                        "code": stripped[:80],
                    })
                    break  # One match per category per line

    return results


def demo_pattern_scanner():
    print("=" * 60)
    print("3. Code Pattern Scanner")
    print("=" * 60)

    sample = '''
# Application config
DATABASE_PASSWORD = "MyDbPass123!"
API_KEY = "sk-proj-abcdef123456789"
DEBUG = True

# SQL query building (vulnerable)
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(f"DELETE FROM sessions WHERE user = '{username}'")

# Command execution (vulnerable)
os.system("ping " + target_host)

# Insecure HTTP
response = requests.get("http://api.example.com/data")

# Hardcoded server
server = "192.168.1.100"

# Safe examples (should not trigger)
password = os.environ.get("DB_PASSWORD")
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
'''

    print(f"\n  Scanning code for security anti-patterns...\n")
    results = scan_code_patterns(sample, "app_config.py")

    by_category: dict[str, list] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    for category, items in by_category.items():
        print(f"  [{category}] ({len(items)} finding{'s' if len(items) > 1 else ''})")
        for item in items:
            print(f"    Line {item['line']:3d}: {item['code']}")
        print()

    print(f"  Total findings: {len(results)} across {len(by_category)} categories")
    print()


# =============================================================================
# 4. Security Checklist Validator (보안 체크리스트 검증)
# =============================================================================

@dataclass
class ChecklistItem:
    name: str
    description: str
    check_fn: object  # callable
    severity: str
    passed: bool = False
    details: str = ""


class SecurityChecklist:
    """Validate a project against a security checklist."""

    def __init__(self):
        self.items: list[ChecklistItem] = []

    def add_check(self, name: str, description: str, check_fn, severity: str):
        self.items.append(ChecklistItem(name, description, check_fn, severity))

    def run_all(self, context: dict) -> list[ChecklistItem]:
        """Run all checks against the given context."""
        for item in self.items:
            try:
                passed, details = item.check_fn(context)
                item.passed = passed
                item.details = details
            except Exception as e:
                item.passed = False
                item.details = f"Check error: {e}"
        return self.items

    def report(self) -> str:
        lines = []
        passed = sum(1 for i in self.items if i.passed)
        total = len(self.items)
        lines.append(f"\n  Security Checklist: {passed}/{total} passed\n")

        for item in self.items:
            symbol = "PASS" if item.passed else "FAIL"
            lines.append(f"  [{symbol}] [{item.severity:6s}] {item.name}")
            lines.append(f"          {item.description}")
            if item.details:
                lines.append(f"          -> {item.details}")
            lines.append("")

        return "\n".join(lines)


# Check functions
def check_no_hardcoded_secrets(ctx):
    code = ctx.get("source_code", "")
    findings = scan_code_patterns(code)
    secret_findings = [f for f in findings if "Secret" in f["category"] or "Password" in f["category"]]
    if secret_findings:
        return False, f"Found {len(secret_findings)} hardcoded secret(s)"
    return True, "No hardcoded secrets detected"


def check_no_debug_mode(ctx):
    code = ctx.get("source_code", "")
    if re.search(r'(?i)\bDEBUG\s*=\s*True\b', code):
        return False, "DEBUG=True found in source"
    return True, "Debug mode not enabled"


def check_https_only(ctx):
    code = ctx.get("source_code", "")
    matches = re.findall(r'http://(?!localhost|127\.0\.0\.1)', code)
    if matches:
        return False, f"Found {len(matches)} insecure HTTP URL(s)"
    return True, "All external URLs use HTTPS"


def check_sql_parameterized(ctx):
    code = ctx.get("source_code", "")
    patterns = [r'execute\s*\(\s*f["\']', r'execute\s*\(.*\+']
    for p in patterns:
        if re.search(p, code):
            return False, "SQL queries use string formatting instead of parameters"
    return True, "SQL queries appear to use parameterized queries"


def check_dependencies_updated(ctx):
    reqs = ctx.get("requirements", "")
    issues = check_dependencies(reqs)
    if issues:
        return False, f"{len(issues)} vulnerable package(s) found"
    return True, "No known vulnerable dependencies"


def check_input_validation(ctx):
    code = ctx.get("source_code", "")
    # Check if there is any form of input validation
    validation_patterns = [r'validate', r'sanitize', r'clean', r'escape', r'strip\(\)']
    for p in validation_patterns:
        if re.search(p, code, re.IGNORECASE):
            return True, "Input validation functions detected"
    return False, "No input validation patterns found"


def demo_security_checklist():
    print("=" * 60)
    print("4. Security Checklist Validator")
    print("=" * 60)

    checklist = SecurityChecklist()
    checklist.add_check("No Hardcoded Secrets", "Source code should not contain hardcoded passwords or keys", check_no_hardcoded_secrets, "HIGH")
    checklist.add_check("Debug Mode Off", "DEBUG should not be True in production", check_no_debug_mode, "MEDIUM")
    checklist.add_check("HTTPS Only", "All external URLs should use HTTPS", check_https_only, "HIGH")
    checklist.add_check("Parameterized SQL", "SQL queries should use parameter binding", check_sql_parameterized, "HIGH")
    checklist.add_check("Dependencies Updated", "No known vulnerable packages", check_dependencies_updated, "HIGH")
    checklist.add_check("Input Validation", "User input should be validated", check_input_validation, "MEDIUM")

    # Context: a project with some issues
    context = {
        "source_code": '''
DATABASE_PASSWORD = "MyPass123"
DEBUG = True
cursor.execute(f"SELECT * FROM users WHERE id = '{uid}'")
response = requests.get("http://api.example.com/data")
user_input = sanitize(request.form["name"])
''',
        "requirements": "django==2.0.13\nflask==3.0.0\n",
    }

    checklist.run_all(context)
    print(checklist.report())


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Security Testing Tools Demo")
    print("  보안 테스트 도구 데모")
    print("=" * 60 + "\n")

    demo_static_analysis()
    demo_dependency_checker()
    demo_pattern_scanner()
    demo_security_checklist()

    print("=" * 60)
    print("  Demo complete. All examples use stdlib only.")
    print("  These tools are educational - use professional SAST/DAST")
    print("  tools (Bandit, Safety, Semgrep) for production scanning.")
    print("=" * 60)
