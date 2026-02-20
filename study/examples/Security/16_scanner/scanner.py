#!/usr/bin/env python3
"""
네트워크 보안 스캐너 (교육용)
Network Security Scanner (Educational)

포트 스캐닝, HTTP 보안 헤더 검사, SSL/TLS 정보 수집, 배너 그래빙,
리포트 생성을 포함하는 교육용 네트워크 보안 스캐너입니다.
모든 스캔에 레이트 리미팅이 적용됩니다.

Implements an educational network security scanner with TCP port scanning,
HTTP security header checking, SSL/TLS info gathering, banner grabbing,
and report generation. Rate limiting is applied between all connections.

============================================================================
ETHICAL USE DISCLAIMER / 윤리적 사용 고지
============================================================================

This tool is provided STRICTLY for EDUCATIONAL purposes only.

- ONLY scan systems you OWN or have EXPLICIT WRITTEN PERMISSION to test.
- Unauthorized port scanning may violate laws (CFAA, Computer Misuse Act, etc.)
- The authors assume NO LIABILITY for misuse of this tool.
- When in doubt, scan localhost (127.0.0.1) only.

이 도구는 오직 교육 목적으로만 제공됩니다.
- 본인이 소유하거나 명시적 서면 허가를 받은 시스템만 스캔하세요.
- 무단 포트 스캐닝은 법률 위반이 될 수 있습니다.
============================================================================
"""

import argparse
import json
import socket
import ssl
import sys
import time
import re
from datetime import datetime, timezone
from urllib.parse import urlparse
from http.client import HTTPSConnection, HTTPConnection


# =============================================================================
# Configuration (설정)
# =============================================================================

DEFAULT_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995,
                 3306, 5432, 6379, 8080, 8443, 27017]

WELL_KNOWN_SERVICES = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS",
    993: "IMAPS", 995: "POP3S", 3306: "MySQL", 5432: "PostgreSQL",
    6379: "Redis", 8080: "HTTP-Alt", 8443: "HTTPS-Alt", 27017: "MongoDB",
}

# Security headers that should be present
RECOMMENDED_HEADERS = {
    "Strict-Transport-Security": {
        "description": "Enforces HTTPS connections",
        "severity": "HIGH",
    },
    "Content-Security-Policy": {
        "description": "Prevents XSS and data injection attacks",
        "severity": "HIGH",
    },
    "X-Content-Type-Options": {
        "description": "Prevents MIME type sniffing",
        "severity": "MEDIUM",
        "expected": "nosniff",
    },
    "X-Frame-Options": {
        "description": "Prevents clickjacking attacks",
        "severity": "MEDIUM",
        "expected_any": ["DENY", "SAMEORIGIN"],
    },
    "Referrer-Policy": {
        "description": "Controls referrer information leakage",
        "severity": "LOW",
    },
    "Permissions-Policy": {
        "description": "Controls browser feature permissions",
        "severity": "LOW",
    },
    "X-XSS-Protection": {
        "description": "Legacy XSS filter (use CSP instead)",
        "severity": "INFO",
        "note": "Modern recommendation: set to '0' and rely on CSP",
    },
}


# =============================================================================
# 1. Port Scanner (포트 스캐너)
# =============================================================================

class PortScanner:
    """
    TCP connect scanner for educational purposes.
    Uses full TCP handshake (connect scan), not SYN scan.
    """

    def __init__(self, target: str, ports: list[int], timeout: float = 1.0,
                 delay: float = 0.1):
        self.target = target
        self.ports = ports
        self.timeout = timeout
        self.delay = delay  # Rate limiting delay between connections
        self.results: list[dict] = []

    def scan(self) -> list[dict]:
        """Scan all configured ports with rate limiting."""
        print(f"\n  Scanning {self.target} ({len(self.ports)} ports)...")
        print(f"  Timeout: {self.timeout}s | Delay: {self.delay}s between probes\n")

        self.results = []
        for i, port in enumerate(self.ports):
            result = self._scan_port(port)
            self.results.append(result)

            if result["state"] == "open":
                service = WELL_KNOWN_SERVICES.get(port, "unknown")
                banner = result.get("banner", "")
                banner_str = f" | Banner: {banner}" if banner else ""
                print(f"    Port {port:5d}/tcp  OPEN    ({service}){banner_str}")

            # Rate limiting
            if i < len(self.ports) - 1:
                time.sleep(self.delay)

        open_count = sum(1 for r in self.results if r["state"] == "open")
        print(f"\n  Scan complete: {open_count} open / {len(self.ports)} scanned")
        return self.results

    def _scan_port(self, port: int) -> dict:
        """Attempt TCP connection to a single port."""
        result = {
            "port": port,
            "service": WELL_KNOWN_SERVICES.get(port, "unknown"),
            "state": "closed",
            "banner": "",
        }

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            error = sock.connect_ex((self.target, port))

            if error == 0:
                result["state"] = "open"
                # Attempt banner grab
                result["banner"] = self._grab_banner(sock, port)
            sock.close()
        except socket.timeout:
            result["state"] = "filtered"
        except OSError:
            result["state"] = "closed"

        return result

    def _grab_banner(self, sock: socket.socket, port: int) -> str:
        """Attempt to read a service banner from an open port."""
        try:
            # Some services send banner on connect, others need a nudge
            if port in (80, 8080):
                # HTTP: send a minimal request
                sock.sendall(b"HEAD / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            elif port in (25, 110, 143):
                pass  # SMTP/POP3/IMAP send banner automatically

            sock.settimeout(0.5)
            banner = sock.recv(256)
            # Clean up banner: remove non-printable chars
            cleaned = re.sub(r'[^\x20-\x7e]', '', banner.decode("utf-8", errors="replace"))
            return cleaned.strip()[:100]
        except (socket.timeout, OSError, UnicodeDecodeError):
            return ""


# =============================================================================
# 2. HTTP Security Header Checker (HTTP 보안 헤더 검사)
# =============================================================================

class HeaderChecker:
    """Check HTTP response headers against security best practices."""

    def __init__(self, target: str):
        self.target = target
        self.response_headers: dict = {}
        self.findings: list[dict] = []

    def check(self) -> list[dict]:
        """Fetch headers and analyze them."""
        print(f"\n  Checking HTTP security headers for: {self.target}")

        self.response_headers = self._fetch_headers()
        if not self.response_headers:
            print("  ERROR: Could not connect to target")
            return []

        print(f"\n  Response Headers Received:")
        for key, value in sorted(self.response_headers.items()):
            print(f"    {key}: {value[:60]}")

        self.findings = self._analyze_headers()

        print(f"\n  Security Header Analysis:")
        for finding in self.findings:
            symbol = "OK" if finding["status"] == "present" else "!!"
            print(f"    [{symbol}] [{finding['severity']:6s}] {finding['header']}")
            print(f"          {finding['message']}")

        present = sum(1 for f in self.findings if f["status"] == "present")
        total = len(self.findings)
        print(f"\n  Score: {present}/{total} recommended headers present")
        return self.findings

    def _fetch_headers(self) -> dict:
        """Fetch HTTP(S) headers from target."""
        try:
            parsed = urlparse(self.target if "://" in self.target else f"https://{self.target}")
            host = parsed.hostname or self.target
            port = parsed.port
            path = parsed.path or "/"

            if parsed.scheme == "https" or (port and port == 443):
                conn = HTTPSConnection(host, port=port or 443, timeout=5)
            else:
                conn = HTTPConnection(host, port=port or 80, timeout=5)

            conn.request("HEAD", path)
            response = conn.getresponse()
            headers = dict(response.getheaders())
            conn.close()
            return headers
        except Exception as e:
            print(f"  Connection error: {e}")
            return {}

    def _analyze_headers(self) -> list[dict]:
        """Compare response headers against recommendations."""
        findings = []
        # Normalize header keys to title case for comparison
        norm_headers = {k.lower(): v for k, v in self.response_headers.items()}

        for header, info in RECOMMENDED_HEADERS.items():
            header_lower = header.lower()
            finding = {
                "header": header,
                "severity": info["severity"],
                "description": info["description"],
            }

            if header_lower in norm_headers:
                value = norm_headers[header_lower]
                finding["status"] = "present"
                finding["value"] = value
                finding["message"] = f"Present: {value[:50]}"

                # Check expected value
                if "expected" in info and info["expected"].lower() != value.lower():
                    finding["message"] += f" (recommended: {info['expected']})"
                if "expected_any" in info:
                    if not any(v.lower() in value.lower() for v in info["expected_any"]):
                        finding["message"] += f" (recommended: {' or '.join(info['expected_any'])})"
            else:
                finding["status"] = "missing"
                finding["message"] = f"MISSING - {info['description']}"

            findings.append(finding)

        # Check for information disclosure headers
        disclosure_headers = ["server", "x-powered-by", "x-aspnet-version"]
        for h in disclosure_headers:
            if h in norm_headers:
                findings.append({
                    "header": h.title(),
                    "severity": "LOW",
                    "status": "warning",
                    "value": norm_headers[h],
                    "message": f"Information disclosure: {norm_headers[h]} (consider removing)",
                    "description": "Reveals server technology information",
                })

        return findings


# =============================================================================
# 3. SSL/TLS Info Gatherer (SSL/TLS 정보 수집)
# =============================================================================

class TLSChecker:
    """Gather SSL/TLS certificate and protocol information."""

    def __init__(self, target: str, port: int = 443):
        self.target = target
        self.port = port
        self.cert_info: dict = {}

    def check(self) -> dict:
        """Connect and gather TLS information."""
        print(f"\n  Checking SSL/TLS for: {self.target}:{self.port}")

        try:
            context = ssl.create_default_context()
            with socket.create_connection((self.target, self.port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=self.target) as ssock:
                    self.cert_info = self._extract_info(ssock)
                    self._print_info()
                    return self.cert_info
        except ssl.SSLCertVerificationError as e:
            print(f"  SSL Certificate Error: {e}")
            self.cert_info = {"error": str(e), "verified": False}
        except (socket.timeout, OSError) as e:
            print(f"  Connection Error: {e}")
            self.cert_info = {"error": str(e)}

        return self.cert_info

    def _extract_info(self, ssock: ssl.SSLSocket) -> dict:
        """Extract TLS session and certificate details."""
        cert = ssock.getpeercert()
        cipher = ssock.cipher()

        info = {
            "protocol": ssock.version(),
            "cipher_suite": cipher[0] if cipher else "unknown",
            "cipher_bits": cipher[2] if cipher else 0,
            "verified": True,
            "subject": dict(x[0] for x in cert.get("subject", ())),
            "issuer": dict(x[0] for x in cert.get("issuer", ())),
            "serial": cert.get("serialNumber", ""),
            "not_before": cert.get("notBefore", ""),
            "not_after": cert.get("notAfter", ""),
            "san": [entry[1] for entry in cert.get("subjectAltName", ())],
        }

        # Check expiry
        if info["not_after"]:
            try:
                expiry = datetime.strptime(info["not_after"], "%b %d %H:%M:%S %Y %Z")
                days_left = (expiry - datetime.utcnow()).days
                info["days_until_expiry"] = days_left
                info["expired"] = days_left < 0
            except ValueError:
                pass

        return info

    def _print_info(self):
        """Print TLS information in a readable format."""
        info = self.cert_info
        print(f"\n  TLS Connection:")
        print(f"    Protocol:     {info.get('protocol', 'N/A')}")
        print(f"    Cipher:       {info.get('cipher_suite', 'N/A')}")
        print(f"    Key bits:     {info.get('cipher_bits', 'N/A')}")
        print(f"    Verified:     {info.get('verified', False)}")

        print(f"\n  Certificate:")
        subject = info.get("subject", {})
        print(f"    Subject:      {subject.get('commonName', 'N/A')}")
        issuer = info.get("issuer", {})
        print(f"    Issuer:       {issuer.get('organizationName', 'N/A')}")
        print(f"    Valid from:   {info.get('not_before', 'N/A')}")
        print(f"    Valid until:  {info.get('not_after', 'N/A')}")

        days = info.get("days_until_expiry")
        if days is not None:
            status = "EXPIRED" if days < 0 else f"{days} days remaining"
            if 0 < days <= 30:
                status += " (WARNING: expiring soon)"
            print(f"    Expiry:       {status}")

        san = info.get("san", [])
        if san:
            print(f"    SAN entries:  {', '.join(san[:5])}")
            if len(san) > 5:
                print(f"                  ... and {len(san) - 5} more")


# =============================================================================
# 4. Report Generator (리포트 생성기)
# =============================================================================

class ReportGenerator:
    """Generate a structured security scan report."""

    def __init__(self, target: str):
        self.target = target
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.sections: list[dict] = []

    def add_port_scan(self, results: list[dict]):
        self.sections.append({"type": "port_scan", "data": results})

    def add_header_check(self, findings: list[dict]):
        self.sections.append({"type": "header_check", "data": findings})

    def add_tls_check(self, info: dict):
        self.sections.append({"type": "tls_check", "data": info})

    def generate_text(self) -> str:
        """Generate a plain-text report."""
        lines = [
            "=" * 70,
            "  SECURITY SCAN REPORT",
            "=" * 70,
            f"  Target:    {self.target}",
            f"  Timestamp: {self.timestamp}",
            f"  Sections:  {len(self.sections)}",
            "",
        ]

        for section in self.sections:
            if section["type"] == "port_scan":
                lines.append("-" * 70)
                lines.append("  PORT SCAN RESULTS")
                lines.append("-" * 70)
                open_ports = [r for r in section["data"] if r["state"] == "open"]
                for r in open_ports:
                    banner = f"  Banner: {r['banner']}" if r["banner"] else ""
                    lines.append(f"    {r['port']:5d}/tcp  {r['state']:8s}  {r['service']}{banner}")
                lines.append(f"\n    Open: {len(open_ports)} / Scanned: {len(section['data'])}")
                lines.append("")

            elif section["type"] == "header_check":
                lines.append("-" * 70)
                lines.append("  HTTP SECURITY HEADERS")
                lines.append("-" * 70)
                for f in section["data"]:
                    symbol = "OK" if f["status"] == "present" else "!!"
                    lines.append(f"    [{symbol}] {f['header']:35s}  {f['message'][:50]}")
                lines.append("")

            elif section["type"] == "tls_check":
                lines.append("-" * 70)
                lines.append("  TLS/SSL INFORMATION")
                lines.append("-" * 70)
                data = section["data"]
                if "error" in data:
                    lines.append(f"    Error: {data['error']}")
                else:
                    lines.append(f"    Protocol:  {data.get('protocol', 'N/A')}")
                    lines.append(f"    Cipher:    {data.get('cipher_suite', 'N/A')}")
                    subj = data.get('subject', {})
                    lines.append(f"    Subject:   {subj.get('commonName', 'N/A')}")
                    days = data.get('days_until_expiry')
                    if days is not None:
                        lines.append(f"    Expiry:    {days} days remaining")
                lines.append("")

        lines.append("=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_json(self) -> str:
        """Generate a JSON report."""
        report = {
            "target": self.target,
            "timestamp": self.timestamp,
            "sections": self.sections,
        }
        return json.dumps(report, indent=2, default=str)


# =============================================================================
# 5. CLI Interface (CLI 인터페이스)
# =============================================================================

def parse_port_range(port_str: str) -> list[int]:
    """Parse port specification: '80', '80,443', '1-1024', 'common'."""
    if port_str.lower() == "common":
        return DEFAULT_PORTS

    ports = set()
    for part in port_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if 1 <= start <= end <= 65535:
                ports.update(range(start, end + 1))
        else:
            p = int(part)
            if 1 <= p <= 65535:
                ports.add(p)
    return sorted(ports)


def run_scan(args):
    """Execute the scan based on CLI arguments."""
    target = args.target
    report = ReportGenerator(target)

    # Resolve hostname
    try:
        ip = socket.gethostbyname(target)
        print(f"\n  Target: {target} ({ip})")
    except socket.gaierror:
        print(f"\n  ERROR: Cannot resolve hostname: {target}")
        return

    # Port scan
    if not args.skip_ports:
        ports = parse_port_range(args.ports)
        scanner = PortScanner(target, ports, timeout=args.timeout, delay=args.delay)
        results = scanner.scan()
        report.add_port_scan(results)

    # HTTP header check
    if not args.skip_headers:
        checker = HeaderChecker(target)
        findings = checker.check()
        if findings:
            report.add_header_check(findings)

    # TLS check
    if not args.skip_tls:
        tls = TLSChecker(target, port=args.tls_port)
        info = tls.check()
        if info:
            report.add_tls_check(info)

    # Generate report
    if args.output:
        if args.format == "json":
            report_text = report.generate_json()
        else:
            report_text = report.generate_text()

        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"\n  Report saved to: {args.output}")
    else:
        print("\n" + report.generate_text())


def run_demo():
    """Run a localhost-only demo when no arguments are provided."""
    print("=" * 60)
    print("  Network Security Scanner - Demo Mode")
    print("  네트워크 보안 스캐너 - 데모 모드")
    print("=" * 60)

    print("""
  ETHICAL USE DISCLAIMER:
  This tool is for EDUCATIONAL purposes only.
  Only scan systems you own or have written permission to test.
  Unauthorized scanning may violate applicable laws.
""")

    # Demo 1: Port scanner on localhost (most ports will be closed)
    print("-" * 60)
    print("  Demo 1: Port Scan (localhost, limited ports)")
    print("-" * 60)

    scanner = PortScanner(
        "127.0.0.1",
        [22, 80, 443, 3000, 5000, 5050, 8080, 8443],
        timeout=0.3,
        delay=0.05,
    )
    port_results = scanner.scan()

    # Demo 2: Show header check logic (without real connection)
    print("\n" + "-" * 60)
    print("  Demo 2: HTTP Security Header Analysis (simulated)")
    print("-" * 60)

    # Simulate response headers from a typical website
    simulated_headers = {
        "Content-Type": "text/html; charset=utf-8",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "Server": "nginx/1.24.0",
    }

    print(f"\n  Simulated response headers:")
    for k, v in simulated_headers.items():
        print(f"    {k}: {v}")

    print(f"\n  Analysis:")
    norm = {k.lower(): v for k, v in simulated_headers.items()}
    for header, info in RECOMMENDED_HEADERS.items():
        h_lower = header.lower()
        if h_lower in norm:
            print(f"    [OK] {header}: {norm[h_lower][:50]}")
        else:
            print(f"    [!!] {header}: MISSING - {info['description']}")

    # Check info disclosure
    if "server" in norm:
        print(f"    [!!] Server header reveals: {norm['server']} (consider removing)")

    # Demo 3: Show TLS check info
    print(f"\n" + "-" * 60)
    print("  Demo 3: SSL/TLS Check (simulated)")
    print("-" * 60)

    simulated_tls = {
        "protocol": "TLSv1.3",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "cipher_bits": 256,
        "verified": True,
        "subject": {"commonName": "example.com"},
        "issuer": {"organizationName": "Let's Encrypt"},
        "days_until_expiry": 45,
    }

    print(f"\n  Simulated TLS info:")
    print(f"    Protocol:  {simulated_tls['protocol']}")
    print(f"    Cipher:    {simulated_tls['cipher_suite']}")
    print(f"    Key bits:  {simulated_tls['cipher_bits']}")
    print(f"    Subject:   {simulated_tls['subject']['commonName']}")
    print(f"    Issuer:    {simulated_tls['issuer']['organizationName']}")
    print(f"    Expiry:    {simulated_tls['days_until_expiry']} days remaining")

    # Demo 4: Report generation
    print(f"\n" + "-" * 60)
    print("  Demo 4: Report Generation")
    print("-" * 60)

    report = ReportGenerator("127.0.0.1")
    report.add_port_scan(port_results)
    print(f"\n{report.generate_text()}")

    # CLI usage info
    print("\n" + "-" * 60)
    print("  CLI Usage Examples:")
    print("-" * 60)
    print("""
  # Scan localhost (safe for testing)
  python scanner.py 127.0.0.1

  # Scan common ports on your own server
  python scanner.py myserver.com --ports common

  # Scan specific port range with custom timeout
  python scanner.py myserver.com --ports 80,443,8080 --timeout 2.0

  # Full scan with JSON report output
  python scanner.py myserver.com --ports 1-1024 --output report.json --format json

  # Skip port scan, only check headers and TLS
  python scanner.py myserver.com --skip-ports

  # Skip TLS check
  python scanner.py myserver.com --skip-tls
""")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Educational Network Security Scanner",
        epilog="DISCLAIMER: Only scan systems you own or have permission to test.",
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Target hostname or IP address (default: run demo)",
    )
    parser.add_argument(
        "--ports", "-p", default="common",
        help="Ports to scan: '80,443', '1-1024', or 'common' (default: common)",
    )
    parser.add_argument(
        "--timeout", "-t", type=float, default=1.0,
        help="Connection timeout in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=0.1,
        help="Delay between port probes in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--tls-port", type=int, default=443,
        help="Port for TLS check (default: 443)",
    )
    parser.add_argument(
        "--skip-ports", action="store_true",
        help="Skip port scanning",
    )
    parser.add_argument(
        "--skip-headers", action="store_true",
        help="Skip HTTP header check",
    )
    parser.add_argument(
        "--skip-tls", action="store_true",
        help="Skip TLS/SSL check",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output report to file",
    )
    parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        help="Report format (default: text)",
    )
    return parser


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Network Security Scanner (Educational)")
    print("  네트워크 보안 스캐너 (교육용)")
    print("=" * 60)

    if args.target is None:
        run_demo()
    else:
        print("""
  DISCLAIMER: This tool is for educational purposes only.
  Only scan systems you own or have written permission to test.
""")
        run_scan(args)

    print("\n" + "=" * 60)
    print("  Scan complete.")
    print("=" * 60)
