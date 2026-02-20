"""
TLS and Certificate Demo
=========================

Educational demonstration of TLS/SSL concepts:
- Certificate parsing from a live server
- TLS connection information display
- Self-signed certificate generation commands
- Certificate chain verification concepts

Uses only Python standard library (ssl, socket).
No external dependencies required.

Note: Requires internet connection for live server examples.
"""

import ssl
import socket
import datetime
import textwrap
import hashlib

print("=" * 65)
print("  TLS and Certificate Demo")
print("=" * 65)
print()


# ============================================================
# Section 1: TLS Connection to a Live Server
# ============================================================

print("-" * 65)
print("  Section 1: TLS Connection Information")
print("-" * 65)

hostname = "www.google.com"
port = 443

print(f"\n  Connecting to {hostname}:{port}...")
print()

try:
    context = ssl.create_default_context()

    with socket.create_connection((hostname, port), timeout=5) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as tls_sock:
            # Connection info
            print(f"  TLS Version:     {tls_sock.version()}")
            print(f"  Cipher Suite:    {tls_sock.cipher()[0]}")
            print(f"  Cipher Bits:     {tls_sock.cipher()[2]}")
            print()

            # Certificate info
            cert = tls_sock.getpeercert()

            # Subject
            subject = dict(x[0] for x in cert.get("subject", ()))
            print(f"  Subject:")
            for key, value in subject.items():
                print(f"    {key}: {value}")

            # Issuer
            issuer = dict(x[0] for x in cert.get("issuer", ()))
            print(f"\n  Issuer:")
            for key, value in issuer.items():
                print(f"    {key}: {value}")

            # Validity period
            not_before = cert.get("notBefore", "N/A")
            not_after = cert.get("notAfter", "N/A")
            print(f"\n  Valid From:      {not_before}")
            print(f"  Valid Until:     {not_after}")

            # Serial number
            serial = cert.get("serialNumber", "N/A")
            print(f"  Serial Number:   {serial}")

            # Subject Alternative Names
            sans = cert.get("subjectAltName", ())
            if sans:
                san_list = [v for _, v in sans[:5]]
                print(f"\n  Subject Alt Names ({len(sans)} total):")
                for san in san_list:
                    print(f"    - {san}")
                if len(sans) > 5:
                    print(f"    ... and {len(sans) - 5} more")

            # DER-encoded certificate for fingerprint
            der_cert = tls_sock.getpeercert(binary_form=True)
            sha256_fp = hashlib.sha256(der_cert).hexdigest()
            sha1_fp = hashlib.sha1(der_cert).hexdigest()
            print(f"\n  SHA-256 Fingerprint:")
            print(f"    {':'.join(sha256_fp[i:i+2] for i in range(0, 32, 2))}...")
            print(f"  SHA-1 Fingerprint:")
            print(f"    {':'.join(sha1_fp[i:i+2] for i in range(0, len(sha1_fp), 2))}")

except socket.timeout:
    print("  Connection timed out. Skipping live server demo.")
except Exception as e:
    print(f"  Connection failed: {e}")
    print("  (This section requires internet access)")

print()


# ============================================================
# Section 2: TLS Protocol Details
# ============================================================

print("-" * 65)
print("  Section 2: TLS Protocol & Supported Configurations")
print("-" * 65)

print(f"""
  Python SSL module information:
  - OpenSSL version:    {ssl.OPENSSL_VERSION}
  - Default protocol:   TLS (auto-negotiated)
""")

# Show supported protocols
ctx = ssl.create_default_context()
print(f"  Default context settings:")
print(f"    Protocol:         TLSv1.2+ (minimum)")
print(f"    Verify mode:      {ctx.verify_mode.name}")
print(f"    Check hostname:   {ctx.check_hostname}")
print()

# Demonstrate cipher suite listing
print("  Enabled cipher suites (top 10):")
ciphers = ctx.get_ciphers()
for i, c in enumerate(ciphers[:10]):
    protocol = c.get("protocol", "?")
    name = c.get("name", "?")
    bits = c.get("alg_bits", 0)
    print(f"    {i+1:2}. [{protocol}] {name} ({bits}-bit)")
if len(ciphers) > 10:
    print(f"    ... and {len(ciphers) - 10} more")
print()


# ============================================================
# Section 3: Certificate Chain Concepts
# ============================================================

print("-" * 65)
print("  Section 3: Certificate Chain of Trust")
print("-" * 65)

print("""
  Certificate Chain Structure:
  ============================

  +---------------------------+
  |    Root CA Certificate    |  Self-signed, pre-installed
  |  (e.g., DigiCert Root)   |  in OS/browser trust store
  +---------------------------+
              |
              | signs
              v
  +---------------------------+
  | Intermediate CA Cert      |  Signed by Root CA
  | (e.g., DigiCert G2)      |  Provides operational isolation
  +---------------------------+
              |
              | signs
              v
  +---------------------------+
  | Server Certificate        |  Signed by Intermediate CA
  | (e.g., *.google.com)     |  Presented during TLS handshake
  +---------------------------+

  Verification Process:
  1. Server sends its certificate + intermediate cert(s)
  2. Client finds the issuer of each cert in the chain
  3. Client verifies each signature using issuer's public key
  4. Chain must end at a trusted Root CA in the trust store
  5. Client checks: expiration, revocation (CRL/OCSP), hostname
""")

# Show trusted CA count
default_certs = ssl.create_default_context()
ca_certs = default_certs.get_ca_certs()
print(f"  Trusted Root CAs in system store: {len(ca_certs)}")
if ca_certs:
    print(f"\n  Sample trusted CAs:")
    seen = set()
    count = 0
    for ca in ca_certs:
        org = dict(x[0] for x in ca.get("issuer", ())).get(
            "organizationName", "Unknown"
        )
        if org not in seen:
            seen.add(org)
            print(f"    - {org}")
            count += 1
            if count >= 8:
                break
    print(f"    ... and more")
print()


# ============================================================
# Section 4: Self-Signed Certificate Commands
# ============================================================

print("-" * 65)
print("  Section 4: Self-Signed Certificate Generation (OpenSSL)")
print("-" * 65)

print("""
  The following OpenSSL commands create certificates for
  development/testing. NOT for production use.

  --- Generate Private Key ---
  $ openssl genrsa -out server.key 2048

  --- Generate Self-Signed Certificate (valid 365 days) ---
  $ openssl req -new -x509 -key server.key -out server.crt \\
      -days 365 -subj "/CN=localhost/O=Dev/C=US"

  --- Generate with Subject Alternative Names ---
  $ openssl req -new -x509 -key server.key -out server.crt \\
      -days 365 -subj "/CN=localhost" \\
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

  --- View Certificate Details ---
  $ openssl x509 -in server.crt -text -noout

  --- Verify Certificate Chain ---
  $ openssl verify -CAfile ca.crt server.crt

  --- Test TLS Connection ---
  $ openssl s_client -connect example.com:443 -servername example.com

  --- Generate Let's Encrypt Certificate (production) ---
  $ certbot certonly --standalone -d yourdomain.com

  Note: For Python development servers:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('server.crt', 'server.key')
""")


# ============================================================
# Section 5: TLS Handshake Visualization
# ============================================================

print("-" * 65)
print("  Section 5: TLS 1.3 Handshake Overview")
print("-" * 65)

print("""
  TLS 1.3 Handshake (1-RTT):
  ===========================

  Client                              Server
    |                                    |
    |  ClientHello                       |
    |  + supported_versions (TLS 1.3)   |
    |  + key_share (X25519/P-256)       |
    |  + signature_algorithms            |
    |  + cipher_suites                   |
    | ---------------------------------> |
    |                                    |
    |                    ServerHello     |
    |              + selected version    |
    |              + selected key_share  |
    |              {EncryptedExtensions} |
    |              {Certificate}         |
    |              {CertificateVerify}   |
    |              {Finished}            |
    | <--------------------------------- |
    |                                    |
    |  {Finished}                        |
    | ---------------------------------> |
    |                                    |
    |  [Application Data] <============> |
    |  (encrypted with derived keys)     |

  Key improvements in TLS 1.3:
  - 1-RTT handshake (was 2-RTT in TLS 1.2)
  - 0-RTT resumption (with replay protection caveats)
  - Removed insecure algorithms (RC4, 3DES, SHA-1, RSA key transport)
  - Forward secrecy mandatory (ephemeral DH only)
  - Simplified cipher suites (only 5 defined)
""")


# ============================================================
# Section 6: Summary
# ============================================================

print("=" * 65)
print("  Summary")
print("=" * 65)
print("""
  TLS Best Practices:
  - Minimum TLS 1.2, prefer TLS 1.3
  - Use strong cipher suites (AES-GCM, ChaCha20-Poly1305)
  - Enable HSTS (HTTP Strict Transport Security)
  - Use certificate pinning for mobile apps
  - Monitor certificate expiration
  - Use Let's Encrypt for free production certificates
  - Never disable certificate verification in production
    (ssl._create_unverified_context is for testing ONLY)

  Common SSL/TLS Errors:
  - CERTIFICATE_VERIFY_FAILED: CA not trusted or expired
  - HOSTNAME_MISMATCH: cert CN/SAN doesn't match hostname
  - TLSV1_ALERT_PROTOCOL_VERSION: server requires newer TLS
""")
