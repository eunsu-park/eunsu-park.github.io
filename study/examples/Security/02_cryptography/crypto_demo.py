"""
Cryptography Fundamentals Demo
==============================

Educational demonstration of core cryptographic primitives:
- AES-GCM symmetric encryption/decryption
- RSA key generation and hybrid encryption
- Digital signatures with Ed25519
- Key exchange simulation with X25519
- Fallback implementations using hashlib/hmac

Requirements:
    pip install cryptography   (optional - fallback provided)

This is a DEFENSIVE/EDUCATIONAL example. All operations demonstrate
how to properly use cryptographic libraries for data protection.
"""

import os
import hashlib
import hmac
import base64
import struct
import json

# ============================================================
# Section 1: Check for cryptography library availability
# ============================================================

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ed25519, x25519
    from cryptography.hazmat.primitives import hashes, serialization
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

print("=" * 65)
print("  Cryptography Fundamentals Demo")
print("=" * 65)
print(f"\n  cryptography library available: {HAS_CRYPTOGRAPHY}")
print()


# ============================================================
# Section 2: AES-GCM Symmetric Encryption
# ============================================================

print("-" * 65)
print("  Section 2: AES-GCM Symmetric Encryption")
print("-" * 65)

if HAS_CRYPTOGRAPHY:
    # Generate a random 256-bit key
    key = AESGCM.generate_key(bit_length=256)
    print(f"\n  AES-256 Key (hex):  {key.hex()[:32]}...")

    # AES-GCM provides both confidentiality and authenticity
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    plaintext = b"Sensitive data: credit card 4111-1111-1111-1111"
    associated_data = b"metadata:user_id=42"

    # Encrypt with authenticated associated data (AAD)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
    print(f"  Nonce (hex):        {nonce.hex()}")
    print(f"  Plaintext:          {plaintext.decode()}")
    print(f"  Ciphertext (hex):   {ciphertext.hex()[:48]}...")
    print(f"  Ciphertext length:  {len(ciphertext)} bytes")
    print(f"    (plaintext {len(plaintext)} + auth tag 16 = {len(plaintext) + 16})")

    # Decrypt
    decrypted = aesgcm.decrypt(nonce, ciphertext, associated_data)
    print(f"  Decrypted:          {decrypted.decode()}")
    print(f"  Match:              {decrypted == plaintext}")

    # Demonstrate tamper detection
    tampered = bytearray(ciphertext)
    tampered[0] ^= 0xFF  # Flip bits in first byte
    try:
        aesgcm.decrypt(nonce, bytes(tampered), associated_data)
        print("  Tamper detection:   FAILED (should not reach here)")
    except Exception:
        print("  Tamper detection:   Authentication tag mismatch detected!")
else:
    print("\n  [cryptography not installed - showing fallback]")

print()

# --- Fallback: XOR stream cipher concept with HMAC auth ---
print("  -- Fallback: HMAC-authenticated encryption concept --")

def fallback_encrypt(key_bytes: bytes, plaintext: bytes) -> dict:
    """
    Educational fallback using HMAC for authentication.
    NOT production-ready - use AES-GCM via cryptography library.
    """
    # Derive separate keys for encryption and MAC
    enc_key = hashlib.sha256(key_bytes + b"enc").digest()
    mac_key = hashlib.sha256(key_bytes + b"mac").digest()

    # Simple XOR stream (educational only)
    iv = os.urandom(16)
    stream = hashlib.sha256(enc_key + iv).digest()
    # Extend stream for longer messages
    extended = stream
    while len(extended) < len(plaintext):
        extended += hashlib.sha256(enc_key + extended[-32:]).digest()
    ct = bytes(p ^ s for p, s in zip(plaintext, extended))

    # HMAC for authentication
    tag = hmac.new(mac_key, iv + ct, hashlib.sha256).digest()
    return {"iv": iv, "ciphertext": ct, "tag": tag}


def fallback_decrypt(key_bytes: bytes, bundle: dict) -> bytes:
    """Decrypt and verify the fallback encryption."""
    enc_key = hashlib.sha256(key_bytes + b"enc").digest()
    mac_key = hashlib.sha256(key_bytes + b"mac").digest()

    # Verify HMAC first (encrypt-then-MAC pattern)
    expected_tag = hmac.new(
        mac_key, bundle["iv"] + bundle["ciphertext"], hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected_tag, bundle["tag"]):
        raise ValueError("Authentication failed - data tampered!")

    # Decrypt
    stream = hashlib.sha256(enc_key + bundle["iv"]).digest()
    extended = stream
    while len(extended) < len(bundle["ciphertext"]):
        extended += hashlib.sha256(enc_key + extended[-32:]).digest()
    return bytes(c ^ s for c, s in zip(bundle["ciphertext"], extended))


fb_key = os.urandom(32)
fb_plain = b"Hello, fallback encryption!"
fb_bundle = fallback_encrypt(fb_key, fb_plain)
fb_decrypted = fallback_decrypt(fb_key, fb_bundle)
print(f"  Plaintext:          {fb_plain.decode()}")
print(f"  IV (hex):           {fb_bundle['iv'].hex()}")
print(f"  Ciphertext (hex):   {fb_bundle['ciphertext'].hex()}")
print(f"  HMAC tag (hex):     {fb_bundle['tag'].hex()[:32]}...")
print(f"  Decrypted:          {fb_decrypted.decode()}")
print(f"  Match:              {fb_decrypted == fb_plain}")
print()


# ============================================================
# Section 3: RSA Key Generation and Hybrid Encryption
# ============================================================

print("-" * 65)
print("  Section 3: RSA Key Generation & Hybrid Encryption")
print("-" * 65)

if HAS_CRYPTOGRAPHY:
    # Generate RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()

    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    print(f"\n  RSA-2048 public key generated")
    print(f"  PEM format (first line): {pub_pem.decode().split(chr(10))[0]}")

    # Hybrid encryption: RSA encrypts an AES key, AES encrypts the data
    # Step 1: Generate random AES session key
    session_key = os.urandom(32)

    # Step 2: RSA-encrypt the session key
    encrypted_session_key = public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    print(f"  Session key (hex):          {session_key.hex()[:24]}...")
    print(f"  RSA-encrypted key length:   {len(encrypted_session_key)} bytes")

    # Step 3: AES-GCM encrypt the actual data with the session key
    aesgcm_hybrid = AESGCM(session_key)
    hybrid_nonce = os.urandom(12)
    hybrid_data = b"Large payload encrypted with AES, key protected by RSA"
    hybrid_ct = aesgcm_hybrid.encrypt(hybrid_nonce, hybrid_data, None)
    print(f"  Hybrid ciphertext length:   {len(hybrid_ct)} bytes")

    # Receiver: RSA-decrypt session key, then AES-decrypt data
    recovered_key = private_key.decrypt(
        encrypted_session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    recovered_plain = AESGCM(recovered_key).decrypt(hybrid_nonce, hybrid_ct, None)
    print(f"  Decrypted payload:          {recovered_plain.decode()}")
    print(f"  Key recovery match:         {recovered_key == session_key}")
else:
    print("\n  [cryptography not installed]")
    print("  RSA hybrid encryption requires the cryptography library.")
    print("  Install with: pip install cryptography")
print()


# ============================================================
# Section 4: Digital Signatures with Ed25519
# ============================================================

print("-" * 65)
print("  Section 4: Digital Signatures (Ed25519)")
print("-" * 65)

if HAS_CRYPTOGRAPHY:
    # Ed25519: fast, secure, small signatures
    signing_key = ed25519.Ed25519PrivateKey.generate()
    verify_key = signing_key.public_key()

    message = b"Transfer $1000 from Alice to Bob"
    signature = signing_key.sign(message)

    print(f"\n  Message:            {message.decode()}")
    print(f"  Signature (hex):    {signature.hex()[:48]}...")
    print(f"  Signature length:   {len(signature)} bytes")

    # Verify valid signature
    try:
        verify_key.verify(signature, message)
        print("  Verification:       VALID")
    except Exception:
        print("  Verification:       INVALID")

    # Verify tampered message
    tampered_msg = b"Transfer $9999 from Alice to Bob"
    try:
        verify_key.verify(signature, tampered_msg)
        print("  Tampered verify:    VALID (should not happen!)")
    except Exception:
        print("  Tampered verify:    INVALID (tamper detected!)")
else:
    print("\n  [cryptography not installed]")
    print("  Ed25519 requires the cryptography library.")
print()

# --- Fallback: HMAC-based message authentication ---
print("  -- Fallback: HMAC-based message authentication --")
secret = os.urandom(32)
msg = b"Important message to authenticate"
mac_tag = hmac.new(secret, msg, hashlib.sha256).digest()
print(f"  Message:            {msg.decode()}")
print(f"  HMAC-SHA256 (hex):  {mac_tag.hex()}")
verify_ok = hmac.compare_digest(
    mac_tag, hmac.new(secret, msg, hashlib.sha256).digest()
)
print(f"  Verification:       {'VALID' if verify_ok else 'INVALID'}")
print()


# ============================================================
# Section 5: Key Exchange with X25519
# ============================================================

print("-" * 65)
print("  Section 5: Key Exchange (X25519 / Diffie-Hellman)")
print("-" * 65)

if HAS_CRYPTOGRAPHY:
    # X25519 Diffie-Hellman key exchange
    alice_private = x25519.X25519PrivateKey.generate()
    alice_public = alice_private.public_key()

    bob_private = x25519.X25519PrivateKey.generate()
    bob_public = bob_private.public_key()

    # Both sides compute the same shared secret
    alice_shared = alice_private.exchange(bob_public)
    bob_shared = bob_private.exchange(alice_public)

    print(f"\n  Alice's shared secret: {alice_shared.hex()[:32]}...")
    print(f"  Bob's shared secret:   {bob_shared.hex()[:32]}...")
    print(f"  Secrets match:         {alice_shared == bob_shared}")

    # Derive an encryption key from the shared secret
    derived_key = hashlib.sha256(alice_shared).digest()
    print(f"  Derived AES key:       {derived_key.hex()[:32]}...")
else:
    print("\n  [cryptography not installed]")
    print("  X25519 requires the cryptography library.")

print()

# --- Fallback: Simplified DH concept with small numbers ---
print("  -- Fallback: Diffie-Hellman concept (small numbers) --")
# Educational only - real DH uses much larger primes
p = 23  # Small prime (real: 2048+ bits)
g = 5   # Generator

alice_secret = 6   # Alice's private key
bob_secret = 15    # Bob's private key

alice_pub = pow(g, alice_secret, p)  # g^a mod p
bob_pub = pow(g, bob_secret, p)      # g^b mod p

alice_computed = pow(bob_pub, alice_secret, p)   # (g^b)^a mod p
bob_computed = pow(alice_pub, bob_secret, p)     # (g^a)^b mod p

print(f"  Parameters:  p={p}, g={g}")
print(f"  Alice:  secret={alice_secret}, public={alice_pub}")
print(f"  Bob:    secret={bob_secret}, public={bob_pub}")
print(f"  Alice computes shared: {alice_computed}")
print(f"  Bob computes shared:   {bob_computed}")
print(f"  Secrets match:         {alice_computed == bob_computed}")
print()


# ============================================================
# Section 6: Summary
# ============================================================

print("=" * 65)
print("  Summary of Cryptographic Primitives")
print("=" * 65)
print("""
  Primitive        | Use Case                  | Key Size
  -----------------+---------------------------+----------
  AES-GCM          | Symmetric encryption      | 128/256-bit
  RSA-OAEP         | Asymmetric / hybrid enc   | 2048+ bit
  Ed25519          | Digital signatures         | 256-bit
  X25519           | Key exchange (ECDH)        | 256-bit
  HMAC-SHA256      | Message authentication     | 256-bit
  PBKDF2/Argon2    | Password hashing           | N/A

  Best Practices:
  - Never implement your own crypto primitives
  - Use authenticated encryption (AES-GCM, not AES-CBC alone)
  - Use hybrid encryption for large data (RSA + AES)
  - Rotate keys regularly
  - Use constant-time comparison for MACs/signatures
""")
