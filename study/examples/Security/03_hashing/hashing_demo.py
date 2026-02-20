"""
Hashing and Message Authentication Demo
========================================

Educational demonstration of hashing concepts:
- SHA-256, SHA-3, BLAKE2 with hashlib
- Password hashing with PBKDF2 (hashlib built-in)
- HMAC generation and verification
- Constant-time comparison
- Simplified Merkle tree implementation

All examples use Python's standard library (hashlib, hmac, secrets).
No external dependencies required.
"""

import hashlib
import hmac
import os
import secrets
import time
import struct

print("=" * 65)
print("  Hashing and Message Authentication Demo")
print("=" * 65)
print()


# ============================================================
# Section 1: Cryptographic Hash Functions
# ============================================================

print("-" * 65)
print("  Section 1: Cryptographic Hash Functions")
print("-" * 65)

message = b"The quick brown fox jumps over the lazy dog"
print(f"\n  Input message: {message.decode()}")
print(f"  Input length:  {len(message)} bytes")
print()

# SHA-256 (SHA-2 family)
sha256_hash = hashlib.sha256(message).hexdigest()
print(f"  SHA-256:    {sha256_hash}")

# SHA-384 (SHA-2 family)
sha384_hash = hashlib.sha384(message).hexdigest()
print(f"  SHA-384:    {sha384_hash[:48]}...")

# SHA-512 (SHA-2 family)
sha512_hash = hashlib.sha512(message).hexdigest()
print(f"  SHA-512:    {sha512_hash[:48]}...")

# SHA-3 (Keccak-based, different internal structure from SHA-2)
sha3_256 = hashlib.sha3_256(message).hexdigest()
print(f"  SHA3-256:   {sha3_256}")

sha3_512 = hashlib.sha3_512(message).hexdigest()
print(f"  SHA3-512:   {sha3_512[:48]}...")

# BLAKE2 (fast, secure, used in libsodium/WireGuard)
blake2b_hash = hashlib.blake2b(message).hexdigest()
print(f"  BLAKE2b:    {blake2b_hash[:48]}...")

blake2s_hash = hashlib.blake2s(message).hexdigest()
print(f"  BLAKE2s:    {blake2s_hash}")
print()

# --- Avalanche effect demonstration ---
print("  -- Avalanche Effect --")
msg1 = b"Hello World"
msg2 = b"Hello World!"  # One character added
h1 = hashlib.sha256(msg1).hexdigest()
h2 = hashlib.sha256(msg2).hexdigest()
print(f"  Input 1: 'Hello World'  -> {h1[:32]}...")
print(f"  Input 2: 'Hello World!' -> {h2[:32]}...")

# Count differing bits
bits1 = bin(int(h1, 16))[2:].zfill(256)
bits2 = bin(int(h2, 16))[2:].zfill(256)
diff_bits = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
print(f"  Bits changed: {diff_bits}/256 ({diff_bits/256*100:.1f}%)")
print(f"  (Ideal avalanche: ~50% bits change)")
print()

# --- Incremental hashing for large data ---
print("  -- Incremental Hashing (for large files) --")
hasher = hashlib.sha256()
chunks = [b"chunk1_", b"chunk2_", b"chunk3_done"]
for chunk in chunks:
    hasher.update(chunk)
incremental = hasher.hexdigest()

all_at_once = hashlib.sha256(b"chunk1_chunk2_chunk3_done").hexdigest()
print(f"  Incremental: {incremental[:32]}...")
print(f"  All-at-once: {all_at_once[:32]}...")
print(f"  Match:       {incremental == all_at_once}")
print()


# ============================================================
# Section 2: Password Hashing with PBKDF2
# ============================================================

print("-" * 65)
print("  Section 2: Password Hashing (PBKDF2)")
print("-" * 65)

print("""
  Why not just SHA-256 for passwords?
  - SHA-256 is TOO FAST (~1 billion hashes/sec on GPU)
  - Attackers can brute-force quickly
  - Password hashing must be SLOW (key stretching)

  Recommended algorithms (best to worst):
  1. Argon2id  (memory-hard, state of the art)
  2. bcrypt    (CPU-hard, widely supported)
  3. PBKDF2    (CPU-hard, NIST approved, in stdlib)
""")

password = "MyS3cur3P@ssw0rd!"
salt = os.urandom(16)  # Unique per password

# PBKDF2 with high iteration count
iterations = 600_000  # NIST recommends >= 600,000 for SHA-256

start = time.time()
derived_key = hashlib.pbkdf2_hmac(
    "sha256",
    password.encode("utf-8"),
    salt,
    iterations,
    dklen=32,
)
elapsed = time.time() - start

print(f"  Password:       {password}")
print(f"  Salt (hex):     {salt.hex()}")
print(f"  Iterations:     {iterations:,}")
print(f"  Derived key:    {derived_key.hex()}")
print(f"  Time:           {elapsed:.3f}s")
print()

# Verify password
print("  -- Password Verification --")
correct_key = hashlib.pbkdf2_hmac(
    "sha256", "MyS3cur3P@ssw0rd!".encode(), salt, iterations, dklen=32
)
wrong_key = hashlib.pbkdf2_hmac(
    "sha256", "WrongPassword".encode(), salt, iterations, dklen=32
)
print(f"  Correct password match: {hmac.compare_digest(derived_key, correct_key)}")
print(f"  Wrong password match:   {hmac.compare_digest(derived_key, wrong_key)}")
print()

# Storage format
stored = f"pbkdf2:sha256:{iterations}${salt.hex()}${derived_key.hex()}"
print(f"  Storage format: {stored[:50]}...")
print(f"  (algorithm:hash:iterations$salt$key)")
print()


# ============================================================
# Section 3: HMAC Generation and Verification
# ============================================================

print("-" * 65)
print("  Section 3: HMAC (Hash-based Message Authentication Code)")
print("-" * 65)

print("""
  HMAC = Hash(Key || Hash(Key || Message))
  - Proves both integrity AND authenticity
  - Requires shared secret key
  - Used in: JWT, API authentication, TLS
""")

api_secret = secrets.token_bytes(32)
payload = b'{"user":"admin","action":"delete","id":42}'

# Generate HMAC
mac = hmac.new(api_secret, payload, hashlib.sha256).hexdigest()
print(f"  API Secret:    {api_secret.hex()[:24]}...")
print(f"  Payload:       {payload.decode()}")
print(f"  HMAC-SHA256:   {mac}")
print()

# Verify HMAC
print("  -- HMAC Verification --")
received_mac = mac  # Simulating received MAC
computed_mac = hmac.new(api_secret, payload, hashlib.sha256).hexdigest()
is_valid = hmac.compare_digest(received_mac, computed_mac)
print(f"  Valid MAC:     {is_valid}")

# Tampered payload
tampered_payload = b'{"user":"admin","action":"delete","id":9999}'
tampered_mac = hmac.new(api_secret, tampered_payload, hashlib.sha256).hexdigest()
is_valid_tampered = hmac.compare_digest(received_mac, tampered_mac)
print(f"  Tampered MAC:  {is_valid_tampered}  (tamper detected!)")
print()


# ============================================================
# Section 4: Constant-Time Comparison
# ============================================================

print("-" * 65)
print("  Section 4: Constant-Time Comparison")
print("-" * 65)

print("""
  Why constant-time comparison matters:
  - Regular '==' comparison short-circuits on first mismatch
  - Attacker can measure response time to guess bytes one by one
  - This is called a "timing side-channel attack"
  - hmac.compare_digest() always compares ALL bytes
""")

secret_token = "a1b2c3d4e5f6g7h8"

# Demonstrate timing difference (educational - results may vary)
def unsafe_compare(a: str, b: str) -> bool:
    """VULNERABLE: short-circuits on first mismatch."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False
    return True


def safe_compare(a: str, b: str) -> bool:
    """SAFE: always compares all characters."""
    return hmac.compare_digest(a.encode(), b.encode())


# Timing test
test_inputs = [
    ("Completely wrong token!!", "no match at start"),
    ("a1b2c3d4XXXXXXXX", "partial match (8 chars)"),
    ("a1b2c3d4e5f6g7h8", "exact match"),
]

print(f"\n  Secret token: {secret_token}")
print()
print(f"  {'Input':<28} {'Unsafe':<12} {'Safe':<12}")
print(f"  {'-'*28} {'-'*12} {'-'*12}")

for test_input, desc in test_inputs:
    padded = test_input.ljust(len(secret_token))[:len(secret_token)]

    # Unsafe comparison timing
    start = time.perf_counter_ns()
    for _ in range(10000):
        unsafe_compare(secret_token, padded)
    unsafe_ns = (time.perf_counter_ns() - start) // 10000

    # Safe comparison timing
    start = time.perf_counter_ns()
    for _ in range(10000):
        safe_compare(secret_token, padded)
    safe_ns = (time.perf_counter_ns() - start) // 10000

    print(f"  {desc:<28} {unsafe_ns:>6}ns     {safe_ns:>6}ns")

print()
print("  Note: Unsafe times may vary with match length; safe times")
print("  should be roughly constant regardless of input.")
print()


# ============================================================
# Section 5: Merkle Tree Implementation
# ============================================================

print("-" * 65)
print("  Section 5: Merkle Tree (Hash Tree)")
print("-" * 65)

print("""
  Merkle trees efficiently verify data integrity:
  - Used in: Git, Bitcoin, IPFS, certificate transparency
  - Verify any single block without downloading entire dataset
  - Proof size: O(log n) hashes for n data blocks
""")


class MerkleTree:
    """Simplified Merkle tree for educational purposes."""

    def __init__(self, data_blocks: list[bytes]):
        self.data_blocks = data_blocks
        self.leaves = [self._hash_leaf(d) for d in data_blocks]
        self.tree = self._build_tree(self.leaves[:])

    @staticmethod
    def _hash_leaf(data: bytes) -> str:
        return hashlib.sha256(b"\x00" + data).hexdigest()

    @staticmethod
    def _hash_pair(left: str, right: str) -> str:
        combined = b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)
        return hashlib.sha256(combined).hexdigest()

    def _build_tree(self, nodes: list[str]) -> list[list[str]]:
        tree = [nodes]
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    next_level.append(self._hash_pair(nodes[i], nodes[i + 1]))
                else:
                    # Odd node: promote to next level
                    next_level.append(nodes[i])
            tree.append(next_level)
            nodes = next_level
        return tree

    @property
    def root(self) -> str:
        return self.tree[-1][0]

    def get_proof(self, index: int) -> list[tuple[str, str]]:
        """Get Merkle proof for a leaf at given index."""
        proof = []
        for level in self.tree[:-1]:
            if index % 2 == 0:
                sibling_idx = index + 1
                direction = "right"
            else:
                sibling_idx = index - 1
                direction = "left"
            if sibling_idx < len(level):
                proof.append((direction, level[sibling_idx]))
            index //= 2
        return proof

    @classmethod
    def verify_proof(cls, leaf_data: bytes, proof: list, root: str) -> bool:
        """Verify a Merkle proof against the root hash."""
        current = cls._hash_leaf(leaf_data)
        for direction, sibling in proof:
            if direction == "right":
                current = cls._hash_pair(current, sibling)
            else:
                current = cls._hash_pair(sibling, current)
        return current == root


# Build a Merkle tree
blocks = [f"Transaction {i}: Alice pays Bob ${i * 10}".encode() for i in range(8)]
tree = MerkleTree(blocks)

print(f"\n  Data blocks: {len(blocks)} transactions")
print(f"  Tree levels: {len(tree.tree)}")
print(f"  Root hash:   {tree.root[:32]}...")
print()

# Show tree structure
print("  Tree structure (abbreviated hashes):")
for i, level in enumerate(tree.tree):
    indent = "  " * (len(tree.tree) - i)
    hashes = " ".join(h[:8] for h in level)
    label = "Root" if i == len(tree.tree) - 1 else f"L{i}"
    print(f"  {indent}{label}: [{hashes}]")
print()

# Generate and verify a Merkle proof
target_idx = 3
proof = tree.get_proof(target_idx)
print(f"  -- Merkle Proof for block {target_idx} --")
print(f"  Block data: {blocks[target_idx].decode()}")
print(f"  Proof path ({len(proof)} hashes):")
for direction, h in proof:
    print(f"    {direction}: {h[:16]}...")

is_valid = MerkleTree.verify_proof(blocks[target_idx], proof, tree.root)
print(f"  Proof valid: {is_valid}")

# Tampered data
is_valid_tampered = MerkleTree.verify_proof(b"TAMPERED DATA", proof, tree.root)
print(f"  Tampered proof valid: {is_valid_tampered}")
print()


# ============================================================
# Section 6: Summary
# ============================================================

print("=" * 65)
print("  Summary")
print("=" * 65)
print("""
  Algorithm     | Output   | Speed    | Use Case
  --------------+----------+----------+---------------------------
  SHA-256       | 256 bit  | Fast     | Data integrity, Git
  SHA-3         | 256 bit  | Medium   | Quantum-resistant backup
  BLAKE2b       | 512 bit  | Fastest  | General purpose, libsodium
  PBKDF2        | Variable | Slow*    | Password hashing
  HMAC-SHA256   | 256 bit  | Fast     | Message authentication
  Merkle Tree   | 256 bit  | O(log n) | Blockchain, Git, IPFS

  * Slow by design - prevents brute-force attacks

  Key Takeaways:
  - Use PBKDF2/bcrypt/Argon2 for passwords (NEVER plain SHA-256)
  - Use HMAC for message authentication (not bare hashes)
  - Use constant-time comparison for security-critical comparisons
  - Merkle trees enable efficient verification of large datasets
""")
