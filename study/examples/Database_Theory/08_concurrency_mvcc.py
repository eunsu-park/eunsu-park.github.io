"""
Concurrency Control: MVCC and 2PL

Demonstrates concurrency control mechanisms:
- Multi-Version Concurrency Control (MVCC): timestamp-based
- Two-Phase Locking (2PL): lock-based
- Read/write conflicts and serialization
- Deadlock detection and prevention

Theory:
- MVCC: Each transaction sees a consistent snapshot via versioning
  - Readers don't block writers, writers don't block readers
  - Uses transaction IDs/timestamps to determine visibility
- 2PL: Transactions acquire locks before accessing data
  - Growing phase: acquire locks
  - Shrinking phase: release locks
  - Strict 2PL: hold all locks until commit
- Conflicts: WW (write-write), WR (write-read), RW (read-write)

Note: This is a simplified simulation for educational purposes.
Production databases implement these with much more sophistication.
"""

import threading
import time
import random
from typing import Dict, Set, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass


class LockType(Enum):
    SHARED = "S"      # Read lock
    EXCLUSIVE = "X"   # Write lock


@dataclass
class Version:
    """A version of a data item in MVCC."""
    value: int
    txn_id: int       # Transaction that created this version
    timestamp: float
    committed: bool = False


class MVCCDatabase:
    """Simplified Multi-Version Concurrency Control implementation."""

    def __init__(self):
        self.data: Dict[str, List[Version]] = {}
        self.current_txn_id = 0
        self.lock = threading.Lock()

    def begin_transaction(self) -> int:
        """Start a new transaction and return its ID."""
        with self.lock:
            self.current_txn_id += 1
            return self.current_txn_id

    def read(self, key: str, txn_id: int) -> Optional[int]:
        """Read value for given key as of transaction's snapshot."""
        with self.lock:
            if key not in self.data:
                return None

            # Find latest committed version visible to this transaction
            # (created by transaction with ID <= txn_id)
            visible_versions = [
                v for v in self.data[key]
                if v.committed and v.txn_id <= txn_id
            ]

            if not visible_versions:
                return None

            # Return latest visible version
            latest = max(visible_versions, key=lambda v: v.txn_id)
            return latest.value

    def write(self, key: str, value: int, txn_id: int):
        """Create new version of data item."""
        with self.lock:
            if key not in self.data:
                self.data[key] = []

            # Create new version (uncommitted)
            version = Version(
                value=value,
                txn_id=txn_id,
                timestamp=time.time(),
                committed=False
            )
            self.data[key].append(version)

    def commit(self, txn_id: int):
        """Commit transaction, making its versions visible."""
        with self.lock:
            for versions in self.data.values():
                for version in versions:
                    if version.txn_id == txn_id:
                        version.committed = True

    def rollback(self, txn_id: int):
        """Rollback transaction, removing its versions."""
        with self.lock:
            for key in self.data:
                self.data[key] = [v for v in self.data[key] if v.txn_id != txn_id]

    def print_versions(self, key: str):
        """Print all versions of a key."""
        with self.lock:
            if key not in self.data:
                print(f"  {key}: (no versions)")
                return

            print(f"  {key}:")
            for v in sorted(self.data[key], key=lambda x: x.txn_id):
                status = "committed" if v.committed else "uncommitted"
                print(f"    T{v.txn_id}: {v.value} ({status})")


class TwoPhaseLockingDatabase:
    """Simplified Two-Phase Locking implementation."""

    def __init__(self):
        self.data: Dict[str, int] = {}
        self.locks: Dict[str, Set[Tuple[int, LockType]]] = {}  # key -> {(txn_id, lock_type)}
        self.lock = threading.Lock()
        self.wait_graph: Dict[int, Set[int]] = {}  # For deadlock detection

    def acquire_lock(self, key: str, txn_id: int, lock_type: LockType, timeout: float = 2.0) -> bool:
        """Acquire lock on key for transaction."""
        start_time = time.time()

        while True:
            with self.lock:
                if key not in self.locks:
                    self.locks[key] = set()

                current_locks = self.locks[key]

                # Check if lock can be granted
                can_grant = True
                if lock_type == LockType.SHARED:
                    # Shared lock: OK if no exclusive locks by other txns
                    for tid, lt in current_locks:
                        if tid != txn_id and lt == LockType.EXCLUSIVE:
                            can_grant = False
                            break
                else:  # EXCLUSIVE
                    # Exclusive lock: OK if no locks by other txns
                    for tid, lt in current_locks:
                        if tid != txn_id:
                            can_grant = False
                            break

                if can_grant:
                    self.locks[key].add((txn_id, lock_type))
                    return True

            # Check timeout
            if time.time() - start_time > timeout:
                return False  # Timeout (potential deadlock)

            time.sleep(0.01)  # Wait before retry

    def release_locks(self, txn_id: int):
        """Release all locks held by transaction."""
        with self.lock:
            for key in self.locks:
                self.locks[key] = {(tid, lt) for tid, lt in self.locks[key] if tid != txn_id}

    def read(self, key: str, txn_id: int) -> Optional[int]:
        """Read value with shared lock."""
        if not self.acquire_lock(key, txn_id, LockType.SHARED):
            raise Exception(f"T{txn_id}: Cannot acquire shared lock on {key} (deadlock?)")

        with self.lock:
            return self.data.get(key)

    def write(self, key: str, value: int, txn_id: int):
        """Write value with exclusive lock."""
        if not self.acquire_lock(key, txn_id, LockType.EXCLUSIVE):
            raise Exception(f"T{txn_id}: Cannot acquire exclusive lock on {key} (deadlock?)")

        with self.lock:
            self.data[key] = value

    def commit(self, txn_id: int):
        """Commit transaction and release locks (strict 2PL)."""
        self.release_locks(txn_id)


def demonstrate_mvcc():
    """Demonstrate MVCC snapshot isolation."""
    print("=" * 60)
    print("MULTI-VERSION CONCURRENCY CONTROL (MVCC)")
    print("=" * 60)
    print()

    db = MVCCDatabase()

    # Setup initial data
    txn0 = db.begin_transaction()
    db.write("x", 100, txn0)
    db.write("y", 200, txn0)
    db.commit(txn0)

    print("Initial state:")
    print("-" * 60)
    db.print_versions("x")
    db.print_versions("y")

    # Concurrent transactions
    print("\n\nConcurrent transactions:")
    print("-" * 60)

    # T1: Read x, sleep, read x again (repeatable read)
    txn1 = db.begin_transaction()
    print(f"T{txn1}: BEGIN")
    x1 = db.read("x", txn1)
    print(f"T{txn1}: READ x = {x1}")

    # T2: Modify x and commit
    txn2 = db.begin_transaction()
    print(f"T{txn2}: BEGIN")
    db.write("x", 150, txn2)
    print(f"T{txn2}: WRITE x = 150")
    db.commit(txn2)
    print(f"T{txn2}: COMMIT")

    # T1 reads again (should still see 100 due to snapshot isolation)
    x1_again = db.read("x", txn1)
    print(f"T{txn1}: READ x = {x1_again} (snapshot isolation - sees old version)")
    db.commit(txn1)
    print(f"T{txn1}: COMMIT")

    print("\n\nVersions after both commits:")
    print("-" * 60)
    db.print_versions("x")

    # T3: Read latest committed value
    txn3 = db.begin_transaction()
    x3 = db.read("x", txn3)
    print(f"\nT{txn3}: READ x = {x3} (sees latest committed version)")
    db.commit(txn3)

    print("\n✓ MVCC allows:")
    print("  - Readers don't block writers")
    print("  - Writers don't block readers")
    print("  - Each transaction sees consistent snapshot")
    print()


def demonstrate_2pl():
    """Demonstrate Two-Phase Locking."""
    print("=" * 60)
    print("TWO-PHASE LOCKING (2PL)")
    print("=" * 60)
    print()

    db = TwoPhaseLockingDatabase()

    # Initialize data
    db.data["x"] = 100
    db.data["y"] = 200

    print("Initial state: x=100, y=200")
    print("-" * 60)

    # Transaction 1: x = x + 10
    print("\nT1: Increment x by 10")
    txn1 = 1
    try:
        x = db.read("x", txn1)
        print(f"T1: READ x = {x} (acquired S lock)")
        time.sleep(0.1)
        db.write("x", x + 10, txn1)
        print(f"T1: WRITE x = {x + 10} (upgraded to X lock)")
        db.commit(txn1)
        print(f"T1: COMMIT (released locks)")
    except Exception as e:
        print(f"T1: ERROR - {e}")
        db.release_locks(txn1)

    print(f"\nFinal x = {db.data['x']}")

    # Demonstrate lock conflict
    print("\n\n" + "=" * 60)
    print("LOCK CONFLICTS")
    print("=" * 60)
    print()

    db.data["balance"] = 1000

    def transfer_out(amount: int, txn_id: int, delay: float):
        """Transfer money out."""
        try:
            print(f"T{txn_id}: Request X lock on balance")
            balance = db.read("balance", txn_id)
            print(f"T{txn_id}: READ balance = {balance}")
            time.sleep(delay)
            db.write("balance", balance - amount, txn_id)
            print(f"T{txn_id}: WRITE balance = {balance - amount}")
            db.commit(txn_id)
            print(f"T{txn_id}: COMMIT")
        except Exception as e:
            print(f"T{txn_id}: ABORTED - {e}")
            db.release_locks(txn_id)

    # Concurrent conflicting transactions
    print("Two transactions trying to modify balance concurrently:")
    print("-" * 60)

    t1 = threading.Thread(target=transfer_out, args=(100, 2, 0.5))
    t2 = threading.Thread(target=transfer_out, args=(200, 3, 0.3))

    t1.start()
    time.sleep(0.1)  # T2 starts slightly after T1
    t2.start()

    t1.join()
    t2.join()

    print(f"\nFinal balance = {db.data['balance']}")
    print("✓ 2PL ensures serializability through locking")
    print()


def demonstrate_deadlock():
    """Demonstrate deadlock scenario."""
    print("=" * 60)
    print("DEADLOCK SCENARIO")
    print("=" * 60)
    print()

    db = TwoPhaseLockingDatabase()
    db.data["x"] = 100
    db.data["y"] = 200

    print("Initial: x=100, y=200")
    print("-" * 60)
    print("\nScenario: T1 and T2 both need locks on x and y")
    print("  T1: Lock x, then y")
    print("  T2: Lock y, then x")
    print("  → Circular wait → Deadlock")
    print()

    deadlock_detected = [False]

    def transaction1():
        txn_id = 1
        try:
            print(f"T{txn_id}: Request lock on x")
            db.read("x", txn_id)
            print(f"T{txn_id}: Acquired lock on x")
            time.sleep(0.2)
            print(f"T{txn_id}: Request lock on y")
            db.read("y", txn_id)
            print(f"T{txn_id}: Acquired lock on y")
            db.commit(txn_id)
            print(f"T{txn_id}: COMMIT")
        except Exception as e:
            print(f"T{txn_id}: DEADLOCK DETECTED - {e}")
            deadlock_detected[0] = True
            db.release_locks(txn_id)

    def transaction2():
        txn_id = 2
        try:
            time.sleep(0.1)  # Start slightly after T1
            print(f"T{txn_id}: Request lock on y")
            db.read("y", txn_id)
            print(f"T{txn_id}: Acquired lock on y")
            time.sleep(0.2)
            print(f"T{txn_id}: Request lock on x")
            db.read("x", txn_id)
            print(f"T{txn_id}: Acquired lock on x")
            db.commit(txn_id)
            print(f"T{txn_id}: COMMIT")
        except Exception as e:
            print(f"T{txn_id}: DEADLOCK DETECTED - {e}")
            deadlock_detected[0] = True
            db.release_locks(txn_id)

    t1 = threading.Thread(target=transaction1)
    t2 = threading.Thread(target=transaction2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if deadlock_detected[0]:
        print("\n✓ Deadlock detected via timeout")
        print("\nDeadlock prevention strategies:")
        print("  1. Lock ordering: always acquire locks in same order")
        print("  2. Timeout: abort transaction if lock wait too long")
        print("  3. Wait-die/Wound-wait: timestamp-based schemes")
    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          CONCURRENCY CONTROL: MVCC AND 2PL                   ║
║  Multi-Version Concurrency Control, Two-Phase Locking        ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_mvcc()
    demonstrate_2pl()
    demonstrate_deadlock()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("MVCC (Multi-Version Concurrency Control):")
    print("  ✓ Readers don't block writers")
    print("  ✓ Snapshot isolation")
    print("  ✗ More storage (multiple versions)")
    print()
    print("2PL (Two-Phase Locking):")
    print("  ✓ Guarantees serializability")
    print("  ✗ Readers block writers (and vice versa)")
    print("  ✗ Deadlock possible")
    print()
    print("Modern databases often use hybrid approaches")
    print("  (e.g., MVCC with 2PL for writes)")
    print("=" * 60)
