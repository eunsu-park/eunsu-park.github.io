"""
Authorization Models Demo
=========================

Educational demonstration of authorization concepts:
- RBAC (Role-Based Access Control)
- ABAC (Attribute-Based Access Control)
- ACL (Access Control List)
- Permission inheritance

Uses only Python standard library. No external dependencies required.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any
from datetime import datetime, time

print("=" * 65)
print("  Authorization Models Demo")
print("=" * 65)
print()


# ============================================================
# Section 1: RBAC (Role-Based Access Control)
# ============================================================

print("-" * 65)
print("  Section 1: RBAC (Role-Based Access Control)")
print("-" * 65)

print("""
  RBAC assigns permissions to roles, then roles to users.
  Users inherit all permissions from their assigned roles.

  User  -->  Role  -->  Permission
  Alice -->  Admin -->  [read, write, delete, manage_users]
  Bob   -->  Editor --> [read, write]
  Carol -->  Viewer --> [read]
""")


class Permission(Flag):
    """Bitwise permission flags for efficient storage."""
    NONE = 0
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    MANAGE_USERS = auto()
    MANAGE_ROLES = auto()
    ADMIN = READ | WRITE | DELETE | MANAGE_USERS | MANAGE_ROLES


@dataclass
class Role:
    """A named role with a set of permissions."""
    name: str
    permissions: Permission
    description: str = ""

    def has_permission(self, perm: Permission) -> bool:
        return (self.permissions & perm) == perm


@dataclass
class RBACUser:
    """A user with one or more roles."""
    username: str
    roles: list[Role] = field(default_factory=list)

    def has_permission(self, perm: Permission) -> bool:
        return any(role.has_permission(perm) for role in self.roles)

    def get_all_permissions(self) -> Permission:
        result = Permission.NONE
        for role in self.roles:
            result |= role.permissions
        return result


# Define roles
admin_role = Role("admin", Permission.ADMIN, "Full system access")
editor_role = Role("editor", Permission.READ | Permission.WRITE, "Can read and write")
viewer_role = Role("viewer", Permission.READ, "Read-only access")
moderator_role = Role(
    "moderator",
    Permission.READ | Permission.WRITE | Permission.DELETE,
    "Can moderate content",
)

# Assign roles to users
alice = RBACUser("alice", [admin_role])
bob = RBACUser("bob", [editor_role])
carol = RBACUser("carol", [viewer_role])
dave = RBACUser("dave", [editor_role, moderator_role])  # Multiple roles

print("\n  Users and their roles:")
for user in [alice, bob, carol, dave]:
    role_names = ", ".join(r.name for r in user.roles)
    perms = user.get_all_permissions()
    perm_names = [p.name for p in Permission if p in perms and p.name != "ADMIN"]
    print(f"    {user.username:<8} roles=[{role_names}]")
    print(f"             permissions={perm_names}")
print()

# Permission checks
print("  Permission checks:")
checks = [
    (alice, Permission.MANAGE_USERS, "Alice: manage users"),
    (bob, Permission.WRITE, "Bob: write content"),
    (bob, Permission.DELETE, "Bob: delete content"),
    (carol, Permission.READ, "Carol: read content"),
    (carol, Permission.WRITE, "Carol: write content"),
    (dave, Permission.DELETE, "Dave: delete content"),
]
for user, perm, desc in checks:
    result = user.has_permission(perm)
    status = "ALLOWED" if result else "DENIED"
    print(f"    {desc:<28} -> {status}")
print()


# ============================================================
# Section 2: ABAC (Attribute-Based Access Control)
# ============================================================

print("-" * 65)
print("  Section 2: ABAC (Attribute-Based Access Control)")
print("-" * 65)

print("""
  ABAC makes access decisions based on attributes of:
  - Subject (user): role, department, clearance level
  - Resource: type, classification, owner
  - Action: read, write, delete
  - Environment: time of day, IP address, location

  More flexible than RBAC but more complex to manage.
""")


@dataclass
class Subject:
    """The entity requesting access."""
    user_id: str
    department: str
    clearance_level: int  # 1=Public, 2=Internal, 3=Confidential, 4=Secret
    roles: list[str] = field(default_factory=list)


@dataclass
class Resource:
    """The object being accessed."""
    resource_id: str
    resource_type: str
    classification: int  # 1=Public, 2=Internal, 3=Confidential, 4=Secret
    owner_department: str
    owner_id: str


@dataclass
class Environment:
    """Contextual attributes."""
    current_time: time
    ip_address: str
    is_vpn: bool = False


class ABACPolicy:
    """A single ABAC policy rule."""

    def __init__(self, name: str, description: str,
                 condition: callable, effect: str = "allow"):
        self.name = name
        self.description = description
        self.condition = condition
        self.effect = effect  # "allow" or "deny"

    def evaluate(self, subject: Subject, resource: Resource,
                 action: str, environment: Environment) -> str | None:
        """Returns effect if condition matches, None otherwise."""
        if self.condition(subject, resource, action, environment):
            return self.effect
        return None


class ABACEngine:
    """Policy evaluation engine using deny-overrides combining."""

    def __init__(self):
        self.policies: list[ABACPolicy] = []

    def add_policy(self, policy: ABACPolicy):
        self.policies.append(policy)

    def evaluate(self, subject: Subject, resource: Resource,
                 action: str, environment: Environment) -> tuple[bool, list[str]]:
        """Evaluate all policies. Deny overrides allow."""
        reasons = []
        has_allow = False

        for policy in self.policies:
            result = policy.evaluate(subject, resource, action, environment)
            if result == "deny":
                reasons.append(f"DENIED by '{policy.name}': {policy.description}")
                return False, reasons
            elif result == "allow":
                has_allow = True
                reasons.append(f"ALLOWED by '{policy.name}'")

        if has_allow:
            return True, reasons
        reasons.append("DENIED: no matching allow policy")
        return False, reasons


# Create ABAC engine with policies
engine = ABACEngine()

# Policy 1: Clearance level must meet or exceed resource classification
engine.add_policy(ABACPolicy(
    "clearance_check",
    "Subject clearance must >= resource classification",
    lambda s, r, a, e: s.clearance_level >= r.classification,
    "allow",
))

# Policy 2: Deny access to Secret resources outside business hours
engine.add_policy(ABACPolicy(
    "business_hours_secret",
    "Secret resources only during business hours (9-18)",
    lambda s, r, a, e: (
        r.classification == 4 and not (time(9, 0) <= e.current_time <= time(18, 0))
    ),
    "deny",
))

# Policy 3: Deny write/delete to resources from other departments
engine.add_policy(ABACPolicy(
    "department_write_restriction",
    "Write/delete only within own department",
    lambda s, r, a, e: (
        a in ("write", "delete") and s.department != r.owner_department
    ),
    "deny",
))

# Policy 4: Require VPN for confidential+ resources
engine.add_policy(ABACPolicy(
    "vpn_required_confidential",
    "Confidential+ resources require VPN",
    lambda s, r, a, e: r.classification >= 3 and not e.is_vpn,
    "deny",
))

# Test scenarios
print("\n  ABAC Policy Evaluation Scenarios:")
print()

scenarios = [
    {
        "desc": "Engineer reads public docs (clearance 2, public resource)",
        "subject": Subject("eng_01", "engineering", 2, ["engineer"]),
        "resource": Resource("doc_1", "document", 1, "engineering", "eng_01"),
        "action": "read",
        "env": Environment(time(10, 0), "10.0.0.1", is_vpn=True),
    },
    {
        "desc": "Engineer reads secret doc without VPN",
        "subject": Subject("eng_02", "engineering", 4, ["engineer"]),
        "resource": Resource("doc_s", "document", 4, "engineering", "eng_02"),
        "action": "read",
        "env": Environment(time(14, 0), "203.0.113.5", is_vpn=False),
    },
    {
        "desc": "Marketing writes to engineering resource",
        "subject": Subject("mkt_01", "marketing", 2, ["analyst"]),
        "resource": Resource("spec_1", "spec", 2, "engineering", "eng_01"),
        "action": "write",
        "env": Environment(time(11, 0), "10.0.0.2", is_vpn=True),
    },
    {
        "desc": "Admin reads secret doc at night via VPN",
        "subject": Subject("adm_01", "security", 4, ["admin"]),
        "resource": Resource("doc_s2", "document", 4, "security", "adm_01"),
        "action": "read",
        "env": Environment(time(23, 0), "10.0.0.3", is_vpn=True),
    },
    {
        "desc": "Low-clearance user reads confidential resource",
        "subject": Subject("int_01", "intern", 1, ["intern"]),
        "resource": Resource("plan_1", "plan", 3, "engineering", "eng_01"),
        "action": "read",
        "env": Environment(time(10, 0), "10.0.0.4", is_vpn=True),
    },
]

for scenario in scenarios:
    allowed, reasons = engine.evaluate(
        scenario["subject"], scenario["resource"],
        scenario["action"], scenario["env"],
    )
    status = "ALLOWED" if allowed else "DENIED"
    print(f"  Scenario: {scenario['desc']}")
    print(f"    Result: {status}")
    for reason in reasons:
        print(f"    Reason: {reason}")
    print()


# ============================================================
# Section 3: ACL (Access Control List)
# ============================================================

print("-" * 65)
print("  Section 3: ACL (Access Control List)")
print("-" * 65)

print("""
  ACL attaches permissions directly to resources.
  Each resource has a list of (subject, permissions) entries.

  Similar to Unix file permissions:
    -rwxr-xr--  owner=rwx  group=r-x  others=r--
""")


class ACLPermission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class ACLEntry:
    """Single entry in an access control list."""
    principal: str      # user_id or group_id
    principal_type: str  # "user" or "group"
    permissions: set[ACLPermission]


class ACL:
    """Access Control List for a resource."""

    def __init__(self, resource_id: str, owner: str):
        self.resource_id = resource_id
        self.owner = owner
        self.entries: list[ACLEntry] = []

    def add_entry(self, principal: str, principal_type: str,
                  permissions: set[ACLPermission]):
        self.entries.append(ACLEntry(principal, principal_type, permissions))

    def check_access(self, user_id: str, groups: list[str],
                     permission: ACLPermission) -> bool:
        """Check if user has the specified permission."""
        # Owner always has full access
        if user_id == self.owner:
            return True

        for entry in self.entries:
            if entry.principal_type == "user" and entry.principal == user_id:
                if permission in entry.permissions:
                    return True
            elif entry.principal_type == "group" and entry.principal in groups:
                if permission in entry.permissions:
                    return True
        return False

    def display(self):
        """Display ACL in a readable format."""
        print(f"    Resource: {self.resource_id}  (owner: {self.owner})")
        for entry in self.entries:
            perms = ", ".join(p.value for p in entry.permissions)
            print(f"      {entry.principal_type}:{entry.principal} -> [{perms}]")


# Create ACL for a document
doc_acl = ACL("project_plan.docx", "alice")
doc_acl.add_entry("bob", "user", {ACLPermission.READ, ACLPermission.WRITE})
doc_acl.add_entry("carol", "user", {ACLPermission.READ})
doc_acl.add_entry("engineering", "group", {ACLPermission.READ})
doc_acl.add_entry("managers", "group",
                   {ACLPermission.READ, ACLPermission.WRITE, ACLPermission.DELETE})

print("\n  Document ACL:")
doc_acl.display()
print()

# ACL checks
acl_checks = [
    ("alice", [], ACLPermission.DELETE, "Alice (owner) delete"),
    ("bob", ["engineering"], ACLPermission.WRITE, "Bob write"),
    ("bob", ["engineering"], ACLPermission.DELETE, "Bob delete"),
    ("carol", [], ACLPermission.READ, "Carol read"),
    ("dave", ["engineering"], ACLPermission.READ, "Dave (engineering group) read"),
    ("dave", ["engineering"], ACLPermission.WRITE, "Dave (engineering group) write"),
    ("eve", ["managers"], ACLPermission.DELETE, "Eve (managers group) delete"),
]

print("  ACL Permission Checks:")
for user_id, groups, perm, desc in acl_checks:
    result = doc_acl.check_access(user_id, groups, perm)
    status = "ALLOWED" if result else "DENIED"
    print(f"    {desc:<36} -> {status}")
print()


# ============================================================
# Section 4: Permission Inheritance
# ============================================================

print("-" * 65)
print("  Section 4: Permission Inheritance")
print("-" * 65)

print("""
  Permissions can be inherited through hierarchies:
  - Role hierarchy:  Admin > Manager > Editor > Viewer
  - Resource hierarchy: Organization > Department > Project > File

  Example:
    /company
      /company/engineering        (engineering team: read/write)
      /company/engineering/proj1  (inherits from parent)
      /company/engineering/proj1/secret.md  (override: restricted)
""")


class PermissionNode:
    """A node in a hierarchical permission tree."""

    def __init__(self, name: str, parent: PermissionNode | None = None):
        self.name = name
        self.parent = parent
        self.children: list[PermissionNode] = []
        self.acl: dict[str, set[str]] = {}  # user -> permissions
        self.inherit: bool = True  # Whether to inherit from parent

        if parent:
            parent.children.append(self)

    def set_permissions(self, user: str, perms: set[str]):
        self.acl[user] = perms

    def get_effective_permissions(self, user: str) -> set[str]:
        """Get permissions considering inheritance chain."""
        # Own explicit permissions
        own_perms = self.acl.get(user, None)

        if own_perms is not None:
            # Explicit permissions override inheritance
            return own_perms

        # Inherit from parent if allowed
        if self.inherit and self.parent:
            return self.parent.get_effective_permissions(user)

        return set()  # No permissions

    def get_full_path(self) -> str:
        if self.parent:
            return f"{self.parent.get_full_path()}/{self.name}"
        return f"/{self.name}"


# Build resource hierarchy
company = PermissionNode("company")
engineering = PermissionNode("engineering", company)
marketing = PermissionNode("marketing", company)
proj1 = PermissionNode("proj1", engineering)
proj2 = PermissionNode("proj2", engineering)
secret_file = PermissionNode("secret.md", proj1)

# Set permissions at various levels
company.set_permissions("admin", {"read", "write", "delete", "manage"})
company.set_permissions("ceo", {"read"})
engineering.set_permissions("eng_team", {"read", "write"})
marketing.set_permissions("mkt_team", {"read", "write"})
proj1.set_permissions("proj1_lead", {"read", "write", "delete"})

# Override: secret.md has restricted access (no inheritance)
secret_file.inherit = False
secret_file.set_permissions("admin", {"read", "write"})
secret_file.set_permissions("proj1_lead", {"read"})

# Display hierarchy and effective permissions
print("\n  Resource Hierarchy and Permissions:")

all_nodes = [company, engineering, marketing, proj1, proj2, secret_file]
all_users = ["admin", "ceo", "eng_team", "mkt_team", "proj1_lead"]

for node in all_nodes:
    depth = 0
    n = node
    while n.parent:
        depth += 1
        n = n.parent
    indent = "  " * depth
    inherit_mark = "" if node.inherit else " [NO INHERIT]"
    print(f"    {indent}{node.name}{inherit_mark}")
    if node.acl:
        for user, perms in node.acl.items():
            print(f"    {indent}  -> {user}: {sorted(perms)}")

print()

# Test effective permissions
print("  Effective Permission Resolution:")
test_cases = [
    ("admin", company, "Admin at /company"),
    ("admin", proj1, "Admin at /company/engineering/proj1 (inherited)"),
    ("admin", secret_file, "Admin at secret.md (explicit override)"),
    ("eng_team", engineering, "eng_team at /engineering"),
    ("eng_team", proj1, "eng_team at /proj1 (inherited from engineering)"),
    ("eng_team", secret_file, "eng_team at secret.md (no inherit)"),
    ("proj1_lead", proj1, "proj1_lead at /proj1"),
    ("proj1_lead", secret_file, "proj1_lead at secret.md (read only)"),
    ("ceo", marketing, "CEO at /marketing (inherited from company)"),
]

for user, node, desc in test_cases:
    perms = node.get_effective_permissions(user)
    perm_str = sorted(perms) if perms else "NONE"
    print(f"    {desc}")
    print(f"      -> {perm_str}")
print()


# ============================================================
# Section 5: Summary
# ============================================================

print("=" * 65)
print("  Summary: Authorization Models Comparison")
print("=" * 65)
print("""
  Model | Complexity | Flexibility | Best For
  ------+------------+-------------+----------------------------
  RBAC  | Low        | Medium      | Most applications
  ABAC  | High       | Very High   | Complex enterprises, gov
  ACL   | Medium     | Medium      | File systems, documents
  ReBAC | Medium     | High        | Social networks, sharing

  RBAC:  Simple, role-centric. Good default choice.
  ABAC:  Policy-based, context-aware. Maximum flexibility.
  ACL:   Per-resource control. Good for file/document systems.

  Key Principles:
  - Principle of Least Privilege: grant minimum needed access
  - Separation of Duties: split critical tasks across roles
  - Defense in Depth: layer authorization checks
  - Deny by Default: require explicit allow
  - Audit everything: log all access decisions
""")
