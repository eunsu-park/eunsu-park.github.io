# Architecture Decision Records (ADR)

Architecture Decision Records (ADRs) capture important architectural decisions made
during a project. Each ADR documents the context, decision, and consequences so that
future team members understand *why* the system is built the way it is.

**Format**: Michael Nygard's ADR template (the most widely adopted convention).

---

## ADR Template

```markdown
# ADR-NNN: [Short noun phrase — what was decided]

**Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-NNN]
**Date**: YYYY-MM-DD
**Deciders**: [List of people involved in the decision]

## Context

What is the issue that motivates this decision?
Describe the forces at play — technical, political, social, project-related.
This section is value-neutral; state facts, not opinions.

## Decision

What is the change being proposed or that was decided?
Write in the active voice: "We will use X" rather than "X will be used".
Include alternatives that were considered and briefly explain why they were rejected.

## Consequences

What becomes easier or harder after this decision?
List positive consequences, negative consequences, and risks.
Be honest about trade-offs — an ADR with no negative consequences is suspect.
```

---

## Guidelines for Writing Effective ADRs

1. **Write ADRs at decision time.** Don't reconstruct history months later.
2. **Keep each ADR focused.** One significant decision per document.
3. **Immutable after acceptance.** Never edit a past ADR; supersede it with a new one.
4. **Link related ADRs.** Cross-reference when decisions build on each other.
5. **Short title, clear noun phrase.** "Use PostgreSQL as primary database" not "Database".
6. **Include rejected alternatives.** This is the most valuable part for future readers.
7. **Store in the repository.** `docs/adr/` next to the code it describes.

---

## Example ADR 1

# ADR-001: Use PostgreSQL as Primary Database

**Status**: Accepted
**Date**: 2025-03-10
**Deciders**: Alice Kim (Tech Lead), Bob Park (Backend), Carol Lee (DBA)

## Context

The e-commerce platform needs a relational database to store users, products, orders,
and inventory. We need ACID transactions for order processing, and the team anticipates
complex queries with joins across multiple tables. Initial data volume is expected to be
~5 million rows across all tables with modest write throughput (~200 writes/second peak).

We evaluated three options:

| Option         | Pros                                    | Cons                                      |
|----------------|-----------------------------------------|-------------------------------------------|
| PostgreSQL     | Mature, full ACID, JSONB, full-text     | Vertical scaling limits at extreme load   |
| MySQL 8        | Wide hosting support, faster simple reads | Weaker JSONB, less expressive SQL         |
| MongoDB        | Flexible schema, horizontal scaling     | No multi-document ACID (without sessions) |

## Decision

We will use **PostgreSQL 16** as the sole primary database for all structured data.

Specific choices within this decision:
- Use `JSONB` columns for semi-structured product attributes, avoiding a separate NoSQL store.
- Use row-level security (RLS) to enforce tenant isolation at the database layer.
- Use a managed instance (AWS RDS) to offload operational concerns (backups, patching).

MySQL was rejected because its JSONB support is inferior and our team has stronger
PostgreSQL expertise. MongoDB was rejected because order processing requires
cross-collection transactions that MongoDB handles less cleanly than PostgreSQL.

## Consequences

**Positive:**
- Full ACID guarantees simplify order and payment logic significantly.
- JSONB support eliminates the need for a separate document store for product attributes.
- RLS provides a robust second line of defence for multi-tenant isolation.
- Extensive tooling ecosystem (pgAdmin, Alembic, SQLAlchemy, pg_stat_statements).

**Negative / Risks:**
- PostgreSQL does not scale writes horizontally without significant architectural changes
  (e.g., Citus, read replicas). If write throughput grows beyond ~5,000 writes/second,
  this decision must be revisited (see ADR-007: Sharding Strategy).
- RDS costs more than a self-hosted instance; budget must account for this.
- Team must maintain migration discipline (Alembic) to prevent schema drift.

---

## Example ADR 2

# ADR-012: Adopt Microservices Architecture for Payment Service

**Status**: Accepted
**Date**: 2025-07-22
**Deciders**: Alice Kim (Tech Lead), David Oh (Platform), Eve Choi (Security)
**Supersedes**: ADR-003 (Monolithic architecture for all services)

## Context

The payment service currently lives inside the main Django monolith (see ADR-003).
Three pain points have emerged that the monolith cannot address cleanly:

1. **Deployment coupling**: A bug fix in the product catalogue requires redeploying
   the entire application, including the payment module. Any deployment carries risk
   for payment processing, which must have 99.95% uptime.
2. **Compliance isolation**: PCI-DSS compliance requires limiting the cardholder data
   environment (CDE) to as few systems as possible. Keeping payment logic in the
   monolith means the entire application is in scope, which triples audit complexity.
3. **Independent scaling**: Checkout traffic spikes (flash sales) require extra payment
   processing capacity without scaling unrelated services.

Alternatives considered:

| Option                          | Assessment                                                  |
|---------------------------------|-------------------------------------------------------------|
| Extract payment as a library    | Solves none of the three pain points                        |
| Strangler Fig to full microservices | Too broad; creates risk across the whole platform       |
| Extract *only* payment service  | Targeted, addresses all three pain points, manageable scope |

## Decision

We will extract the payment service into a **standalone microservice** using the
Strangler Fig pattern, while keeping the rest of the platform as a monolith.

Implementation decisions:
- **Language/Framework**: Python 3.12 + FastAPI (async, matches team skills).
- **Communication**: Synchronous REST for checkout flow; async events via RabbitMQ
  for post-payment notifications (receipt emails, inventory deduction).
- **Data**: Dedicated PostgreSQL schema, not shared with the monolith (database-per-service).
- **Auth**: Service-to-service calls authenticated with short-lived JWTs signed by
  an internal CA (not user-facing OAuth tokens).
- **Deployment**: Separate Kubernetes Deployment with HPA; isolated network policy
  limits ingress to the API gateway and the monolith only.

## Consequences

**Positive:**
- Payment service can be deployed independently — zero downtime for product catalogue releases.
- PCI-DSS CDE scope reduced from the whole monolith to one small service and its database.
- Can scale payment pods independently during flash sales without scaling catalogue services.
- Failure in the monolith (e.g., OOM) does not crash in-flight payment transactions.

**Negative / Risks:**
- **Distributed systems complexity**: Network failures between the monolith and payment
  service must be handled explicitly (retries, idempotency keys, circuit breakers).
  The monolith today handles this with a local function call.
- **Operational overhead**: Additional Kubernetes Deployment, service, and network policy
  to maintain. Observability must be extended (distributed tracing with OpenTelemetry).
- **Data consistency**: Without a shared database, refunds and order status updates require
  event-driven coordination. The team must implement the Saga pattern for the checkout flow.
- **Scope creep risk**: Extracting one service is justified. Extracting everything into
  microservices is not — this ADR explicitly does not authorise a full decomposition.
  Future extractions require separate ADRs.

**Follow-up actions:**
- ADR-013: Event schema for payment events on RabbitMQ
- ADR-014: Saga pattern implementation for checkout/refund flow
- Runbook: Payment service on-call playbook (escalation, rollback procedure)
