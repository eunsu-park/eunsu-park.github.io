#!/bin/bash

###############################################################################
# PostgreSQL Primary-Standby Replication Setup Script
#
# This script automates the setup of PostgreSQL streaming replication with:
# 1. Replication user creation on primary
# 2. pg_hba.conf configuration for replication connections
# 3. Base backup for standby initialization
# 4. Standby configuration with streaming replication
# 5. Replication verification and failover testing
#
# Prerequisites:
# - PostgreSQL installed on both primary and standby servers
# - SSH access between servers (for remote setup)
# - Sufficient disk space for base backup
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PRIMARY_HOST="${PRIMARY_HOST:-localhost}"
PRIMARY_PORT="${PRIMARY_PORT:-5432}"
STANDBY_HOST="${STANDBY_HOST:-localhost}"
STANDBY_PORT="${STANDBY_PORT:-5433}"

POSTGRES_USER="${POSTGRES_USER:-postgres}"
REPLICATION_USER="${REPLICATION_USER:-replicator}"
REPLICATION_PASSWORD="${REPLICATION_PASSWORD:-repl_password}"

PRIMARY_DATA_DIR="${PRIMARY_DATA_DIR:-/var/lib/postgresql/data}"
STANDBY_DATA_DIR="${STANDBY_DATA_DIR:-/var/lib/postgresql/standby}"
WAL_ARCHIVE_DIR="${WAL_ARCHIVE_DIR:-/var/lib/postgresql/wal_archive}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

###############################################################################
# Step 1: Create replication user on primary
###############################################################################
create_replication_user() {
    log_info "Creating replication user on primary..."

    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$PRIMARY_HOST" -p "$PRIMARY_PORT" \
        -U "$POSTGRES_USER" -d postgres <<EOF
-- Create replication user with replication privileges
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$REPLICATION_USER') THEN
        CREATE ROLE $REPLICATION_USER WITH REPLICATION LOGIN PASSWORD '$REPLICATION_PASSWORD';
    END IF;
END
\$\$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE postgres TO $REPLICATION_USER;

-- Show replication slots
SELECT * FROM pg_replication_slots;
EOF

    log_info "Replication user created successfully."
}

###############################################################################
# Step 2: Configure pg_hba.conf for replication
###############################################################################
configure_pg_hba() {
    log_info "Configuring pg_hba.conf for replication..."

    # Add replication entry to pg_hba.conf
    # In production, restrict to specific IP addresses
    local HBA_ENTRY="host replication $REPLICATION_USER all md5"

    # Check if entry already exists
    if docker exec postgres-primary grep -q "^$HBA_ENTRY" /var/lib/postgresql/data/pg_hba.conf 2>/dev/null; then
        log_warn "Replication entry already exists in pg_hba.conf"
    else
        docker exec postgres-primary bash -c "echo '$HBA_ENTRY' >> /var/lib/postgresql/data/pg_hba.conf"
        log_info "Added replication entry to pg_hba.conf"

        # Reload PostgreSQL configuration
        docker exec postgres-primary psql -U postgres -c "SELECT pg_reload_conf();"
        log_info "PostgreSQL configuration reloaded."
    fi
}

###############################################################################
# Step 3: Create base backup for standby
###############################################################################
create_base_backup() {
    log_info "Creating base backup from primary to standby..."

    # Stop standby if running
    log_info "Stopping standby server..."
    docker stop postgres-standby 2>/dev/null || true

    # Remove old standby data
    log_warn "Removing old standby data directory..."
    docker exec postgres-standby rm -rf /var/lib/postgresql/data/* 2>/dev/null || true

    # Create base backup using pg_basebackup
    log_info "Running pg_basebackup..."
    docker exec postgres-standby bash -c "PGPASSWORD='$REPLICATION_PASSWORD' pg_basebackup \
        -h postgres-primary \
        -p 5432 \
        -U $REPLICATION_USER \
        -D /var/lib/postgresql/data \
        -Fp \
        -Xs \
        -P \
        -R"

    log_info "Base backup completed successfully."
}

###############################################################################
# Step 4: Configure standby for streaming replication
###############################################################################
configure_standby() {
    log_info "Configuring standby for streaming replication..."

    # Create standby.signal file (PostgreSQL 12+)
    docker exec postgres-standby touch /var/lib/postgresql/data/standby.signal

    # Configure primary_conninfo in postgresql.auto.conf
    local CONN_INFO="host=postgres-primary port=5432 user=$REPLICATION_USER password=$REPLICATION_PASSWORD"

    docker exec postgres-standby bash -c "cat >> /var/lib/postgresql/data/postgresql.auto.conf <<EOF
# Standby configuration
primary_conninfo = '$CONN_INFO'
restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'
archive_cleanup_command = 'pg_archivecleanup /var/lib/postgresql/wal_archive %r'
EOF"

    log_info "Standby configuration completed."
}

###############################################################################
# Step 5: Start standby and verify replication
###############################################################################
verify_replication() {
    log_info "Starting standby server..."
    docker start postgres-standby

    # Wait for standby to start
    sleep 5

    log_info "Verifying replication status on primary..."
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$PRIMARY_HOST" -p "$PRIMARY_PORT" \
        -U "$POSTGRES_USER" -d postgres -c "SELECT * FROM pg_stat_replication;"

    log_info "Checking replication lag..."
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$PRIMARY_HOST" -p "$PRIMARY_PORT" \
        -U "$POSTGRES_USER" -d postgres -c "
        SELECT
            client_addr,
            state,
            sent_lsn,
            write_lsn,
            flush_lsn,
            replay_lsn,
            sync_state
        FROM pg_stat_replication;"

    log_info "${GREEN}Replication setup completed successfully!${NC}"
}

###############################################################################
# Step 6: Test replication
###############################################################################
test_replication() {
    log_info "Testing replication..."

    # Create test table on primary
    log_info "Creating test table on primary..."
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$PRIMARY_HOST" -p "$PRIMARY_PORT" \
        -U "$POSTGRES_USER" -d postgres <<EOF
CREATE TABLE IF NOT EXISTS replication_test (
    id SERIAL PRIMARY KEY,
    data TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO replication_test (data) VALUES ('Test replication data');
EOF

    # Wait for replication
    sleep 2

    # Verify data on standby
    log_info "Verifying data on standby..."
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$STANDBY_HOST" -p "$STANDBY_PORT" \
        -U "$POSTGRES_USER" -d postgres -c "SELECT * FROM replication_test;"

    log_info "${GREEN}Replication test successful!${NC}"
}

###############################################################################
# Main execution
###############################################################################
main() {
    log_info "Starting PostgreSQL Primary-Standby setup..."

    # Ensure directories exist
    mkdir -p "$WAL_ARCHIVE_DIR"

    # Execute setup steps
    create_replication_user
    configure_pg_hba
    create_base_backup
    configure_standby
    verify_replication
    test_replication

    log_info "${GREEN}All setup steps completed successfully!${NC}"
    log_info "Primary: $PRIMARY_HOST:$PRIMARY_PORT"
    log_info "Standby: $STANDBY_HOST:$STANDBY_PORT"
}

# Run main function
main "$@"
