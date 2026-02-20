# PostgreSQL Examples

This directory contains practical PostgreSQL examples demonstrating replication, high availability, and production configurations.

## Files

### 08_primary_standby_compose.yml

Docker Compose configuration for PostgreSQL streaming replication setup.

**Features:**
- Primary-Standby architecture with streaming replication
- WAL archiving for Point-In-Time Recovery (PITR)
- Health checks for both instances
- Shared volume for WAL archive
- Network isolation

**Usage:**
```bash
# Start the cluster
docker-compose -f 08_primary_standby_compose.yml up -d

# Check logs
docker-compose -f 08_primary_standby_compose.yml logs -f

# Connect to primary
docker exec -it postgres-primary psql -U postgres

# Connect to standby
docker exec -it postgres-standby psql -U postgres

# Stop the cluster
docker-compose -f 08_primary_standby_compose.yml down
```

**Ports:**
- Primary: 5432
- Standby: 5433

### 09_primary_standby_setup.sh

Automated setup script for PostgreSQL Primary-Standby replication.

**What it does:**
1. Creates replication user on primary
2. Configures pg_hba.conf for replication access
3. Creates base backup using pg_basebackup
4. Configures standby with streaming replication
5. Verifies replication status
6. Tests replication with sample data

**Usage:**
```bash
# Make script executable first
chmod +x 09_primary_standby_setup.sh

# With Docker Compose
./09_primary_standby_setup.sh

# Custom configuration
PRIMARY_HOST=db1.example.com \
STANDBY_HOST=db2.example.com \
REPLICATION_USER=repl_user \
./09_primary_standby_setup.sh
```

**Configuration Variables:**
- `PRIMARY_HOST`: Primary server hostname (default: localhost)
- `PRIMARY_PORT`: Primary server port (default: 5432)
- `STANDBY_HOST`: Standby server hostname (default: localhost)
- `STANDBY_PORT`: Standby server port (default: 5433)
- `REPLICATION_USER`: Replication user name (default: replicator)
- `REPLICATION_PASSWORD`: Replication user password

## Replication Concepts

### Streaming Replication

PostgreSQL streaming replication continuously streams WAL (Write-Ahead Log) records from the primary to standby servers.

**Advantages:**
- Near real-time replication
- Low latency
- Automatic synchronization
- Read replicas for load distribution

### WAL Archiving

WAL archiving provides a backup mechanism for point-in-time recovery.

**Key Settings:**
- `wal_level=replica`: Enable WAL logging for replication
- `archive_mode=on`: Enable WAL archiving
- `archive_command`: Command to archive WAL files
- `max_wal_senders`: Maximum number of concurrent replication connections

### Monitoring Replication

Check replication status on primary:
```sql
SELECT * FROM pg_stat_replication;
```

Check replication lag:
```sql
SELECT
    client_addr,
    state,
    pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes
FROM pg_stat_replication;
```

Check recovery status on standby:
```sql
SELECT pg_is_in_recovery();
SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();
```

## Failover Scenarios

### Manual Failover

1. **Stop primary** (simulated failure):
   ```bash
   docker stop postgres-primary
   ```

2. **Promote standby to primary**:
   ```bash
   docker exec postgres-standby pg_ctl promote -D /var/lib/postgresql/data
   ```

3. **Verify standby is now primary**:
   ```bash
   docker exec postgres-standby psql -U postgres -c "SELECT pg_is_in_recovery();"
   ```
   Should return `f` (false) indicating it's now a primary.

### Automatic Failover

For production environments, consider using:
- **Patroni**: HA solution with automatic failover
- **repmgr**: Replication manager for PostgreSQL
- **Pacemaker + Corosync**: Cluster resource manager

## Best Practices

1. **Network Security**
   - Use SSL/TLS for replication connections
   - Restrict `pg_hba.conf` entries to specific IP addresses
   - Use strong passwords for replication users

2. **Monitoring**
   - Monitor replication lag regularly
   - Set up alerts for replication failures
   - Track WAL archive size

3. **Backup Strategy**
   - Maintain both WAL archives and base backups
   - Test restore procedures regularly
   - Use multiple backup retention policies

4. **Performance**
   - Adjust `max_wal_senders` based on number of replicas
   - Configure `hot_standby_feedback` to prevent query conflicts
   - Monitor disk I/O on standby servers

## Troubleshooting

### Replication Not Starting

Check primary logs:
```bash
docker logs postgres-primary
```

Verify replication user permissions:
```sql
SELECT rolname, rolreplication FROM pg_roles WHERE rolname = 'replicator';
```

### Replication Lag Increasing

Check network connectivity:
```bash
docker exec postgres-standby ping postgres-primary
```

Monitor replication state:
```sql
SELECT * FROM pg_stat_replication;
```

### Standby Cannot Connect

Verify pg_hba.conf entry:
```bash
docker exec postgres-primary cat /var/lib/postgresql/data/pg_hba.conf | grep replication
```

Test connection from standby:
```bash
docker exec postgres-standby psql -h postgres-primary -U replicator -d postgres -c "SELECT 1;"
```

## References

- [PostgreSQL Replication Documentation](https://www.postgresql.org/docs/current/high-availability.html)
- [pg_basebackup](https://www.postgresql.org/docs/current/app-pgbasebackup.html)
- [Streaming Replication](https://www.postgresql.org/docs/current/warm-standby.html#STREAMING-REPLICATION)
