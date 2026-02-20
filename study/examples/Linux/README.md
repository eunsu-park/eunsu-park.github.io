# Linux Examples

Example scripts illustrating operational and diagnostic concepts covered in the Linux study topic.

## Scripts

### `disaster_recovery.sh`

Simulates a disaster recovery planning checklist. Teaches key DR concepts through hands-on verification:

- **Backup verification** — checks backup age against RPO threshold and validates file size
- **Service health checks** — confirms critical services are running using `systemctl` or `pgrep`
- **Database backup simulation** — mimics `pg_dump` output validation
- **Network connectivity** — ensures recovery-path hosts are reachable
- **Disk space monitoring** — flags filesystems that would block a restore
- **Recovery simulation** — walks through a step-by-step tabletop exercise (failover scenario)

```bash
chmod +x disaster_recovery.sh
./disaster_recovery.sh --check      # run all health checks
./disaster_recovery.sh --simulate   # walk through recovery steps
./disaster_recovery.sh --report     # full readiness report with pass/warn/fail counts
```

### `performance_diagnostics.sh`

Diagnoses CPU, memory, disk I/O, and network bottlenecks. Explains what each metric means and how to act on it:

- **CPU analysis** — load average, load/core ratio, top CPU consumers, zombie processes
- **Memory analysis** — RAM utilization, page cache, swap usage, top memory consumers
- **Disk I/O** — filesystem usage, `iostat` throughput and `%util`, `/proc/diskstats` fallback
- **Network analysis** — TCP connection states, listening ports, interface byte counters
- **Summary report** — consolidates all findings with severity (CRIT / WARN / OK) and remediation hints

```bash
chmod +x performance_diagnostics.sh
./performance_diagnostics.sh --all      # full analysis + bottleneck summary
./performance_diagnostics.sh --cpu      # CPU only
./performance_diagnostics.sh --memory   # memory only
./performance_diagnostics.sh --disk     # disk I/O only
./performance_diagnostics.sh --network  # network only
```

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| `bash` 4+ | Script runtime | built-in (macOS ships bash 3; use `brew install bash`) |
| `ps`, `df`, `ping` | Core metrics | built-in on Linux and macOS |
| `iostat` | Disk I/O statistics | `sudo apt install sysstat` / `brew install sysstat` |
| `ss` | Socket statistics | `sudo apt install iproute2` (Linux); use `netstat` on macOS |
| `netstat` | Network connections | built-in on macOS; `sudo apt install net-tools` on Linux |
| `systemctl` | Service status | systemd-based Linux distros only |
| `free` | Memory summary | Linux only; macOS uses `vm_stat` (handled automatically) |

Both scripts detect the OS at runtime and fall back gracefully when a tool is unavailable.
