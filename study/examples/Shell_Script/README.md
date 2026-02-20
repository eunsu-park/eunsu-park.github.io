# Shell Script Examples

This directory contains practical examples for advanced bash scripting topics.

## Directory Structure

```
Shell_Script/
├── 02_parameter_expansion/    # Parameter expansion and declare
├── 03_arrays/                 # Array manipulation
├── 05_function_library/       # Reusable function libraries
├── 06_io_redirection/         # Advanced I/O redirection
├── 08_regex/                  # Regex validation and extraction
├── 09_process_management/     # Parallel execution and cleanup
├── 10_error_handling/         # Error framework and safe operations
├── 11_argument_parsing/       # getopts, getopt, CLI tools
├── 13_disaster_recovery/      # Disaster recovery backup/restore
├── 13_testing/                # Bats tests and test runner
├── 14_perf_diagnosis/         # Performance bottleneck diagnosis
├── 14_task_runner/            # Makefile-like task runner project
├── 15_deployment/             # Deployment automation scripts
├── 16_monitoring/             # System monitoring dashboard
└── README.md                  # This file
```

## Examples Overview

### 02_parameter_expansion/

Advanced string manipulation and variable attributes.

#### string_ops.sh
Demonstrates parameter expansion for string operations:
- Path component extraction (filename, directory, extension)
- Batch file renaming patterns
- Config file parsing (key=value format)
- URL parsing (protocol, host, port, path)
- Default values and substitutions
- String length and substrings
- Case conversion

**Run:**
```bash
cd 02_parameter_expansion
./string_ops.sh
```

#### declare_demo.sh
Shows various uses of the `declare` builtin:
- Integer arithmetic with `declare -i`
- Readonly variables with `declare -r`
- Case conversion with `declare -l/-u`
- Name references with `declare -n`
- Array declarations (indexed and associative)
- Export variables with `declare -x`
- Function-local variables

**Run:**
```bash
cd 02_parameter_expansion
./declare_demo.sh
```

### 03_arrays/

Practical applications of bash arrays.

#### assoc_arrays.sh
Demonstrates associative arrays (hash maps):
- Word frequency counter
- Config file loader (key=value into associative array)
- Simple phonebook (add, lookup, delete, list operations)
- Nested data structures using key prefixes

**Run:**
```bash
cd 03_arrays
./assoc_arrays.sh
```

**Features:**
- Reads from stdin or uses sample data
- Interactive-style demonstrations
- Practical real-world use cases

#### csv_parser.sh
CSV file parser with advanced features:
- Parse CSV files handling quoted fields
- Query rows by column value
- Pretty-print as formatted table
- Extract column values
- Handle commas inside quoted fields

**Run:**
```bash
cd 03_arrays
./csv_parser.sh
```

**Usage as library:**
```bash
source csv_parser.sh
parse_csv "data.csv"
print_table
query_by_column "Name" "Alice"
get_column "Email"
```

### 05_function_library/

Reusable function libraries for common tasks.

#### lib/logging.sh
Comprehensive logging library:
- Log levels: DEBUG, INFO, WARN, ERROR
- Colored console output with timestamps
- Optional file logging
- Configurable log level filtering
- Section headers and separators

**Features:**
- `log_debug()`, `log_info()`, `log_warn()`, `log_error()`
- `set_log_level(level_name)`
- `enable_file_logging(filename)`
- `disable_file_logging()`
- `log_separator()`, `log_section(title)`

#### lib/validation.sh
Input validation functions:
- Email validation
- IP address validation (IPv4)
- Port number validation (1-65535)
- URL validation
- Hostname validation
- MAC address validation
- Hex color validation (#RGB, #RRGGBB)
- Semantic version validation (semver)
- Date validation (YYYY-MM-DD)
- Integer checks (is_integer, is_positive, is_non_negative, is_in_range)
- Path validation (file/directory exists)
- Command existence check
- String blank/not-blank checks

#### demo.sh
Demonstrates both libraries with comprehensive examples.

**Run:**
```bash
cd 05_function_library
./demo.sh
```

**Usage in your scripts:**
```bash
#!/usr/bin/env bash
source "lib/logging.sh"
source "lib/validation.sh"

set_log_level DEBUG
enable_file_logging "/var/log/myapp.log"

log_info "Starting application"

if ! validate_email "$user_email"; then
    log_error "Invalid email address"
    exit 1
fi

log_info "Email validated successfully"
```

### 06_io_redirection/

Advanced I/O redirection techniques.

#### fd_demo.sh
File descriptor demonstrations:
- Custom file descriptors (exec 3>file, 4>file)
- Redirecting stdout and stderr separately
- Swapping stdout and stderr
- Logging to both console and file simultaneously
- Read/write file descriptors
- Saving and restoring stdout
- Here-documents with file descriptors
- Noclobber mode and forced overwrites

**Run:**
```bash
cd 06_io_redirection
./fd_demo.sh
```

#### process_sub.sh
Process substitution examples:
- Comparing output of two commands with `diff <(cmd1) <(cmd2)`
- Feeding multiple inputs to commands
- Avoiding subshell variable scope issues
- Reading multiple streams simultaneously
- Write process substitution `>(cmd)`
- Complex data processing pipelines
- Practical log analysis example

**Run:**
```bash
cd 06_io_redirection
./process_sub.sh
```

**Key patterns:**
```bash
# Compare outputs
diff <(cmd1) <(cmd2)

# Multiple inputs
paste <(gen1) <(gen2) <(gen3)

# Avoid subshell variable loss
while read line; do
    ((count++))
done < <(command)

# Split output to multiple destinations
command | tee >(proc1) >(proc2) > output.txt
```

#### fifo_demo.sh
Named pipe (FIFO) demonstrations:
- Basic FIFO usage
- Producer-consumer pattern
- Bidirectional communication (two FIFOs)
- Load balancing with multiple workers
- Pipeline stages using FIFOs
- Timeout handling with FIFOs
- Automatic cleanup with trap

**Run:**
```bash
cd 06_io_redirection
./fifo_demo.sh
```

**Features:**
- Creates temporary FIFOs in /tmp
- Background processes for producers/consumers
- Clean signal handling and cleanup
- Practical patterns for IPC

### 08_regex/

Regular expression patterns for validation and data extraction.

#### validate.sh
Input validation using bash regex (`=~` and `BASH_REMATCH`):
- Email, IPv4, date, semver, and URL validation
- Comprehensive pass/fail demonstrations

#### extract.sh
Data extraction using regex:
- URL, email, and phone number extraction from text
- Log line parsing with structured output
- CSV parsing with quoted field handling

**Run:**
```bash
cd 08_regex
./validate.sh
./extract.sh
```

### 09_process_management/

Parallel execution and signal handling patterns.

#### parallel.sh
Parallel execution with concurrency control:
- `run_parallel()` with configurable concurrency limit
- PID tracking and exit code collection
- Sequential vs parallel performance comparison

#### cleanup.sh
Signal handling and cleanup patterns:
- Trap handlers for EXIT, INT, TERM, HUP
- Automatic temp file/directory cleanup
- Lock file management
- Graceful shutdown for long-running scripts

**Run:**
```bash
cd 09_process_management
./parallel.sh
./cleanup.sh
```

### 10_error_handling/

Reusable error handling frameworks.

#### error_framework.sh
Comprehensive error handling:
- Predefined error codes (E_SUCCESS, E_INVALID_ARG, etc.)
- Stack trace with BASH_LINENO and FUNCNAME
- Try/catch simulation using subshells

#### safe_ops.sh
Safe file and command operations:
- `safe_cd()`, `safe_rm()`, `safe_write()`, `safe_cp()`
- `retry()` with exponential backoff
- `require_cmd()` for dependency checking

**Run:**
```bash
cd 10_error_handling
./error_framework.sh
./safe_ops.sh
```

### 11_argument_parsing/

CLI argument parsing patterns.

#### getopts_demo.sh
POSIX getopts demonstration:
- Short options: `-v`, `-o FILE`, `-n NUM`, `-h`
- Proper usage function and error handling
- Positional argument processing

#### cli_tool.sh
Professional CLI tool with:
- Short and long options (`--verbose`, `--help`, `--version`)
- Progress bar and spinner implementations
- Colored output with verbose/quiet modes

**Run:**
```bash
cd 11_argument_parsing
./getopts_demo.sh -v -o output.txt file1 file2
./cli_tool.sh --help
```

### 13_disaster_recovery/

Disaster recovery backup and restore scripts.

#### dr_backup.sh
Automated disaster recovery backup script (~120 lines):
- Full system backup (filesystem, databases, config)
- PostgreSQL database dumps with pg_dump
- Compression (gzip) and encryption (gpg)
- Remote backup transfer via rsync
- Backup rotation policy (daily/weekly/monthly)
- Email notifications on success/failure
- Comprehensive logging with timestamps
- Incremental backup support

**Features:**
- Configurable backup sources
- Database backup with custom format
- Secure GPG encryption
- Remote transfer to backup server
- Automatic cleanup of old backups
- Error handling and validation

**Run:**
```bash
cd 13_disaster_recovery
./dr_backup.sh                  # Full backup
./dr_backup.sh -n               # Local only (no remote)
./dr_backup.sh -i               # Incremental backup
./dr_backup.sh --help           # Show usage
```

#### dr_restore.sh
Restore from disaster recovery backups (~100 lines):
- List available backups by date
- Selective restoration (all/filesystem/database/config)
- Backup integrity verification with checksums
- Decryption and decompression
- Database restoration with pg_restore
- Dry-run mode for safety
- Comprehensive logging

**Features:**
- Interactive backup selection
- SHA256 checksum verification
- Selective component restore
- Safe dry-run testing
- Automatic decryption/decompression

**Run:**
```bash
cd 13_disaster_recovery
./dr_restore.sh --list                      # List backups
./dr_restore.sh -d 20240215_120000          # Restore specific backup
./dr_restore.sh -t database                 # Restore databases only
./dr_restore.sh -n -d 20240215_120000       # Dry-run mode
./dr_restore.sh -v -d 20240215_120000       # Verify integrity
```

### 13_testing/

Shell script testing with Bats framework.

#### math_lib.sh
A math library to be tested: `add()`, `subtract()`, `multiply()`, `divide()`, `factorial()`, `is_prime()`, `gcd()`, `lcm()`.

#### test_math_lib.bats
Complete Bats test suite with 50+ test cases covering normal, edge, and error cases.

#### run_tests.sh
Automated test runner: runs Bats tests, ShellCheck validation, and generates summary reports.

**Run:**
```bash
cd 13_testing
./run_tests.sh          # Run all tests
bats test_math_lib.bats # Run Bats tests directly
```

### 14_perf_diagnosis/

Performance bottleneck diagnosis tools.

#### bottleneck_finder.sh
System performance bottleneck analyzer (~150 lines):
- CPU usage analysis with mpstat/top
- Load average monitoring (1/5/15 min)
- Memory and swap usage analysis
- Disk I/O statistics with iostat
- Network interface statistics
- Process resource consumption
- Automatic bottleneck detection
- Color-coded warnings (green/yellow/red)

**Features:**
- Comprehensive system metrics collection
- Configurable thresholds (warning/critical)
- Top CPU and memory consumers
- Zombie process detection
- Network connection analysis
- Summary report with recommendations
- Export to file capability

**Run:**
```bash
cd 14_perf_diagnosis
./bottleneck_finder.sh                      # Full analysis
./bottleneck_finder.sh -a cpu               # CPU analysis only
./bottleneck_finder.sh -a memory            # Memory analysis only
./bottleneck_finder.sh -o report.txt        # Export to file
./bottleneck_finder.sh -v                   # Verbose mode
```

**Analysis Types:**
- `cpu`: CPU usage, load average, top consumers
- `memory`: RAM/swap usage, memory consumers
- `disk`: Disk space, I/O statistics
- `network`: Interface stats, connections
- `process`: Process counts, top CPU/memory users
- `all`: Complete system analysis (default)

### 14_task_runner/

Complete task runner project (from Lesson 14).

#### task.sh
Makefile-like task runner:
- Task discovery via `task::*` naming convention
- Dependency resolution with `depends_on`
- Help generation from `##` comments
- Colored output with timestamps

**Run:**
```bash
cd 14_task_runner
./task.sh --help
./task.sh build
./task.sh deploy
```

### 15_deployment/

Deployment automation scripts (from Lesson 15).

#### deploy.sh
Production-grade deployment automation:
- SSH-based remote execution and file syncing
- Rolling deployment with health checks
- Rollback support with release history
- Multi-environment support (staging/production)

#### entrypoint.sh
Docker container entrypoint script:
- Environment variable validation
- Template processing with envsubst
- Wait-for-it functionality for dependencies
- Signal handling for graceful shutdown

**Run:**
```bash
cd 15_deployment
./deploy.sh --help
./deploy.sh status --env staging
```

### 16_monitoring/

System monitoring tools (from Lesson 16).

#### monitor.sh
Interactive terminal dashboard:
- Real-time CPU, memory, disk, load metrics
- Terminal UI with tput (colors, boxes, cursor positioning)
- Color-coded alerts (green/yellow/red)
- Cross-platform support (Linux and macOS)

#### health_check.sh
Cron-safe health check automation:
- System resource monitoring with thresholds
- Process and HTTP endpoint checks
- Webhook alerting (Slack-compatible)
- Idempotent design for repeated cron execution

**Run:**
```bash
cd 16_monitoring
./monitor.sh            # Interactive dashboard
./health_check.sh       # One-shot health check (for cron)
```

## General Usage Notes

### Making Scripts Executable

All scripts are already set up with the shebang `#!/usr/bin/env bash`, but you may need to make them executable:

```bash
chmod +x script_name.sh
```

### Script Safety

All scripts use:
```bash
set -euo pipefail
```

This means:
- `-e`: Exit on error
- `-u`: Error on undefined variables
- `-o pipefail`: Pipelines fail if any command fails

### Sourcing vs Executing

Some scripts can be both:
- **Executed**: `./script.sh` - Runs the demo
- **Sourced**: `source script.sh` - Loads functions for use in your shell

Libraries in `05_function_library/lib/` should be sourced, not executed.

### Temporary Files

Scripts that create temporary files use `/tmp` and clean up after themselves. The FIFO demo uses `trap` to ensure cleanup even if interrupted.

### Platform Notes

These scripts are tested on:
- Linux (bash 4.0+)
- macOS (bash 3.2+ with some limitations on features like `declare -l/-u`)

Some features require bash 4.0+:
- Associative arrays
- `declare -l/-u` (case conversion)
- `${var,,}` and `${var^^}` syntax

## Best Practices Demonstrated

1. **Error Handling**: Using `set -euo pipefail` and checking return codes
2. **Input Validation**: Comprehensive validation library
3. **Logging**: Structured logging with levels and colors
4. **Code Reuse**: Separating reusable functions into libraries
5. **Documentation**: Comments explaining complex operations
6. **Cleanup**: Proper resource cleanup with trap handlers
7. **Portability**: Using `#!/usr/bin/env bash` for compatibility

## Learning Path

Suggested order to study these examples:

1. **02_parameter_expansion/** - Master string manipulation and declare
2. **03_arrays/** - Understand associative arrays and CSV parsing
3. **05_function_library/** - Build reusable code
4. **06_io_redirection/** - Advanced I/O, process substitution, FIFOs
5. **08_regex/** - Regex validation and data extraction
6. **09_process_management/** - Parallel execution and signal handling
7. **10_error_handling/** - Error frameworks and safe operations
8. **11_argument_parsing/** - CLI tool development
9. **13_testing/** - Bats testing framework
10. **13_disaster_recovery/** - Backup and restore automation
11. **14_task_runner/** - Complete task runner project
12. **14_perf_diagnosis/** - Performance analysis tools
13. **15_deployment/** - Deployment automation
14. **16_monitoring/** - System monitoring dashboard

## Additional Resources

- Bash Reference Manual: https://www.gnu.org/software/bash/manual/
- Advanced Bash-Scripting Guide: https://tldp.org/LDP/abs/html/
- ShellCheck (linting): https://www.shellcheck.net/

## License

These examples are provided as educational material under the MIT License.
