Explain benchmark charts and bottleneck signals.

Use deterministic chart status and bottleneck rules as the primary evidence.

Chart interpretation rules:
- Empty chart: identify the missing or empty CSV that feeds it.
- Performance overview: use CPU, memory, disk throughput, IOPS, and utilization.
- CPU-disk correlation: distinguish CPU pressure from disk wait and queueing.
- Disk threshold charts: compare observed values with configured baselines when available.
- Per-method attribution: focus on configured workload methods, success/failure counts, and latency percentiles.
- Sync-health: explain height diff, reported lag, boolean health, or stale progress depending on the chain mode.

Bottleneck rules:
- High CPU + low IO wait + low disk utilization suggests CPU bottleneck.
- High CPU + high IO wait + high disk await/utilization suggests disk bottleneck.
- High disk utilization + high await suggests queueing latency.
- High per-method failure rate suggests RPC instability or fake-node fixture mismatch.
- Sync lag above threshold for the configured duration suggests node health risk.

Always include confidence and missing evidence.
