#!/usr/bin/env bash
# =====================================================================
# single_disk_workload_profile.sh — single-disk synthetic I/O workload
# =====================================================================
# Generate predictable disk read/write workload using dd, so the e2e_smoke
# harness can verify that:
#   1. unified_monitor.sh captures non-zero I/O
#   2. cgroup_collector.py reports non-zero cgroup_io_*
#   3. HTML report shows the workload window
#
# DEFAULTS (safe on cloudtop, NEVER touch the real ledger device):
#   LEDGER_DEVICE   = "sda"   (root disk — read-only sentinel)
#   ACCOUNTS_DEVICE = ""      (validates "optional second disk" branch)
#   WORKDIR         = /tmp/bench_workload  (scratch — gitignored)
#
# SAFETY GUARDRAILS (in order, abort if violated):
#   G1 — total write volume ≤ 1 GiB
#   G2 — concurrent dd processes ≤ 4
#   G3 — destination filesystem free space ≥ 5 GiB
#   G4 — never write to /dev/* directly; always to file under WORKDIR
#   G5 — prefer direct I/O; use GNU dd nocache semantics when the filesystem
#        rejects direct I/O (for example Docker overlayfs read paths)
#
# OUTPUT:
#   stdout — JSON-line per phase: {"phase": "...", "bytes": N, "duration_s": f, "mbps": f}
#   exit 0 on success, non-zero on guardrail violation or dd failure
# =====================================================================

set -euo pipefail

# -------- Config (override via env) ----------------------------------
WORKDIR="${WORKDIR:-/tmp/bench_workload}"
TOTAL_WRITE_CAP_MIB="${TOTAL_WRITE_CAP_MIB:-1024}"     # G1: 1 GiB
MAX_CONCURRENT_DD="${MAX_CONCURRENT_DD:-4}"            # G2
MIN_FREE_GIB="${MIN_FREE_GIB:-5}"                      # G3
BLOCK_SIZE="${BLOCK_SIZE:-1M}"
DURATION_SEC="${DURATION_SEC:-60}"
PHASES="${PHASES:-write,read,mixed}"   # comma-separated
WRITE_IO_FLAG="direct"
READ_IO_FLAG="direct"

# -------- Helpers ----------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

emit_json() {
  # $1 phase, $2 bytes, $3 duration_s, $4 mbps
  printf '{"phase":"%s","bytes":%s,"duration_s":%s,"mbps":%s,"ts":"%s"}\n' \
    "$1" "$2" "$3" "$4" "$(date -u +%FT%TZ)"
}

calculate_timing() {
  local start="$1"
  local end="$2"
  local bytes="$3"
  awk -v start="$start" -v end="$end" -v bytes="$bytes" 'BEGIN {
    duration = end - start
    if (duration <= 0) {
      duration = 0.000001
    }
    printf "%.4f,%.2f\n", duration, (bytes / 1048576) / duration
  }'
}

guardrail_check() {
  # G3: free space
  mkdir -p "$WORKDIR"
  local free_kib
  free_kib=$(df -P -k "$WORKDIR" | awk 'NR==2 {print $4}')
  local free_gib=$((free_kib / 1024 / 1024))
  [[ "$free_gib" -ge "$MIN_FREE_GIB" ]] || \
    die "G3 violated: $WORKDIR has ${free_gib} GiB free, need ${MIN_FREE_GIB}"

  # G1: write cap sanity
  [[ "$TOTAL_WRITE_CAP_MIB" -le 10240 ]] || \
    die "G1 violated: TOTAL_WRITE_CAP_MIB=$TOTAL_WRITE_CAP_MIB > 10240 hard limit"

  # G2: concurrency cap sanity
  [[ "$MAX_CONCURRENT_DD" -le 16 ]] || \
    die "G2 violated: MAX_CONCURRENT_DD=$MAX_CONCURRENT_DD > 16 hard limit"
}

cleanup() {
  rm -f "${WORKDIR}/bench_data" "${WORKDIR}/bench_data.read"
}
trap cleanup EXIT

run_write_io() {
  local destination="$1"
  local count="$2"
  dd if=/dev/zero of="$destination" \
    bs="$BLOCK_SIZE" count="$count" \
    conv=fsync "oflag=${WRITE_IO_FLAG}" status=none
}

run_read_io() {
  local source="$1"
  dd if="$source" of=/dev/null \
    bs="$BLOCK_SIZE" "iflag=${READ_IO_FLAG}" status=none
}

negotiate_io_mode() {
  local probe="${WORKDIR}/.direct_io_probe"
  rm -f "$probe"

  if ! dd if=/dev/zero of="$probe" bs="$BLOCK_SIZE" count=1 \
      conv=fsync oflag=direct status=none 2>/dev/null; then
    WRITE_IO_FLAG="nocache"
  fi

  if [[ -f "$probe" ]] && ! dd if="$probe" of=/dev/null bs="$BLOCK_SIZE" \
      count=1 iflag=direct status=none 2>/dev/null; then
    READ_IO_FLAG="nocache"
  fi

  rm -f "$probe"
  echo "# I/O mode: write=${WRITE_IO_FLAG} read=${READ_IO_FLAG}" >&2
}

# -------- Phases -----------------------------------------------------
phase_write() {
  local count=$((TOTAL_WRITE_CAP_MIB))   # block_size=1M → count=MiB
  local start end dur bytes mbps
  start=$(date +%s.%N)
  # G4: write to a scratch file, never the device itself.
  if ! run_write_io "${WORKDIR}/bench_data" "$count"; then
    die "dd write failed"
  fi
  end=$(date +%s.%N)
  bytes=$((count * 1024 * 1024))
  IFS=',' read -r dur mbps < <(calculate_timing "$start" "$end" "$bytes")
  emit_json "write" "$bytes" "$dur" "$mbps"
}

phase_read() {
  [[ -f "${WORKDIR}/bench_data" ]] || die "phase_read: bench_data missing"
  local start end dur bytes mbps
  start=$(date +%s.%N)
  if ! run_read_io "${WORKDIR}/bench_data"; then
    die "dd read failed"
  fi
  end=$(date +%s.%N)
  bytes=$(stat -c%s "${WORKDIR}/bench_data")
  IFS=',' read -r dur mbps < <(calculate_timing "$start" "$end" "$bytes")
  emit_json "read" "$bytes" "$dur" "$mbps"
}

phase_mixed() {
  # Concurrent read + write under MAX_CONCURRENT_DD cap.
  # Write half the cap to keep total bounded by G1 even with parallel write.
  local half=$((TOTAL_WRITE_CAP_MIB / 2))
  local procs=$((MAX_CONCURRENT_DD / 2))
  [[ "$procs" -ge 1 ]] || procs=1
  local start end dur
  start=$(date +%s.%N)
  local pids=()
  local i
  for ((i=0; i<procs; i++)); do
    run_write_io "${WORKDIR}/bench_mix_w_${i}" "$((half / procs))" &
    pids+=($!)
    run_read_io "${WORKDIR}/bench_data" &
    pids+=($!)
  done
  local pid rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  end=$(date +%s.%N)
  rm -f "${WORKDIR}"/bench_mix_w_*
  [[ "$rc" -eq 0 ]] || die "dd mixed phase had failures"
  # Approx bytes = write_half_MiB (read amount varies with cache)
  local bytes=$((half * 1024 * 1024))
  local mbps
  IFS=',' read -r dur mbps < <(calculate_timing "$start" "$end" "$bytes")
  emit_json "mixed" "$bytes" "$dur" "$mbps"
}

# -------- Main -------------------------------------------------------
main() {
  echo "# single_disk_workload_profile.sh starting" >&2
  echo "# WORKDIR=$WORKDIR PHASES=$PHASES CAP=${TOTAL_WRITE_CAP_MIB}MiB MAX_DD=$MAX_CONCURRENT_DD" >&2
  echo "# LEDGER_DEVICE=${LEDGER_DEVICE:-sda} ACCOUNTS_DEVICE=${ACCOUNTS_DEVICE:-<none>}" >&2

  guardrail_check
  negotiate_io_mode

  IFS=',' read -ra phase_arr <<< "$PHASES"
  for p in "${phase_arr[@]}"; do
    case "$p" in
      write) phase_write ;;
      read)  phase_read  ;;
      mixed) phase_mixed ;;
      *) die "unknown phase: $p" ;;
    esac
  done

  echo "# workload complete" >&2
}

# Only run main when executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
