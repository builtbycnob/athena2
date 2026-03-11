#!/usr/bin/env python3
"""Monitor a running ATHENA simulation by parsing its log output.

Usage:
    python scripts/monitor.py <log_file>
    python scripts/monitor.py  # auto-detect latest task output
"""

import re
import sys
import os
import glob
import time


def parse_log(path: str) -> dict:
    with open(path) as f:
        lines = f.readlines()

    total_runs = 0
    concurrency = 1
    runs: list[dict] = []
    active_runs: dict[str, dict] = {}
    llm_calls = 0
    llm_tokens = 0
    llm_time = 0.0
    retries = 0
    repairs = []
    truncations = 0
    failure_artifacts = []
    cached_tokens = 0

    for line in lines:
        # Total runs + concurrency
        m = re.search(r"Starting (\d+) runs(?: \(concurrency=(\d+)\))?", line)
        if m:
            total_runs = int(m.group(1))
            if m.group(2):
                concurrency = int(m.group(2))

        # New run start
        m = re.search(r"\[MC\] Run (\d+)/(\d+): (.+)", line)
        if m:
            run_info = {
                "n": int(m.group(1)),
                "id": m.group(3),
                "agents": {},
                "status": "running",
                "time": None,
            }
            active_runs[m.group(3)] = run_info

        # Agent done — match to any active run by run_id prefix in the tag
        m = re.search(r"\[(.+?)\]\s+(Appellant|Respondent|Judge): done \((\d+\.\d+)s, valid=(\w+)\)", line)
        if m:
            agent = m.group(2).lower()
            tag = m.group(1)
            for rid, rinfo in active_runs.items():
                if tag in rid or rid in tag:
                    rinfo["agents"][agent] = {
                        "time": float(m.group(3)),
                        "valid": m.group(4) == "True",
                    }
                    break

        # Agent failed
        m = re.search(r"\[(.+?)\]\s+(Appellant|Respondent|Judge): FAILED", line)
        if m:
            tag = m.group(1)
            for rid, rinfo in active_runs.items():
                if tag in rid or rid in tag:
                    rinfo["agents"][m.group(2).lower()] = {"failed": True}
                    break

        # Run OK — new format: OK (Xs) — run_id
        m = re.search(r"\[MC\]\s+OK \((\d+\.\d+)s\) — (.+)", line)
        if m:
            run_id = m.group(2).strip()
            rinfo = active_runs.pop(run_id, None)
            if rinfo is None:
                # Fallback: legacy format "OK (Xs) — N/M succeeded so far"
                rinfo = next(iter(active_runs.values()), None)
                if rinfo:
                    active_runs.pop(rinfo["id"], None)
            if rinfo:
                rinfo["status"] = "ok"
                rinfo["time"] = float(m.group(1))
                runs.append(rinfo)

        # Run FAIL
        m = re.search(r"\[MC\]\s+FAIL \((\d+\.\d+)s\): (.+)", line)
        if m:
            elapsed = float(m.group(1))
            error = m.group(2)
            # Find the active run that failed (match by most recent start)
            rinfo = None
            for rid, ri in list(active_runs.items()):
                rinfo = ri
                active_runs.pop(rid, None)
                break
            if rinfo:
                rinfo["status"] = "fail"
                rinfo["time"] = elapsed
                rinfo["error"] = error
                runs.append(rinfo)

        # Run EXCEPTION
        m = re.search(r"\[MC\]\s+EXCEPTION \((\d+\.\d+)s\): (.+)", line)
        if m:
            elapsed = float(m.group(1))
            error = m.group(2)
            rinfo = None
            for rid, ri in list(active_runs.items()):
                rinfo = ri
                active_runs.pop(rid, None)
                break
            if rinfo:
                rinfo["status"] = "exception"
                rinfo["time"] = elapsed
                rinfo["error"] = error
                runs.append(rinfo)

        # LLM call
        m = re.search(r"\[LLM\] Call #(\d+): (\d+) prompt → (\d+) output tok, (\d+\.\d+)s \((\d+\.\d+) tok/s\)", line)
        if m:
            llm_calls = int(m.group(1))
            llm_tokens += int(m.group(3))
            llm_time += float(m.group(4))
            # Detect retries: multiple LLM calls without an agent "done" between them
            # (simplified: count LLM calls vs completed agents)

        # LLM cached tokens (oMLX prefix sharing)
        m = re.search(r"cached=(\d+)", line)
        if m:
            cached_tokens += int(m.group(1))

        # LLM repair
        m = re.search(r"\[LLM\] JSON repaired: (.+)", line)
        if m:
            repairs.append(m.group(1))

        # LLM truncation + retry
        m = re.search(r"\[LLM\] Truncated at", line)
        if m:
            truncations += 1

        # LLM failure artifact
        m = re.search(r"\[LLM\] Failure artifact saved: (.+)", line)
        if m:
            failure_artifacts.append(m.group(1))

        # Done line
        m = re.search(r"\[MC\] Done in (\d+\.\d+)s — (\d+)/(\d+)", line)
        if m:
            pass  # runs already tracked

    succeeded = sum(1 for r in runs if r["status"] == "ok")
    failed = sum(1 for r in runs if r["status"] in ("fail", "exception"))
    completed = succeeded + failed

    return {
        "total_runs": total_runs,
        "concurrency": concurrency,
        "completed": completed,
        "succeeded": succeeded,
        "failed": failed,
        "runs": runs,
        "active_runs": active_runs,
        "llm_calls": llm_calls,
        "llm_tokens": llm_tokens,
        "llm_time": llm_time,
        "repairs": repairs,
        "truncations": truncations,
        "failure_artifacts": failure_artifacts,
        "cached_tokens": cached_tokens,
    }


def format_report(data: dict) -> str:
    lines = []
    total = data["total_runs"] or 60
    concurrency = data.get("concurrency", 1)
    completed = data["completed"]
    succeeded = data["succeeded"]
    failed = data["failed"]
    remaining = total - completed
    in_progress = len(data.get("active_runs", {}))

    # Progress bar
    pct = completed / total * 100 if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * completed / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"\n  ATHENA Simulation Monitor (concurrency={concurrency})")
    lines.append(f"  {'─' * 45}")
    lines.append(f"  [{bar}] {pct:.0f}%")
    lines.append(f"  {completed}/{total} completed | {succeeded} ok | {failed} fail | {in_progress} active | {remaining} remaining")

    # Timing
    ok_runs = [r for r in data["runs"] if r["status"] == "ok" and r["time"]]
    if ok_runs:
        avg_time = sum(r["time"] for r in ok_runs) / len(ok_runs)
        eta_s = avg_time * remaining / concurrency
        eta_min = eta_s / 60
        total_elapsed = sum(r["time"] for r in data["runs"] if r["time"])
        lines.append(f"")
        lines.append(f"  Avg run time:  {avg_time:.0f}s")
        lines.append(f"  Elapsed:       {total_elapsed/60:.1f} min")
        lines.append(f"  ETA:           {eta_min:.0f} min ({remaining} runs × {avg_time:.0f}s / {concurrency} workers)")

    # LLM stats
    if data["llm_calls"] > 0:
        avg_tok_s = data["llm_tokens"] / data["llm_time"] if data["llm_time"] > 0 else 0
        calls_per_run = data["llm_calls"] / completed if completed > 0 else 0
        lines.append(f"")
        lines.append(f"  LLM calls:     {data['llm_calls']} ({calls_per_run:.1f}/run)")
        lines.append(f"  Total tokens:  {data['llm_tokens']:,}")
        lines.append(f"  Avg speed:     {avg_tok_s:.1f} tok/s")
        if data["cached_tokens"] > 0:
            cache_pct = data["cached_tokens"] / data["llm_tokens"] * 100 if data["llm_tokens"] > 0 else 0
            lines.append(f"  Cached tokens: {data['cached_tokens']:,} ({cache_pct:.0f}% prefix hit)")

    # Repair stats
    if data["repairs"] or data["truncations"] or data["failure_artifacts"]:
        lines.append(f"")
        lines.append(f"  Robustness:")
        if data["repairs"]:
            # Count repair types
            type_counts = {}
            for r in data["repairs"]:
                for fix in r.split(", "):
                    type_counts[fix] = type_counts.get(fix, 0) + 1
            type_str = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
            lines.append(f"    Repairs:     {len(data['repairs'])} ({type_str})")
        if data["truncations"]:
            lines.append(f"    Truncations: {data['truncations']} (retried with 2x budget)")
        if data["failure_artifacts"]:
            lines.append(f"    Artifacts:   {len(data['failure_artifacts'])} saved")
            for a in data["failure_artifacts"][-3:]:
                lines.append(f"      {a}")

    # Active runs
    active = data.get("active_runs", {})
    if active:
        lines.append(f"")
        lines.append(f"  Active runs ({len(active)}):")
        for rid, cr in list(active.items())[:8]:
            agents_done = list(cr["agents"].keys())
            status = f"agents: {', '.join(agents_done)}" if agents_done else "starting..."
            lines.append(f"    Run {cr['n']}/{total} — {cr['id']} ({status})")

    # Failed runs summary
    failed_runs = [r for r in data["runs"] if r["status"] in ("fail", "exception")]
    if failed_runs:
        lines.append(f"")
        lines.append(f"  Failed runs:")
        for r in failed_runs[-5:]:  # last 5
            err = r.get("error", "unknown")[:60]
            lines.append(f"    {r['id']}: {err}")

    lines.append("")
    return "\n".join(lines)


def find_latest_log() -> str | None:
    """Find the most recent task output file."""
    pattern = "/private/tmp/claude-*/tasks/*.output"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_log()
        if not path:
            print("No log file found. Usage: python scripts/monitor.py <log_file>")
            sys.exit(1)
        print(f"Auto-detected: {path}")

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    data = parse_log(path)
    print(format_report(data))


if __name__ == "__main__":
    main()
