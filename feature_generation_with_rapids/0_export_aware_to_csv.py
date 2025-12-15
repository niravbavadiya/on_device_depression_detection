#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export AWARE MySQL tables to weekly CSV chunks, grouped by participant (pid).

- Discovers all unique device_ids from the AWARE DB.
- Reads / updates a participant CSV mapping device_id -> pid.
- For each device_id/pid, exports weekly chunks into:

    <out_root>/<pid>/<YYYY-MM-DD_YYYY-MM-DD>/<table>.csv
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Set, Dict
import pandas as pd
import mysql.connector as mc

DEFAULT_TABLES = [
    "locations",
    "screen",
    "calls",
    "bluetooth",
    "wifi",
    "fused_geofences",   # duplicated into each chunk (no timestamp)
]

# ---------- time helpers (same as your existing script) ----------
def parse_time(s: str) -> int:
    """Return unix ms from ISO8601 or unix s/ms string."""
    s = s.strip()
    if s.isdigit() and len(s) >= 12:
        return int(s)  # assume ms
    if s.isdigit():
        return int(s) * 1000  # seconds -> ms
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception as e:
        raise SystemExit(f"Could not parse datetime '{s}': {e}")

def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def align_to_monday_00(dt: datetime) -> datetime:
    """Floor to Monday 00:00:00 (UTC). Monday=0."""
    dt0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    days_back = dt0.weekday()  # 0..6, Monday=0
    return dt0 - timedelta(days=days_back)

def next_monday(dt: datetime) -> datetime:
    dt0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    ahead = (7 - dt0.weekday()) % 7
    ahead = 7 if ahead == 0 else ahead
    return dt0 + timedelta(days=ahead)

def monday_windows(start_utc: datetime, end_utc_excl: datetime) -> List[Tuple[datetime, datetime]]:
    """Half-open windows [Monday00, nextMonday00) covering [start, end)."""
    if start_utc >= end_utc_excl:
        return []
    m_start = align_to_monday_00(start_utc)
    m_end   = next_monday(end_utc_excl - timedelta(seconds=1))
    windows = []
    cur = m_start
    while cur < m_end:
        nxt = cur + timedelta(days=7)
        windows.append((cur, nxt))
        cur = nxt
    return windows

# ---------- DB helpers ----------
def detect_global_range_mysql(conn, tables: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Find min/max timestamp (ms) across timestamped tables that exist."""
    min_ms, max_ms = None, None
    cur = conn.cursor()
    for t in tables:
        try:
            cur.execute("SHOW TABLES LIKE %s", (t,))
            if not cur.fetchone():
                continue
            cur.execute(f"SHOW COLUMNS FROM `{t}`")
            cols = [r[0] for r in cur.fetchall()]
            if "timestamp" not in cols:
                continue
            cur.execute(f"SELECT MIN(`timestamp`), MAX(`timestamp`) FROM `{t}`")
            row = cur.fetchone()
            if not row:
                continue
            mn, mx = row
            if mn is None or mx is None:
                continue
            mn = int(mn); mx = int(mx)
            min_ms = mn if (min_ms is None or mn < min_ms) else min_ms
            max_ms = mx if (max_ms is None or mx > max_ms) else max_ms
        except Exception:
            continue
    cur.close()
    return min_ms, max_ms

def get_unique_device_ids(conn, tables: List[str]) -> Set[str]:
    """Collect all distinct device_id values across tables that have a device_id column."""
    device_ids: Set[str] = set()
    cur = conn.cursor()
    for t in tables:
        try:
            cur.execute("SHOW TABLES LIKE %s", (t,))
            if not cur.fetchone():
                continue
            cur.execute(f"SHOW COLUMNS FROM `{t}`")
            cols = [r[0] for r in cur.fetchall()]
            if "device_id" not in cols:
                continue
            cur.execute(f"SELECT DISTINCT `device_id` FROM `{t}` WHERE `device_id` IS NOT NULL")
            for (dev,) in cur.fetchall():
                if dev:
                    device_ids.add(str(dev))
        except Exception as e:
            print(f"[warn] Could not inspect device IDs from table '{t}': {e}")
            continue
    cur.close()
    return device_ids

# ---------- participant CSV helpers ----------
PARTICIPANT_COLS = [
    "device_id",
    "fitbit_id",
    "empatica_id",
    "pid",
    "label",
    "platform",
    "start_date",
    "end_date",
]

def load_or_init_participants(csv_path: str) -> pd.DataFrame:
    """Load participant CSV or create an empty one with the required columns."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype=str)
        df = df.fillna("")
        # Ensure all required cols exist
        for col in PARTICIPANT_COLS:
            if col not in df.columns:
                df[col] = ""
        df = df[PARTICIPANT_COLS]
        print(f"[info] Loaded participant CSV with {len(df)} rows from: {csv_path}")
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame(columns=PARTICIPANT_COLS)
        print(f"[info] Participant CSV not found. Will create new at: {csv_path}")
    return df

def next_pid(existing_pids: List[str]) -> str:
    """Generate the next PID as P001, P002, ... based on existing PIDs."""
    max_n = 0
    for p in existing_pids:
        if not isinstance(p, str):
            continue
        p = p.strip()
        if p.startswith("P") and p[1:].isdigit():
            n = int(p[1:])
            if n > max_n:
                max_n = n
    new_n = max_n + 1 if max_n > 0 else 1
    return f"P{new_n:03d}"

def update_participants_with_devices(
    participants: pd.DataFrame,
    device_ids: Set[str],
    platform: str = "android",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Ensure every device_id in device_ids has a row in participants.
    Returns updated dataframe and mapping device_id -> pid.
    """
    participants = participants.copy()
    participants["device_id"] = participants["device_id"].astype(str).fillna("")
    participants["pid"] = participants["pid"].astype(str).fillna("")
    participants["label"] = participants["label"].astype(str).fillna("")

    # Existing mapping
    dev_to_pid: Dict[str, str] = {}
    for _, row in participants.iterrows():
        dev = row["device_id"].strip()
        pid = row["pid"].strip()
        if dev and pid:
            dev_to_pid[dev] = pid

    # Add new devices
    existing_pids = participants["pid"].tolist()
    added_rows = []
    for dev in sorted(device_ids):
        if not dev:
            continue
        if dev in dev_to_pid:
            continue
        new_pid = next_pid(existing_pids)
        existing_pids.append(new_pid)
        dev_to_pid[dev] = new_pid
        added_rows.append({
            "device_id": dev,
            "fitbit_id": "",
            "empatica_id": "",
            "pid": new_pid,
            "label": new_pid,      # label = pid for now
            "platform": platform,  # android for now
            "start_date": "",
            "end_date": "",
        })

    if added_rows:
        print(f"[info] Adding {len(added_rows)} new participant(s) for unseen device_id(s).")
        participants = pd.concat([participants, pd.DataFrame(added_rows)], ignore_index=True)

    # Sort by device_id for nicer CSV
    participants = participants.sort_values(by=["device_id"]).reset_index(drop=True)

    return participants, dev_to_pid

# ---------- export core (extended with optional device_id filter) ----------
def make_query(table: str, since_ms: Optional[int], until_ms: Optional[int]) -> str:
    return f"SELECT * FROM `{table}`"

def export_table(
    conn,
    table: str,
    out_dir: str,
    chunksize: int,
    mode: str,
    since_ms: Optional[int],
    until_ms: Optional[int],
    device_id: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if table == "wifi":
        out_path = os.path.join(out_dir, "wifi_visible.csv")
    else:
        out_path = os.path.join(out_dir, f"{table}.csv")
    if mode == "w" and os.path.exists(out_path):
        os.remove(out_path)

    # Table exists?
    cur = conn.cursor()
    cur.execute("SHOW TABLES LIKE %s", (table,))
    ok = cur.fetchone()
    if not ok:
        print(f"[warn] Table '{table}' not found. Skipping.")
        cur.close()
        return

    # Discover columns
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    cols = [r[0] for r in cur.fetchall()]
    cur.close()
    has_ts = "timestamp" in cols
    has_dev = "device_id" in cols

    # Build SQL + params
    base_sql = make_query(table, since_ms, until_ms)
    where = []
    params: List[object] = []

    if has_ts and since_ms is not None:
        where.append("`timestamp` >= %s")
        params.append(since_ms)
    if has_ts and until_ms is not None:
        where.append("`timestamp` < %s")
        params.append(until_ms)

    if device_id is not None and has_dev:
        where.append("`device_id` = %s")
        params.append(device_id)

    if where:
        base_sql += " WHERE " + " AND ".join(where)

    print(f"[info] Exporting {table} -> {out_path} (device_id={device_id})")
    wrote_any = False
    total = 0

    try:
        it = pd.read_sql(base_sql, conn, params=params, chunksize=chunksize)
        first = True
        for chunk in it:
            wrote_any = True
            n = len(chunk)
            total += n
            chunk.to_csv(
                out_path,
                index=False,
                mode=("w" if first and mode == "w" else "a"),
                header=(first or (mode == "w" and not os.path.exists(out_path))),
            )
            first = False

        if not wrote_any:
            df = pd.read_sql(base_sql, conn, params=params)
            n = len(df)
            if n > 0:
                df.to_csv(
                    out_path,
                    index=False,
                    mode=("w" if mode == "w" else "a"),
                    header=True,
                )
                total += n
                wrote_any = True

    except ValueError:
        df = pd.read_sql(base_sql, conn, params=params)
        n = len(df)
        if n > 0:
            df.to_csv(
                out_path,
                index=False,
                mode=("w" if mode == "w" else "a"),
                header=True,
            )
            total += n
            wrote_any = True

    # If still no rows, write header-only CSV
    if not wrote_any:
        pd.DataFrame(columns=cols).to_csv(
            out_path,
            index=False,
            mode=("a" if mode == "a" else "w"),
            header=True,
        )
        print(f"[ok] {table}: 0 rows (wrote header only, device_id={device_id})")
    else:
        print(f"[ok] {table}: {total} rows (device_id={device_id})")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Export AWARE MySQL tables to weekly CSV chunks, grouped by participant (pid) "
            "using a participant mapping CSV."
        )
    )
    ap.add_argument("--host", default=os.environ.get("AW_DB_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("AW_DB_PORT", "3306")))
    ap.add_argument("--user", default=os.environ.get("AW_DB_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("AW_DB_PASS", ""))
    ap.add_argument("--db", default=os.environ.get("AW_DB_NAME", "aware"))
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES),
                    help=f"Comma-separated list of tables to export. Default: {','.join(DEFAULT_TABLES)}")
    ap.add_argument("--since", help="Lower bound on timestamp (ISO8601 or unix ms/s). If omitted, auto-detect from DB.", default=None)
    ap.add_argument("--until", help="Upper bound (exclusive) on timestamp (ISO8601 or unix ms/s). If omitted, auto-detect from DB.", default=None)
    ap.add_argument("--out", help="Root output folder (defaults to your project aware_raw_data).")
    ap.add_argument(
        "--participants_csv",
        help=(
            "Path to participant_files CSV (device_id,fitbit_id,empatica_id,pid,label,platform,start_date,end_date). "
            "If omitted, defaults to '<project_root>/rapids_out/external/participant_files/participant_files.csv'."
        ),
        default=None,
    )
    ap.add_argument("--chunksize", type=int, default=250000, help="Chunk size for streaming reads (default 250k)")
    ap.add_argument("--append", action="store_true", help="Append within a week folder (normally not needed).")
    args = ap.parse_args()

    # Default out_root: aware_raw_data under your project
    default_out = r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\feature_generation_thesis"
    out_root = args.out or os.path.join(default_out, "aware_raw_data")
    os.makedirs(out_root, exist_ok=True)
    print(f"[info] Root output directory (aware_raw_data): {out_root}")

    # Derive default participants CSV if not provided
    if args.participants_csv:
        participants_csv = args.participants_csv
    else:
        project_root = os.path.dirname(out_root)  # assume aware_raw_data is under project root
        participants_csv = os.path.join(
            project_root,
            "rapids_out",
            "external",
            "participant_files",
            "participant_files.csv",
        )
    print(f"[info] Participant CSV path: {participants_csv}")

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    # Connect once
    print(f"[info] Connecting to MySQL {args.host}:{args.port} / {args.db} as {args.user}")
    try:
        conn = mc.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.db,
            autocommit=True,
        )
    except Exception as e:
        print(f"[error] Failed to connect to MySQL: {e}")
        sys.exit(1)

    # Determine global range
    if args.since and args.until:
        start_ms = parse_time(args.since)
        end_ms   = parse_time(args.until)
    else:
        mn, mx = detect_global_range_mysql(conn, tables)
        if mn is None or mx is None:
            print("[fatal] Could not detect any timestamps in DB. Provide --since and --until.")
            conn.close()
            sys.exit(2)
        start_ms = mn
        end_ms   = mx + 1  # exclusive

    start_dt = ms_to_dt(start_ms)
    end_dt   = ms_to_dt(end_ms)

    # Build Monday-aligned weekly windows
    windows = monday_windows(start_dt, end_dt)
    if not windows:
        print("[info] No windows to export.")
        conn.close()
        return

    print(f"[info] Will export {len(windows)} weekly chunk(s): {windows[0][0].date()} -> {windows[-1][1].date()}")

    mode = "a" if args.append else "w"

    # Get all unique device_ids from DB
    all_device_ids = get_unique_device_ids(conn, tables)
    print(f"[info] Found {len(all_device_ids)} unique device_id(s) in DB.")

    # Load / update participant CSV
    participants_df = load_or_init_participants(participants_csv)
    participants_df, dev_to_pid = update_participants_with_devices(participants_df, all_device_ids, platform="android")

    # Save updated participant CSV
    participants_df.to_csv(participants_csv, index=False)
    print(f"[ok] Participant CSV updated: {participants_csv}")

    # Filter mapping to only devices that actually exist in DB (just in case CSV has extras)
    device_pid_pairs = [(dev, pid) for dev, pid in dev_to_pid.items() if dev in all_device_ids and pid]

    if not device_pid_pairs:
        print("[info] No device/pid pairs to export. Nothing to do.")
        conn.close()
        return

    # Export each week for each device_id/pid
    for i, (w_start, w_end) in enumerate(windows, 1):
        since_dt = max(w_start, start_dt)
        until_dt = min(w_end,   end_dt)
        since_ms = dt_to_ms(since_dt)
        until_ms = dt_to_ms(until_dt)

        mon_label = w_start.date().isoformat()
        sun_label = (w_end - timedelta(days=1)).date().isoformat()

        print(f"\n=== Week {i}/{len(windows)}: {mon_label} -> {sun_label} ===")
        print(f"[info] Window: {since_dt.isoformat()} -> {until_dt.isoformat()}  (UTC)")

        for dev, pid in device_pid_pairs:
            pid_root = os.path.join(out_root, pid)
            week_dir = os.path.join(pid_root, f"{mon_label}_{sun_label}")
            os.makedirs(week_dir, exist_ok=True)

            print(f"\n[info] Exporting for device_id={dev}, pid={pid}")
            print(f"[info] Folder: {week_dir}")

            for t in tables:
                try:
                    export_table(
                        conn,
                        t,
                        week_dir,
                        args.chunksize,
                        mode,
                        since_ms,
                        until_ms,
                        device_id=dev,
                    )
                except Exception as e:
                    print(f"[warn] Failed to export {t} for pid={pid}, week {mon_label}->{sun_label}: {e}")

    conn.close()
    print("\n[done] Weekly exports complete.")
    print("[info] Check subfolders under (by pid):", out_root)

if __name__ == "__main__":
    main()
