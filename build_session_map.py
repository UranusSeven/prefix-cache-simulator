#!/usr/bin/env python3
"""
Build a trace_id -> session_id map from raw header logs.

Extracts Session-Id from the `msg` field (request header dump) and
pairs it with the `trace_id` field. Outputs a JSON file.

Usage:
    python build_session_map.py <header_log.jsonl> -o session_map.json
"""

import argparse
import json
import re
import sys


SESSION_ID_RE = re.compile(r"Session-Id:\[([^\]]+)\]")


def main():
    parser = argparse.ArgumentParser(
        description="Build trace_id -> session_id map from header logs"
    )
    parser.add_argument("log_file", help="JSONL header log file")
    parser.add_argument(
        "-o", "--output", default="session_map.json",
        help="Output JSON file (default: session_map.json)",
    )
    args = parser.parse_args()

    session_map = {}
    no_session = 0
    total = 0

    with open(args.log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            trace_id = entry.get("trace_id", "")
            msg = entry.get("msg", "")
            if not trace_id or not msg:
                continue

            total += 1
            m = SESSION_ID_RE.search(msg)
            if m:
                session_map[trace_id] = m.group(1)
            else:
                no_session += 1

    with open(args.output, "w") as f:
        json.dump(session_map, f)

    print(f"Processed {total} lines")
    print(f"  Mapped: {len(session_map)} trace_id -> session_id")
    print(f"  No Session-Id: {no_session}")
    print(f"  Unique sessions: {len(set(session_map.values()))}")
    print(f"  Written to: {args.output}")


if __name__ == "__main__":
    main()
