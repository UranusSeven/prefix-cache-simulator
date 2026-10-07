#!/usr/bin/env python3
"""
WekaTrace adapter: convert semianalysisai/cc-traces-weka-* traces into the
prefix-cache-simulator's native pre-blocked JSONL format.

Input: traces.jsonl where each line is a session trace:
  {"id": ..., "block_size": 64, "hash_id_scope": "local",
   "requests": [{"t": ..., "model": ..., "in": ..., "out": ...,
                 "hash_ids": [...], "type": "s" | "subagent" | "n", ...}]}

Main-turn requests (type "s") and subagent inner requests (nested under
type "subagent" entries, type "n") are flattened into one stream per
session, sorted by relative timestamp t.

Output: one JSONL line per model request:
  {"trace_id": "<session>-<seq>", "timestamp": "<zero-padded t>",
   "model": ..., "group": "<session>", "namespace": "<session>",
   "block_size": <out block size>, "prompt_tokens": <in>,
   "block_ids": [...]}

hash_id_scope is "local" (ids are only unique within a session), so the
session id is carried in `namespace` — cross-session false hits are
impossible by construction.

With --superblock-tokens S (multiple of the trace block_size), every
S/64 consecutive source blocks are regrouped into one superblock identified
by the comma-joined source ids, and `block_size` becomes S. Use this to
model a target engine with coarser blocks than the trace's 64-token
granularity (e.g. Kimi-K3 TP8, where the mamba state page forces a
1536-token block).
"""

import argparse
import json
import sys
import time


def iter_model_requests(row: dict):
    """Yield (t, request) for every model request in a trace row, including
    subagent inner requests."""
    for req in row.get("requests", []):
        if req.get("type") == "subagent":
            for inner in req.get("requests", []):
                if "hash_ids" in inner and "in" in inner:
                    yield inner
        elif "hash_ids" in req and "in" in req:
            yield req


def main():
    ap = argparse.ArgumentParser(
        description="Convert WekaTrace cc-traces to prefix-cache-simulator "
        "pre-blocked JSONL"
    )
    ap.add_argument("traces_file", help="WekaTrace traces.jsonl")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL path")
    ap.add_argument(
        "--superblock-tokens", type=int, default=None,
        help="Regroup N consecutive source blocks into one superblock of "
        "this many tokens (must be a multiple of the trace block_size). "
        "Use to model a target engine whose block size is larger than the "
        "trace's 64-token granularity, e.g. 1536 for Kimi-K3 TP8.",
    )
    args = ap.parse_args()

    t0 = time.time()
    n_sessions = 0
    n_requests = 0
    total_tokens = 0

    with open(args.traces_file, "r", encoding="utf-8") as f, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            session_id = row["id"]
            block_size = row.get("block_size", 64)
            if row.get("hash_id_scope", "local") != "local":
                print(
                    f"warning: session {session_id} has "
                    f"hash_id_scope={row.get('hash_id_scope')}; block ids "
                    f"are namespaced per session anyway",
                    file=sys.stderr,
                )

            group_by = 1
            out_block_size = block_size
            if args.superblock_tokens:
                if args.superblock_tokens % block_size:
                    sys.exit(
                        f"--superblock-tokens {args.superblock_tokens} is not "
                        f"a multiple of trace block_size {block_size}"
                    )
                group_by = args.superblock_tokens // block_size
                out_block_size = args.superblock_tokens

            reqs = list(iter_model_requests(row))
            reqs.sort(key=lambda r: r["t"])

            for i, r in enumerate(reqs):
                ids = r["hash_ids"]
                if group_by > 1:
                    # A superblock is identified by the ordered tuple of its
                    # source block ids; equality of superblocks == equality
                    # of all contained 64-token blocks.
                    ids = [
                        ",".join(map(str, ids[j : j + group_by]))
                        for j in range(0, len(ids), group_by)
                    ]
                out.write(json.dumps({
                    "trace_id": f"{session_id}-{i:05d}",
                    # zero-pad so lexicographic timestamp sort == numeric
                    "timestamp": f"{r['t']:020.6f}",
                    "model": r.get("model", ""),
                    "group": session_id,
                    "namespace": session_id,
                    "block_size": out_block_size,
                    "prompt_tokens": r["in"],
                    "block_ids": ids,
                }) + "\n")
                n_requests += 1
                total_tokens += r["in"]
            n_sessions += 1

            if n_sessions % 50 == 0:
                print(
                    f"\r  {n_sessions} sessions, {n_requests} requests "
                    f"({n_requests / max(time.time() - t0, 1e-9):.0f} req/s)",
                    end="", flush=True,
                )

    print(
        f"\r  done: {n_sessions} sessions -> {n_requests} requests, "
        f"{total_tokens:,} input tokens in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
