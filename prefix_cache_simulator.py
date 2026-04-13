#!/usr/bin/env python3
"""
Prefix Cache Hit Rate Simulator

Simulates an LRU prefix cache over a replay of tokenized requests.
Computes theoretical cache hit rate and diagnoses low-hit sessions.
"""

import argparse
import difflib
import hashlib
import json
import pickle
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedRequest:
    trace_id: str
    timestamp: str
    messages: list       # raw messages list (OpenAI chat format)
    prompt: str          # flattened text for fuzzy matching in session analysis
    model: str = ""


def _decode_unicode_escapes(raw: str) -> str:
    """Decode \\uXXXX and other escape sequences in a raw string."""
    try:
        # Use the codec that handles \\n, \\t, \\uXXXX etc.
        return raw.encode("utf-8").decode("unicode_escape")
    except Exception:
        return raw


def _parse_request_body(entry: dict) -> Optional[dict]:
    """Extract the request body dict from a log entry, handling multiple formats."""
    # New format: 'body' field
    if "body" in entry:
        b = entry["body"]
        if isinstance(b, dict):
            return b
        if isinstance(b, str):
            try:
                return json.loads(b)
            except json.JSONDecodeError:
                return None
        return None

    # Old format: 'request_body' is an escaped JSON string
    raw = entry.get("request_body")
    if not isinstance(raw, str):
        return None

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: decode unicode escapes then parse
    decoded = _decode_unicode_escapes(raw)
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return None


def _extract_messages_from_body(body: dict) -> Optional[tuple[list, str]]:
    """
    Extract messages and a flattened prompt string from a parsed request body.

    Returns (messages, prompt_text) where messages is the raw OpenAI-format
    list and prompt_text is a flattened string for fuzzy matching.
    Returns None if the body cannot be parsed.
    """
    # Simple prompt field — no messages to apply chat template to
    if "prompt" in body and isinstance(body["prompt"], str) and body["prompt"]:
        prompt = body["prompt"]
        return [{"role": "user", "content": prompt}], prompt

    # Messages array (OpenAI format)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    # Validate messages and build flattened text for fuzzy matching
    clean_messages = []
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                clean_messages.append({"role": role, "content": content})
                text_parts.append(content)
        elif isinstance(content, list):
            # Array of content parts — skip multimodal
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type", "") != "text":
                    return None  # multimodal, skip entire request
                t = item.get("text", "")
                if t:
                    parts.append(t)
            if parts:
                joined = "".join(parts)
                clean_messages.append({"role": role, "content": joined})
                text_parts.append(joined)

    if not clean_messages:
        return None

    return clean_messages, "\n".join(text_parts)


def parse_log_file(path: str) -> list[ParsedRequest]:
    """Parse a JSONL log file into a list of ParsedRequests, sorted by timestamp."""
    requests = []
    skipped_multimodal = 0
    errors = 0
    t0 = time.time()

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            body = _parse_request_body(entry)
            if body is None:
                errors += 1
                continue

            result = _extract_messages_from_body(body)
            if result is None:
                # Could be multimodal skip or just empty
                skipped_multimodal += 1
                continue

            messages, prompt_text = result
            trace_id = entry.get("trace_id", f"line_{lineno}")
            timestamp = entry.get("__TIMESTAMP__", entry.get("ts", ""))
            model = entry.get("model_name", body.get("model", ""))

            requests.append(ParsedRequest(
                trace_id=trace_id,
                timestamp=str(timestamp),
                messages=messages,
                prompt=prompt_text,
                model=model,
            ))

            if lineno % 500 == 0:
                elapsed = time.time() - t0
                print(
                    f"\r  parsing: {lineno} lines, {len(requests)} valid, "
                    f"{skipped_multimodal} skipped, {errors} errors "
                    f"({lineno / max(elapsed, 1e-9):.0f} lines/s)",
                    end="", flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"\r  parsed {lineno} lines -> {len(requests)} requests, "
        f"{skipped_multimodal} skipped(multimodal), {errors} errors "
        f"in {elapsed:.1f}s"
    )

    # Sort by timestamp for chronological replay
    requests.sort(key=lambda r: r.timestamp)
    return requests


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Chained-hash block key generation (aligned with vLLM)
# ---------------------------------------------------------------------------

# vLLM's NONE_HASH is the parent hash for the first block.
# In production it's either random or hash_fn(PYTHONHASHSEED).
# For reproducible simulation, we use a fixed seed.
_NONE_HASH = hashlib.sha256(
    pickle.dumps("prefix_cache_simulator_none_hash",
                 protocol=pickle.HIGHEST_PROTOCOL)
).digest()


def _hash_block_tokens(
    parent_block_hash: bytes,
    curr_block_token_ids: tuple[int, ...],
    extra_keys: tuple | None = None,
) -> bytes:
    """
    Compute block hash matching vLLM's hash_block_tokens (sha256 default).

    See vllm/vllm/v1/core/kv_cache_utils.py::hash_block_tokens
    and vllm/vllm/utils/hashing.py::sha256.
    """
    input_bytes = pickle.dumps(
        (parent_block_hash, curr_block_token_ids, extra_keys),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return hashlib.sha256(input_bytes).digest()


def compute_block_keys(token_ids: list[int], block_size: int) -> list[bytes]:
    """
    Split token_ids into blocks of block_size and produce a chained hash key
    for each block.  Block i's key depends on all blocks 0..i (prefix-dependent).

    Uses the same hashing scheme as vLLM's prefix caching (sha256 over
    pickle-serialized (parent_hash, token_ids_tuple, extra_keys)).

    Returns a list of bytes keys, one per block.
    """
    keys = []
    prev_hash = _NONE_HASH
    for start in range(0, len(token_ids), block_size):
        block = tuple(token_ids[start : start + block_size])
        block_hash = _hash_block_tokens(prev_hash, block)
        keys.append(block_hash)
        prev_hash = block_hash
    return keys


# ---------------------------------------------------------------------------
# LRU prefix cache
# ---------------------------------------------------------------------------

class LRUPrefixCache:
    """
    LRU cache keyed by chained block hashes.
    Capacity is measured in number of blocks.
    """

    def __init__(self, capacity_blocks: int):
        self.capacity = capacity_blocks
        self.cache: OrderedDict[bytes, None] = OrderedDict()

    def query(
        self, block_keys: list[int], block_size: int
    ) -> int:
        """
        Simulate a prefix-cache lookup for one request.

        Returns the number of cached (hit) tokens.
        After the lookup, ALL blocks are written/updated in the cache.
        """
        cached_tokens = 0

        # Phase 1: prefix match — sequential check, stop at first miss
        for i, key in enumerate(block_keys):
            if key in self.cache:
                # Hit — count tokens in this block
                # Last block may be shorter, but we count block_size for
                # uniformity (the caller tracks actual token counts).
                cached_tokens += block_size
                # Touch: move to end (most recently used)
                self.cache.move_to_end(key)
            else:
                break  # first miss stops prefix matching

        # Phase 2: write all blocks into cache
        for key in block_keys:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                self.cache[key] = None
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)  # evict LRU

        return cached_tokens


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    trace_id: str
    prompt_tokens: int
    cached_tokens: int

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class SimulationResult:
    block_size: int
    cache_capacity_blocks: int
    total_prompt_tokens: int
    total_cached_tokens: int
    per_request: list[RequestResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (
            self.total_cached_tokens / self.total_prompt_tokens
            if self.total_prompt_tokens
            else 0.0
        )


def run_simulation(
    tokenizer,
    requests: list[ParsedRequest],
    block_size: int,
    cache_capacity_tokens: int,
    keep_prompts: bool = False,
) -> SimulationResult:
    """
    Tokenize and simulate in a single streaming pass.

    Each request is tokenized, simulated, then its messages/prompt are freed
    to keep peak memory low.  If *keep_prompts* is True the ``prompt`` field
    is preserved for session analysis.
    """
    cache_capacity_blocks = cache_capacity_tokens // block_size
    cache = LRUPrefixCache(cache_capacity_blocks)

    result = SimulationResult(
        block_size=block_size,
        cache_capacity_blocks=cache_capacity_blocks,
        total_prompt_tokens=0,
        total_cached_tokens=0,
    )

    t0 = time.time()
    total_tokens = 0
    fallback_count = 0
    for i, req in enumerate(requests):
        # --- tokenize one request ---
        try:
            tokens = tokenizer.apply_chat_template(
                req.messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
        except Exception:
            tokens = tokenizer.encode(req.prompt, add_special_tokens=False)
            fallback_count += 1

        total_tokens += len(tokens)

        # Free heavy fields immediately; keep prompt only if needed later
        req.messages = None  # type: ignore[assignment]
        if not keep_prompts:
            req.prompt = None  # type: ignore[assignment]

        if not tokens:
            continue

        # --- simulate ---
        block_keys = compute_block_keys(tokens, block_size)
        cached_tokens = cache.query(block_keys, block_size)

        prompt_tokens = len(tokens)
        cached_tokens = min(cached_tokens, prompt_tokens)

        result.total_prompt_tokens += prompt_tokens
        result.total_cached_tokens += cached_tokens
        result.per_request.append(
            RequestResult(
                trace_id=req.trace_id,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
            )
        )

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"\r  processing: {i + 1}/{len(requests)} "
                f"hit_rate={result.hit_rate:.4f} "
                f"cache_blocks={len(cache.cache):,} "
                f"({(i + 1) / max(elapsed, 1e-9):.0f} req/s)",
                end="", flush=True,
            )

    elapsed = time.time() - t0
    fallback_msg = f", {fallback_count} fallback" if fallback_count else ""
    print(
        f"\r  done: {len(result.per_request)} requests, "
        f"{total_tokens:,} tokens in {elapsed:.1f}s{fallback_msg}"
    )
    return result


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    session_id: str
    trace_ids: list[str] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        return (
            self.total_cached_tokens / self.total_prompt_tokens
            if self.total_prompt_tokens
            else 0.0
        )


def analyze_sessions(
    sim_result: SimulationResult,
    session_map: dict[str, str],
    requests: list[ParsedRequest],
    top_k: int = 5,
):
    """Analyze per-session cache hit rates and diagnose low-hit sessions."""
    # Build session stats
    sessions: dict[str, SessionStats] = {}
    # Map trace_id -> prompt text for fuzzy matching
    trace_to_prompt: dict[str, str] = {r.trace_id: r.prompt for r in requests}

    for rr in sim_result.per_request:
        sid = session_map.get(rr.trace_id)
        if sid is None:
            continue
        if sid not in sessions:
            sessions[sid] = SessionStats(session_id=sid)
        s = sessions[sid]
        s.trace_ids.append(rr.trace_id)
        s.total_prompt_tokens += rr.prompt_tokens
        s.total_cached_tokens += rr.cached_tokens

    if not sessions:
        print("\n  No sessions found in session map.")
        return

    # Filter to sessions with >= 2 requests (need context to diagnose)
    multi_req_sessions = [s for s in sessions.values() if len(s.trace_ids) >= 2]
    multi_req_sessions.sort(key=lambda s: s.hit_rate)

    print(f"\n{'='*70}")
    print("SESSION ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total sessions: {len(sessions)}")
    print(f"  Sessions with >=2 requests: {len(multi_req_sessions)}")

    if not multi_req_sessions:
        print("  No multi-request sessions to analyze.")
        return

    # Summary distribution
    hit_rates = [s.hit_rate for s in multi_req_sessions]
    avg_hr = sum(hit_rates) / len(hit_rates)
    print(f"  Average in-session hit rate: {avg_hr:.4f}")

    # Bottom-k sessions
    bottom = multi_req_sessions[:top_k]
    print(f"\n  Bottom {len(bottom)} sessions by hit rate:")
    print(f"  {'Session ID':<40} {'Reqs':>5} {'Tokens':>10} {'Cached':>10} {'HitRate':>8}")
    print(f"  {'-'*40} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
    for s in bottom:
        print(
            f"  {s.session_id:<40} {len(s.trace_ids):>5} "
            f"{s.total_prompt_tokens:>10,} {s.total_cached_tokens:>10,} "
            f"{s.hit_rate:>8.4f}"
        )

    # Fuzzy matching diagnosis for each bottom session
    # Build a trace_id -> RequestResult lookup
    rr_map = {rr.trace_id: rr for rr in sim_result.per_request}

    for s in bottom:
        print(f"\n  --- Session: {s.session_id} (hit_rate={s.hit_rate:.4f}) ---")
        for i in range(1, len(s.trace_ids)):
            prev_tid = s.trace_ids[i - 1]
            curr_tid = s.trace_ids[i]
            prev_prompt = trace_to_prompt.get(prev_tid, "")
            curr_prompt = trace_to_prompt.get(curr_tid, "")

            # Quick ratio first (fast), then ratio if interesting
            ratio = difflib.SequenceMatcher(
                None, prev_prompt, curr_prompt
            ).quick_ratio()

            curr_rr = rr_map.get(curr_tid)
            curr_hr = curr_rr.hit_rate if curr_rr else 0.0

            # Only report if similarity is high but cache hit is low
            if ratio > 0.3:
                # Get more accurate ratio
                accurate_ratio = difflib.SequenceMatcher(
                    None, prev_prompt, curr_prompt
                ).ratio()
                status = "OK" if curr_hr > 0.5 else "LOW HIT"
                print(
                    f"    req[{i-1}]->[{i}]: "
                    f"similarity={accurate_ratio:.3f}  "
                    f"cache_hit={curr_hr:.3f}  "
                    f"tokens={curr_rr.prompt_tokens if curr_rr else 0}  "
                    f"[{status}]"
                )
                if accurate_ratio > 0.7 and curr_hr < 0.3:
                    # Show where the prompts diverge
                    prev_lines = prev_prompt[:500].splitlines(keepends=True)
                    curr_lines = curr_prompt[:500].splitlines(keepends=True)
                    diff = list(difflib.unified_diff(
                        prev_lines, curr_lines,
                        fromfile=f"req[{i-1}]", tofile=f"req[{i}]",
                        n=1,
                    ))
                    if diff:
                        print("    Divergence (first 500 chars):")
                        for d in diff[:20]:
                            print(f"      {d.rstrip()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prefix Cache Hit Rate Simulator"
    )
    parser.add_argument("log_file", help="JSONL log file path")
    parser.add_argument(
        "--tokenizer", required=True, help="HuggingFace tokenizer path or name"
    )
    parser.add_argument(
        "--block-size", type=int, default=16,
        help="Tokens per cache block (default: 16)",
    )
    parser.add_argument(
        "--cache-capacity", type=int, default=3_200_000_000,
        help="LRU cache capacity in tokens (default: 3,200,000,000)",
    )
    parser.add_argument(
        "--session-map",
        help="JSON file mapping trace_id -> session_id",
    )
    parser.add_argument(
        "--top-k-sessions", type=int, default=5,
        help="Number of low-hit sessions to analyze (default: 5)",
    )
    args = parser.parse_args()

    # 1. Parse log
    print(f"[1/3] Parsing log file: {args.log_file}")
    requests = parse_log_file(args.log_file)
    if not requests:
        print("No valid requests found. Exiting.")
        sys.exit(1)

    # 2. Load tokenizer
    print(f"\n[2/3] Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # 3. Tokenize + simulate in one streaming pass (avoids holding all token
    #    lists in memory at once).
    keep_prompts = bool(args.session_map)
    print(
        f"\n[3/3] Tokenizing & simulating "
        f"({len(requests)} requests, "
        f"block_size={args.block_size}, "
        f"cache_capacity={args.cache_capacity:,} tokens / "
        f"{args.cache_capacity // args.block_size:,} blocks)"
    )
    result = run_simulation(
        tokenizer, requests, args.block_size, args.cache_capacity,
        keep_prompts=keep_prompts,
    )

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Total requests:      {len(result.per_request):,}")
    print(f"  Total prompt tokens: {result.total_prompt_tokens:,}")
    print(f"  Total cached tokens: {result.total_cached_tokens:,}")
    print(f"  Block hit rate:      {result.hit_rate:.6f} ({result.hit_rate*100:.2f}%)")
    if result.per_request:
        req_hit_rate = sum(r.hit_rate for r in result.per_request) / len(result.per_request)
    else:
        req_hit_rate = 0.0
    print(f"  Request hit rate:    {req_hit_rate:.6f} ({req_hit_rate*100:.2f}%)")
    print(f"  Block size:          {args.block_size} tokens")
    print(f"  Cache capacity:      {result.cache_capacity_blocks:,} blocks")

    # Per-request hit rate distribution
    if result.per_request:
        hrs = [r.hit_rate for r in result.per_request]
        hrs.sort()
        print(f"\n  Per-request hit rate distribution:")
        print(f"    min:    {hrs[0]:.4f}")
        print(f"    p25:    {hrs[len(hrs)//4]:.4f}")
        print(f"    median: {hrs[len(hrs)//2]:.4f}")
        print(f"    p75:    {hrs[3*len(hrs)//4]:.4f}")
        print(f"    max:    {hrs[-1]:.4f}")
        print(f"    mean:   {sum(hrs)/len(hrs):.4f}")

    # Session analysis (optional)
    if args.session_map:
        print(f"\nSession analysis (map: {args.session_map})")
        with open(args.session_map, "r") as f:
            session_map = json.load(f)
        analyze_sessions(result, session_map, requests, args.top_k_sessions)
    else:
        print("\nSession analysis skipped (no --session-map provided)")


if __name__ == "__main__":
    main()
