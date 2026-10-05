# Prefix Cache Simulator

Simulate an LRU prefix cache over a replay of tokenized LLM requests to estimate theoretical cache hit rates.

## How it works

1. **Parse** JSONL request logs (OpenAI-format `messages`)
2. **Tokenize** prompts using a HuggingFace tokenizer
3. **Split** token sequences into fixed-size blocks (default 16 tokens)
4. **Chain-hash** each block — block *i*'s key depends on all blocks 0..*i*, so only true prefix matches produce cache hits
5. **Simulate** an LRU cache (default 200M tokens capacity) — sequential prefix matching stops at the first miss, then all blocks are written/updated
6. **Analyze** per-session cache hit rates and run fuzzy string matching to diagnose why similar prompts miss the cache

## Usage

```bash
pip install -r requirements.txt

# Basic simulation
python prefix_cache_simulator.py <log.jsonl> \
  --tokenizer <hf-tokenizer-path-or-name> \
  --block-size 16 \
  --cache-capacity 200000000

# With session analysis
python build_session_map.py <header_log.jsonl> -o session_map.json
python prefix_cache_simulator.py <log.jsonl> \
  --tokenizer <hf-tokenizer-path-or-name> \
  --session-map session_map.json
```

## Arguments

### `prefix_cache_simulator.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `log_file` | yes | — | JSONL log file (each line has `request_body` or `body` with OpenAI-format messages) |
| `--tokenizer` | yes | — | HuggingFace tokenizer path or model name |
| `--block-size` | no | 16 | Tokens per cache block |
| `--cache-capacity` | no | 200,000,000 | LRU cache capacity in tokens |
| `--session-map` | no | — | JSON file mapping `trace_id` → `session_id` |
| `--top-k-sessions` | no | 5 | Number of low-hit sessions to diagnose |

### Mamba state checkpoint mode (hybrid linear-attention models)

Hybrid models like **Kimi-K3 / Kimi-Linear** (KDA linear attention + MLA
full attention) can only reuse a cached prefix if a **mamba state
checkpoint** exists at the resume position — KV blocks alone are not enough.
Enable checkpoint-aware simulation with two flags:

```bash
python prefix_cache_simulator.py <log.jsonl> \
  --tokenizer <hf-tokenizer> \
  --block-size 16 \
  --mamba-state-interval 16 \
  --mamba-state-size 4000
```

| Argument | Default | Description |
|---|---|---|
| `--mamba-state-interval` | off | Store a mamba state checkpoint every N tokens; the final partial interval counts as a checkpoint. Set equal to `--block-size` to mirror vLLM's `mamba_cache_mode=align` |
| `--mamba-state-size` | — | Capacity cost of one checkpoint, in token-equivalents (required with `--mamba-state-interval`) |
| `--no-tail-cache` | off | Never cache the final partial block (vLLM align mode *does* cache the prompt's partial tail) |

Semantics, mirroring vLLM v1's `MambaManager`:

- **Hit rule**: the reusable prefix is the *newest cached checkpoint* covered
  by the KV prefix match (checkpoints are scanned right-to-left — an SSM
  only needs the latest state to resume). No cached checkpoint → 0 hits,
  even if all KV blocks are cached.
- **Capacity**: one shared LRU pool measured in tokens. Each KV block entry
  costs `block_size` tokens; each checkpoint costs `--mamba-state-size`
  tokens. Convert state bytes to token-equivalents with:

  ```
  mamba_state_size = total_state_bytes_across_mamba_layers
                     / total_kv_bytes_per_token_across_attn_layers
  ```

  e.g. 45 KDA layers × 4.14 MiB state ÷ (15 MLA layers × 1152 B/token)
  ≈ 10,800 tokens per checkpoint.

Not modeled (second order for hit-rate estimation): the 2+spec resident
blocks each in-flight request holds, and chunked-prefill boundary alignment.

### `build_session_map.py`

Extracts trace ID → session ID mappings from request header logs (where `msg` contains the HTTP headers).

```bash
python build_session_map.py <header_log.jsonl> -o session_map.json
```
