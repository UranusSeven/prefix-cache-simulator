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

### `build_session_map.py`

Extracts `trace_id` → `Session-Id` mappings from request header logs (where `msg` contains the HTTP headers).

```bash
python build_session_map.py <header_log.jsonl> -o session_map.json
```

## Log format

Each JSONL line should contain:
- `trace_id` — unique request identifier
- `request_body` (string, JSON-encoded) or `body` (object) — with a `messages` array in OpenAI chat format
- `__TIMESTAMP__` or `ts` — for chronological ordering
