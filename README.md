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

### Hybrid (attention + mamba state) mode

For hybrid models like **Kimi-K3 / Kimi-Linear** (KDA linear attention + MLA
full attention), vLLM caches one *mamba state checkpoint* per block boundary
per mamba layer alongside the attention KV, in a single unified block pool
(`mamba_cache_mode=align`). Mamba states are large (the fp32 recurrent state
is `num_heads × head_dim²` per layer), so the per-block memory cost — not the
token capacity — is what limits the cache.

Enable hybrid mode with `--cache-capacity-bytes` plus either a model config
or explicit layer sizes. The hit rule is unchanged (chained-hash prefix
match); what changes is that capacity is a byte budget divided by the real
per-block cost, and the block size is raised if one attention page cannot
hold one mamba state (exactly what vLLM does in
`_align_hybrid_block_size`).

```bash
# Derive everything from the model's HF config.json (Kimi-Linear style)
python prefix_cache_simulator.py <log.jsonl> \
  --tokenizer <hf-tokenizer> \
  --model-config <path-to-config.json> \
  --cache-capacity-bytes 80GiB \
  --dtype bf16 --tp-size 1 --num-speculative-tokens 0

# Or specify the memory model explicitly
python prefix_cache_simulator.py <log.jsonl> \
  --tokenizer <hf-tokenizer> \
  --cache-capacity-bytes 80GiB \
  --num-attn-layers 15 --num-mamba-layers 45 \
  --mamba-state-bytes 4341760 --attn-bytes-per-token 1152
```

| Argument | Default | Description |
|---|---|---|
| `--cache-capacity-bytes` | — | Pool capacity in bytes (`80GiB`, `1.5GB`, …). Enables hybrid mode |
| `--model-config` | — | HF `config.json` with `linear_attn_config`; derives layer counts, KDA state bytes, MLA/full-attn KV bytes |
| `--dtype` | bf16 | dtype for conv state and KV (recurrent state is always fp32, as in vLLM) |
| `--num-speculative-tokens` | 0 | Widens the conv-state window (`kernel-1+num_spec`) |
| `--tp-size` | 1 | Per-block state/KV divides by TP size |
| `--kernel-block-alignment` | 16 | Alignment used when raising block size to fit a mamba page |
| `--num-attn-layers` / `--num-mamba-layers` | — | Explicit layer counts (override config) |
| `--mamba-state-bytes` | — | Explicit mamba state bytes per layer per block |
| `--attn-bytes-per-token` | — | Explicit attention KV bytes per layer per token |
| `--no-tail-cache` | off | Never cache the final partial block (vLLM align mode *does* cache the prompt's partial tail) |

KDA state formula (per layer per block, mirroring vLLM's
`MambaStateShapeCalculator.kda_state_shape`):

```
conv_state      = (H·d + 2·Hk·dk) × (conv_kernel − 1 + num_spec) × dtype_bytes
recurrent_state = H × d × d × 4          # fp32, always
```

Per-block pool cost = `(num_attn_layers + num_mamba_layers) × unified_page`,
where `unified_page = max(attn_page, mamba_state)` after the block-size raise.

Not modeled (second order for hit-rate estimation): the 2+spec resident
blocks each in-flight request holds, and chunked-prefill boundary alignment.

### `build_session_map.py`

Extracts trace ID → session ID mappings from request header logs (where `msg` contains the HTTP headers).

```bash
python build_session_map.py <header_log.jsonl> -o session_map.json
```
