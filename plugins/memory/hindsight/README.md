# Hindsight Memory Provider

Long-term memory with knowledge graph, entity resolution, and multi-strategy retrieval. Supports cloud and local (embedded) modes.

## Requirements

- **Cloud:** API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
- **Local:** API key for a supported LLM provider (OpenAI, Anthropic, Gemini, Groq, MiniMax, or Ollama). Embeddings and reranking run locally — no additional API keys needed.

## Setup

```bash
hermes memory setup    # select "hindsight"
```

The setup wizard will install dependencies automatically via `uv` and walk you through configuration.

Or manually (cloud mode with defaults):
```bash
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

### Cloud Mode

Connects to the Hindsight Cloud API. Requires an API key from [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io).

### Local Mode

Runs an embedded Hindsight server with built-in PostgreSQL. Requires an LLM API key (e.g. Groq, OpenAI, Anthropic) for memory extraction and synthesis. The daemon starts automatically in the background on first use and stops after 5 minutes of inactivity.

Daemon startup logs: `~/.hermes/logs/hindsight-embed.log`
Daemon runtime logs: `~/.hindsight/profiles/<profile>.log`

## Config

Config file: `~/.hermes/hindsight/config.json`

### Connection

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` or `local` |
| `api_url` | `https://api.hindsight.vectorize.io` | API URL (cloud mode) |
| `api_url` | `http://localhost:8888` | API URL (local mode, unused — daemon manages its own port) |

### Memory

| Key | Default | Description |
|-----|---------|-------------|
| `bank_id` | `hermes` | Memory bank name |
| `budget` | `mid` | Recall thoroughness: `low` / `mid` / `high` |

### Integration

| Key | Default | Description |
|-----|---------|-------------|
| `memory_mode` | `hybrid` | How memories are integrated into the agent |
| `prefetch_method` | `recall` | Method for automatic context injection |
| `retain_tags` | `[]` | Tags attached to retained documents |
| `retain_source` | `""` | `metadata.source` value attached to retained documents |
| `retain_user_prefix` | `User` | Label used before user turns in retained transcripts |
| `retain_assistant_prefix` | `Assistant` | Label used before assistant turns in retained transcripts |
| `retain_chunk_every_n_turns` | `0` | Also retain a sliding conversation window every N turns (`0` disables) |
| `retain_chunk_overlap_turns` | `0` | Extra prior turns included in chunked conversation windows |

**memory_mode:**
- `hybrid` — automatic context injection + tools available to the LLM
- `context` — automatic injection only, no tools exposed
- `tools` — tools only, no automatic injection

**prefetch_method:**
- `recall` — injects raw memory facts (fast)
- `reflect` — injects LLM-synthesized summary (slower, more coherent)

**auto-retain behavior:**
- Every turn is retained immediately as its own document with metadata such as `session_id`, `platform`, `user_id`, `agent_identity`, `turn_index`, and `retained_at`.
- If `retain_chunk_every_n_turns` is set to `2` or higher, Hermes also emits periodic sliding-window documents with context `conversation_window` so Hindsight can see more local conversation context without re-uploading the full session on every turn.

### Local Mode LLM

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `openai` | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `minimax`, `ollama` |
| `llm_model` | per-provider | Model name (e.g. `gpt-4o-mini`, `openai/gpt-oss-120b`) |

The LLM API key is stored in `~/.hermes/.env` as `HINDSIGHT_LLM_API_KEY`.

## Tools

Available in `hybrid` and `tools` memory modes:

| Tool | Description |
|------|-------------|
| `hindsight_retain` | Store information with auto entity extraction |
| `hindsight_recall` | Multi-strategy search (semantic + entity graph) |
| `hindsight_reflect` | Cross-memory synthesis (LLM-powered) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HINDSIGHT_API_KEY` | API key for Hindsight Cloud |
| `HINDSIGHT_LLM_API_KEY` | LLM API key for local mode |
| `HINDSIGHT_API_URL` | Override API endpoint |
| `HINDSIGHT_BANK_ID` | Override bank name |
| `HINDSIGHT_BUDGET` | Override recall budget |
| `HINDSIGHT_MODE` | Override mode (`cloud` / `local`) |
| `HINDSIGHT_RETAIN_TAGS` | Comma-separated tags applied to retained documents |
| `HINDSIGHT_RETAIN_SOURCE` | Value written to `metadata.source` |
| `HINDSIGHT_RETAIN_USER_PREFIX` | Label used before user turns in retained transcripts |
| `HINDSIGHT_RETAIN_ASSISTANT_PREFIX` | Label used before assistant turns in retained transcripts |
| `HINDSIGHT_RETAIN_CHUNK_EVERY_N_TURNS` | Also retain a sliding conversation window every N turns (`0` disables) |
| `HINDSIGHT_RETAIN_CHUNK_OVERLAP_TURNS` | Extra prior turns included in chunked conversation windows |
