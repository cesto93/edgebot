# AGENTS.md

## Project

Interactive AI shell TUI (Bubbletea) with 5 LLM providers: Ollama, Gemini, OpenRouter, LitertLM, Llamacpp. OpenAI-compatible HTTP providers (Ollama, Gemini, OpenRouter) share `OpenAICaller` (`llm/openai.go`) via `NewProviderCaller` factory (`llm/providers.go`). Two in-process providers run no HTTP server: `llamacpp` uses `LlamacppCaller` (`llm/llamacpp.go`) with the `yzma` Go binding, and `litertlm` uses `LitertLMCaller` (`llm/litertlm.go`) with the `litertlm-go` binding.

Entry point: `main.go` → `cmd.Execute()`. Default command launches Bubbletea TUI in `cmd/shell.go` (MVU pattern).

## Commands

```
make build              # go build -o edgebot .
make install            # go install .
make install-yzma       # install yzma CLI + llama.cpp libs to ~/.edgebot/lib
make install-litertlm   # download LiteRT-LM C-API lib + aux/GPU libs to ~/.edgebot/lib
make proto              # regenerate service/proto/*.pb.go (needs protoc + plugins)
make coverage           # test + HTML report
make docker-build       # build edgebot:latest image (docker build -t edgebot:latest .)
make docker-up          # start edgebot via docker compose (docker compose up -d)
make docker-down        # stop edgebot compose stack (docker compose down)
go fmt ./... && go vet ./... && go build -o edgebot . && go test ./...
```

Docker: `Dockerfile` (multi-stage `golang:1.26-bookworm` → `debian:bookworm-slim`, `CGO_ENABLED=0`, `ENTRYPOINT ["docker-entrypoint.sh"]` + `unzip` per le lib LiteRT-LM) and `docker-compose.yml` (bot-only: single `edgebot-bot` service, `image: ghcr.io/cesto93/edgebot:latest` + `build: .`, `command: ["bot"]`, `restart: unless-stopped`; provider/model via env overrides `LLM_PROVIDER=litertlm`/`LLM_MODEL=gemma-4-E2B-it` plus `LITERTLM_MODEL_REPO=litert-community/gemma-4-E2B-it-litert-lm`, `LITERTLM_MODEL_FILE=gemma-4-E2B-it.litertlm`, `LITERTLM_BACKEND=cpu` (no Ollama wiring) — l'entrypoint `docker-entrypoint.sh` installa/verifica le lib (`make install-litertlm` equivalent; re-download se mancanti/vuote/non-ELF, fail-fast con `ls` diagnostico se una lib required resta inutilizzabile), fa `edgebot models pull` del `.litertlm` se assente (un fallimento non blocca il force della config) e forza `config --provider litertlm --model gemma-4-E2B-it --backend cpu`; mounts `./:/workspace` plus named volumes `edgebot-config`/`edgebot-data` for `~/.config/edgebot` and `~/.edgebot`, `env_file: .env`).

## Packages

| Directory | Contents |
|-----------|----------|
| `cmd/` | Cobra commands: default (TUI shell), `config`, `commit`, `models` (with `pull` subcommand), `commands`, `agents`, `stats`, `context`, `service`, `bot` (Telegram) |
| `config/` | Viper YAML config, model lists (OpenRouter free models fetched live, 10 min cache), `.env` loading via `gotenv`. Free OpenRouter models filtered by zero pricing and `architecture.output_modalities` (audio-only excluded); `InputTypes` from `input_modalities` |
| `llm/` | `Agent`, `Caller`, `RawCaller` (adds `CallStructured`), `ToolExecutor`, 6 tool definitions, `NewProviderCaller`/`NewProviderCallerRaw` factory (shared `newProviderCaller` helper), `ProviderConfig`, system prompts (`BuildPrompt`, `PlanPrompt`, `BotPrompt`, `ChatPrompt`), `LlamacppCaller`, `LitertLMCaller`. Tool dispatch via `ToolExecutorPolicy` (`llm/executor.go`, pluggable confirm/execute hooks) shared by shell/CLI/service; `NewConfirmPolicy` helper; `NoopExecutor` runs nothing. Shared `decodeDataURL` in `llm/decode.go` (validates `data:` prefix/`;base64`, supports padded+unpadded). Agents: `build` (all tools), `plan` (ReadFile only), `bot` (ReadFile+KV only with Telegram prompt), `chat` (no tools) via `GetAgentDefs`/`GetAgentDef` (returns cloned tool maps); `NewAgentFor` intersects agent allowed tools with user toggles; `NewAgentForSession` adds backend + think effort + AGENTS.md + skills (chat skips AGENTS.md/skills). `OpenAICaller` has 60s timeout, 10-hop tool limit, `TrimSuffix` baseURL, `LimitReader` 10MiB, empty `ToolCallID` fallback, `response_format` omitted after hop 0, explicit `Body.Close` (no defer-in-loop), persists `usage` via `stats.RecordUsage`. Gemini thought signatures (`llm/types.go` `ExtraContent`, `llm/openai.go` `ensureGeminiThoughtSignatures`): `tool_calls[].extra_content.google.thought_signature` (+ message/part level) is replayed verbatim each hop; missing first-call signatures on Gemini are backfilled with `skip_thought_signature_validator` so old transcripts don't 400. Unified think effort (`llm/think.go`: none/minimal/low/medium/high/xhigh/max, `""` = provider default) is sent as `reasoning.effort` on OpenRouter and `reasoning_effort` on other OpenAI-compatible providers (Ollama/Gemini; Gemini rejects unknown `reasoning` with 400), maps to `NoThink` on llamacpp when `none`, and is ignored (debug-logged) on litertlm. `WithThink` factory variants (`NewProviderCallerWithThink`, `NewProviderCallerRawWithThink`, `NewOpenAICallerWithThink`) carry it; plain `NewProviderCaller` defaults to unset. `IsAllowedCommandForPolicy` trims first; `GetAgentDefs` clones maps |
| `tools/` | `RunCommand` (bash -c), `ReadFile`, `WriteFile`, KV store (bbolt with 1s timeout), `GetDistro`, `GetShell` |
| `stats/` | Persistent token usage store (bbolt at `~/.config/edgebot/usage.db`): `RecordUsage`, `GetStats`, `Reset` |
| `service/` | gRPC service over a unix socket at `~/.edgebot/service.sock` (`MaxMsgSize` in `socket.go`). `server.go` (`Server`, swappable `callLLM`) builds the agent via `llm.NewAgentForSession` and runs tools via `ServiceExecutor`; `client.go` (`Client`, `IsActive`, `Chat`, `Stop`, `ErrUnavailable`); `convert.go` maps `llm.Message` ↔ proto including Gemini `thought_signature` (tool/message/part level). `ChatRequest.think_effort` carries the session think effort (invalid values rejected as response errors). Wire types in `service/proto/` (committed; `make proto` to regenerate) |

## Config layering

1. `~/.config/edgebot/.env` (global)
2. `./.env` (local overrides)
3. `config.yaml` from `./` or `~/.config/edgebot/`

`.env.example` documents the recognized env vars. Defaults: provider=ollama, model=granite4:3b-h, log_level=info, confirm=true, allowed_commands=ls,pwd, agent=build, agent_files=true, skills=true, llm.think_effort="" (provider default). litertlm backend defaults to `cpu`. Explicit env overrides win over config.yaml: `LLM_PROVIDER` → `llm.provider`, `LLM_MODEL` → `llm.model` (auto-detects provider via `LookupModelInfo` when `LLM_PROVIDER` is unset, mirroring `edgebot config --model`), `LITERTLM_BACKEND` → `litertlm.backend` (see `applyEnvOverrides`). `docker-compose.yml` sets `LLM_PROVIDER=litertlm`/`LLM_MODEL=gemma-4-E2B-it` for the bot stack. All 6 tools enabled by default. Set via `edgebot config --think-effort low` (empty resets).

`agent_files` toggles AGENTS.md support: `llm.GetAgentFiles(true)` loads `~/.config/edgebot/AGENTS.md` and `./AGENTS.md` as extra system-prompt instructions. Toggle via `edgebot config --agent-files=false`.

`skills` toggles skills support: `llm.GetSkills(true)` scans `~/.agents/skills/*/SKILL.md` and `./skills/*/SKILL.md` (standard agent-skills layout with YAML frontmatter `name`/`description`). Only a compact index (`llm.GetSkillsPrompt`) is injected into the system prompt; the model reads full SKILL.md contents on demand via `ReadFile`. Toggle via `edgebot config --skills=false`.

Logging uses `log/slog`; call `config.InitLogger(cfg.LogLevel)` after `LoadConfig()`.

## Testing

```bash
go test ./...
go test -v -run TestName ./package
go test -cover ./...
```

Conventions: table-driven tests, `t.Run()` subtests, function variable mocking (`userConfigDirFunc`, `loadEnvFunc`, `dbPathFunc`).

## TUI conventions (cmd/shell.go)

- `/` commands: help, get-config, config, models, agent, reset, add-cmd, exit, quit
- `@filepath` for image attachments (base64 encoded)
- Tool execution requires confirmation unless command is in `allowed_commands`
- `/agent` opens a selection menu; `m.cfg.Agent` is persisted to config. `ElaborateMessage` builds the agent via `llm.NewAgentForSession(...)`
- Transcript renders in a `bubbles/viewport` (scroll: PgUp/PgDn, Shift+Up/Down, Home/End, mouse wheel; `followOutput` tails new turns, hint shown when detached). User/AI turns render as rounded bordered blocks (`renderUserBlock`/`renderAIBlock`); input + confirm/loading stay pinned in the footer
- Lipgloss styles: `promptStyle`, `systemStyle`, `userStyle`, `aiStyle`, `errorStyle`, `cmdStyle`, `helpStyle`, `dimStyle`

## Debug flag (cmd/cmd.go)

Persistent `--debug` flag on the root command. `cmd.initLogger(cfg)` temporarily forces debug level in memory. All subcommands must call `initLogger(cfg)` after `config.LoadConfig()`.

## Config command (cmd/config.go)

- Usable as `edgebot config` (shows current config via `PrintConfig`) or `edgebot config --flag value`
- Flags: `--provider`, `--model`, `--agent`, `--agent-files`, `--skills`, `--log-level`, `--confirm`, `--allowed-commands`, `--backend`, `--think-effort`, `--enable-tool`, `--disable-tool`, `--add-cmd`, `--rm-cmd`
- `--model` without `--provider` auto-detects provider via `config.LookupModelInfo`; `--add-cmd` uses `name=prompt` format

## Commands command (cmd/commands.go)

- Usable as `edgebot commands` (lists custom commands) or `edgebot commands --run <name> [args...]`; `-o` / `--output` writes structured output to a file
- `--run` looks up via `config.LoadCommands` (merges `.edgebot/commands/*.md` files and config `commands` map)
- Command args that are existing files are read into the prompt (`cmd/input.go`): images become multimodal `[]ContentPart`, `.txt`/`.md`/`.pdf` appended as text
- **Structured commands**: a command file whose frontmatter has a `schema: <path>` field (resolved relative to the command file dir) runs via `CallStructured` with a `response_format: json_schema` envelope, bypassing the service. This replaces the removed `extract` command.
- Regular commands use `llm.NewAgentForSession` with `llm.ToolExecutorPolicy{}` (no confirmation); prints final assistant text to stdout

## Commit command (cmd/commit.go)

- Usable as `edgebot commit` or via `go run . commit`
- `-A` / `--all` stages all changes; `-d` / `--dry-run` prints without committing (and unstages if used with `-A`)
- Sends `git log --oneline -5` + `git diff --cached` as context; uses `llm.NewProviderCaller` with `llm.NoopExecutor` (no tools)
- Diff truncated to `commitMaxDiffChars` runes (marked) so prompts fit small local contexts; `<think>` sections stripped via `stripThinkBlock` (also drops dangling unclosed tags); llamacpp capped to `commitMaxTokens` with `NoThink` set
- Strips markdown code fences, writes to a temp file, runs `git commit -F <file>`

## Extract command (cmd/extract.go)

> Removed. Structured extraction is now a structured command (frontmatter `schema:` field). See `cmd/commands.go` and `cmd/shell.go`. File reading helpers live in `cmd/input.go` (`readInputFile`, `readPDF`, `isImage`, `encodeImage`, `buildCommandContent`, `buildCommandTextAndImages`).

## Models command (cmd/models.go, cmd/models_pull.go)

- Usable as `edgebot models` or `edgebot models pull <repo> <model> [mmproj]`
- Lists models in a table (Model, Provider, Size, Input Types); current model prefixed with `* `
- `-s` / `--set <model>` sets the current model; `-d` / `--delete <model>` deletes a local GGUF/`.litertlm` file (also removes paired mmproj for llamacpp)
- For llamacpp, Size from GGUF file info. Input types: gemini hardcoded; openrouter from API `input_modalities`; llamacpp `text, image` when an `mmproj-*` file matches; ollama/litertlm `-`
- `pull` downloads from HuggingFace; `.litertlm` → `~/.edgebot/models/litertlm/`, else `.gguf` → `~/.edgebot/models/llamacpp/`
- Validates `repo` (`owner/name`) and `filename` (no path traversal); `http.Client` with 10 min timeout; progress bar; auto-updates config via `config.SaveModelWithProvider` (second file auto-detected as vision projector); cleans up partial files on failure (explicit `Close` before `Remove`)

## Skills command (cmd/skills.go)

- Usable as `edgebot skills`
- Lists `llm.GetSkills(cfg.Skills)` in a table (NAME, DESCRIPTION, PATH)
- Warns when `skills` is disabled in config
- `--pull <git-url>` shallow-clones the repo (`git clone --depth 1`, via the shared `execCommand` mock) and installs every directory containing a `SKILL.md` into `~/.agents/skills/`; existing skills with the same dir name are replaced (reported as Updated), then the list prints. Handles both `skills/<name>/SKILL.md` repos (e.g. run-llama/llamaparse-agent-skills) and root-level skill dirs

## Agents command (cmd/agents.go)

- Usable as `edgebot agents`
- Lists `llm.GetAgentDefs()` in a table (AGENT, DESCRIPTION, TOOLS); current agent prefixed with `* `; sorted; `text/tabwriter`
- `-s` / `--set <agent>` validates against `llm.GetAgentDefs()` and persists via `config.SaveConfig`

## Stats command (cmd/stats.go)

- Usable as `edgebot stats`
- Aggregated token usage table (CALLS, INPUT, OUTPUT, CACHED, REASONING, TOTAL, COST + TOTAL row) from `stats.GetStats()`
- `--reset` clears all usage

## Context command (cmd/context.go)

- Usable as `edgebot context`
- Shows AGENTS.md files read into context (path, word count, token estimate) plus the active agent's system prompt size
- `--prompt` prints the system prompt; `--agents` prints the AGENTS.md texts; `--skills` prints the skills index sent to the agent (with per-skill descriptions)
- Uses `llm.GetAgentFileInfo(cfg.AgentFiles)`; warns when `agent_files` is disabled. Also lists skills via `llm.GetSkills(cfg.Skills)` and their index token cost; warns when `skills` is disabled

## Service command (cmd/service.go)

- Usable as `edgebot service` (foreground), `service --stop`, or `service --status`
- grpc-go server on unix socket `~/.edgebot/service.sock`; stale socket removed when no live service answers `Ping`
- Sessions route through the service when `service.IsActive()` (shell, custom commands, commit); `cmd/service_helpers.go` provides `chatRequestFromConfig` and `chatWithServiceFallback` (falls back to local on `service.ErrUnavailable`)

## Bot command (cmd/bot.go)

- Usable as `edgebot bot` (Telegram long-polling bot)
- Token from `--token` or `TELEGRAM_BOT_TOKEN` env (loaded via `.env`); `--allow-from` / `TELEGRAM_ALLOWED_CHAT_IDS` restricts to chat IDs or @usernames (empty = allow everyone); validates via `getMe`
- Long polls `getUpdates` (30s), per-chat `[]llm.Message` history with `/reset` and `/help` handling; sends `typing` chat action and splits replies >4096 chars
- Pure `net/http` Telegram client (no external deps): `telegramClient` (`getMe`, `getUpdates`, `sendMessage`, `sendChatAction`)
- Forwards to dedicated `bot` agent (`llm.NewAgentForSession("bot",...).CallLLM` with `BotPrompt` and tools `ReadFile, KVGet, KVList, KVSet` only) via `botExecutor` (mirrors `ServiceExecutor` confirm policy); falls back to service when `service.IsActive()` via `service.Chat` + `ErrUnavailable` check (forces `req.Agent="bot"`)

## CI workflows (.github/workflows)

- `ci.yml`: runs format/vet/build/test
- `docker.yml`: on push to `main`/`master` or `v*.*.*` tags, builds multi-arch (`linux/amd64`, `linux/arm64`) image via Buildx/QEMU and pushes to `ghcr.io/<owner>/edgebot` (`latest` on default branch, short SHA, tag ref)

## Key gotchas

- `config.LoadConfig()` may return partial defaults on error — check both return values
- `.env` files loaded at startup via `gotenv.Load()` — place API keys there, not in config.yaml
- KV store at `~/.config/edgebot/kv_store.db`; usage stats at `~/.config/edgebot/usage.db`
 - System prompts are Go constants in `llm/prompt.go`, copied to `~/.edgebot/BUILDPROMPT.md`/`PLANPROMPT.md`/`BOTPROMPT.md`/`CHATPROMPT.md` on first run — always read from there, never local file
- 100 char soft line limit, Go 1.26.0+
 - Llamacpp: `model` config is the GGUF filename without extension (`.gguf` appended as fallback). Run `make install-yzma` for libs; place GGUFs in `~/.edgebot/models/llamacpp/`. Structured output uses a GBNF grammar (`jsonSchemaToGBNF`, `llm/gbnf.go`) with `llama.SamplerInitGrammar` (grammar sampler freed via `SamplerFree`; it must be added to the sampler chain BEFORE greedy — reversed order lets greedy pick invalid tokens like a reasoning model's `<think>` prefix, which aborts the process via an uncaught C++ exception). `applyChatTemplate` (`llm/llamacpp.go`) falls back to the builtin `chatml` template with a `slog.Warn` when the model's own template fails to render (e.g. Spark-X2.5 uses Jinja features newer than the bundled llama.cpp); both text (`buildPrompt`) and vision (`buildVisionPrompt`) paths share it. Image input uses yzma's `pkg/mtmd` (`setupVision`, `buildVisionPrompt`/`generateVision`) when a vision projector (`config.FindLlamacppMMProj()`) is present; init failures degrade to text-only; token counts use `llama.Tokenize` for both paths. mmproj files are excluded from `config.GetLlamacppModels()`. Audio input is not supported. `generateVision` no longer double-frees bitmap on failure. Prefill is chunked into `n_batch` pieces with explicit positions (`decodePrompt` via `BatchInit`/`Batch.Add`); prompts longer than `n_ctx` fail cleanly instead of tripping `GGML_ASSERT`. Default sampler chain is penalties (`1.1`/`64`) + greedy. `MaxTokens` overrides the generation cap, `NoThink` pre-fills an empty `<think>` block for short-form tasks.
 - LitertLM: loads libs from `$LITERTLM_LIB` (default `~/.edgebot/lib`) and `.litertlm` models from `~/.edgebot/models/litertlm/`. Binding dlopens fixed filenames: `libGemmaModelConstraintProvider.so` + `liblitertlm_c_cpu.so` (cpu) / `liblitertlm_c.so` (gpu) — NOT `liblitert-lm.so`. Run `make install-litertlm` to fetch them. `client()` pre-flights the backend-selected main lib + Gemma provider (missing/empty → actionable error naming the file). Backend from config `litertlm.backend` (default `cpu`), overridable via `$LITERTLM_BACKEND`. A single client is cached per (lib dir, model path, backend) with `\x00` delimiter, created on `context.Background()`, lock not held during fetch. Tools are manual-dispatch `RawTool`s (max 5 hops, errors on limit); multimodal uses `SendMulti`; `CallStructured` prompt-engines the schema.
- `/models` menu scans `config.GetLlamacppModels()` / `config.GetLitertLMModels()`; `config.IsLlamacppModel()` / `IsLitertLMModel()` used by `SaveModelWithProvider` for provider auto-detection.
 - Service: `service.IsActive()` pings the socket; `ServiceExecutor` denies non-allowed `RunCommand`s and all `WriteFile` when `confirm` is true; structured commands never route through the service. Wire types regenerate via `make proto` (not needed to build).
 - Shared `~/.edgebot` dirs resolved via `config.EdgebotDir()`, `config.LibDir()`, `config.ModelsDir(provider)`. Helpers: `config.GetCommandName(cmd)`, `config.FormatFileSize(b)` (capped at `P`, handles `b<0`).
 - `config.GetOpenRouterModels()` is mutex-protected (10 min TTL, lock not held during fetch); `config.LoadCommands` merges home then cwd (cwd wins); `cmd/commands` lists merged commands. `SaveModelWithProvider` uses `LookupModelInfo`.
 - `cmd/bot.go`: `splitTelegramMessage` is rune-aware (4096 char limit, rune-level cut); `parseAllowList` returns nil for empty allowlist; Telegram poll uses bounded concurrency (10); `sendMessageChunk` validates `json.Marshal`/`ReadAll` errors; `sendChatAction` defers close + checks status; per-chat history capped at 50; `KV` bbolt has 1s timeout. `botExecutor`/`ServiceExecutor` share `llm.NewConfirmPolicy`.
 - `cmd/shell.go`: `ElaborateMessage` uses `sync.Mutex` for `messages`, explicit `Body.Close` pattern, `sync.Once` for `cancelChan`, `select` non-blocking confirmation send, history `Up=+1/Down=-1` with `saveHistory` trailing newline + error handling. `runStructuredMessage` uses shared `structuredResponseFormat`/`prettyJSONOrRaw` via `cmd/structured_helpers.go`. `View` word-wraps the transcript to terminal width (`wrapShellText`/`wrapShellTextWithPrefix`, `viewWidth`, 80 fallback) since Bubbletea alt-screen clips over-wide lines instead of soft-wrapping. User/AI turns are bordered blocks (`renderTurnBlock`, lipgloss `Width` excludes the border so blocks use `outer-2`); the viewport height is recomputed in `View` from the footer height via `lipgloss.Height`. `scrollHint` shows a persistent `scroll %` indicator whenever content overflows (plain Up/Down navigate input history, so scrolling would otherwise be undiscoverable).
 - `cmd/models_pull.go`: `validateRepo` uses strict regex, `O_EXCL` create to avoid TOCTOU, error body included on non-200.
 - `cmd/commit.go`: `Backend` included in service request, staged changes reset via named-return defer on dry-run/error, fence stripping via regex, `git commit` var renamed `gitCommit`.
