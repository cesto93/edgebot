# Edgebot

An interactive shell powered by AI (**Ollama, Gemini, OpenRouter, LitertLM, Llamacpp**) that helps you with commands, explanations, and automation directly from your terminal.

## Features

- **Interactive AI Chat**: Ask questions about shell commands, scripting, or general knowledge.
- **Rich TUI**: Built with Bubbletea, featuring a modern interface with command history, autocomplete, and real-time feedback.
- **System Awareness**: Automatically detects your Linux distribution and shell to provide tailored advice.
- **Autonomous Tool Use**: The AI can execute shell commands (`RunCommand`), read files (`ReadFile`), and write files (`WriteFile`) autonomously (with your confirmation).
- **Multi-Provider Support**: Ollama, Google Gemini, OpenRouter, LitertLM, and in-process Llamacpp (via yzma).
- **Configurable**: Easily switch models and providers via a YAML configuration file.

## Prerequisites

- **Go**: Version 1.25.5 or later.
- **Providers**:
  - **Ollama**: Must be running locally (default, port 11434).
  - **Gemini**: Requires `GEMINI_API_KEY` environment variable.
  - **OpenRouter**: Requires `OPEN_ROUTE_KEY` environment variable.
  - **LitertLM**: In-process LiteRT-LM via litertlm-go. Requires a `.litertlm` model and the LiteRT-LM shared libraries (see LitertLM Provider).
  - **Llamacpp**: In-process llama.cpp via yzma. Requires GGUF model and yzma libs in `~/.edgebot/models/llamacpp/` (see Installation).
- **LLM Model**: By default, it expects the `granite4:3b-h` model, but this can be changed in the config.

## Installation

You can build and install the binary using the provided Makefile:

```bash
# Clone the repository
git clone https://github.com/yourusername/edgebot.git
cd edgebot

# Build the binary
make build

# Or install it to your $GOPATH/bin (includes yzma libs)
make install

# Install yzma CLI and llama.cpp libraries to ~/.edgebot/models/llamacpp/
make install-yzma
```

## Usage

Start the interactive shell by running:

```bash
edgebot
```

### Interactive Commands

Within the `edgebot >` prompt, you can use the following commands (with or without the `/` prefix):

- **Type anything**: Send a request to the AI (e.g., "how do I find large files?").
- **`help`**: Show the help menu.
- **`get-config`**: See current model and configuration file location.
- **`models`**: Open a menu to select the model.
- **`reset`**: Clear the chat history.
- **`exit` or `quit`**: Close the shell.

### TUI Shortcuts

- **Arrows (↑/↓)**: Navigate through command history.
- **Tab**: Trigger autocomplete for commands.
- **Ctrl+C**: Stop the current operation or exit.
- **Esc**: Cancel the current request or clear the input.

### Extract Structured Data

Structured extraction is now a **structured command**: a custom command whose frontmatter has a `schema:` field pointing to a JSON schema. It sends a document or image to the LLM and gets back only the data you asked for, shaped by that schema. The schema file is a standard [JSON Schema](https://json-schema.org/) document.

Create a command file, e.g. `.edgebot/commands/invoice.md`:

```markdown
---
description: Extract invoice data
schema: invoice_schema.json
---
Extract the invoice from the provided file.
```

The `schema:` path is resolved relative to the command file's directory. Then run it, passing input files as arguments:

```bash
edgebot commands --run invoice invoice.pdf
edgebot commands --run invoice notes.txt --output result.json
edgebot commands --run invoice receipt.png
```

For example, to pull the key fields out of an invoice, save this schema as `invoice_schema.json` next to the command:

```json
{
  "type": "object",
  "properties": {
    "invoice_number": { "type": "string" },
    "vendor":          { "type": "string" },
    "date":            { "type": "string", "format": "date" },
    "total_amount":    { "type": "number" }
  },
  "required": ["invoice_number", "vendor", "total_amount"]
}
```

Then run:

```bash
edgebot commands --run invoice invoice.pdf
```

Which returns something like:

```json
{
  "invoice_number": "INV-2024-001",
  "vendor": "Acme Corp",
  "date": "2024-03-15",
  "total_amount": 1234.56
}
```

Supported inputs: `.txt`, `.md`, `.pdf` (extracted via `pdftotext`), and images `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` (requires a vision-capable model). File arguments may be used by any command, structured or not. Use `-o result.json` to write the output to a file instead of stdout.

## Configuration

The application looks for a `.env` file in:
1. The current directory.
2. `~/.config/edgebot/.env`

Required environment variables depending on your provider:
- **Ollama**: No API key needed (local)
- **Gemini**: `GEMINI_API_KEY`
- **OpenRouter**: `OPEN_ROUTE_KEY`

The application also looks for a `config.yaml` file in:
1. The current directory.
2. `~/.config/edgebot/config.yaml`

Default configuration:

```yaml
llm:
  provider: "ollama"
  model: "granite4:3b-h"
shell:
  confirm: true
  allowed_commands: "ls,pwd"
```

The provider/model can also be selected without editing `config.yaml` via
explicit environment variables (they override the file, e.g. from
`docker-compose.yml`):

```bash
LLM_PROVIDER=litertlm
LLM_MODEL=gemma-4-E2B-it
LITERTLM_BACKEND=cpu
```

When only `LLM_MODEL` is set, the provider is auto-detected like
`edgebot config --model` does.

### Configuration Options

- **`llm.provider`**: The AI provider to use (`ollama`, `gemini`, `openrouter`, `litertlm` or `llamacpp`).
- **`llm.model`**: The specific model name. For `litertlm`, this is a `.litertlm` filename without extension. For `llamacpp`, this is a GGUF filename (e.g., `granite4-3b-h.Q4_K_M.gguf`).
- **`litertlm.backend`**: The LiteRT-LM inference backend (`cpu` or `gpu`, default `cpu`).
- **`shell.confirm`**: If `true`, the application will always ask for confirmation before executing an AI-suggested command.
- **`shell.allowed_commands`**: A comma-separated list of safe commands that the AI can execute without requiring user confirmation (e.g., "ls,pwd,date").

## LitertLM Provider

The `litertlm` provider runs LiteRT-LM inference **in-process** using the [litertlm-go](https://github.com/vladimirvivien/litertlm-go) Go binding. It loads the LiteRT-LM shared libraries and a `.litertlm` model directly into the edgebot process — no server binary needed.

### Setup

1. Download the LiteRT-LM C API release and place it in `~/.edgebot/lib/`, then fetch the prebuilt aux/GPU libraries:
   ```bash
   make install-litertlm
   ```
   This downloads the official `litert_lm_c_api` release from `google-ai-edge/LiteRT-LM`, extracts `lib/linux_x86_64/liblitert-lm.so`, renames it to `liblitertlm_c_cpu.so` (the filename the binding actually dlopens), and fetches the prebuilt aux/GPU libs into `~/.edgebot/lib/`. No symlink is needed.

2. Place a `.litertlm` model file in `~/.edgebot/models/litertlm/`. You can download one with `edgebot models pull`:
   ```bash
   edgebot models pull <repo> <filename.litertlm>
   ```
   Files ending in `.litertlm` are saved to `~/.edgebot/models/litertlm/` and auto-registered as LiteRT-LM models; any other file is treated as a GGUF model for the `llamacpp` provider.

3. Configure `~/.config/edgebot/config.yaml`:
   ```yaml
   llm:
     provider: "litertlm"
     model: "gemma-4-E2B-it"
   litertlm:
     backend: "cpu"
   ```

   The `.litertlm` extension is appended automatically if omitted. The backend (`cpu`/`gpu`, default `cpu`) can also be set via `LITERTLM_BACKEND`.

### Limitations

- **Structured output** (structured commands) uses schema prompting, not the binding's native structured-output path.
- **No API key or base URL** needed — purely local.

## Llamacpp Provider

The `llamacpp` provider runs llama.cpp inference **in-process** using the [yzma](https://github.com/hybridgroup/yzma) Go binding. Unlike other providers that send HTTP requests to a server, it loads the model directly into the edgebot process — no external service needed.

### Setup

1. Install yzma and the llama.cpp shared libraries:
   ```bash
   make install-yzma
   ```
   This downloads the llama.cpp `.so` files to `~/.edgebot/models/llamacpp/`.

2. Place a GGUF model file in `~/.edgebot/models/llamacpp/`. For example:
   ```bash
   wget -O ~/.edgebot/models/llamacpp/granite4-3b-h.Q4_K_M.gguf <url>
   ```

3. Configure `~/.config/edgebot/config.yaml`:
   ```yaml
   llm:
     provider: "llamacpp"
     model: "granite4-3b-h.Q4_K_M.gguf"
   ```

   The `.gguf` extension is appended automatically if omitted.

### Image input

Attach images with `@filepath` as usual. For image input to work the model must be a vision model and a matching vision projector (`mmproj`) GGUF must be available:

1. Place the `mmproj` file next to your model in `~/.edgebot/models/llamacpp/`. It is auto-detected when its name contains `mmproj` (e.g. `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf`) and is excluded from the model list.

2. Requires llama.cpp libraries ≥ b10273 (v1.23.0) — run `make install-yzma` again if you installed an older version.

Vision models with matching GGUFs include moondream2 and Qwen2.5-VL. Without a projector, image requests return a clear error while text-only inference keeps working.

### Limitations

- **Structured output** (structured commands) is supported via a GBNF grammar derived from the JSON schema.
- **Image input** is supported for vision models with an `mmproj` (see above).
- **Audio input** is not supported.
- **No API key or base URL** needed — purely local.

## Service Modality

`edgebot service` runs a lightweight gRPC server that exposes the full
edgebot logic (prompts, AGENTS.md files, skills, tools, and LLM calls) over a unix
socket at `~/.edgebot/service.sock`. When the service is running, other
edgebot sessions detect it and route their requests through it instead of
calling the LLM locally.

```bash
edgebot service           # start the service (foreground)
edgebot service --stop    # stop a running service
edgebot service --status  # show whether a service is running
```

Sessions route through the service automatically when it is active — no extra
configuration needed. The interactive shell, `edgebot commands --run`, and
`edgebot commit` all use it when available and fall back to local execution
if the service becomes unreachable. The session's own model/provider/agent
settings are honored (e.g. switching models with `/models` still works), while
the service supplies the prompts, AGENTS.md context, skills index, tool execution, and the
LLM calls from its working directory.

Tool confirmation is honored inside the service: with the default
`shell.confirm: true`, commands outside `allowed_commands` and all file
writes are denied (the service cannot prompt interactively). Set
`edgebot config --confirm=false` to let the service auto-execute everything.

## How it works

Edgebot uses a unified OpenAI-compatible API to communicate with your LLM models. It uses a system prompt to inform the LLM about your environment (e.g., "running on Ubuntu 22.04 using /bin/bash"). 

When the LLM decides it needs to perform an action, it can use the following tools:
- **`RunCommand`**: Executes a shell command and returns the output.
- **`ReadFile`**: Reads the content of a specified file.
- **`WriteFile`**: Writes content to a specified file.
- **`KVSet`/`KVGet`/`KVList`**: Persistent key-value store (bbolt).

Commands are executed via `bash -c`. For safety, execution of commands not in `allowed_commands` always requires explicit user confirmation (`y/N`).

## License

MIT
