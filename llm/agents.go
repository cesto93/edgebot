package llm

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
)

// Agent represents an AI agent with its prompt, model, provider, and tools.
type Agent struct {
	Prompt     string
	AgentFiles string
	Skills     string
	Model      string
	Provider   string
	Tools      []any
	Backend    string
	// ThinkEffort is the unified reasoning-effort level ("" = provider default).
	ThinkEffort ThinkEffort
}

// GetAllTools returns the full list of tools for the agent.
func GetAllTools() []any {
	return []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "RunCommand",
				"description": "Execute a shell command and return its output",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command": map[string]any{
							"type":        "string",
							"description": "The shell command to execute (e.g., 'ls -la', 'echo hello')",
						},
					},
					"required": []string{"command"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "WriteFile",
				"description": "Write content to a file at the specified path",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "The absolute or relative path to the file",
						},
						"content": map[string]any{
							"type":        "string",
							"description": "The content to write to the file",
						},
					},
					"required": []string{"path", "content"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "ReadFile",
				"description": "Read the content of a file at the specified path",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "The absolute or relative path to the file",
						},
					},
					"required": []string{"path"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "KVSet",
				"description": "Save a value to the KV store with a given key",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"key": map[string]any{
							"type":        "string",
							"description": "The key to save",
						},
						"value": map[string]any{
							"type":        "string",
							"description": "The value to save",
						},
					},
					"required": []string{"key", "value"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "KVGet",
				"description": "Retrieve a value from the KV store by key",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"key": map[string]any{
							"type":        "string",
							"description": "The key to retrieve",
						},
					},
					"required": []string{"key"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "KVList",
				"description": "List all keys currently in the KV store",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	}
}

// GetEnabledTools filters the full list of tools based on the enabledMap.
func GetEnabledTools(enabledMap map[string]bool) []any {
	allTools := GetAllTools()
	if enabledMap == nil {
		return allTools
	}

	var enabledTools []any
	for _, t := range allTools {
		if toolMap, ok := t.(map[string]any); ok {
			if function, ok := toolMap["function"].(map[string]any); ok {
				if name, ok := function["name"].(string); ok {
					if enabled, exists := enabledMap[name]; !exists || enabled {
						enabledTools = append(enabledTools, t)
					}
				}
			}
		}
	}
	return enabledTools
}

// GetToolDescriptions returns a map of tool name to description extracted from GetAllTools.
func GetToolDescriptions() map[string]string {
	descs := make(map[string]string)
	for _, t := range GetAllTools() {
		if toolMap, ok := t.(map[string]any); ok {
			if function, ok := toolMap["function"].(map[string]any); ok {
				name, _ := function["name"].(string)
				desc, _ := function["description"].(string)
				if name != "" {
					descs[name] = desc
				}
			}
		}
	}
	return descs
}

type PromptData struct {
	Distro string
	Shell  string
	Cwd    string
	Tools  string
}

func buildToolDescriptions(tools []any) string {
	var sb bytes.Buffer
	for _, t := range tools {
		toolMap, ok := t.(map[string]any)
		if !ok {
			continue
		}
		function, ok := toolMap["function"].(map[string]any)
		if !ok {
			continue
		}
		name, _ := function["name"].(string)
		desc, _ := function["description"].(string)
		sb.WriteString("- ")
		sb.WriteString(name)
		sb.WriteString(": ")
		sb.WriteString(desc)
		sb.WriteString("\n")
	}
	return sb.String()
}

// GetDefaultSystemPrompt returns the default system prompt based on distro, shell, and current directory.
func GetDefaultSystemPrompt(enabledTools map[string]bool) string {
	return GetAgentSystemPrompt("build", GetEnabledTools(enabledTools))
}

// readPromptFile reads BUILDPROMPT.md from ~/.edgebot/BUILDPROMPT.md.
// Falls back to the embedded default if the file cannot be read.
func readPromptFile() []byte {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("Cannot determine home directory, using embedded prompt", "err", err)
		return GetDefaultPromptBytes()
	}

	raw, err := os.ReadFile(filepath.Join(home, ".edgebot", "BUILDPROMPT.md"))
	if err != nil {
		slog.Warn("Cannot read ~/.edgebot/BUILDPROMPT.md, using embedded prompt", "err", err)
		return GetDefaultPromptBytes()
	}

	return raw
}

// readPlanPromptFile reads PLANPROMPT.md from ~/.edgebot/PLANPROMPT.md.
// Falls back to the embedded plan prompt if the file cannot be read.
func readPlanPromptFile() []byte {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("Cannot determine home directory, using embedded prompt", "err", err)
		return []byte(PlanPrompt)
	}

	raw, err := os.ReadFile(filepath.Join(home, ".edgebot", "PLANPROMPT.md"))
	if err != nil {
		slog.Warn("Cannot read ~/.edgebot/PLANPROMPT.md, using embedded prompt", "err", err)
		return []byte(PlanPrompt)
	}

	return raw
}

// NewAgent creates a new Agent with the given parameters.
func NewAgent(model, provider string, toolsEnabled map[string]bool) *Agent {
	return &Agent{
		Prompt:   GetDefaultSystemPrompt(toolsEnabled),
		Model:    model,
		Provider: provider,
		Tools:    GetEnabledTools(toolsEnabled),
	}
}

// AgentDef describes a named agent: which tools it is allowed to use.
type AgentDef struct {
	Name        string
	Description string
	Tools       map[string]bool
}

var buildAgentTools = map[string]bool{
	"RunCommand": true,
	"WriteFile":  true,
	"ReadFile":   true,
	"KVSet":      false,
	"KVGet":      false,
	"KVList":     false,
}

var planAgentTools = map[string]bool{
	"RunCommand": false,
	"WriteFile":  false,
	"ReadFile":   true,
	"KVSet":      false,
	"KVGet":      false,
	"KVList":     false,
}

var botAgentTools = map[string]bool{
	"RunCommand": false,
	"WriteFile":  false,
	"ReadFile":   true,
	"KVSet":      true,
	"KVGet":      true,
	"KVList":     true,
}

var chatAgentTools = map[string]bool{
	"RunCommand": false,
	"WriteFile":  false,
	"ReadFile":   false,
	"KVSet":      false,
	"KVGet":      false,
	"KVList":     false,
}

// GetAgentDefs returns the list of built-in agents. The first entry (build) is
// the default agent.
func GetAgentDefs() []AgentDef {
	clone := func(m map[string]bool) map[string]bool {
		c := make(map[string]bool, len(m))
		for k, v := range m {
			c[k] = v
		}
		return c
	}
	return []AgentDef{
		{
			Name:        "build",
			Description: "Full access: all tools enabled with the default prompt",
			Tools:       clone(buildAgentTools),
		},
		{
			Name:        "plan",
			Description: "Read-only planning: cannot write files or launch commands",
			Tools:       clone(planAgentTools),
		},
		{
			Name:        "bot",
			Description: "Telegram bot: read-only with KV store (ReadFile, KVGet, KVList, KVSet)",
			Tools:       clone(botAgentTools),
		},
		{
			Name:        "chat",
			Description: "Plain chat: no tools, conversation only",
			Tools:       clone(chatAgentTools),
		},
	}
}

// GetAgentDef returns the agent definition for the given name, falling back to
// the default build agent for empty or unknown names.
func GetAgentDef(name string) AgentDef {
	if name == "" {
		name = "build"
	}
	for _, def := range GetAgentDefs() {
		if def.Name == name {
			return def
		}
	}
	return GetAgentDefs()[0]
}

// EnabledTools merges the agent's allowed tools with the user's tool toggles
// (cfg.Tools). A tool is available only when the agent permits it AND the user
// has not disabled it.
func (d AgentDef) EnabledTools(cfgTools map[string]bool) []any {
	merged := make(map[string]bool)
	for _, t := range GetAllTools() {
		if toolMap, ok := t.(map[string]any); ok {
			if function, ok := toolMap["function"].(map[string]any); ok {
				if name, ok := function["name"].(string); ok {
					enabled := d.Tools[name]
					if v, exists := cfgTools[name]; exists {
						enabled = enabled && v
					}
					merged[name] = enabled
				}
			}
		}
	}
	return GetEnabledTools(merged)
}

// NewAgentFor creates a new Agent for the named agent, restricted to the tools
// the agent is allowed to use (intersected with the user's tool toggles).
func NewAgentFor(agentName, model, provider string, cfgTools map[string]bool) *Agent {
	def := GetAgentDef(agentName)
	tools := def.EnabledTools(cfgTools)
	return &Agent{
		Prompt:   GetAgentSystemPrompt(def.Name, tools),
		Model:    model,
		Provider: provider,
		Tools:    tools,
	}
}

// NewAgentForSession creates an Agent like NewAgentFor, additionally applying
// the session's inference backend, think effort, AGENTS.md support, and
// skills support settings. The chat agent is conversation-only, so it skips
// AGENTS.md and skills context entirely to keep the prompt minimal.
func NewAgentForSession(agentName, model, provider string, cfgTools map[string]bool, backend string, agentFiles, skills bool, thinkEffort ThinkEffort) *Agent {
	a := NewAgentFor(agentName, model, provider, cfgTools)
	a.Backend = backend
	a.ThinkEffort = thinkEffort
	if GetAgentDef(agentName).Name == "chat" {
		return a
	}
	a.AgentFiles = GetAgentFiles(agentFiles)
	a.Skills = GetSkillsPrompt(skills)
	return a
}
