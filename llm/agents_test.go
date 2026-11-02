package llm

import (
	"strings"
	"testing"
)

func TestGetAllTools(t *testing.T) {
	tools := GetAllTools()
	if len(tools) == 0 {
		t.Fatal("GetAllTools() returned empty slice")
	}

	expectedNames := []string{"RunCommand", "WriteFile", "ReadFile", "KVSet", "KVGet", "KVList"}
	foundNames := make(map[string]bool)

	for _, tool := range tools {
		toolMap, ok := tool.(map[string]any)
		if !ok {
			t.Fatalf("Tool is not map[string]any: %T", tool)
		}
		function, ok := toolMap["function"].(map[string]any)
		if !ok {
			t.Fatal("Tool missing 'function' key")
		}
		name, ok := function["name"].(string)
		if !ok {
			t.Fatal("Tool function missing 'name' key")
		}
		foundNames[name] = true

		if _, ok := function["description"].(string); !ok {
			t.Errorf("Tool %q missing 'description' key", name)
		}
		if _, ok := function["parameters"].(map[string]any); !ok {
			t.Errorf("Tool %q missing 'parameters' key", name)
		}
	}

	for _, expected := range expectedNames {
		if !foundNames[expected] {
			t.Errorf("GetAllTools() missing tool %q", expected)
		}
	}
}

func TestGetEnabledTools(t *testing.T) {
	tests := []struct {
		name        string
		enabledMap  map[string]bool
		expectCount int
	}{
		{
			name:        "nil map returns all tools",
			enabledMap:  nil,
			expectCount: 6,
		},
		{
			name:        "empty map returns all tools",
			enabledMap:  map[string]bool{},
			expectCount: 6,
		},
		{
			name: "disable one tool",
			enabledMap: map[string]bool{
				"RunCommand": false,
			},
			expectCount: 5,
		},
		{
			name: "enable only two tools",
			enabledMap: map[string]bool{
				"RunCommand": true,
				"ReadFile":   true,
				"WriteFile":  false,
				"KVSet":      false,
				"KVGet":      false,
				"KVList":     false,
			},
			expectCount: 2,
		},
		{
			name: "explicitly enabled not in all tools still works",
			enabledMap: map[string]bool{
				"NonExistent": true,
			},
			expectCount: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetEnabledTools(tt.enabledMap)
			if len(got) != tt.expectCount {
				t.Errorf("GetEnabledTools() returned %d tools, want %d", len(got), tt.expectCount)
			}
		})
	}
}

func TestGetEnabledToolsReturnsCorrectTools(t *testing.T) {
	enabledMap := map[string]bool{
		"RunCommand": true,
		"ReadFile":   true,
		"WriteFile":  false,
		"KVSet":      false,
		"KVGet":      false,
		"KVList":     false,
	}

	got := GetEnabledTools(enabledMap)
	names := make(map[string]bool)
	for _, tool := range got {
		toolMap := tool.(map[string]any)
		function := toolMap["function"].(map[string]any)
		names[function["name"].(string)] = true
	}

	if !names["RunCommand"] {
		t.Error("Expected RunCommand to be enabled")
	}
	if !names["ReadFile"] {
		t.Error("Expected ReadFile to be enabled")
	}
	if names["WriteFile"] {
		t.Error("Expected WriteFile to be disabled")
	}
	if names["KVSet"] {
		t.Error("Expected KVSet to be disabled")
	}
}

func TestGetToolDescriptions(t *testing.T) {
	descs := GetToolDescriptions()
	if len(descs) == 0 {
		t.Fatal("GetToolDescriptions() returned empty map")
	}

	expectedTools := []string{"RunCommand", "WriteFile", "ReadFile", "KVSet", "KVGet", "KVList"}
	for _, name := range expectedTools {
		desc, ok := descs[name]
		if !ok {
			t.Errorf("GetToolDescriptions() missing tool %q", name)
			continue
		}
		if desc == "" {
			t.Errorf("Tool %q has empty description", name)
		}
	}
}

func TestGetToolDescriptionsConsistency(t *testing.T) {
	allTools := GetAllTools()
	descs := GetToolDescriptions()

	if len(allTools) != len(descs) {
		t.Errorf("GetAllTools() returned %d tools, GetToolDescriptions() returned %d entries", len(allTools), len(descs))
	}
}

func TestBuildToolDescriptions(t *testing.T) {
	tools := GetAllTools()
	result := buildToolDescriptions(tools)

	if result == "" {
		t.Fatal("buildToolDescriptions() returned empty string")
	}

	if !strings.Contains(result, "- RunCommand:") {
		t.Error("buildToolDescriptions() missing RunCommand")
	}
	if !strings.Contains(result, "- WriteFile:") {
		t.Error("buildToolDescriptions() missing WriteFile")
	}
	if !strings.Contains(result, "- ReadFile:") {
		t.Error("buildToolDescriptions() missing ReadFile")
	}
}

func TestBuildToolDescriptionsEmpty(t *testing.T) {
	result := buildToolDescriptions([]any{})
	if result != "" {
		t.Errorf("buildToolDescriptions() with empty input returned %q, want empty string", result)
	}
}

func TestBuildToolDescriptionsSkipsInvalid(t *testing.T) {
	input := []any{
		"not a map",
		map[string]any{"wrong": "structure"},
		map[string]any{
			"function": map[string]any{
				"name":        "TestTool",
				"description": "A test tool",
			},
		},
	}

	result := buildToolDescriptions(input)
	if !strings.Contains(result, "TestTool") {
		t.Error("buildToolDescriptions() should include valid tool")
	}
	if strings.Contains(result, "not a map") {
		t.Error("buildToolDescriptions() should skip invalid entries")
	}
}

func TestNewAgent(t *testing.T) {
	agent := NewAgent("test-model", "ollama", nil)

	if agent == nil {
		t.Fatal("NewAgent() returned nil")
	}
	if agent.Model != "test-model" {
		t.Errorf("NewAgent().Model = %q, want %q", agent.Model, "test-model")
	}
	if agent.Provider != "ollama" {
		t.Errorf("NewAgent().Provider = %q, want %q", agent.Provider, "ollama")
	}
	if agent.Prompt == "" {
		t.Error("NewAgent().Prompt is empty")
	}
	if len(agent.Tools) == 0 {
		t.Error("NewAgent().Tools is empty")
	}
}

func TestNewAgentWithDisabledTools(t *testing.T) {
	enabledMap := map[string]bool{
		"RunCommand": true,
		"ReadFile":   true,
		"WriteFile":  false,
		"KVSet":      false,
		"KVGet":      false,
		"KVList":     false,
	}

	agent := NewAgent("test-model", "ollama", enabledMap)
	if len(agent.Tools) != 2 {
		t.Errorf("NewAgent() with disabled tools returned %d tools, want 2", len(agent.Tools))
	}
}

func toolNames(tools []any) map[string]bool {
	names := make(map[string]bool)
	for _, tool := range tools {
		toolMap, ok := tool.(map[string]any)
		if !ok {
			continue
		}
		function, ok := toolMap["function"].(map[string]any)
		if !ok {
			continue
		}
		if name, ok := function["name"].(string); ok {
			names[name] = true
		}
	}
	return names
}

func TestGetAgentDefs(t *testing.T) {
	defs := GetAgentDefs()
	if len(defs) < 2 {
		t.Fatalf("GetAgentDefs() returned %d agents, want at least 2", len(defs))
	}
	if defs[0].Name != "build" {
		t.Errorf("GetAgentDefs()[0].Name = %q, want %q", defs[0].Name, "build")
	}

	names := make(map[string]bool)
	for _, def := range defs {
		if def.Name == "" {
			t.Error("agent definition missing name")
		}
		if def.Description == "" {
			t.Errorf("agent %q missing description", def.Name)
		}
		if len(def.Tools) == 0 {
			t.Errorf("agent %q has no tools", def.Name)
		}
		names[def.Name] = true
	}
	if !names["build"] {
		t.Error("GetAgentDefs() missing build agent")
	}
	if !names["plan"] {
		t.Error("GetAgentDefs() missing plan agent")
	}
}

func TestGetAgentDef(t *testing.T) {
	if def := GetAgentDef("build"); def.Name != "build" {
		t.Errorf("GetAgentDef(build).Name = %q, want build", def.Name)
	}
	if def := GetAgentDef("plan"); def.Name != "plan" {
		t.Errorf("GetAgentDef(plan).Name = %q, want plan", def.Name)
	}
	if def := GetAgentDef(""); def.Name != "build" {
		t.Errorf("GetAgentDef(\"\").Name = %q, want build", def.Name)
	}
	if def := GetAgentDef("nonexistent"); def.Name != "build" {
		t.Errorf("GetAgentDef(nonexistent).Name = %q, want build", def.Name)
	}
}

func TestPlanAgentTools(t *testing.T) {
	def := GetAgentDef("plan")
	names := toolNames(def.EnabledTools(nil))

	if names["RunCommand"] {
		t.Error("plan agent should not enable RunCommand")
	}
	if names["WriteFile"] {
		t.Error("plan agent should not enable WriteFile")
	}
	for _, disabled := range []string{"KVSet", "KVGet", "KVList"} {
		if names[disabled] {
			t.Errorf("plan agent should not enable %s", disabled)
		}
	}
	if !names["ReadFile"] {
		t.Error("plan agent should enable ReadFile")
	}
}

func TestBuildAgentTools(t *testing.T) {
	def := GetAgentDef("build")
	names := toolNames(def.EnabledTools(nil))

	for _, expected := range []string{"RunCommand", "WriteFile", "ReadFile"} {
		if !names[expected] {
			t.Errorf("build agent should enable %s", expected)
		}
	}
}

func TestAgentEnabledToolsRespectsUserToggle(t *testing.T) {
	def := GetAgentDef("build")
	names := toolNames(def.EnabledTools(map[string]bool{"ReadFile": false}))

	if names["ReadFile"] {
		t.Error("user-disabled ReadFile should stay disabled for build agent")
	}
	if !names["RunCommand"] {
		t.Error("RunCommand should remain enabled for build agent")
	}
}

func TestNewAgentFor(t *testing.T) {
	plan := NewAgentFor("plan", "test-model", "ollama", nil)
	if plan == nil {
		t.Fatal("NewAgentFor(plan) returned nil")
	}
	if len(plan.Tools) != 1 {
		t.Errorf("NewAgentFor(plan) returned %d tools, want 1", len(plan.Tools))
	}
	if !strings.Contains(plan.Prompt, "planning agent") {
		t.Errorf("plan agent prompt missing planning role, got: %s", plan.Prompt)
	}

	build := NewAgentFor("build", "test-model", "ollama", nil)
	if len(build.Tools) != 3 {
		t.Errorf("NewAgentFor(build) returned %d tools, want 3", len(build.Tools))
	}
	if !strings.Contains(build.Prompt, "expert shell assistant") {
		t.Errorf("build agent prompt missing default role, got: %s", build.Prompt)
	}
}

func TestNewAgentForSessionChatSkipsContext(t *testing.T) {
	agent := NewAgentForSession("chat", "test-model", "ollama", nil, "cpu", true, true, "")
	if agent.AgentFiles != "" {
		t.Errorf("chat agent should skip AGENTS.md, got %q", agent.AgentFiles)
	}
	if agent.Skills != "" {
		t.Errorf("chat agent should skip skills, got %q", agent.Skills)
	}
	if len(agent.Tools) != 0 {
		t.Errorf("NewAgentForSession(chat) returned %d tools, want 0", len(agent.Tools))
	}
}

func TestNewAgentForUnknownFallsBackToBuild(t *testing.T) {
	agent := NewAgentFor("nonexistent", "test-model", "ollama", nil)
	if len(agent.Tools) != 3 {
		t.Errorf("NewAgentFor(nonexistent) returned %d tools, want 3", len(agent.Tools))
	}
}
