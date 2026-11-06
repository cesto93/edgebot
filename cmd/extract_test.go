//go:build integration

package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"edgebot/config"
)

func TestRunStructuredCommandWithRealLLM(t *testing.T) {
	// The structured command runs locally against the llm model in
	// ~/.edgebot/models/llamacpp (see integration_llamacpp_test.go).
	tests := []struct {
		name  string
		model string
	}{
		{
			name:  "llamacpp-" + integrationLLMModel,
			model: integrationLLMModel,
		},
	}

	schema := `{
		"type": "object",
		"properties": {
			"invoice_number": { "type": "string" },
			"vendor":         { "type": "string" },
			"total_amount":   { "type": "number" }
		},
		"required": ["invoice_number", "vendor", "total_amount"]
	}`

	input := `Invoice INV-2024-001
Vendor: Acme Corp
Total: $1,234.56`

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requireLlamacppModel(t, tt.model)

			tmpDir := t.TempDir()

			origDir, err := os.Getwd()
			if err != nil {
				t.Fatal(err)
			}
			os.Chdir(tmpDir)
			defer os.Chdir(origDir)

			if err := os.WriteFile(filepath.Join(tmpDir, "invoice.txt"), []byte(input), 0644); err != nil {
				t.Fatal(err)
			}

			commandsDir := filepath.Join(tmpDir, ".edgebot", "commands")
			if err := os.MkdirAll(commandsDir, 0755); err != nil {
				t.Fatal(err)
			}

			schemaPath := filepath.Join(commandsDir, "schema.json")
			if err := os.WriteFile(schemaPath, []byte(schema), 0644); err != nil {
				t.Fatal(err)
			}
			cmdContent := "---\ndescription: Extract invoice data\nschema: schema.json\n---\nExtract structured data from the invoice.\n"
			if err := os.WriteFile(filepath.Join(commandsDir, "invoice.md"), []byte(cmdContent), 0644); err != nil {
				t.Fatal(err)
			}

			configYAML := fmt.Sprintf("llm:\n  provider: llamacpp\n  model: %s\n", tt.model)
			if err := os.WriteFile(filepath.Join(tmpDir, "config.yaml"), []byte(configYAML), 0644); err != nil {
				t.Fatal(err)
			}

			cfg, err := config.LoadConfig()
			if err != nil {
				t.Fatalf("failed to load config: %v", err)
			}
			commandOutput = ""
			if err := runCustomCommand(cfg, "invoice", []string{"invoice.txt"}); err != nil {
				t.Fatalf("runCustomCommand() error = %v", err)
			}
		})
	}
}
