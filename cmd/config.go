package cmd

import (
	"fmt"
	"slices"
	"strings"

	"edgebot/config"
	"edgebot/llm"

	"github.com/spf13/cobra"
)

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Show or modify configuration settings",
	Long:  `Show the current edgebot configuration, or modify it by passing flags to set specific config fields.`,
	Example: `  edgebot config
  edgebot config --provider gemini --model gemini-2.0-flash
  edgebot config --decision-model lev-1
  edgebot config --log-level debug
  edgebot config --agent plan
  edgebot config --agent-files=false
  edgebot config --confirm=false
  edgebot config --allowed-commands "ls,pwd,git,curl"
  edgebot config --backend gpu
  edgebot config --think-effort low
  edgebot config --add-cmd "hello=say hello world"
  edgebot config --rm-cmd "hello"
  edgebot config --enable-tool WriteFile
  edgebot config --disable-tool KVGet`,
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := config.LoadConfig()
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		changed := false
		fl := cmd.Flags()

		if fl.Changed("provider") {
			v, _ := fl.GetString("provider")
			cfg.LLM.Provider = v
			changed = true
		}

		if fl.Changed("model") {
			v, _ := fl.GetString("model")
			if config.IsDecisionModel(v) {
				cfg.LLM.DecisionModel = v
				fmt.Printf("Note: %q is a decision model; set as the decision default (used by `edgebot decide`) instead of the chat model.\n", v)
			} else {
				cfg.LLM.Model = v
				if !fl.Changed("provider") {
					if info := config.LookupModelInfo(v); info != nil {
						cfg.LLM.Provider = info.Provider
						cfg.LLM.InputTypes = info.InputTypes
					}
				}
			}
			changed = true
		}

		if fl.Changed("decision-model") {
			v, _ := fl.GetString("decision-model")
			if err := config.SaveDecisionModel(v); err != nil {
				return err
			}
			fmt.Printf("Decision model set to %s (used by `edgebot decide`)\n", v)
			// Reload so a later save below does not clobber the change.
			cfg, err = config.LoadConfig()
			if err != nil {
				return fmt.Errorf("failed to reload config: %w", err)
			}
		}

		if fl.Changed("agent") {
			v, _ := fl.GetString("agent")
			cfg.Agent = v
			changed = true
		}

		if fl.Changed("agent-files") {
			v, _ := fl.GetBool("agent-files")
			cfg.AgentFiles = v
			changed = true
		}

		if fl.Changed("skills") {
			v, _ := fl.GetBool("skills")
			cfg.Skills = v
			changed = true
		}

		if fl.Changed("log-level") {
			v, _ := fl.GetString("log-level")
			cfg.LogLevel = v
			changed = true
		}

		if fl.Changed("confirm") {
			v, _ := fl.GetBool("confirm")
			cfg.Shell.Confirm = v
			changed = true
		}

		if fl.Changed("allowed-commands") {
			v, _ := fl.GetStringSlice("allowed-commands")
			cfg.Shell.AllowedCommands = v
			changed = true
		}

		if fl.Changed("backend") {
			v, _ := fl.GetString("backend")
			cfg.LitertLM.Backend = v
			changed = true
		}

		if fl.Changed("think-effort") {
			v, _ := fl.GetString("think-effort")
			effort, err := llm.ParseThinkEffort(v)
			if err != nil {
				return err
			}
			cfg.LLM.ThinkEffort = string(effort)
			changed = true
		}

		if fl.Changed("enable-tool") {
			v, _ := fl.GetString("enable-tool")
			if !isKnownTool(v) {
				return fmt.Errorf("unknown tool %q", v)
			}
			if cfg.Tools == nil {
				cfg.Tools = make(map[string]bool)
			}
			cfg.Tools[v] = true
			changed = true
		}

		if fl.Changed("disable-tool") {
			v, _ := fl.GetString("disable-tool")
			if !isKnownTool(v) {
				return fmt.Errorf("unknown tool %q", v)
			}
			if cfg.Tools == nil {
				cfg.Tools = make(map[string]bool)
			}
			cfg.Tools[v] = false
			changed = true
		}

		if fl.Changed("add-cmd") {
			val, _ := fl.GetString("add-cmd")
			name, prompt, found := strings.Cut(val, "=")
			if !found || name == "" {
				return fmt.Errorf("invalid format for --add-cmd, expected name=prompt")
			}
			if cfg.Commands == nil {
				cfg.Commands = make(map[string]string)
			}
			cfg.Commands[name] = prompt
			changed = true
		}

		if fl.Changed("rm-cmd") {
			v, _ := fl.GetString("rm-cmd")
			delete(cfg.Commands, v)
			changed = true
		}

		if !changed {
			PrintConfig()
			return nil
		}

		if err := config.SaveConfig(cfg); err != nil {
			return fmt.Errorf("failed to save config: %w", err)
		}

		fmt.Println("Configuration updated successfully.")
		return nil
	},
}

func init() {
	configCmd.Flags().String("provider", "", "Set LLM provider (ollama, gemini, openrouter, litertlm)")
	configCmd.Flags().String("model", "", "Set LLM model name")
	configCmd.Flags().String("decision-model", "", "Set default decision model for `edgebot decide` (local System One GGUF)")
	configCmd.Flags().String("agent", "", "Set active agent (build, plan, chat)")
	configCmd.Flags().Bool("agent-files", false, "Enable or disable AGENTS.md support")
	configCmd.Flags().Bool("skills", false, "Enable or disable skills (~/.agents/skills, ./skills) support")
	configCmd.Flags().String("log-level", "", "Set log level (debug, info, warn, error)")
	configCmd.Flags().Bool("confirm", false, "Require confirmation for tool execution")
	configCmd.Flags().StringSlice("allowed-commands", nil, "Comma-separated list of commands that skip confirmation")
	configCmd.Flags().String("backend", "", "LiteRT-LM inference backend (cpu, gpu)")
	configCmd.Flags().String("think-effort", "", "Reasoning effort (none, minimal, low, medium, high, xhigh, max; empty = provider default)")
	configCmd.Flags().String("enable-tool", "", "Enable a tool by name")
	configCmd.Flags().String("disable-tool", "", "Disable a tool by name")
	configCmd.Flags().String("add-cmd", "", "Add a custom command (format: name=prompt)")
	configCmd.Flags().String("rm-cmd", "", "Remove a custom command by name")

	rootCmd.AddCommand(configCmd)
}

func isKnownTool(name string) bool {
	known := []string{"RunCommand", "WriteFile", "ReadFile", "KVSet", "KVGet", "KVList"}
	return slices.Contains(known, name)
}
