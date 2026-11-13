package cmd

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"edgebot/config"
	"edgebot/llm"

	"github.com/spf13/cobra"
)

var agentsSet string

var agentsCmd = &cobra.Command{
	Use:   "agents",
	Short: "List all available agents",
	RunE: func(cmd *cobra.Command, args []string) error {
		return runAgents(cmd)
	},
}

func init() {
	rootCmd.AddCommand(agentsCmd)
	agentsCmd.Flags().StringVarP(&agentsSet, "set", "s", "", "Set the current agent")
}

func runAgents(cmd *cobra.Command) error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	if cmd.Flags().Changed("set") {
		if strings.TrimSpace(agentsSet) == "" {
			return fmt.Errorf("--set requires a non-empty agent name")
		}
		found := false
		for _, def := range llm.GetAgentDefs() {
			if def.Name == agentsSet {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("agent %q not found", agentsSet)
		}
		cfg.Agent = agentsSet
		if err := config.SaveConfig(cfg); err != nil {
			return fmt.Errorf("failed to set agent: %w", err)
		}
		fmt.Printf("Agent set to %s\n", agentsSet)
		return nil
	}

	defs := llm.GetAgentDefs()
	if len(defs) == 0 {
		fmt.Println("No agents found.")
		return nil
	}

	sorted := make([]llm.AgentDef, len(defs))
	copy(sorted, defs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "\tAGENT\tDESCRIPTION\tTOOLS")
	fmt.Fprintln(w, "\t-----\t-----------\t-----")
	for _, def := range sorted {
		marker := ""
		if def.Name == cfg.Agent {
			marker = "*"
		}
		var toolNames []string
		for name, enabled := range def.Tools {
			if enabled {
				toolNames = append(toolNames, name)
			}
		}
		sort.Strings(toolNames)
		tools := strings.Join(toolNames, ", ")
		if tools == "" {
			tools = "-"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", marker, def.Name, def.Description, tools)
	}
	// A flush failure means the table may be truncated; report it.
	if err := w.Flush(); err != nil {
		return fmt.Errorf("failed to write agents table: %w", err)
	}

	return nil
}
