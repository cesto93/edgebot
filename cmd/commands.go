package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"text/tabwriter"
	"time"

	"edgebot/config"
	"edgebot/llm"

	"github.com/spf13/cobra"
)

var runCommandName string
var commandOutput string

var commandsCmd = &cobra.Command{
	Use:   "commands",
	Short: "List all available custom commands",
	RunE: func(cmd *cobra.Command, args []string) error {
		return runCommands(args)
	},
}

func init() {
	commandsCmd.Flags().StringVar(&runCommandName, "run", "", "run a custom command by name")
	commandsCmd.Flags().StringVarP(&commandOutput, "output", "o", "", "output file for structured commands (default: stdout)")
	rootCmd.AddCommand(commandsCmd)
}

func runCommands(args []string) error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	if runCommandName != "" {
		return runCustomCommand(cfg, runCommandName, args)
	}

	cmds := config.LoadCommands(cfg)

	if len(cmds) == 0 {
		fmt.Println("No commands found.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "COMMAND\tDESCRIPTION")
	fmt.Fprintln(w, "-------\t-----------")
	for _, c := range cmds {
		desc := c.Description
		if desc == "" {
			desc = "-"
		}
		fmt.Fprintf(w, "%s\t%s\n", c.Name, desc)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}

	return nil
}

// runCustomCommand runs a custom command through the running service when
// available, falling back to a local LLM call otherwise.
func runCustomCommand(cfg *config.Config, name string, extraArgs []string) error {
	cmds := config.LoadCommands(cfg)
	var found *config.CommandInfo
	for i := range cmds {
		if cmds[i].Name == name {
			found = &cmds[i]
			break
		}
	}
	if found == nil {
		return fmt.Errorf("command not found: %s", name)
	}

	if found.Schema != "" {
		return runStructuredCommand(cfg, found, extraArgs)
	}

	content, err := buildCommandContent(found.Prompt, extraArgs)
	if err != nil {
		return err
	}

	messages := []llm.Message{
		{Role: "user", Content: content},
	}

	slog.Debug("provider", "name", cfg.LLM.Provider, "model", cfg.LLM.Model)
	slog.Debug("user prompt", "prompt", found.Prompt)

	llmStart := time.Now()
	resultMessages, err := customCommandCallLLM(cfg, messages)
	llmDuration := time.Since(llmStart)
	if err != nil {
		return err
	}

	slog.Debug("timing", "llm", llmDuration)

	responseStr, err := lastAssistantText(resultMessages)
	if err != nil {
		return err
	}

	fmt.Println(responseStr)
	return nil
}

// runStructuredCommand runs a command that carries a JSON schema in its
// frontmatter. The schema is loaded and the LLM is called with a structured
// output response_format; the JSON result is printed (or written to
// commandOutput). Structured commands bypass the service, like extract did.
func runStructuredCommand(cfg *config.Config, cmd *config.CommandInfo, args []string) error {
	schemaRaw, err := loadSchema(cmd.Schema)
	if err != nil {
		return err
	}

	content, err := buildCommandContent(cmd.Prompt, args)
	if err != nil {
		return err
	}

	responseFormat := structuredResponseFormat(schemaRaw)

	systemPrompt := "You extract structured data from documents and images. Return only valid JSON matching the provided schema."

	messages := []llm.Message{
		{Role: "user", Content: content},
	}

	slog.Debug("provider", "name", cfg.LLM.Provider, "model", cfg.LLM.Model)
	slog.Debug("schema", "path", cmd.Schema)

	caller := llm.NewProviderCallerRawWithThink(cfg.LLM.Provider, cfg.LLM.Model, llm.NoopExecutor{}, llm.ThinkEffort(cfg.LLM.ThinkEffort))
	if lc, ok := caller.(*llm.LitertLMCaller); ok {
		lc.Backend = cfg.LitertLM.Backend
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	llmStart := time.Now()
	resultMessages, err := caller.CallStructured(ctx, systemPrompt, messages, nil, responseFormat)
	llmDuration := time.Since(llmStart)
	if err != nil {
		return fmt.Errorf("LLM call failed: %w", err)
	}

	slog.Debug("timing", "llm", llmDuration)

	contentStr, err := lastAssistantText(resultMessages)
	if err != nil {
		return err
	}

	result := prettyJSONOrRaw(contentStr)

	if commandOutput != "" {
		if err := os.WriteFile(commandOutput, []byte(result+"\n"), 0644); err != nil {
			return fmt.Errorf("failed to write output: %w", err)
		}
	} else {
		fmt.Println(result)
	}

	return nil
}

// customCommandCallLLM runs the custom command through the running service
// when available, falling back to a local LLM call otherwise.
func customCommandCallLLM(cfg *config.Config, messages []llm.Message) ([]llm.Message, error) {
	req := chatRequestFromConfig(cfg, messages)
	return chatWithServiceFallback(req, func(err error) error {
		return fmt.Errorf("LLM call failed: %w", err)
	}, func() ([]llm.Message, error) {
		agent := llm.NewAgentForSession(cfg.Agent, cfg.LLM.Model, cfg.LLM.Provider, cfg.Tools, cfg.LitertLM.Backend, cfg.AgentFiles, cfg.Skills, llm.ThinkEffort(cfg.LLM.ThinkEffort))
		slog.Debug("system prompt", "prompt", agent.Prompt)
		return agent.CallLLM(context.Background(), &llm.ToolExecutorPolicy{}, messages)
	})
}
