package cmd

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"edgebot/config"

	"github.com/spf13/cobra"
)

var modelsSet string
var modelsDelete string

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List all available models from all providers",
	RunE: func(cmd *cobra.Command, args []string) error {
		return runModels(cmd)
	},
}

func init() {
	rootCmd.AddCommand(modelsCmd)
	modelsCmd.Flags().StringVarP(&modelsSet, "set", "s", "", "Set the current model")
	modelsCmd.Flags().StringVarP(&modelsDelete, "delete", "d", "", "Delete a locally downloaded model (llamacpp or litertlm)")
}

func runModels(cmd *cobra.Command) error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	if cmd.Flags().Changed("delete") {
		if strings.TrimSpace(modelsDelete) == "" {
			return fmt.Errorf("--delete requires a non-empty model name")
		}
		paths, err := config.DeleteLocalModel(modelsDelete)
		if err != nil {
			return err
		}
		for _, p := range paths {
			fmt.Printf("Deleted %s\n", p)
		}
		if cfg.LLM.Model == modelsDelete {
			fmt.Printf("Note: %q is still the configured model; run 'edgebot config --model <model>' to pick another.\n", modelsDelete)
		}
		return nil
	}

	if cmd.Flags().Changed("set") {
		if strings.TrimSpace(modelsSet) == "" {
			return fmt.Errorf("--set requires a non-empty model name")
		}
		info := config.LookupModelInfo(modelsSet)
		if info == nil {
			return fmt.Errorf("model %q not found", modelsSet)
		}
		kind, err := config.SetDefaultModel(modelsSet, info.Provider)
		if err != nil {
			return fmt.Errorf("failed to set model: %w", err)
		}
		if kind == config.ModelTypeDecision {
			fmt.Printf("Decision model set to %s (used by `edgebot decide`)\n", modelsSet)
			return nil
		}
		fmt.Printf("Model set to %s (provider: %s)\n", modelsSet, info.Provider)
		return nil
	}

	models := config.GetAllAvailableModels()
	if len(models) == 0 {
		fmt.Println("No models found.")
		return nil
	}

	sort.Slice(models, func(i, j int) bool {
		if models[i].Provider != models[j].Provider {
			return models[i].Provider < models[j].Provider
		}
		return models[i].Name < models[j].Name
	})

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "\tMODEL\tPROVIDER\tTYPE\tSIZE\tINPUT TYPES")
	fmt.Fprintln(w, "\t-----\t--------\t----\t----\t-----------")
	for _, m := range models {
		marker := ""
		if m.Name == cfg.LLM.Model && m.Provider == cfg.LLM.Provider {
			marker += "*"
		}
		if m.Provider == "llamacpp" && cfg.LLM.DecisionModel != "" && m.Name == cfg.LLM.DecisionModel {
			marker += "D"
		}
		size := m.Size
		if size == "" {
			size = "-"
		}
		inputTypes := strings.Join(m.InputTypes, ", ")
		if inputTypes == "" {
			inputTypes = "-"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", marker, m.Name, m.Provider, m.ModelType(), size, inputTypes)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	fmt.Println("* current chat model, D decision model used by `edgebot decide`")

	return nil
}
