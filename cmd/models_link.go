package cmd

import (
	"fmt"
	"strings"

	"edgebot/config"

	"github.com/spf13/cobra"
)

var modelsLinkCmd = &cobra.Command{
	Use:   "link <model> <mmproj>",
	Short: "Link a local vision projector (mmproj) to a local model",
	Long: `Record that a local GGUF model uses a local vision projector (mmproj).

Both names refer to files in the llamacpp models dir
(~/.edgebot/models/llamacpp/); the ".gguf" suffix is optional. The coupling is
saved in mmproj.json in that directory and is the only source runtime uses:
projectors are never inferred from filenames.

This is normally done automatically by "edgebot models pull" when a model and
its projector are downloaded together.`,
	Args: cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		model := strings.TrimSuffix(args[0], ".gguf")
		mmproj := strings.TrimSuffix(args[1], ".gguf")
		if err := config.SaveMMProjLink(model, mmproj); err != nil {
			return err
		}
		path, err := config.MMProjLinksFile()
		if err != nil {
			return err
		}
		fmt.Printf("Linked %s -> %s (saved in %s)\n", model, mmproj, path)
		return nil
	},
}

func init() {
	modelsCmd.AddCommand(modelsLinkCmd)
}
