package cmd

import (
	"fmt"
	"os"

	"edgebot/config"

	"github.com/spf13/cobra"
)

var debug bool

var rootCmd = &cobra.Command{
	Use:   "edgebot",
	Short: "AI Shell is an interactive shell powered by AI",
	Long:  `An interactive shell powered by AI that can help you with commands and explanations.`,
	Example: `  edgebot
  edgebot config
  edgebot commit
  edgebot models
  edgebot ocr document.pdf
  edgebot commands --run extract invoice.pdf`,
	Run: func(cmd *cobra.Command, args []string) {
		if err := RunShell(); err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	},
}

func init() {
	rootCmd.PersistentFlags().BoolVar(&debug, "debug", false, "enable debug logging (temporarily overrides the configured log level)")
	rootCmd.AddCommand(commitCmd)
}

// initLogger configures the global slog level, temporarily forcing debug mode
// when the --debug flag is passed. The configured log level is not modified.
func initLogger(cfg *config.Config) {
	level := cfg.LogLevel
	if debug {
		level = "debug"
	}
	config.InitLogger(level)
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
