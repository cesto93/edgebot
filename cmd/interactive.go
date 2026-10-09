package cmd

import (
	"fmt"
	"sort"
	"strings"

	"edgebot/config"
	"edgebot/llm"
)

const (
	ColorBlue   = "\033[34m"
	ColorCyan   = "\033[36m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorReset  = "\033[0m"
	ColorBold   = "\033[1m"
	Prompt      = ColorBold + ColorGreen + "edgebot > " + ColorReset
)

func PrintConfig() {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("%sError loading config: %v%s\n", ColorYellow, err, ColorReset)
		return
	}

	fmt.Printf("%sCurrent Configuration:%s\n", ColorBold+ColorCyan, ColorReset)
	fmt.Printf("Log Level: %s%s%s\n", ColorGreen, cfg.LogLevel, ColorReset)
	fmt.Printf("Agent: %s%s%s\n", ColorGreen, cfg.Agent, ColorReset)
	fmt.Printf("Agent Files: %s%v%s\n", ColorGreen, cfg.AgentFiles, ColorReset)
	fmt.Printf("Skills: %s%v%s\n", ColorGreen, cfg.Skills, ColorReset)
	fmt.Printf("Provider: %s%s%s\n", ColorGreen, cfg.LLM.Provider, ColorReset)
	fmt.Printf("Model: %s%s%s\n", ColorGreen, cfg.LLM.Model, ColorReset)
	decisionModel := cfg.LLM.DecisionModel
	if decisionModel == "" {
		decisionModel = "(not set)"
	}
	fmt.Printf("Decision Model: %s%s%s\n", ColorGreen, decisionModel, ColorReset)
	thinkEffort := cfg.LLM.ThinkEffort
	if thinkEffort == "" {
		thinkEffort = "(provider default)"
	}
	fmt.Printf("Think Effort: %s%s%s\n", ColorGreen, thinkEffort, ColorReset)
	fmt.Printf("Confirm Commands: %s%v%s\n", ColorGreen, cfg.Shell.Confirm, ColorReset)
	fmt.Printf("Allowed Commands: %s%s%s\n", ColorGreen, strings.Join(cfg.Shell.AllowedCommands, ","), ColorReset)

	if len(cfg.Tools) > 0 {
		fmt.Printf("\n%sTools:%s\n", ColorBold+ColorCyan, ColorReset)
		toolDescs := llm.GetToolDescriptions()
		var toolNames []string
		for name := range cfg.Tools {
			toolNames = append(toolNames, name)
		}
		sort.Strings(toolNames)
		for _, name := range toolNames {
			status := "disabled"
			if cfg.Tools[name] {
				status = "enabled"
			}
			desc := toolDescs[name]
			fmt.Printf("  %s%-12s%s %s(%s)%s", ColorGreen, name, ColorReset, ColorBlue, status, ColorReset)
			if desc != "" {
				fmt.Printf(" %s- %s%s", ColorYellow, desc, ColorReset)
			}
			fmt.Println()
		}
	}

	if len(cfg.Commands) > 0 {
		fmt.Printf("\n%sCommands:%s\n", ColorBold+ColorCyan, ColorReset)
		var cmdNames []string
		for name := range cfg.Commands {
			cmdNames = append(cmdNames, name)
		}
		sort.Strings(cmdNames)
		for _, name := range cmdNames {
			prompt := cfg.Commands[name]
			fmt.Printf("  %s/%-11s%s %s- %s%s\n", ColorGreen, name, ColorReset, ColorYellow, prompt, ColorReset)
		}
	}

	if cfg.ConfigFile != "" {
		fmt.Printf("\nConfig file: %s%s%s\n", ColorBlue, cfg.ConfigFile, ColorReset)
	} else {
		fmt.Printf("\nConfig file: %sNone (using defaults)%s\n", ColorYellow, ColorReset)
	}
}
