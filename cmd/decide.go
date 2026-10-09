package cmd

import (
	"fmt"
	"io"
	"os"
	"strings"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/stats"

	"github.com/hybridgroup/yzma/exp/decide"
	"github.com/spf13/cobra"
)

var decideModel string
var decideRequest string
var decideState string
var decideStateFile string
var decideChoice string
var decideScore string
var decideNoul string
var decideOutput string
var decideBothOrders bool

var decideCmd = &cobra.Command{
	Use:   "decide",
	Short: "Ask a System One decision model a typed question",
	Long: `Score a state with a local decision model (llamacpp provider).

Decision models answer typed questions with calibrated probabilities instead
of text. The model must be a self-describing System One GGUF (openjev, lev,
laya, kev, ...) in ~/.edgebot/models/llamacpp/ — pull one with
'edgebot models pull <repo> <model.gguf>' first. The model comes from --model
or the configured decision default ('edgebot config --decision-model').

Two input modes (mutually exclusive):

- Full request: --request takes a TypeSafe /v1/systemone JSON file ('-' for
  stdin) with a "state" value and a "questions" object mapping ids to
  {type, instructions, criteria} questions of type choice, score or noul.

- Single question: --state (or --state-file) plus exactly one of --choice,
  --score or --noul, each "instructions|..." with pipe-separated options:
  choice takes name or name=description options, score takes 2-10 level
  descriptions, noul takes the false and true descriptions.

The response is printed as indented JSON (answers plus token usage), or
written to --output.`,
	Example: `  edgebot decide --request request.json
  edgebot decide --state '{"ticket": "charged twice"}' --choice "Who handles this?|billing=payments|technical=bugs"
  edgebot decide --state "The film was excellent." --score "Sentiment?|negative|positive"
  edgebot decide --state "The meeting is on Thursday." --noul "Holds?|does not hold|holds"`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runDecide()
	},
}

func init() {
	rootCmd.AddCommand(decideCmd)
	decideCmd.Flags().StringVar(&decideModel, "model", "", "decision model name (default: the configured decision model)")
	decideCmd.Flags().StringVar(&decideRequest, "request", "", "TypeSafe /v1/systemone JSON request file ('-' for stdin)")
	decideCmd.Flags().StringVar(&decideState, "state", "", "state to judge (plain text or JSON)")
	decideCmd.Flags().StringVar(&decideStateFile, "state-file", "", "read the state from a file")
	decideCmd.Flags().StringVar(&decideChoice, "choice", "", `"instructions|name|name=description|..." choice question`)
	decideCmd.Flags().StringVar(&decideScore, "score", "", `"instructions|level 0|level 1|..." score question`)
	decideCmd.Flags().StringVar(&decideNoul, "noul", "", `"instructions|false description|true description" true/false question`)
	decideCmd.Flags().BoolVar(&decideBothOrders, "both-orders", false, "read choice questions in both option orders and average (cancels first-option bias)")
	decideCmd.Flags().StringVarP(&decideOutput, "output", "o", "", "write the JSON response to a file instead of stdout")
}

func runDecide() error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	model, err := resolveDecideModel(cfg, decideModel)
	if err != nil {
		return err
	}

	req, err := decideRequestFromFlags()
	if err != nil {
		return err
	}

	modelDir, err := config.ModelsDir("llamacpp")
	if err != nil {
		return fmt.Errorf("cannot determine models dir: %w", err)
	}
	modelPath, err := llm.ResolveDecisionModelPath(modelDir, model)
	if err != nil {
		return err
	}
	libDir, err := config.LibDir()
	if err != nil {
		return fmt.Errorf("cannot determine lib dir: %w", err)
	}

	d, err := llm.OpenDecisionModel(libDir, modelPath, decide.Options{BothOrders: decideBothOrders})
	if err != nil {
		return err
	}
	defer d.Close()

	resp, err := d.Answer(req)
	if err != nil {
		return fmt.Errorf("decide: %w", err)
	}
	stats.RecordUsage("llamacpp", model, stats.Usage{
		PromptTokens: resp.Usage.InputTokens,
		TotalTokens:  resp.Usage.InputTokens,
	})

	out := llm.FormatDecisionResponse(resp)
	if decideOutput != "" {
		if err := os.WriteFile(decideOutput, []byte(out+"\n"), 0644); err != nil {
			return fmt.Errorf("failed to write output: %w", err)
		}
		return nil
	}
	fmt.Println(out)
	return nil
}

// resolveDecideModel picks the decision model: the --model flag wins, then
// the configured decision default, then (for configs predating the decision
// default) the chat model when it points at a decision GGUF.
func resolveDecideModel(cfg *config.Config, flag string) (string, error) {
	if strings.TrimSpace(flag) != "" {
		return strings.TrimSpace(flag), nil
	}
	if strings.TrimSpace(cfg.LLM.DecisionModel) != "" {
		return cfg.LLM.DecisionModel, nil
	}
	if cfg.LLM.Provider == "llamacpp" && config.IsDecisionModel(cfg.LLM.Model) {
		return cfg.LLM.Model, nil
	}
	return "", fmt.Errorf("no decision model configured (use --model or 'edgebot config --decision-model <name>')")
}

// decideRequestFromFlags builds the decision request from either --request or
// the single-question flags (--state/--state-file plus one question type).
func decideRequestFromFlags() (*decide.Request, error) {
	if decideRequest != "" {
		if decideState != "" || decideStateFile != "" || decideChoice != "" || decideScore != "" || decideNoul != "" {
			return nil, fmt.Errorf("--request cannot be combined with --state, --state-file, --choice, --score or --noul")
		}
		data, err := readDecideRequest(decideRequest)
		if err != nil {
			return nil, err
		}
		req, err := decide.ParseRequest(data)
		if err != nil {
			return nil, fmt.Errorf("decide: %w", err)
		}
		return req, nil
	}

	state, err := decideStateFromFlags()
	if err != nil {
		return nil, err
	}
	q, err := decideQuestionFromFlags()
	if err != nil {
		return nil, err
	}
	return &decide.Request{
		State:     llm.ParseDecisionState(state),
		Questions: []decide.NamedQuestion{{ID: "question", Question: q}},
	}, nil
}

func readDecideRequest(path string) ([]byte, error) {
	if path == "-" {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return nil, fmt.Errorf("failed to read request from stdin: %w", err)
		}
		return data, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read request file: %w", err)
	}
	return data, nil
}

func decideStateFromFlags() (string, error) {
	if decideState != "" && decideStateFile != "" {
		return "", fmt.Errorf("--state and --state-file are mutually exclusive")
	}
	if decideStateFile != "" {
		data, err := os.ReadFile(decideStateFile)
		if err != nil {
			return "", fmt.Errorf("failed to read state file: %w", err)
		}
		if strings.TrimSpace(string(data)) == "" {
			return "", fmt.Errorf("--state-file %q is empty", decideStateFile)
		}
		return string(data), nil
	}
	if strings.TrimSpace(decideState) == "" {
		return "", fmt.Errorf("--state or --state-file is required (or use --request)")
	}
	return decideState, nil
}

func decideQuestionFromFlags() (decide.Question, error) {
	set := 0
	for _, s := range []string{decideChoice, decideScore, decideNoul} {
		if s != "" {
			set++
		}
	}
	if set == 0 {
		return decide.Question{}, fmt.Errorf("one of --choice, --score or --noul is required (or use --request)")
	}
	if set > 1 {
		return decide.Question{}, fmt.Errorf("--choice, --score and --noul are mutually exclusive")
	}
	switch {
	case decideChoice != "":
		return llm.BuildChoiceQuestion(decideChoice)
	case decideScore != "":
		return llm.BuildScoreQuestion(decideScore)
	default:
		return llm.BuildNoulQuestion(decideNoul)
	}
}
