package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"edgebot/config"
	"edgebot/llm"

	"github.com/spf13/cobra"
)

const (
	ocrDefaultDPI    = "200"
	ocrDefaultCtx    = 8192
	ocrDefaultTokens = 4096
)

// OCR task prompts from the MiniCPM-OCR model card. OCR models expect the task
// as the user message, without a system prompt.
const (
	ocrTextPrompt    = "Text Recognition:"
	ocrTablePrompt   = "Table Recognition:"
	ocrFormulaPrompt = "Formula Recognition:"
)

var (
	ocrModel  string
	ocrMMProj string
	ocrMode   string
	ocrPrompt string
	ocrDPI    string
	ocrPages  string
	ocrOutput string
)

var ocrCmd = &cobra.Command{
	Use:   "ocr <image|pdf>...",
	Short: "Extract text from images and PDFs with the configured OCR model",
	Long: `Extract text from images and PDFs using the OCR model configured with
'llm.ocr_model' (set it with 'edgebot config --ocr-model <model>').

Images are sent directly to the model. PDFs are rasterized page by page with
pdftoppm (poppler-utils) and OCR'd in order. The default prompt is
"Text Recognition:"; use --mode table or --mode formula for tables and
display formulas.`,
	Example: `  edgebot ocr page.png
  edgebot ocr document.pdf
  edgebot ocr --mode table invoice.pdf -o invoice.md
  edgebot ocr --pages 1-3 scanned.pdf
  edgebot ocr --model MiniCPM-OCR-1_0-Q4_K_M scan.jpg`,
	Args: cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runOCR(args)
	},
}

func init() {
	rootCmd.AddCommand(ocrCmd)
	ocrCmd.Flags().StringVarP(&ocrModel, "model", "m", "", "OCR model (default: llm.ocr_model)")
	ocrCmd.Flags().StringVar(&ocrMMProj, "mmproj", "", "vision projector (mmproj) override")
	ocrCmd.Flags().StringVar(&ocrMode, "mode", "text", "recognition mode: text, table or formula")
	ocrCmd.Flags().StringVarP(&ocrPrompt, "prompt", "p", "", "custom OCR prompt (overrides --mode)")
	ocrCmd.Flags().StringVar(&ocrDPI, "dpi", ocrDefaultDPI, "PDF rasterization DPI")
	ocrCmd.Flags().StringVar(&ocrPages, "pages", "", "PDF page range, e.g. 1-3 or 5")
	ocrCmd.Flags().StringVarP(&ocrOutput, "output", "o", "", "write the extracted text to a file (default: stdout)")
}

func runOCR(args []string) error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	modelName := strings.TrimSpace(ocrModel)
	if modelName == "" {
		modelName = cfg.LLM.OCRModel
	}
	if modelName == "" {
		return fmt.Errorf("no OCR model configured: pull one with 'edgebot models pull <repo> <model.gguf> [mmproj.gguf]' or run 'edgebot config --ocr-model <model>'")
	}

	info := config.LookupModelInfo(modelName)
	if info == nil {
		return fmt.Errorf("OCR model %q not found (run 'edgebot models' to list available models)", modelName)
	}
	if info.ModelType() != config.ModelTypeOCR {
		fmt.Fprintf(os.Stderr, "Warning: %q is not classified as an OCR model; continuing anyway.\n", modelName)
	}

	prompt, err := ocrTaskPrompt()
	if err != nil {
		return err
	}

	caller := llm.NewProviderCallerWithThink(info.Provider, modelName, llm.NoopExecutor{}, "")
	if lc, ok := caller.(*llm.LlamacppCaller); ok {
		lc.CtxSize = ocrDefaultCtx
		lc.MaxTokens = ocrDefaultTokens
		lc.MMProj = strings.TrimSpace(ocrMMProj)
		if lc.MMProj == "" {
			lc.MMProj = cfg.LLM.OCRMMProj
		}
	} else if lc, ok := caller.(*llm.LitertLMCaller); ok {
		lc.Backend = cfg.LitertLM.Backend
	} else if strings.TrimSpace(ocrMMProj) != "" {
		fmt.Fprintf(os.Stderr, "Warning: --mmproj only applies to the llamacpp provider, ignoring it.\n")
	}

	slog.Debug("ocr", "provider", info.Provider, "model", modelName, "prompt", prompt)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var sections []string
	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			return fmt.Errorf("cannot read %s: %w", arg, err)
		}
		if info.IsDir() {
			return fmt.Errorf("%s is a directory (pass image or PDF files)", arg)
		}

		if isImage(arg) {
			text, err := ocrImage(ctx, caller, prompt, arg)
			if err != nil {
				return err
			}
			sections = append(sections, text)
			continue
		}

		if strings.EqualFold(filepath.Ext(arg), ".pdf") {
			text, err := ocrPDF(ctx, caller, prompt, arg)
			if err != nil {
				return err
			}
			sections = append(sections, text)
			continue
		}

		return fmt.Errorf("unsupported input %q (supported: images and .pdf)", arg)
	}

	result := strings.TrimSpace(strings.Join(sections, "\n\n---\n\n"))
	if result == "" {
		return fmt.Errorf("empty OCR result")
	}

	if ocrOutput != "" {
		if err := os.WriteFile(ocrOutput, []byte(result+"\n"), 0o644); err != nil {
			return fmt.Errorf("failed to write output: %w", err)
		}
		return nil
	}
	fmt.Println(result)
	return nil
}

// ocrTaskPrompt maps --mode to the OCR model's task prompt, unless an explicit
// --prompt was given.
func ocrTaskPrompt() (string, error) {
	if p := strings.TrimSpace(ocrPrompt); p != "" {
		return p, nil
	}
	switch strings.ToLower(strings.TrimSpace(ocrMode)) {
	case "", "text":
		return ocrTextPrompt, nil
	case "table":
		return ocrTablePrompt, nil
	case "formula":
		return ocrFormulaPrompt, nil
	default:
		return "", fmt.Errorf("invalid --mode %q (want text, table or formula)", ocrMode)
	}
}

// ocrImage runs the OCR model on a single image file and returns the extracted
// text.
func ocrImage(ctx context.Context, caller llm.Caller, prompt, path string) (string, error) {
	encoded, err := encodeImage(path)
	if err != nil {
		return "", fmt.Errorf("failed to read image %s: %w", path, err)
	}
	messages := []llm.Message{{
		Role: "user",
		Content: []llm.ContentPart{
			{Type: "text", Text: prompt},
			{Type: "image_url", ImageURL: &llm.ContentImage{URL: encoded}},
		},
	}}
	result, err := caller.Call(ctx, "", messages, nil)
	if err != nil {
		return "", fmt.Errorf("OCR failed for %s: %w", path, err)
	}
	return ocrResponseText(result)
}

// ocrPDF rasterizes the selected pages of a PDF with pdftoppm and OCRs them in
// order, joining the page texts with a blank line.
func ocrPDF(ctx context.Context, caller llm.Caller, prompt, path string) (string, error) {
	dir, err := os.MkdirTemp("", "edgebot-ocr-")
	if err != nil {
		return "", fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	pages, err := rasterizePDF(path, ocrDPI, ocrPages, dir)
	if err != nil {
		return "", err
	}

	var texts []string
	for _, page := range pages {
		text, err := ocrImage(ctx, caller, prompt, page)
		if err != nil {
			return "", err
		}
		if text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n\n"), nil
}

// rasterizePDF renders PDF pages to PNG files in dir using pdftoppm, returning
// the page paths in page order.
func rasterizePDF(path, dpi, pages, dir string) ([]string, error) {
	if _, err := strconv.Atoi(strings.TrimSpace(dpi)); err != nil {
		return nil, fmt.Errorf("invalid --dpi %q", dpi)
	}
	prefix := filepath.Join(dir, "page")
	args := []string{"-png", "-r", strings.TrimSpace(dpi)}
	if strings.TrimSpace(pages) != "" {
		first, last, err := parsePageRange(pages)
		if err != nil {
			return nil, err
		}
		args = append(args, "-f", strconv.Itoa(first), "-l", strconv.Itoa(last))
	}
	args = append(args, path, prefix)

	out, err := execCommand("pdftoppm", args...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("pdftoppm failed (is poppler-utils installed?): %w: %s", err, strings.TrimSpace(string(out)))
	}

	files, err := filepath.Glob(prefix + "*.png")
	if err != nil {
		return nil, fmt.Errorf("failed to list rendered pages: %w", err)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("pdftoppm produced no pages for %s", path)
	}
	sort.Slice(files, func(i, j int) bool {
		return pdfPageNumber(files[i]) < pdfPageNumber(files[j])
	})
	return files, nil
}

// parsePageRange parses "N" or "FIRST-LAST" into a pdftoppm page range.
func parsePageRange(s string) (int, int, error) {
	s = strings.TrimSpace(s)
	first, last, ok := strings.Cut(s, "-")
	if !ok {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			return 0, 0, fmt.Errorf("invalid --pages %q (want N or FIRST-LAST)", s)
		}
		return n, n, nil
	}
	a, errA := strconv.Atoi(strings.TrimSpace(first))
	b, errB := strconv.Atoi(strings.TrimSpace(last))
	if errA != nil || errB != nil || a < 1 || b < a {
		return 0, 0, fmt.Errorf("invalid --pages %q (want N or FIRST-LAST)", s)
	}
	return a, b, nil
}

// pdfPageNumber extracts the trailing page number from a pdftoppm output file
// (e.g. /tmp/x/page-12.png -> 12), so pages sort numerically.
func pdfPageNumber(path string) int {
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	i := strings.LastIndex(base, "-")
	if i < 0 {
		return 0
	}
	n, err := strconv.Atoi(base[i+1:])
	if err != nil {
		return 0
	}
	return n
}

// ocrResponseText extracts the assistant text from a caller result.
func ocrResponseText(messages []llm.Message) (string, error) {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Role != "assistant" {
			continue
		}
		switch content := msg.Content.(type) {
		case string:
			return stripThinkBlock(content), nil
		case []llm.ContentPart:
			var b strings.Builder
			for _, part := range content {
				if part.Type == "text" && part.Text != "" {
					if b.Len() > 0 {
						b.WriteString("\n")
					}
					b.WriteString(part.Text)
				}
			}
			return stripThinkBlock(b.String()), nil
		}
	}
	return "", fmt.Errorf("empty response from OCR model")
}
