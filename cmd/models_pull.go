package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"edgebot/config"

	"github.com/spf13/cobra"
)

var modelsPullCmd = &cobra.Command{
	Use:   "pull <repo> [model] [mmproj]",
	Short: "Download model file(s) from a HuggingFace repo",
	Long: `Download one or more model files from a HuggingFace repository to the local models directory.

The repo is the HuggingFace repository path (e.g., unsloth/Qwen3.5-2B-GGUF).
When no filenames are given, the repo is queried and the available model files
are listed so one can be selected interactively. If the repo also contains a
vision projector (mmproj) it is included automatically; when several exist, one
is selected interactively.

The filenames are the specific model files to download (e.g., Qwen3.5-2B-Q4_K_M.gguf).

The destination is chosen by each filename extension: .litertlm files are saved
to ~/.edgebot/models/litertlm/ and registered as LiteRT-LM models; anything
else is saved to ~/.edgebot/models/llamacpp/ as a GGUF model.

Up to two files may be downloaded per repo: when a single file is pulled the
config is updated to use this model; when two files are pulled the first is
treated as the model and the second as its vision projector (mmproj), which is
auto-detected by scanning the models dir.`,
	Args: cobra.RangeArgs(1, 3),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runPull(args[0], args[1:])
	},
}

func init() {
	modelsCmd.AddCommand(modelsPullCmd)
}

// repoFile is a file entry returned by the HuggingFace repository tree API.
type repoFile struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
}

var (
	// fetchRepoFiles is swappable in tests.
	fetchRepoFiles           = fetchHuggingFaceFiles
	pullInput      io.Reader = os.Stdin
	pullOutput     io.Writer = os.Stdout
)

func runPull(repo string, filenames []string) error {
	if err := validateRepo(repo); err != nil {
		return err
	}
	if len(filenames) == 0 {
		selected, err := selectRepoFiles(repo, bufio.NewReader(pullInput), pullOutput)
		if err != nil {
			return err
		}
		filenames = selected
	}
	for _, f := range filenames {
		if err := validateFilename(f); err != nil {
			return err
		}
	}
	var lastErr error
	downloaded := make([]bool, len(filenames))
	for i, filename := range filenames {
		if err := downloadFile(repo, filename); err != nil {
			lastErr = err
			fmt.Fprintf(os.Stderr, "Failed to download %s: %v\n", filename, err)
			continue
		}
		downloaded[i] = true
	}
	if downloaded[0] {
		if err := updateConfig(filenames, downloaded); err != nil {
			fmt.Printf("Warning: failed to update config: %v\n", err)
		}
	}
	return lastErr
}

// selectRepoFiles queries the repo and asks which model (and, when present,
// which vision projector) to download. A single candidate is chosen without
// prompting; a vision projector is assumed to be wanted whenever one exists.
func selectRepoFiles(repo string, in *bufio.Reader, out io.Writer) ([]string, error) {
	files, err := fetchRepoFiles(repo)
	if err != nil {
		return nil, err
	}
	models, projectors := splitRepoFiles(files)
	if len(models) == 0 {
		return nil, fmt.Errorf("no .gguf or .litertlm model files found in %s", repo)
	}

	model, err := promptRepoFile(in, out, repo, "model file", models)
	if err != nil {
		return nil, err
	}
	selected := []string{model.Path}
	if providerForFilename(model.Path) == "llamacpp" && len(projectors) > 0 {
		projector, err := promptRepoFile(in, out, repo, "vision projector (mmproj)", projectors)
		if err != nil {
			return nil, err
		}
		selected = append(selected, projector.Path)
	}
	return selected, nil
}

// splitRepoFiles separates downloadable model files from vision projectors.
// Only files at the repo root are considered, since filenames are saved
// directly into the local models directory. Results are sorted by name.
func splitRepoFiles(files []repoFile) (models, projectors []repoFile) {
	for _, f := range files {
		if f.Type != "" && f.Type != "file" {
			continue
		}
		name := f.Path
		if name == "" || strings.Contains(name, "/") || strings.Contains(name, "\\") {
			continue
		}
		lower := strings.ToLower(name)
		if !strings.HasSuffix(lower, ".gguf") && !strings.HasSuffix(lower, ".litertlm") {
			continue
		}
		if strings.HasSuffix(lower, ".gguf") && strings.Contains(lower, "mmproj") {
			projectors = append(projectors, f)
			continue
		}
		models = append(models, f)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Path < models[j].Path })
	sort.Slice(projectors, func(i, j int) bool { return projectors[i].Path < projectors[j].Path })
	return models, projectors
}

// promptRepoFile shows the available files and reads a selection by number or
// exact name. When only one candidate exists it is returned immediately.
func promptRepoFile(in *bufio.Reader, out io.Writer, repo, kind string, options []repoFile) (repoFile, error) {
	if len(options) == 1 {
		fmt.Fprintf(out, "Using %s: %s\n", kind, options[0].Path)
		return options[0], nil
	}

	fmt.Fprintf(out, "\nAvailable %s files in %s:\n", kind, repo)
	for i, f := range options {
		size := ""
		if f.Size > 0 {
			size = fmt.Sprintf(" (%s)", config.FormatFileSize(f.Size))
		}
		fmt.Fprintf(out, "  %d) %s%s\n", i+1, f.Path, size)
	}
	for {
		fmt.Fprintf(out, "Select a %s [1-%d]: ", kind, len(options))
		line, err := in.ReadString('\n')
		if err != nil && err != io.EOF {
			return repoFile{}, fmt.Errorf("reading selection: %w", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			if err == io.EOF {
				return repoFile{}, fmt.Errorf("no %s selected", kind)
			}
			fmt.Fprintln(out, "Please enter a number or a filename.")
			continue
		}
		if n, convErr := strconv.Atoi(line); convErr == nil {
			if n >= 1 && n <= len(options) {
				return options[n-1], nil
			}
		}
		for _, f := range options {
			if f.Path == line {
				return f, nil
			}
		}
		fmt.Fprintf(out, "Invalid selection %q\n", line)
		if err == io.EOF {
			return repoFile{}, fmt.Errorf("invalid %s selection %q", kind, line)
		}
	}
}

// fetchHuggingFaceFiles lists the files in a HuggingFace repo via its tree API.
func fetchHuggingFaceFiles(repo string) ([]repoFile, error) {
	url := fmt.Sprintf("https://huggingface.co/api/models/%s/tree/main", repo)
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to list files in %s: %w", repo, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		if len(body) > 0 {
			return nil, fmt.Errorf("failed to list files in %s: %s: %s", repo, resp.Status, strings.TrimSpace(string(body)))
		}
		return nil, fmt.Errorf("failed to list files in %s: %s", repo, resp.Status)
	}

	var files []repoFile
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&files); err != nil {
		return nil, fmt.Errorf("failed to parse file list for %s: %w", repo, err)
	}
	return files, nil
}

var repoRe = regexp.MustCompile(`^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$`)

func validateRepo(repo string) error {
	if strings.TrimSpace(repo) == "" || strings.Contains(repo, "..") || strings.Contains(repo, "\\") || strings.Contains(repo, " ") {
		return fmt.Errorf("invalid repo %q", repo)
	}
	if !repoRe.MatchString(repo) {
		return fmt.Errorf("invalid repo %q: expected owner/name", repo)
	}
	return nil
}

func validateFilename(filename string) error {
	if filename == "" || filename == "." || strings.Contains(filename, "/") || strings.Contains(filename, "\\") || strings.Contains(filename, "..") {
		return fmt.Errorf("invalid filename %q", filename)
	}
	return nil
}

// updateConfig records the downloaded file(s) in the config: the first is the
// model, an optional second is its vision projector (mmproj), which is
// auto-detected by scanning the models dir. A GGUF carrying System One
// decision metadata becomes the decision default used by `edgebot decide`; an
// OCR GGUF becomes the `edgebot ocr` default and remembers its projector.
func updateConfig(filenames []string, downloaded []bool) error {
	if len(filenames) == 0 {
		return nil
	}
	filename := filenames[0]
	provider := providerForFilename(filename)
	modelName := strings.TrimSuffix(filename, filepath.Ext(filename))
	if provider == "llamacpp" {
		if dir, err := config.ModelsDir(provider); err == nil {
			path := filepath.Join(dir, filename)
			if config.IsDecisionModelFile(path) {
				if err := config.SaveDecisionModel(modelName); err != nil {
					return err
				}
				fmt.Printf("Decision model set to %s (used by `edgebot decide`)\n", modelName)
				return nil
			}
			if config.IsOCRModelFile(path) {
				if err := config.SaveOCRModel(modelName); err != nil {
					return err
				}
				if len(filenames) > 1 && len(downloaded) > 1 && downloaded[1] {
					cfg, err := config.LoadConfig()
					if err != nil {
						return fmt.Errorf("failed to load config: %w", err)
					}
					cfg.LLM.OCRMMProj = strings.TrimSuffix(filenames[1], filepath.Ext(filenames[1]))
					if err := config.SaveConfig(cfg); err != nil {
						return fmt.Errorf("failed to save OCR projector: %w", err)
					}
				}
				fmt.Printf("OCR model set to %s (used by `edgebot ocr`)\n", modelName)
				return nil
			}
		}
	}
	return config.SaveModelWithProvider(modelName, provider)
}

func providerForFilename(filename string) string {
	if strings.HasSuffix(strings.ToLower(filename), ".litertlm") {
		return "litertlm"
	}
	return "llamacpp"
}

func downloadFile(repo, filename string) error {
	provider := providerForFilename(filename)

	destDir, err := config.ModelsDir(provider)
	if err != nil {
		return fmt.Errorf("cannot determine models directory: %w", err)
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	destPath := filepath.Join(destDir, filename)

	if info, err := os.Stat(destPath); err == nil {
		if info.IsDir() {
			return fmt.Errorf("destination is a directory: %s", destPath)
		}
		return fmt.Errorf("model file already exists at %s", destPath)
	}

	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename)
	fmt.Printf("Downloading %s ...\n", url)

	out, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("model file already exists at %s", destPath)
		}
		return fmt.Errorf("failed to create output file: %w", err)
	}
	cleanup := func() {
		out.Close()
		if rmErr := os.Remove(destPath); rmErr != nil && !os.IsNotExist(rmErr) {
			slog.Warn("failed to remove partial file", "path", destPath, "err", rmErr)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		cleanup()
		return fmt.Errorf("failed to create request: %w", err)
	}
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		cleanup()
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		cleanup()
		if len(body) > 0 {
			return fmt.Errorf("download failed: %s: %s", resp.Status, strings.TrimSpace(string(body)))
		}
		return fmt.Errorf("download failed: %s", resp.Status)
	}

	contentLength := resp.ContentLength
	buf := make([]byte, 32*1024)
	var written int64
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				cleanup()
				return fmt.Errorf("write error: %w", werr)
			}
			written += int64(n)
			if contentLength > 0 {
				pct := float64(written) / float64(contentLength) * 100
				fmt.Fprintf(os.Stderr, "\rDownloading... %.1f%% (%s/%s)", pct, config.FormatFileSize(written), config.FormatFileSize(contentLength))
			} else {
				fmt.Fprintf(os.Stderr, "\rDownloading... %s", config.FormatFileSize(written))
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			cleanup()
			return fmt.Errorf("download error: %w", err)
		}
	}
	if err := out.Close(); err != nil {
		cleanup()
		return fmt.Errorf("failed to close file: %w", err)
	}
	fmt.Fprintf(os.Stderr, "\n")

	fmt.Printf("Model saved to %s\n", destPath)
	return nil
}
