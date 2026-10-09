package cmd

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"edgebot/config"

	"github.com/spf13/cobra"
)

var modelsPullCmd = &cobra.Command{
	Use:   "pull <repo> <model> [mmproj]",
	Short: "Download model file(s) from a HuggingFace repo",
	Long: `Download one or more model files from a HuggingFace repository to the local models directory.

The repo is the HuggingFace repository path (e.g., unsloth/Qwen3.5-2B-GGUF).
The filenames are the specific model files to download (e.g., Qwen3.5-2B-Q4_K_M.gguf).

The destination is chosen by each filename extension: .litertlm files are saved
to ~/.edgebot/models/litertlm/ and registered as LiteRT-LM models; anything
else is saved to ~/.edgebot/models/llamacpp/ as a GGUF model.

Up to two files may be downloaded per repo: when a single file is pulled the
config is updated to use this model; when two files are pulled the first is
treated as the model and the second as its vision projector (mmproj), which is
auto-detected by scanning the models dir.`,
	Args: cobra.RangeArgs(2, 3),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runPull(args[0], args[1:])
	},
}

func init() {
	modelsCmd.AddCommand(modelsPullCmd)
}

func runPull(repo string, filenames []string) error {
	if err := validateRepo(repo); err != nil {
		return err
	}
	for _, f := range filenames {
		if err := validateFilename(f); err != nil {
			return err
		}
	}
	var lastErr error
	for i, filename := range filenames {
		if err := downloadFile(repo, filename); err != nil {
			lastErr = err
			fmt.Fprintf(os.Stderr, "Failed to download %s: %v\n", filename, err)
			continue
		}
		if err := updateConfig(filename, i, len(filenames)); err != nil {
			fmt.Printf("Warning: failed to update config: %v\n", err)
		}
	}
	return lastErr
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

// updateConfig records the downloaded file in the config. With a single file
// the model itself is saved; with two files the first is saved as the model
// and the second is only downloaded — it is auto-detected as the vision
// projector (mmproj) by scanning the models dir. A GGUF carrying System One
// decision metadata becomes the decision default used by `edgebot decide`
// instead of the chat model.
func updateConfig(filename string, index, total int) error {
	if total == 2 && index == 1 {
		return nil
	}
	provider := providerForFilename(filename)
	modelName := strings.TrimSuffix(filename, filepath.Ext(filename))
	if provider == "llamacpp" {
		if dir, err := config.ModelsDir(provider); err == nil {
			if config.IsDecisionModelFile(filepath.Join(dir, filename)) {
				if err := config.SaveDecisionModel(modelName); err != nil {
					return err
				}
				fmt.Printf("Decision model set to %s (used by `edgebot decide`)\n", modelName)
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
