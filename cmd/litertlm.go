package cmd

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"edgebot/config"

	"github.com/spf13/cobra"
	"github.com/vladimirvivien/litertlm-go/pkg/libfetch"
)

// litertlmRuntimeVersion is the default LiteRT-LM release to stage. It
// tracks the version supported by the pinned litertlm-go binding (C API
// 1.0.0, unified liblitert-lm shared library).
const litertlmRuntimeVersion = libfetch.DefaultVersion

type litertlmInstallOptions struct {
	Version  string
	Platform string
	Backend  string
	Dir      string
	Force    bool
}

var litertlmInstallOpts litertlmInstallOptions

var litertlmCmd = &cobra.Command{
	Use:   "litertlm",
	Short: "Manage the LiteRT-LM native runtime",
}

var litertlmInstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Download and stage the LiteRT-LM shared library",
	Long: `Download the official LiteRT-LM C-API release and stage its unified
shared library (liblitert-lm.so on Linux) in ~/.edgebot/lib.

The download uses the same libfetch helper as the litertlm-go binding, so the
staged runtime always matches the version the binding was built against. The
destination is $LITERTLM_LIB when set, otherwise ~/.edgebot/lib. Existing
unified libraries are reused; pass --force to re-download.`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, _ []string) error {
		_, err := runLitertLMInstall(cmd.Context(), cmd.OutOrStdout(), litertlmInstallOpts)
		return err
	},
}

func init() {
	f := litertlmInstallCmd.Flags()
	f.StringVar(&litertlmInstallOpts.Version, "version", "", "LiteRT-LM release version (default $LITERTLM_VERSION or "+litertlmRuntimeVersion+")")
	f.StringVar(&litertlmInstallOpts.Platform, "platform", "", "target platform, e.g. linux_x86_64/linux_arm64 (default: host arch)")
	f.StringVar(&litertlmInstallOpts.Backend, "backend", "", "backend hint for fallback prebuilts: cpu or gpu (default cpu)")
	f.StringVar(&litertlmInstallOpts.Dir, "dir", "", "destination directory (default $LITERTLM_LIB or ~/.edgebot/lib)")
	f.BoolVar(&litertlmInstallOpts.Force, "force", false, "re-download even when the library is already present")

	litertlmCmd.AddCommand(litertlmInstallCmd)
	rootCmd.AddCommand(litertlmCmd)
}

// runLitertLMInstall stages the LiteRT-LM runtime and returns the directory
// containing it. Legacy pre-1.0.0 libraries are removed after a successful
// install so the binding cannot pick up a stale, ABI-incompatible runtime.
func runLitertLMInstall(ctx context.Context, out io.Writer, opts litertlmInstallOptions) (string, error) {
	version := firstNonEmpty(opts.Version, os.Getenv("LITERTLM_VERSION"), litertlmRuntimeVersion)

	platform := firstNonEmpty(opts.Platform, os.Getenv("LITERTLM_PLAT"))
	if platform == "" {
		p, err := libfetch.Platform()
		if err != nil {
			return "", fmt.Errorf("detect LiteRT-LM platform: %w", err)
		}
		platform = p
	}

	backend := firstNonEmpty(opts.Backend, os.Getenv("LITERTLM_BACKEND"), "cpu")

	dir := firstNonEmpty(opts.Dir, os.Getenv("LITERTLM_LIB"))
	if dir == "" {
		d, err := config.LibDir()
		if err != nil {
			return "", fmt.Errorf("resolve LiteRT-LM lib dir: %w", err)
		}
		dir = d
	}

	if opts.Force {
		for _, name := range litertlmLibNames(platform) {
			if err := os.Remove(filepath.Join(dir, name)); err == nil {
				fmt.Fprintf(out, "Removed %s\n", filepath.Join(dir, name))
			}
		}
	}

	fmt.Fprintf(out, "Installing LiteRT-LM %s (%s) into %s\n", version, platform, dir)
	installed, err := libfetch.Fetch(ctx,
		libfetch.WithVersion(version),
		libfetch.WithPlatform(platform),
		libfetch.WithBackend(backend),
		libfetch.WithDir(dir),
		libfetch.WithLogf(func(format string, args ...any) {
			fmt.Fprintf(out, format+"\n", args...)
		}),
	)
	if err != nil {
		return "", fmt.Errorf("install LiteRT-LM %s: %w", version, err)
	}

	// C API 1.0.0 bundles the former auxiliary plugins inside the unified
	// library; drop any leftovers from a pre-1.0.0 install.
	for _, name := range litertlmLegacyLibNames(platform) {
		if err := os.Remove(filepath.Join(installed, name)); err == nil {
			fmt.Fprintf(out, "Removed legacy library %s\n", filepath.Join(installed, name))
		}
	}

	mainLib := filepath.Join(installed, litertlmUnifiedLibName(platform))
	fmt.Fprintf(out, "LiteRT-LM runtime ready: %s\n", mainLib)
	return installed, nil
}

// litertlmUnifiedLibName is the unified C API 1.0.0 library filename that the
// litertlm-go loader opens (short name "litert-lm").
func litertlmUnifiedLibName(platform string) string {
	switch {
	case strings.HasPrefix(platform, "windows_"):
		return "litert-lm.dll"
	case strings.HasPrefix(platform, "macos_"):
		return "liblitert-lm.dylib"
	default:
		return "liblitert-lm.so"
	}
}

// litertlmLibNames are the files whose removal forces a fresh download.
func litertlmLibNames(platform string) []string {
	return append([]string{litertlmUnifiedLibName(platform)}, litertlmLegacyLibNames(platform)...)
}

// litertlmLegacyLibNames are the pre-C-API-1.0.0 main and plugin libraries.
// They are no longer needed: the unified runtime bundles their functionality.
func litertlmLegacyLibNames(platform string) []string {
	ext := ".so"
	mainPrefix := "lib"
	auxPrefix := "lib"
	switch {
	case strings.HasPrefix(platform, "windows_"):
		ext = ".dll"
		mainPrefix = ""
	case strings.HasPrefix(platform, "macos_"):
		ext = ".dylib"
	}
	return []string{
		mainPrefix + "litertlm_c_cpu" + ext,
		mainPrefix + "litertlm_c" + ext,
		auxPrefix + "GemmaModelConstraintProvider" + ext,
		auxPrefix + "LiteRt" + ext,
		auxPrefix + "LiteRtWebGpuAccelerator" + ext,
		auxPrefix + "LiteRtTopKWebGpuSampler" + ext,
		auxPrefix + "LiteRtMetalAccelerator" + ext,
		auxPrefix + "LiteRtTopKMetalSampler" + ext,
	}
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
