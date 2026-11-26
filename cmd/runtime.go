package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	debuginfo "runtime/debug"
	"strings"
	"text/tabwriter"
	"time"

	"edgebot/config"
	"edgebot/internal/atomicfile"

	"github.com/spf13/cobra"
	"github.com/vladimirvivien/litertlm-go/pkg/libfetch"
)

// Runtime names used by `edgebot runtime` and by the in-process providers.
const (
	runtimeLlamacpp = "llamacpp"
	runtimeLitertLM = "litertlm"

	// yzmaModule is the llama.cpp binding; `runtime install llamacpp` shells out
	// to its CLI instead of linking pkg/download (which pulls the cloud SDKs
	// that go-getter needs) so the edgebot binary stays small.
	yzmaModule         = "github.com/hybridgroup/yzma"
	yzmaInstallRecord  = "yzma-install.json"
	litertlmRecordName = "litertlm-install.json"
)

// litertlmRuntimeVersion is the default LiteRT-LM release to stage. It
// tracks the version supported by the pinned litertlm-go binding (C API
// 1.0.0, unified liblitert-lm shared library).
const litertlmRuntimeVersion = libfetch.DefaultVersion

// execCommandContext runs a command with a context so installs stop on
// cancellation. Kept as a variable so tests can stub it.
var execCommandContext = exec.CommandContext

type llamacppInstallOptions struct {
	Version   string
	Processor string
	Dir       string
	Force     bool
	Verify    string
}

type litertlmInstallOptions struct {
	Version  string
	Platform string
	Backend  string
	Dir      string
	Force    bool
}

var (
	llamacppInstallOpts llamacppInstallOptions
	litertlmInstallOpts litertlmInstallOptions
	runtimeShowDir      string
)

var runtimeCmd = &cobra.Command{
	Use:   "runtime",
	Short: "Manage the native runtimes used by the in-process providers",
}

var runtimeInstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Download and stage a native runtime",
	Long: `Download and stage a native runtime in the edgebot library directory
(default ~/.edgebot/lib).

Supported runtimes:
  llamacpp   llama.cpp shared libraries used by the llamacpp provider
  litertlm   LiteRT-LM shared library used by the litertlm provider`,
}

var runtimeShowCmd = &cobra.Command{
	Use:   "show",
	Short: "Show the installed native runtimes",
	Args:  cobra.NoArgs,
	RunE: func(cmd *cobra.Command, _ []string) error {
		return runRuntimeShow(cmd.OutOrStdout(), runtimeShowDir)
	},
}

var runtimeInstallLlamacppCmd = &cobra.Command{
	Use:   "llamacpp",
	Short: "Download and stage the llama.cpp shared libraries",
	Long: `Download the llama.cpp release supported by the pinned yzma binding
and extract its shared libraries into the edgebot library directory.

The download is delegated to the yzma CLI, which installs the build matching
the host OS/arch (CUDA is picked when nvidia-smi is available, then ROCm, then
CPU). When the CLI is not on PATH it is built into a temporary directory with
the yzma version this edgebot binary was built against, so the libraries always
match the binding. The destination is ~/.edgebot/lib, where the llamacpp
provider loads the libraries from. Existing installations are reused; pass
--force to re-download.`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, _ []string) error {
		_, err := runLlamacppInstall(cmd.Context(), cmd.OutOrStdout(), llamacppInstallOpts)
		return err
	},
}

var runtimeInstallLitertLMCmd = &cobra.Command{
	Use:   "litertlm",
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
	f := runtimeInstallLlamacppCmd.Flags()
	f.StringVar(&llamacppInstallOpts.Version, "version", "", "llama.cpp release tag (default: the release pinned by the yzma binding)")
	f.StringVar(&llamacppInstallOpts.Processor, "processor", "", "processor to install: cpu, cuda, vulkan, rocm, metal, openvino (default: auto-detect)")
	f.StringVar(&llamacppInstallOpts.Dir, "dir", "", "destination directory (default ~/.edgebot/lib)")
	f.BoolVar(&llamacppInstallOpts.Force, "force", false, "re-download even when the libraries are already present")
	f.StringVar(&llamacppInstallOpts.Verify, "verify", "available", "digest verification: available, require or off")

	f = runtimeInstallLitertLMCmd.Flags()
	f.StringVar(&litertlmInstallOpts.Version, "version", "", "LiteRT-LM release version (default $LITERTLM_VERSION or "+litertlmRuntimeVersion+")")
	f.StringVar(&litertlmInstallOpts.Platform, "platform", "", "target platform, e.g. linux_x86_64/linux_arm64 (default: host arch)")
	f.StringVar(&litertlmInstallOpts.Backend, "backend", "", "backend hint for fallback prebuilts: cpu or gpu (default cpu)")
	f.StringVar(&litertlmInstallOpts.Dir, "dir", "", "destination directory (default $LITERTLM_LIB or ~/.edgebot/lib)")
	f.BoolVar(&litertlmInstallOpts.Force, "force", false, "re-download even when the library is already present")

	runtimeShowCmd.Flags().StringVar(&runtimeShowDir, "dir", "", "library directory to inspect (default $LITERTLM_LIB or ~/.edgebot/lib)")

	runtimeInstallCmd.AddCommand(runtimeInstallLlamacppCmd, runtimeInstallLitertLMCmd)
	runtimeCmd.AddCommand(runtimeInstallCmd, runtimeShowCmd)
	rootCmd.AddCommand(runtimeCmd)
}

// runLlamacppInstall stages the llama.cpp runtime and returns the directory
// containing it. The actual download is the yzma CLI's job, so edgebot only
// has to point it at the provider's lib directory and verify the result.
func runLlamacppInstall(ctx context.Context, out io.Writer, opts llamacppInstallOptions) (string, error) {
	dir, err := resolveRuntimeDir(runtimeLlamacpp, opts.Dir)
	if err != nil {
		return "", fmt.Errorf("resolve llama.cpp lib dir: %w", err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create llama.cpp lib dir: %w", err)
	}

	libPath := filepath.Join(dir, llamacppLibraryName(runtime.GOOS))
	if !opts.Force {
		if st, err := os.Stat(libPath); err == nil && st.Size() > 0 {
			fmt.Fprintf(out, "llama.cpp already installed at %s\n", dir)
			return dir, nil
		}
	}

	yzmaBin, cleanup, err := yzmaBinary(ctx, out)
	if err != nil {
		return "", err
	}
	defer cleanup()

	args := []string{"install", "--lib", dir}
	if opts.Force {
		args = append(args, "--upgrade")
	}
	if opts.Version != "" {
		args = append(args, "--version", opts.Version)
	}
	if opts.Processor != "" {
		args = append(args, "--processor", opts.Processor)
	}
	if opts.Verify != "" {
		args = append(args, "--verify", opts.Verify)
	}

	cmd := execCommandContext(ctx, yzmaBin, args...)
	cmd.Stdout = out
	cmd.Stderr = out
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("install llama.cpp: %w", err)
	}

	if st, err := os.Stat(libPath); err != nil || st.Size() == 0 {
		return "", fmt.Errorf("llama.cpp install completed but %s is missing or empty", libPath)
	}
	fmt.Fprintf(out, "llama.cpp runtime ready: %s\n", libPath)
	return dir, nil
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

	dir, err := resolveRuntimeDir(runtimeLitertLM, opts.Dir)
	if err != nil {
		return "", fmt.Errorf("resolve LiteRT-LM lib dir: %w", err)
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

	record := litertlmInstallRecord{
		Version:   version,
		Platform:  platform,
		Backend:   backend,
		Installed: time.Now().UTC(),
	}
	if err := writeLitertlmInstallRecord(installed, record); err != nil {
		return "", err
	}

	mainLib := filepath.Join(installed, litertlmUnifiedLibName(platform))
	if st, err := os.Stat(mainLib); err != nil || st.Size() == 0 {
		return "", fmt.Errorf("LiteRT-LM install completed but %s is missing or empty", mainLib)
	}
	fmt.Fprintf(out, "LiteRT-LM runtime ready: %s\n", mainLib)
	return installed, nil
}

// yzmaBinary returns the yzma CLI path, building the version edgebot depends
// on into a temporary GOBIN when the CLI is not already installed. The
// returned cleanup removes that temporary install.
func yzmaBinary(ctx context.Context, out io.Writer) (path string, cleanup func(), err error) {
	noop := func() {}
	if bin, err := exec.LookPath("yzma"); err == nil {
		return bin, noop, nil
	}

	version := yzmaDependencyVersion()
	if version == "" {
		return "", nil, fmt.Errorf("yzma CLI not found in PATH and its version is not recorded in the edgebot build: install it with `go install %s@latest`", yzmaModule)
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		return "", nil, fmt.Errorf("yzma CLI not found in PATH and the Go toolchain needed to install %s@%s is unavailable: %w", yzmaModule, version, err)
	}

	tmp, err := os.MkdirTemp("", "edgebot-yzma-*")
	if err != nil {
		return "", nil, fmt.Errorf("create yzma install dir: %w", err)
	}
	cleanup = func() { _ = os.RemoveAll(tmp) }

	fmt.Fprintf(out, "yzma CLI not found, installing %s %s ...\n", yzmaModule, version)
	cmd := execCommandContext(ctx, goBin, "install", yzmaModule+"@"+version)
	cmd.Env = append(os.Environ(), "GOBIN="+tmp)
	cmd.Stdout = out
	cmd.Stderr = out
	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("install yzma %s: %w", version, err)
	}

	bin := filepath.Join(tmp, "yzma")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if _, err := os.Stat(bin); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("yzma binary not found after install: %w", err)
	}
	return bin, cleanup, nil
}

// yzmaDependencyVersion reports the yzma version recorded in the edgebot build
// info, so the CLI always matches the binding that will load the libraries.
func yzmaDependencyVersion() string {
	info, ok := debuginfo.ReadBuildInfo()
	if !ok {
		return ""
	}
	for _, dep := range info.Deps {
		if dep.Path == yzmaModule {
			return dep.Version
		}
	}
	return ""
}

// runtimeStatus is one row of `edgebot runtime show`.
type runtimeStatus struct {
	Name      string
	Installed bool
	Version   string
	Backend   string
	Path      string
	Size      int64
}

// runRuntimeShow prints the state of the native runtimes. Each runtime uses
// the same directory the providers load from, so the report never claims a
// library is installed where runtime loading would not find it.
func runRuntimeShow(out io.Writer, dir string) error {
	llamacppDir, err := resolveRuntimeDir(runtimeLlamacpp, dir)
	if err != nil {
		return fmt.Errorf("resolve llama.cpp lib dir: %w", err)
	}
	litertlmDir, err := resolveRuntimeDir(runtimeLitertLM, dir)
	if err != nil {
		return fmt.Errorf("resolve LiteRT-LM lib dir: %w", err)
	}
	states := []runtimeStatus{
		llamacppRuntimeStatus(llamacppDir),
		litertlmRuntimeStatus(litertlmDir),
	}

	w := tabwriter.NewWriter(out, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "RUNTIME\tSTATUS\tVERSION\tBACKEND\tSIZE\tLIBRARY")
	fmt.Fprintln(w, "-------\t------\t-------\t-------\t----\t-------")
	var missing []string
	for _, st := range states {
		status := "missing"
		version, backend, size := "-", "-", "-"
		if st.Installed {
			status = "installed"
			version = firstNonEmpty(st.Version, "-")
			backend = firstNonEmpty(st.Backend, "-")
			size = config.FormatFileSize(st.Size)
		} else {
			missing = append(missing, st.Name)
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", st.Name, status, version, backend, size, st.Path)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("failed to write runtime table: %w", err)
	}

	for _, name := range missing {
		fmt.Fprintf(out, "\nInstall %s with: edgebot runtime install %s\n", name, name)
	}
	return nil
}

func llamacppRuntimeStatus(dir string) runtimeStatus {
	st := runtimeStatus{Name: runtimeLlamacpp}
	if !st.stat(filepath.Join(dir, llamacppLibraryName(runtime.GOOS))) {
		return st
	}
	if record, err := readYzmaInstallRecord(dir); err == nil {
		st.Version = record.Tag
		st.Backend = record.Processor
	}
	return st
}

func litertlmRuntimeStatus(dir string) runtimeStatus {
	st := runtimeStatus{Name: runtimeLitertLM}
	platform, _ := libfetch.Platform()
	if !st.stat(filepath.Join(dir, litertlmUnifiedLibName(platform))) {
		return st
	}
	if record, err := readLitertlmInstallRecord(dir); err == nil {
		st.Version = record.Version
		st.Backend = record.Backend
	}
	return st
}

// stat records path and, when a non-empty library exists there, marks the
// runtime installed and reports its size.
func (st *runtimeStatus) stat(path string) bool {
	st.Path = path
	info, err := os.Stat(path)
	if err != nil || info.Size() == 0 {
		return false
	}
	st.Installed = true
	st.Size = info.Size()
	return true
}

// resolveRuntimeDir returns the library directory for a runtime: the explicit
// override first, then the provider-specific env var (LITERTLM_LIB, honoured
// by the litertlm loader), then the shared ~/.edgebot/lib.
func resolveRuntimeDir(name, override string) (string, error) {
	if override != "" {
		return override, nil
	}
	if name == runtimeLitertLM {
		if dir := os.Getenv("LITERTLM_LIB"); dir != "" {
			return dir, nil
		}
	}
	return config.LibDir()
}

// llamacppLibraryName is the shared library the yzma loader opens on goos.
func llamacppLibraryName(goos string) string {
	switch goos {
	case "windows":
		return "llama.dll"
	case "darwin":
		return "libllama.dylib"
	default:
		return "libllama.so"
	}
}

// yzmaRecord is the subset of yzma-install.json that `runtime show` needs.
type yzmaRecord struct {
	Tag       string `json:"tag"`
	Processor string `json:"processor"`
}

func readYzmaInstallRecord(dir string) (*yzmaRecord, error) {
	body, err := os.ReadFile(filepath.Join(dir, yzmaInstallRecord))
	if err != nil {
		return nil, err
	}
	var record yzmaRecord
	if err := json.Unmarshal(body, &record); err != nil {
		return nil, fmt.Errorf("decode yzma install record: %w", err)
	}
	return &record, nil
}

// litertlmInstallRecord remembers what `edgebot runtime install litertlm`
// staged, since libfetch itself does not leave a version record.
type litertlmInstallRecord struct {
	Version   string    `json:"version"`
	Platform  string    `json:"platform"`
	Backend   string    `json:"backend"`
	Installed time.Time `json:"installed"`
}

func writeLitertlmInstallRecord(dir string, record litertlmInstallRecord) error {
	body, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return fmt.Errorf("encode LiteRT-LM install record: %w", err)
	}
	body = append(body, '\n')
	if err := atomicfile.Write(filepath.Join(dir, litertlmRecordName), body, 0o644); err != nil {
		return fmt.Errorf("write LiteRT-LM install record: %w", err)
	}
	return nil
}

func readLitertlmInstallRecord(dir string) (*litertlmInstallRecord, error) {
	body, err := os.ReadFile(filepath.Join(dir, litertlmRecordName))
	if err != nil {
		return nil, err
	}
	var record litertlmInstallRecord
	if err := json.Unmarshal(body, &record); err != nil {
		return nil, fmt.Errorf("decode LiteRT-LM install record: %w", err)
	}
	return &record, nil
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
