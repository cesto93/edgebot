package cmd

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"edgebot/config"
)

func TestSaveDownloadedMMProjLink(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("create models dir: %v", err)
	}
	for _, name := range []string{"vision-Q4.gguf", "mmproj-vision-f16.gguf"} {
		if err := os.WriteFile(filepath.Join(modelsDir, name), []byte("x"), 0644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	if err := saveDownloadedMMProjLink(
		[]string{"vision-Q4.gguf", "mmproj-vision-f16.gguf"},
		[]bool{true, true},
	); err != nil {
		t.Fatalf("saveDownloadedMMProjLink() = %v", err)
	}
	got, err := config.FindLlamacppMMProjForModel("vision-Q4")
	if err != nil {
		t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
	}
	want := filepath.Join(modelsDir, "mmproj-vision-f16.gguf")
	if got != want {
		t.Errorf("projector = %q, want %q", got, want)
	}

	t.Run("projector download failed", func(t *testing.T) {
		if err := saveDownloadedMMProjLink(
			[]string{"lonely.gguf", "mmproj-vision-f16.gguf"},
			[]bool{true, false},
		); err != nil {
			t.Fatalf("saveDownloadedMMProjLink() = %v", err)
		}
		got, err := config.FindLlamacppMMProjForModel("lonely")
		if err != nil {
			t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
		}
		if got != "" {
			t.Errorf("projector = %q, want none (download failed)", got)
		}
	})
}

func TestSplitRepoFiles(t *testing.T) {
	files := []repoFile{
		{Type: "file", Path: "README.md"},
		{Type: "file", Path: "model-Q4_K_M.gguf", Size: 1024},
		{Type: "file", Path: "mmproj-model-f16.gguf", Size: 512},
		{Type: "file", Path: "MMProj-F32.gguf", Size: 1024},
		{Type: "file", Path: "model.litertlm", Size: 2048},
		{Type: "file", Path: "nested/model.gguf"},
		{Type: "directory", Path: "subdir"},
	}

	models, projectors := splitRepoFiles(files)

	var modelNames, projectorNames []string
	for _, f := range models {
		modelNames = append(modelNames, f.Path)
	}
	for _, f := range projectors {
		projectorNames = append(projectorNames, f.Path)
	}

	wantModels := []string{"model-Q4_K_M.gguf", "model.litertlm"}
	if strings.Join(modelNames, ",") != strings.Join(wantModels, ",") {
		t.Errorf("models = %v, want %v", modelNames, wantModels)
	}
	wantProjectors := []string{"MMProj-F32.gguf", "mmproj-model-f16.gguf"}
	if strings.Join(projectorNames, ",") != strings.Join(wantProjectors, ",") {
		t.Errorf("projectors = %v, want %v", projectorNames, wantProjectors)
	}
}

func TestPromptRepoFileSingleOption(t *testing.T) {
	var out bytes.Buffer
	got, err := promptRepoFile(bufio.NewReader(strings.NewReader("")), &out, "owner/repo", "model file",
		[]repoFile{{Path: "only.gguf", Size: 2048}})
	if err != nil {
		t.Fatalf("promptRepoFile: %v", err)
	}
	if got.Path != "only.gguf" {
		t.Errorf("selected %q, want only.gguf", got.Path)
	}
	if !strings.Contains(out.String(), "Using model file: only.gguf") {
		t.Errorf("output missing selection notice: %q", out.String())
	}
}

func TestPromptRepoFile(t *testing.T) {
	options := []repoFile{
		{Path: "a.gguf", Size: 1024},
		{Path: "b.gguf", Size: 2048},
	}

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "by number", input: "2\n", want: "b.gguf"},
		{name: "by filename", input: "a.gguf\n", want: "a.gguf"},
		{name: "invalid then valid", input: "9\nnope\n1\n", want: "a.gguf"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			got, err := promptRepoFile(bufio.NewReader(strings.NewReader(tt.input)), &out, "owner/repo", "model file", options)
			if err != nil {
				t.Fatalf("promptRepoFile: %v", err)
			}
			if got.Path != tt.want {
				t.Errorf("selected %q, want %q", got.Path, tt.want)
			}
		})
	}

	t.Run("shows sizes", func(t *testing.T) {
		var out bytes.Buffer
		_, err := promptRepoFile(bufio.NewReader(strings.NewReader("1\n")), &out, "owner/repo", "model file", options)
		if err != nil {
			t.Fatalf("promptRepoFile: %v", err)
		}
		if !strings.Contains(out.String(), "1.0 KB") || !strings.Contains(out.String(), "2.0 KB") {
			t.Errorf("output missing sizes: %q", out.String())
		}
	})

	t.Run("eof", func(t *testing.T) {
		var out bytes.Buffer
		_, err := promptRepoFile(bufio.NewReader(strings.NewReader("")), &out, "owner/repo", "model file", options)
		if err == nil {
			t.Fatal("expected an error on EOF")
		}
	})
}

func TestSelectRepoFiles(t *testing.T) {
	origFetch := fetchRepoFiles
	defer func() { fetchRepoFiles = origFetch }()

	tests := []struct {
		name      string
		files     []repoFile
		input     string
		want      []string
		wantInOut string
	}{
		{
			name: "model and single projector",
			files: []repoFile{
				{Path: "model-Q4.gguf"},
				{Path: "mmproj-f16.gguf"},
			},
			want:      []string{"model-Q4.gguf", "mmproj-f16.gguf"},
			wantInOut: "vision projector",
		},
		{
			name: "multiple models and projectors",
			files: []repoFile{
				{Path: "model-B.gguf"},
				{Path: "model-A.gguf"},
				{Path: "mmproj-b.gguf"},
				{Path: "mmproj-a.gguf"},
			},
			input: "2\n1\n",
			want:  []string{"model-B.gguf", "mmproj-a.gguf"},
		},
		{
			name: "litertlm skips projector",
			files: []repoFile{
				{Path: "model.litertlm"},
				{Path: "mmproj-f16.gguf"},
			},
			want: []string{"model.litertlm"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fetchRepoFiles = func(string) ([]repoFile, error) { return tt.files, nil }
			var out bytes.Buffer
			got, err := selectRepoFiles("owner/repo", bufio.NewReader(strings.NewReader(tt.input)), &out)
			if err != nil {
				t.Fatalf("selectRepoFiles: %v", err)
			}
			if strings.Join(got, ",") != strings.Join(tt.want, ",") {
				t.Errorf("selected %v, want %v", got, tt.want)
			}
			if tt.wantInOut != "" && !strings.Contains(out.String(), tt.wantInOut) {
				t.Errorf("output missing %q: %q", tt.wantInOut, out.String())
			}
			if tt.name == "litertlm skips projector" && strings.Contains(out.String(), "vision projector") {
				t.Errorf("unexpected projector prompt: %q", out.String())
			}
		})
	}

	t.Run("no models", func(t *testing.T) {
		fetchRepoFiles = func(string) ([]repoFile, error) {
			return []repoFile{{Path: "README.md"}}, nil
		}
		_, err := selectRepoFiles("owner/repo", bufio.NewReader(strings.NewReader("")), &bytes.Buffer{})
		if err == nil {
			t.Fatal("expected an error when no model files exist")
		}
	})

	t.Run("fetch error", func(t *testing.T) {
		wantErr := errors.New("boom")
		fetchRepoFiles = func(string) ([]repoFile, error) { return nil, wantErr }
		_, err := selectRepoFiles("owner/repo", bufio.NewReader(strings.NewReader("")), &bytes.Buffer{})
		if !errors.Is(err, wantErr) {
			t.Fatalf("err = %v, want %v", err, wantErr)
		}
	})
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestDownloadFileRejectsTruncatedBody(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))

	origBase, origClient := huggingFaceBaseURL, pullHTTPClient
	t.Cleanup(func() {
		huggingFaceBaseURL = origBase
		pullHTTPClient = origClient
	})
	huggingFaceBaseURL = "https://example.invalid"
	pullHTTPClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			Status:        "200 OK",
			ContentLength: 100,
			Header:        make(http.Header),
			Body:          io.NopCloser(strings.NewReader("short")),
		}, nil
	})}

	err := downloadFile("owner/repo", "model.gguf")
	if err == nil || !strings.Contains(err.Error(), "incomplete download") {
		t.Fatalf("downloadFile() error = %v, want an incomplete-download error", err)
	}

	dest := filepath.Join(home, ".edgebot", "models", "llamacpp", "model.gguf")
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Fatalf("truncated file was not removed: stat err = %v", statErr)
	}
}

func TestDownloadFileCompletesMatchingBody(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))

	origBase, origClient := huggingFaceBaseURL, pullHTTPClient
	t.Cleanup(func() {
		huggingFaceBaseURL = origBase
		pullHTTPClient = origClient
	})
	huggingFaceBaseURL = "https://example.invalid"
	pullHTTPClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			Status:        "200 OK",
			ContentLength: 5,
			Header:        make(http.Header),
			Body:          io.NopCloser(strings.NewReader("hello")),
		}, nil
	})}

	if err := downloadFile("owner/repo", "model.gguf"); err != nil {
		t.Fatalf("downloadFile() = %v", err)
	}
	dest := filepath.Join(home, ".edgebot", "models", "llamacpp", "model.gguf")
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read downloaded file: %v", err)
	}
	if string(got) != "hello" {
		t.Errorf("file = %q, want %q", got, "hello")
	}
}
