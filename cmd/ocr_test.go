package cmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"edgebot/llm"
)

func TestOCRTaskPrompt(t *testing.T) {
	origMode, origPrompt := ocrMode, ocrPrompt
	defer func() { ocrMode, ocrPrompt = origMode, origPrompt }()

	tests := []struct {
		name    string
		mode    string
		prompt  string
		want    string
		wantErr bool
	}{
		{"default text", "", "", ocrTextPrompt, false},
		{"text", "text", "", ocrTextPrompt, false},
		{"table", "table", "", ocrTablePrompt, false},
		{"formula case insensitive", " Formula ", "", ocrFormulaPrompt, false},
		{"custom overrides mode", "table", "Read all text:", "Read all text:", false},
		{"invalid mode", "handwriting", "", "", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ocrMode, ocrPrompt = tc.mode, tc.prompt
			got, err := ocrTaskPrompt()
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ocrTaskPrompt() = %q, want error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ocrTaskPrompt() = %v", err)
			}
			if got != tc.want {
				t.Errorf("ocrTaskPrompt() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParsePageRange(t *testing.T) {
	tests := []struct {
		in        string
		first     int
		last      int
		wantError bool
	}{
		{"3", 3, 3, false},
		{" 2 - 7 ", 2, 7, false},
		{"0", 0, 0, true},
		{"5-2", 0, 0, true},
		{"a-b", 0, 0, true},
		{"", 0, 0, true},
	}
	for _, tc := range tests {
		first, last, err := parsePageRange(tc.in)
		if tc.wantError {
			if err == nil {
				t.Errorf("parsePageRange(%q) = %d,%d, want error", tc.in, first, last)
			}
			continue
		}
		if err != nil {
			t.Errorf("parsePageRange(%q) = %v", tc.in, err)
			continue
		}
		if first != tc.first || last != tc.last {
			t.Errorf("parsePageRange(%q) = %d,%d, want %d,%d", tc.in, first, last, tc.first, tc.last)
		}
	}
}

func TestPDFPageNumber(t *testing.T) {
	tests := map[string]int{
		"/tmp/page-1.png":  1,
		"/tmp/page-10.png": 10,
		"/tmp/page.png":    0,
	}
	for path, want := range tests {
		if got := pdfPageNumber(path); got != want {
			t.Errorf("pdfPageNumber(%q) = %d, want %d", path, got, want)
		}
	}
}

func TestRasterizePDF(t *testing.T) {
	origExec := execCommand
	defer func() { execCommand = origExec }()

	var gotArgs []string
	execCommand = func(command string, args ...string) *exec.Cmd {
		if command != "pdftoppm" {
			return exec.Command(command, args...)
		}
		gotArgs = append([]string{}, args...)
		prefix := args[len(args)-1]
		// Deliberately create page 10 before 2 so the sort is exercised.
		script := fmt.Sprintf("touch %q %q %q",
			prefix+"-1.png", prefix+"-10.png", prefix+"-2.png")
		return exec.Command("sh", "-c", script)
	}

	dir := t.TempDir()
	files, err := rasterizePDF("doc.pdf", "200", "1-3", dir)
	if err != nil {
		t.Fatalf("rasterizePDF() = %v", err)
	}
	want := []string{"page-1.png", "page-2.png", "page-10.png"}
	if len(files) != len(want) {
		t.Fatalf("rasterizePDF() = %v, want %v", files, want)
	}
	for i := range want {
		if filepath.Base(files[i]) != want[i] {
			t.Errorf("page[%d] = %q, want %q", i, filepath.Base(files[i]), want[i])
		}
	}
	joined := strings.Join(gotArgs, " ")
	for _, wantArg := range []string{"-png", "-r 200", "-f 1", "-l 3"} {
		if !strings.Contains(joined, wantArg) {
			t.Errorf("pdftoppm args = %v, missing %q", gotArgs, wantArg)
		}
	}
}

func TestRasterizePDFErrors(t *testing.T) {
	origExec := execCommand
	defer func() { execCommand = origExec }()

	if _, err := rasterizePDF("doc.pdf", "abc", "", t.TempDir()); err == nil {
		t.Error("rasterizePDF() with invalid DPI = nil error")
	}
	if _, err := rasterizePDF("doc.pdf", "200", "bad", t.TempDir()); err == nil {
		t.Error("rasterizePDF() with invalid pages = nil error")
	}

	execCommand = func(command string, args ...string) *exec.Cmd {
		return exec.Command("sh", "-c", "echo boom >&2; exit 1")
	}
	if _, err := rasterizePDF("doc.pdf", "200", "", t.TempDir()); err == nil ||
		!strings.Contains(err.Error(), "pdftoppm failed") {
		t.Errorf("rasterizePDF() error = %v, want pdftoppm failure", err)
	}
}

func TestOCRResponseText(t *testing.T) {
	tests := []struct {
		name     string
		messages []llm.Message
		want     string
		wantErr  bool
	}{
		{
			name:     "string with think",
			messages: []llm.Message{{Role: "assistant", Content: "  hello  </think>world "}},
			want:     "hello  world",
		},
		{
			name: "content parts",
			messages: []llm.Message{{Role: "assistant", Content: []llm.ContentPart{
				{Type: "text", Text: "line one"},
				{Type: "text", Text: "line two"},
			}}},
			want: "line one\nline two",
		},
		{
			name:     "no assistant",
			messages: []llm.Message{{Role: "user", Content: "hi"}},
			wantErr:  true,
		},
		{
			name:     "empty",
			messages: nil,
			wantErr:  true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ocrResponseText(tc.messages)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ocrResponseText() = %q, want error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ocrResponseText() = %v", err)
			}
			if got != tc.want {
				t.Errorf("ocrResponseText() = %q, want %q", got, tc.want)
			}
		})
	}
}

type fakeCaller struct {
	messages []llm.Message
}

func (f *fakeCaller) Call(_ context.Context, _ string, messages []llm.Message, _ []any) ([]llm.Message, error) {
	f.messages = messages
	return []llm.Message{{Role: "assistant", Content: "recognized"}}, nil
}

func TestOCRImage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "page.png")
	if err := os.WriteFile(path, []byte("not-a-real-png"), 0o644); err != nil {
		t.Fatal(err)
	}

	caller := &fakeCaller{}
	got, err := ocrImage(context.Background(), caller, ocrTablePrompt, path)
	if err != nil {
		t.Fatalf("ocrImage() = %v", err)
	}
	if got != "recognized" {
		t.Errorf("ocrImage() = %q, want recognized", got)
	}
	if len(caller.messages) != 1 || caller.messages[0].Role != "user" {
		t.Fatalf("messages = %+v, want one user message", caller.messages)
	}
	parts, ok := caller.messages[0].Content.([]llm.ContentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("content = %#v, want text+image parts", caller.messages[0].Content)
	}
	if parts[0].Text != ocrTablePrompt {
		t.Errorf("prompt = %q, want %q", parts[0].Text, ocrTablePrompt)
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil ||
		!strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("image part = %+v, want base64 png data URL", parts[1])
	}
}
