//go:build integration

package cmd

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"edgebot/config"
)

// The three llamacpp models installed locally (see 'edgebot models'):
// one llm, one OCR and one decision model.
const (
	integrationLLMModel      = "LFM3-1.7B-QAD-Q4_0"
	integrationOCRModel      = "MiniCPM-OCR-1_0-Q4_K_M"
	integrationDecisionModel = "d1-2B-Q4_K_M"
)

// requireLlamacppModel asserts the model GGUF is installed locally, failing
// with an actionable message otherwise (models live in
// ~/.edgebot/models/llamacpp/<name>.gguf).
func requireLlamacppModel(t *testing.T, model string) {
	t.Helper()
	dir, err := config.ModelsDir("llamacpp")
	if err != nil {
		t.Fatalf("cannot resolve llamacpp models dir: %v", err)
	}
	// Mirror ResolveDecisionModelPath: name as-is, then with a .gguf suffix.
	for _, path := range []string{
		filepath.Join(dir, model+".gguf"),
		filepath.Join(dir, model),
	} {
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return
		}
	}
	t.Fatalf("llamacpp model %q not found: expected %s\nInstall it with 'edgebot models pull <huggingface-repo> %s.gguf' (see 'edgebot models')", model, filepath.Join(dir, model+".gguf"), model)
}

// requireOCRMMProj resolves the MiniCPM-OCR vision projector: the projector
// recorded in llm.ocr_mmproj wins, then MiniCPM-OCR's mmproj-model-f16-4x.gguf
// next to the model GGUF. The d1 decision projector is skipped (it belongs to
// the other tested model, so a unique-projector fallback would be wrong). The
// config value resolution mirrors llm.LlamacppCaller.resolveMMProj.
func requireOCRMMProj(t *testing.T) string {
	t.Helper()
	dir, err := config.ModelsDir("llamacpp")
	if err != nil {
		t.Fatalf("cannot resolve llamacpp models dir: %v", err)
	}
	if cfg, err := config.LoadConfig(); err == nil {
		if p := strings.TrimSpace(cfg.LLM.OCRMMProj); p != "" {
			if !filepath.IsAbs(p) {
				p = filepath.Join(dir, p)
			}
			if _, err := os.Stat(p); err != nil && filepath.Ext(p) == "" {
				if _, err := os.Stat(p + ".gguf"); err == nil {
					p += ".gguf"
				}
			}
			if _, err := os.Stat(p); err == nil {
				return p
			}
			t.Fatalf("llm.ocr_mmproj %q not found", cfg.LLM.OCRMMProj)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot list llamacpp models dir: %v", err)
	}
	for _, entry := range entries {
		name := strings.ToLower(entry.Name())
		if entry.IsDir() || !strings.HasSuffix(name, ".gguf") || !strings.Contains(name, "mmproj") {
			continue
		}
		if strings.Contains(name, "d1") {
			continue // projector of the decision model, not the OCR model
		}
		return filepath.Join(dir, entry.Name())
	}
	t.Fatalf("vision projector for %q not found in %s: expected mmproj-model-f16-4x.gguf or 'edgebot config --ocr-mmproj <path>'", integrationOCRModel, dir)
	return ""
}

// captureStdout runs fn capturing everything it prints to stdout.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	os.Stdout = old
	_ = w.Close()
	out := <-done
	_ = r.Close()
	return out
}

// writeOCRTestImage renders simple block digits ("1 2 3 4") into a large white
// PNG, so the OCR model has actual glyphs to recognize.
func writeOCRTestImage(t *testing.T) string {
	t.Helper()
	const digitWidth, digitHeight, scale = 3, 5, 48 // 3x5 bitmap upscaled
	digits := map[rune][digitHeight][digitWidth]byte{
		'1': {{0, 1, 0}, {1, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 1, 1}},
		'2': {{1, 1, 1}, {0, 0, 1}, {1, 1, 1}, {1, 0, 0}, {1, 1, 1}},
		'3': {{1, 1, 1}, {0, 0, 1}, {1, 1, 1}, {0, 0, 1}, {1, 1, 1}},
		'4': {{1, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 0, 1}},
	}
	text := "1234"
	margin, gap := scale, -scale/2 // digits touch slightly like seven-segment text
	w := margin*2 + len(text)*digitWidth*scale + (len(text)-1)*gap
	h := margin*2 + digitHeight*scale
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetGray(x, y, color.Gray{Y: 255})
		}
	}
	black := color.Gray{Y: 0}
	x0 := margin
	for _, ch := range text {
		g, ok := digits[ch]
		if !ok {
			continue
		}
		for row := 0; row < digitHeight; row++ {
			for col := 0; col < digitWidth; col++ {
				if g[row][col] == 1 {
					for dy := 0; dy < scale; dy++ {
						for dx := 0; dx < scale; dx++ {
							img.SetGray(x0+col*scale+dx, margin+row*scale+dy, black)
						}
					}
				}
			}
		}
		x0 += digitWidth*scale + gap
	}

	path := filepath.Join(t.TempDir(), "ocr-input.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create test image: %v", err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("failed to encode test image: %v", err)
	}
	return path
}

// TestRunOCRWithRealModel exercises the OCR command end to end with the local
// MiniCPM-OCR GGUF: vision projector + model load, image decode and text
// generation (runOCR errors on an empty result).
func TestRunOCRWithRealModel(t *testing.T) {
	requireLlamacppModel(t, integrationOCRModel)
	mmproj := requireOCRMMProj(t)
	imagePath := writeOCRTestImage(t)

	tmpDir := t.TempDir()
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(origDir)

	origModel, origMMProj, origMode, origOutput := ocrModel, ocrMMProj, ocrMode, ocrOutput
	defer func() { ocrModel, ocrMMProj, ocrMode, ocrOutput = origModel, origMMProj, origMode, origOutput }()
	ocrModel = integrationOCRModel
	ocrMMProj = mmproj
	ocrMode = "text"
	ocrOutput = ""

	var runErr error
	out := captureStdout(t, func() {
		runErr = runOCR([]string{imagePath})
	})
	if runErr != nil {
		t.Fatalf("runOCR() error = %v", runErr)
	}
	if text := strings.TrimSpace(out); text == "" {
		t.Fatalf("runOCR() printed no text")
	} else {
		t.Logf("OCR output: %q", text)
	}
}

// TestRunDecideWithRealModel exercises the decide command end to end with the
// local d1-2B decision GGUF. The model is known to fail on the previous yzma
// version (lfm3-d1 decision type unsupported), so it runs in a subprocess
// (a crash doesn't kill the test binary) and a failure is reported as a
// FAILURE, not a skip: the test stays red until a yzma upgrade fixes it.
func TestRunDecideWithRealModel(t *testing.T) {
	requireLlamacppModel(t, integrationDecisionModel)

	root, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
			break
		}
		root = filepath.Dir(root)
		if root == "/" {
			t.Fatalf("go.mod not found")
		}
	}

	cmd := exec.Command("go", "run", ".", "decide",
		"--model", integrationDecisionModel,
		"--state", "patient reports fever and headache",
		"--noul", "Does the patient have a fever?|no fever|fever",
	)
	cmd.Dir = root
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	if err := cmd.Run(); err != nil {
		// Fail (not skip) on purpose: the lfm3-d1 decision type is unsupported
		// on the previous yzma version (known bug) and this test is the signal
		// to notice when a yzma upgrade fixes it.
		t.Fatalf("decision model %q failed (known bug on the previous yzma version, fix expected on upgrade): %v\n%s", integrationDecisionModel, err, buf.String())
	}
	t.Logf("decide output:\n%s", buf.String())
}
