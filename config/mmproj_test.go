package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveMMProjLinkPersists(t *testing.T) {
	dir := mockOCRDirs(t)

	// The ".gguf" suffix is optional and trimmed in the stored link.
	if err := SaveMMProjLink(
		"MiniCPM-OCR-1_0-Q4_K_M.gguf",
		"mmproj-model-f16-4x.gguf",
	); err != nil {
		t.Fatalf("SaveMMProjLink() = %v", err)
	}

	links, err := loadMMProjLinks(dir)
	if err != nil {
		t.Fatalf("loadMMProjLinks() = %v", err)
	}
	if got := links["MiniCPM-OCR-1_0-Q4_K_M"]; got != "mmproj-model-f16-4x" {
		t.Errorf("stored link = %q, want mmproj-model-f16-4x", got)
	}

	got, err := FindLlamacppMMProjForModel("MiniCPM-OCR-1_0-Q4_K_M")
	if err != nil {
		t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
	}
	want := filepath.Join(dir, "mmproj-model-f16-4x.gguf")
	if got != want {
		t.Errorf("projector = %q, want %q", got, want)
	}
}

func TestFindLlamacppMMProjForModelUsesOnlyLinks(t *testing.T) {
	dir := mockOCRDirs(t)

	// A projector named after the model is still not coupled until linked.
	writeTestGGUF(t, filepath.Join(dir, "mmproj-plain-vl.gguf"), []ggufKV{
		{"general.name", ggufTypeString, ggufStr("plain vl projector")},
	})
	got, err := FindLlamacppMMProjForModel("plain-vl")
	if err != nil {
		t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
	}
	if got != "" {
		t.Errorf("projector = %q, want none (not linked)", got)
	}

	if err := SaveMMProjLink("plain-vl", "mmproj-plain-vl"); err != nil {
		t.Fatalf("SaveMMProjLink() = %v", err)
	}
	got, err = FindLlamacppMMProjForModel("plain-vl")
	if err != nil {
		t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
	}
	want := filepath.Join(dir, "mmproj-plain-vl.gguf")
	if got != want {
		t.Errorf("projector = %q, want %q", got, want)
	}

	// An unrelated model never inherits the link.
	if got, err := FindLlamacppMMProjForModel("MiniCPM-OCR-1_0-Q4_K_M"); err != nil {
		t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
	} else if got != "" {
		t.Errorf("unlinked model projector = %q, want none", got)
	}
}

func TestFindLlamacppMMProjForModelMissingFile(t *testing.T) {
	dir := mockOCRDirs(t)
	if err := SaveMMProjLink("MiniCPM-OCR-1_0-Q4_K_M", "mmproj-model-f16-4x"); err != nil {
		t.Fatalf("SaveMMProjLink() = %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "mmproj-model-f16-4x.gguf")); err != nil {
		t.Fatalf("remove projector: %v", err)
	}

	got, err := FindLlamacppMMProjForModel("MiniCPM-OCR-1_0-Q4_K_M")
	if err != nil {
		t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
	}
	if got != "" {
		t.Errorf("projector = %q, want none (linked file is missing)", got)
	}
}

func TestSaveMMProjLinkRejectsInvalidNames(t *testing.T) {
	mockOCRDirs(t)

	tests := []struct {
		name      string
		model     string
		projector string
	}{
		{"path traversal model", "../evil", "mmproj-model-f16-4x"},
		{"path traversal projector", "plain-vl", "../../evil"},
		{"empty projector", "plain-vl", "  "},
		{"missing model file", "absent-model", "mmproj-model-f16-4x"},
		{"missing projector file", "plain-vl", "absent-mmproj"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := SaveMMProjLink(tc.model, tc.projector)
			if err == nil {
				t.Fatalf("SaveMMProjLink(%q, %q) = nil, want error", tc.model, tc.projector)
			}
		})
	}
}

func TestDeleteLocalModelRemovesLink(t *testing.T) {
	dir := mockOCRDirs(t)
	if err := SaveMMProjLink("MiniCPM-OCR-1_0-Q4_K_M", "mmproj-model-f16-4x"); err != nil {
		t.Fatalf("SaveMMProjLink() = %v", err)
	}

	removed, err := DeleteLocalModel("MiniCPM-OCR-1_0-Q4_K_M")
	if err != nil {
		t.Fatalf("DeleteLocalModel() = %v", err)
	}
	if len(removed) != 2 {
		t.Fatalf("DeleteLocalModel() removed %d files, want model + projector: %v", len(removed), removed)
	}
	for _, path := range removed {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("%s still exists after delete (err=%v)", path, err)
		}
	}

	links, err := loadMMProjLinks(dir)
	if err != nil {
		t.Fatalf("loadMMProjLinks() = %v", err)
	}
	if _, ok := links["MiniCPM-OCR-1_0-Q4_K_M"]; ok {
		t.Errorf("registry still links the deleted model: %v", links)
	}
}
