package config

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ggufKV is one metadata entry of a synthetic test GGUF file.
type ggufKV struct {
	key   string
	typ   uint32
	value []byte
}

func ggufStr(s string) []byte {
	buf := make([]byte, 8+len(s))
	binary.LittleEndian.PutUint64(buf, uint64(len(s)))
	copy(buf[8:], s)
	return buf
}

func ggufU32(v uint32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, v)
	return buf
}

func ggufStrArray(elems ...string) []byte {
	out := make([]byte, 0)
	typ := make([]byte, 4)
	binary.LittleEndian.PutUint32(typ, ggufTypeString)
	out = append(out, typ...)
	count := make([]byte, 8)
	binary.LittleEndian.PutUint64(count, uint64(len(elems)))
	out = append(out, count...)
	for _, e := range elems {
		out = append(out, ggufStr(e)...)
	}
	return out
}

// writeTestGGUF writes a minimal GGUF file (header + metadata, no tensors).
func writeTestGGUF(t *testing.T, path string, kvs []ggufKV) {
	t.Helper()
	var out []byte
	out = append(out, 'G', 'G', 'U', 'F')
	ver := make([]byte, 4)
	binary.LittleEndian.PutUint32(ver, 3)
	out = append(out, ver...)
	out = append(out, make([]byte, 8)...) // tensor count
	count := make([]byte, 8)
	binary.LittleEndian.PutUint64(count, uint64(len(kvs)))
	out = append(out, count...)
	for _, kv := range kvs {
		out = append(out, ggufStr(kv.key)...)
		typ := make([]byte, 4)
		binary.LittleEndian.PutUint32(typ, kv.typ)
		out = append(out, typ...)
		out = append(out, kv.value...)
	}
	if err := os.WriteFile(path, out, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestIsDecisionModelFile(t *testing.T) {
	dir := t.TempDir()

	plain := filepath.Join(dir, "plain.gguf")
	writeTestGGUF(t, plain, []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("llama")},
		{"general.name", ggufTypeString, ggufStr("Plain 3B")},
		{"llama.context_length", 4, ggufU32(4096)},
	})

	decide := filepath.Join(dir, "decide.gguf")
	writeTestGGUF(t, decide, []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("lev")},
		{"tokenizer.ggml.tokens", ggufTypeArray, ggufStrArray("A", "B", "hello")},
		{"lev.decision.type", ggufTypeString, ggufStr("lev")},
	})

	truncated := filepath.Join(dir, "truncated.gguf")
	writeTestGGUF(t, truncated, []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("llama")},
	})
	data, err := os.ReadFile(truncated)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(truncated, data[:len(data)/2], 0644); err != nil {
		t.Fatal(err)
	}

	junk := filepath.Join(dir, "junk.gguf")
	if err := os.WriteFile(junk, []byte("definitely not a gguf"), 0644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		path string
		want bool
	}{
		{"decision metadata after array", decide, true},
		{"plain llm", plain, false},
		{"truncated file", truncated, false},
		{"non-gguf content", junk, false},
		{"missing file", filepath.Join(dir, "absent.gguf"), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isDecisionModelFile(tc.path); got != tc.want {
				t.Errorf("isDecisionModelFile() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestModelType(t *testing.T) {
	if got := (ModelInfo{}).ModelType(); got != ModelTypeLLM {
		t.Errorf("zero ModelInfo ModelType() = %q, want %q", got, ModelTypeLLM)
	}
	if got := (ModelInfo{Type: ModelTypeDecision}).ModelType(); got != ModelTypeDecision {
		t.Errorf("decision ModelInfo ModelType() = %q, want %q", got, ModelTypeDecision)
	}
	if got := (ModelInfo{Type: "other"}).ModelType(); got != ModelTypeLLM {
		t.Errorf("unknown type ModelType() = %q, want %q", got, ModelTypeLLM)
	}
}

func TestGetLlamacppModelsDecisionType(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	writeTestGGUF(t, filepath.Join(modelsDir, "lev-1.gguf"), []ggufKV{
		{"lev.decision.type", ggufTypeString, ggufStr("lev")},
	})
	writeTestGGUF(t, filepath.Join(modelsDir, "plain-3b.gguf"), []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("llama")},
	})

	models := GetLlamacppModels()
	if len(models) != 2 {
		t.Fatalf("GetLlamacppModels() returned %d models, want 2: %+v", len(models), models)
	}
	byName := map[string]ModelInfo{}
	for _, m := range models {
		byName[m.Name] = m
	}
	if got := byName["lev-1"].ModelType(); got != ModelTypeDecision {
		t.Errorf("lev-1 ModelType() = %q, want %q", got, ModelTypeDecision)
	}
	if got := byName["plain-3b"].ModelType(); got != ModelTypeLLM {
		t.Errorf("plain-3b ModelType() = %q, want %q", got, ModelTypeLLM)
	}
}

func TestGetLlamacppModelsExcludesMMProjTypes(t *testing.T) {
	// Decision detection must not promote mmproj files into the list: they
	// are still filtered out by name.
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestGGUF(t, filepath.Join(modelsDir, "mmproj-lev.gguf"), []ggufKV{
		{"lev.decision.type", ggufTypeString, ggufStr("lev")},
	})

	for _, m := range GetLlamacppModels() {
		if strings.Contains(strings.ToLower(m.Name), "mmproj") {
			t.Errorf("GetLlamacppModels() listed mmproj model %q", m.Name)
		}
	}
	if got := len(GetLlamacppModels()); got != 0 {
		t.Errorf("GetLlamacppModels() = %d models, want 0", got)
	}
}
