package llm

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// shimKV is one metadata entry of a synthetic test GGUF file.
type shimKV struct {
	key   string
	typ   uint32
	value []byte
}

const shimTypeString = 8
const shimTypeArray = 9

func shimStr(s string) []byte {
	buf := make([]byte, 8+len(s))
	binary.LittleEndian.PutUint64(buf, uint64(len(s)))
	copy(buf[8:], s)
	return buf
}

func shimStrArray(elems ...string) []byte {
	out := make([]byte, 0, 16+128*len(elems))
	out = binary.LittleEndian.AppendUint32(out, shimTypeString)
	out = binary.LittleEndian.AppendUint64(out, uint64(len(elems)))
	for _, e := range elems {
		out = append(out, shimStr(e)...)
	}
	return out
}

// writeShimTestGGUF writes a minimal GGUF file (header + metadata, no tensors).
func writeShimTestGGUF(t *testing.T, path string, kvs []shimKV) {
	t.Helper()
	var out []byte
	out = append(out, 'G', 'G', 'U', 'F')
	out = binary.LittleEndian.AppendUint32(out, 3) // version
	out = append(out, make([]byte, 8)...)          // tensor count
	out = binary.LittleEndian.AppendUint64(out, uint64(len(kvs)))
	for _, kv := range kvs {
		out = append(out, shimStr(kv.key)...)
		out = binary.LittleEndian.AppendUint32(out, kv.typ)
		out = append(out, kv.value...)
	}
	if err := os.WriteFile(path, out, 0644); err != nil {
		t.Fatal(err)
	}
}

func shimDecisionGGUF(t *testing.T, dir, name, typ string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	writeShimTestGGUF(t, path, []shimKV{
		{"general.architecture", shimTypeString, shimStr("lfm3")},
		{"lfm3.decision.type", shimTypeString, shimStr(typ)},
		{"lfm3.decision.temperature.noul", shimTypeString, shimStr("0.6")},
		{"tokenizer.ggml.tokens", shimTypeArray, shimStrArray("a", "b")},
	})
	return path
}

func TestShimFromError(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		wantFrom  string
		wantTo    string
		wantMatch bool
	}{
		{"nil", nil, "", "", false},
		{"unsupported known type", fmt.Errorf(`decide: decision type "lfm3-d1" is not supported`), "lfm3-d1", "lfm2-d1", true},
		{"unsupported unknown type", fmt.Errorf(`decide: decision type "h2o-lightning" is not supported`), "", "", false},
		{"other error", fmt.Errorf("decide: the model has no systemone template"), "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			from, to, ok := shimFromError(tt.err)
			if ok != tt.wantMatch || from != tt.wantFrom || to != tt.wantTo {
				t.Errorf("shimFromError(%v) = (%q, %q, %v), want (%q, %q, %v)",
					tt.err, from, to, ok, tt.wantFrom, tt.wantTo, tt.wantMatch)
			}
		})
	}
}

func TestFindDecisionTypeValue(t *testing.T) {
	dir := t.TempDir()
	path := shimDecisionGGUF(t, dir, "d1-2B.gguf", "lfm3-d1")

	offset, value, err := findDecisionTypeValue(path)
	if err != nil {
		t.Fatalf("findDecisionTypeValue: %v", err)
	}
	if value != "lfm3-d1" {
		t.Errorf("value = %q, want lfm3-d1", value)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(strings.Index(string(data), "lfm3-d1")); offset != want {
		t.Errorf("offset = %d, want %d", offset, want)
	}
}

func TestFindDecisionTypeValueMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "plain.gguf")
	writeShimTestGGUF(t, path, []shimKV{
		{"general.architecture", shimTypeString, shimStr("llama")},
		{"general.name", shimTypeString, shimStr("Plain 3B")},
	})
	if _, _, err := findDecisionTypeValue(path); err == nil {
		t.Error("findDecisionTypeValue = nil error, want missing decision type")
	}

	bad := filepath.Join(dir, "bad.gguf")
	if err := os.WriteFile(bad, []byte("not a gguf"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := findDecisionTypeValue(bad); err == nil {
		t.Error("findDecisionTypeValue(bad) = nil error, want not a GGUF")
	}
}

func TestShimDecisionModel(t *testing.T) {
	dir := t.TempDir()
	src := shimDecisionGGUF(t, dir, "d1-2B-Q4_K_M.gguf", "lfm3-d1")

	shimPath, err := shimDecisionModel(src, "lfm3-d1", "lfm2-d1")
	if err != nil {
		t.Fatalf("shimDecisionModel: %v", err)
	}
	if !strings.HasSuffix(shimPath, "d1-2B-Q4_K_M.lfm2-d1shim.gguf") {
		t.Errorf("shim path = %q, want the cached copy name", shimPath)
	}
	_, value, err := findDecisionTypeValue(shimPath)
	if err != nil {
		t.Fatalf("findDecisionTypeValue(shim): %v", err)
	}
	if value != "lfm2-d1" {
		t.Errorf("shim value = %q, want lfm2-d1", value)
	}

	srcInfo, err := os.Stat(src)
	if err != nil {
		t.Fatal(err)
	}
	shimInfo, err := os.Stat(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	if shimInfo.Size() != srcInfo.Size() {
		t.Errorf("shim size = %d, want the same-length copy %d", shimInfo.Size(), srcInfo.Size())
	}

	// A second call reuses the cached copy instead of re-copying: it keeps
	// its own mtime, which a rebuild would refresh to "now".
	orig, err := os.Stat(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	reused, err := shimDecisionModel(src, "lfm3-d1", "lfm2-d1")
	if err != nil {
		t.Fatalf("shimDecisionModel reuse: %v", err)
	}
	if reused != shimPath {
		t.Errorf("reuse path = %q, want %q", reused, shimPath)
	}
	shimInfo, err = os.Stat(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	if !shimInfo.ModTime().Equal(orig.ModTime()) {
		t.Errorf("reuse rebuilt the cached copy (mtime %v, was %v)",
			shimInfo.ModTime(), orig.ModTime())
	}
	_, value, err = findDecisionTypeValue(shimPath)
	if err != nil {
		t.Fatalf("findDecisionTypeValue(reuse): %v", err)
	}
	if value != "lfm2-d1" {
		t.Errorf("reuse value = %q, want lfm2-d1", value)
	}

	// A re-pulled source (forwarded mtime) triggers a rebuild.
	future := srcInfo.ModTime().Add(2 * time.Hour)
	if err := os.Chtimes(src, future, future); err != nil {
		t.Fatal(err)
	}
	staleInfo, err := os.Stat(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := shimDecisionModel(src, "lfm3-d1", "lfm2-d1"); err != nil {
		t.Fatalf("shimDecisionModel rebuild: %v", err)
	}
	shimInfo, err = os.Stat(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	if !shimInfo.ModTime().After(staleInfo.ModTime()) {
		t.Errorf("stale cached copy was not rebuilt (mtime %v)", shimInfo.ModTime())
	}
	_, value, err = findDecisionTypeValue(shimPath)
	if err != nil {
		t.Fatalf("findDecisionTypeValue(rebuild): %v", err)
	}
	if value != "lfm2-d1" {
		t.Errorf("rebuild value = %q, want lfm2-d1", value)
	}
}

func TestShimDecisionModelWrongType(t *testing.T) {
	dir := t.TempDir()
	src := shimDecisionGGUF(t, dir, "lev-1.gguf", "lev")

	if _, err := shimDecisionModel(src, "lfm3-d1", "lfm2-d1"); err == nil {
		t.Error("shimDecisionModel = nil error, want a decision type mismatch")
	}
}

func TestShimDecisionModelLengthMismatch(t *testing.T) {
	dir := t.TempDir()
	src := shimDecisionGGUF(t, dir, "d1-2B-Q4_K_M.gguf", "lfm3-d1")

	_, err := shimDecisionModel(src, "lfm3-d1", "lfm2-d1-omni")
	if err == nil {
		t.Fatal("shimDecisionModel = nil error, want a length mismatch rejection")
	}
	if !strings.Contains(err.Error(), "same value length") {
		t.Errorf("error = %v, want the same-value-length message", err)
	}
}
