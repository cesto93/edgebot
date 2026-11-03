package llm

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
)

// yzma's exp/decide switches on the "<arch>.decision.type" GGUF metadata of a
// self-describing model and does not learn new LiquidAI decision types at our
// release pace. When it rejects such a type, the shim below aliases it to the
// family twin yzma already supports: both pick one label code per option and
// read the logits of that token at the end of the prompt. The alias lives in
// a cached copy of the GGUF whose decision.type value is rewritten in place,
// so the model file itself is never modified.
//
// The rewrite replaces the value with one of the same length, which keeps the
// GGUF byte layout intact. A new entry must keep that invariant.
var decisionTypeShims = map[string]string{
	"lfm3-d1": "lfm2-d1",
}

// shimFromError reports the alias pair yzma's unsupported-type error names.
func shimFromError(err error) (from, to string, ok bool) {
	if err == nil {
		return "", "", false
	}
	msg := err.Error()
	for from, to := range decisionTypeShims {
		if strings.Contains(msg, fmt.Sprintf("decision type %q is not supported", from)) {
			return from, to, true
		}
	}
	return "", "", false
}

// shimDecisionModel returns the path of a cached copy of src whose
// decision.type metadata is rewritten from from to to. The copy lives next
// to src as "<name>.<to>shim.gguf" (without ".gguf"), so repeated decide runs
// reuse it instead of re-copying the model; it is rebuilt when the source has
// been re-pulled since (newer mtime or different size).
func shimDecisionModel(src, from, to string) (string, error) {
	if len(from) != len(to) {
		return "", fmt.Errorf("decide: shim %q -> %q keeps the same value length", from, to)
	}
	offset, value, err := findDecisionTypeValue(src)
	if err != nil {
		return "", fmt.Errorf("decide: scan %s: %w", src, err)
	}
	if value != from {
		return "", fmt.Errorf("decide: %s decision type is %q, want %q", src, value, from)
	}

	dst := strings.TrimSuffix(src, ".gguf") + "." + to + "shim.gguf"
	srcInfo, err := os.Stat(src)
	if err != nil {
		return "", fmt.Errorf("decide: stat %s: %w", src, err)
	}
	if info, err := os.Stat(dst); err == nil {
		if info.Size() == srcInfo.Size() && !info.ModTime().Before(srcInfo.ModTime()) {
			slog.Debug("decide shim reuses cached copy", "path", dst)
			return dst, nil
		}
		if err := os.Remove(dst); err != nil {
			return "", fmt.Errorf("decide: stale shim %s: %w", dst, err)
		}
	}

	if err := writeShim(src, dst, offset, to); err != nil {
		return "", err
	}
	slog.Warn("decision type not supported yet, answering through its shim twin",
		"model", src, "type", from, "shim", to, "copy", dst)
	return dst, nil
}

// writeShim copies src to dst and rewrites the decision.type value bytes at
// offset. Both files are closed explicitly and the partial copy is removed
// when anything fails, mirroring the models pull command.
func writeShim(src, dst string, offset int64, to string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("decide: open %s: %w", src, err)
	}
	out, err := os.Create(dst)
	if err != nil {
		in.Close()
		return fmt.Errorf("decide: create %s: %w", dst, err)
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		in.Close()
		os.Remove(dst)
		return fmt.Errorf("decide: copy %s: %w", dst, err)
	}
	if _, err := out.WriteAt([]byte(to), offset); err != nil {
		out.Close()
		in.Close()
		os.Remove(dst)
		return fmt.Errorf("decide: patch %s: %w", dst, err)
	}
	if err := out.Close(); err != nil {
		in.Close()
		os.Remove(dst)
		return fmt.Errorf("decide: close %s: %w", dst, err)
	}
	if err := in.Close(); err != nil {
		return fmt.Errorf("decide: close %s: %w", src, err)
	}
	return nil
}

// findDecisionTypeValue reads the "<arch>.decision.type" metadata string of a
// GGUF file and reports the byte offset of its value, so the shim can rewrite
// it in place. Only the metadata entries at the start of the file are parsed;
// tensor data is never touched. A bound on the consumed bytes keeps a corrupt
// file from stalling the scan.
const ggufShimMaxBytes = 64 << 20

func findDecisionTypeValue(path string) (int64, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, "", err
	}
	defer f.Close()

	s := &ggufShimReader{f: f}
	if magic := s.uint(4); s.err == nil && magic != 0x46554747 { // "GGUF" little-endian
		return 0, "", fmt.Errorf("%s is not a GGUF file", path)
	}
	s.uint(4) // version
	s.uint(8) // tensor count
	nKV := s.uint(8)
	if s.err != nil {
		return 0, "", s.err
	}
	if nKV > 100000 {
		return 0, "", fmt.Errorf("implausible metadata count %d in %s", nKV, path)
	}
	for range nKV {
		key := s.str()
		typ := uint32(s.uint(4))
		if s.err != nil {
			return 0, "", s.err
		}
		if strings.HasSuffix(key, ".decision.type") && typ == 8 {
			n := s.uint(8)
			valueOff := s.off
			return valueOff, s.strOf(n), nil
		}
		s.skipValue(typ)
		if s.err != nil {
			return 0, "", s.err
		}
	}
	return 0, "", fmt.Errorf("%s has no %s decision type key", path, `"<arch>.decision.type"`)
}

// ggufShimReader reads GGUF primitives from the start of a file, counting the
// consumed bytes in off so a corrupt file can be bounded.
type ggufShimReader struct {
	f    *os.File
	err  error
	off  int64
	used int64
}

func (s *ggufShimReader) fail(err error) {
	if s.err == nil {
		s.err = err
	}
}

func (s *ggufShimReader) uint(n int) uint64 {
	if s.err != nil {
		return 0
	}
	var b [8]byte
	if _, err := io.ReadFull(s.f, b[:n]); err != nil {
		s.fail(err)
		return 0
	}
	s.off += int64(n)
	s.used += int64(n)
	var v uint64
	for i := n - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// str reads a GGUF string (uint64 length + bytes).
func (s *ggufShimReader) str() string {
	return s.strOf(s.uint(8))
}

func (s *ggufShimReader) strOf(n uint64) string {
	if s.err != nil {
		return ""
	}
	if n > uint64(ggufShimMaxBytes-s.used) {
		s.fail(fmt.Errorf("metadata string too large (%d bytes)", n))
		return ""
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(s.f, buf); err != nil {
		s.fail(err)
		return ""
	}
	s.off += int64(n)
	s.used += int64(n)
	return string(buf)
}

// skipValueOfType discards one metadata value of the given GGUF type
// (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
func (s *ggufShimReader) skipValue(typ uint32) {
	switch typ {
	case 0, 1, 7: // uint8, int8, bool
		s.uint(1)
	case 2, 3: // uint16, int16
		s.uint(2)
	case 4, 5, 6: // uint32, int32, float32
		s.uint(4)
	case 10, 11, 12: // uint64, int64, float64
		s.uint(8)
	case 8: // string
		s.strOf(s.uint(8))
	case 9: // array
		elem := uint32(s.uint(4))
		count := s.uint(8)
		s.skipArrayElems(elem, count)
	default:
		s.fail(fmt.Errorf("unknown metadata type %d", typ))
	}
}

func (s *ggufShimReader) skipArrayElems(elem uint32, count uint64) {
	if s.err != nil {
		return
	}
	switch elem {
	case 0, 1, 7:
		s.skip(count)
	case 2, 3:
		s.skip(count * 2)
	case 4, 5, 6:
		s.skip(count * 4)
	case 10, 11, 12:
		s.skip(count * 8)
	case 8: // strings
		for range count {
			s.strOf(s.uint(8))
			if s.err != nil {
				return
			}
		}
	case 9:
		s.fail(fmt.Errorf("nested arrays are not valid GGUF metadata"))
	default:
		s.fail(fmt.Errorf("unknown array element type %d", elem))
	}
}

// skip discards n bytes via Seek without allocating.
func (s *ggufShimReader) skip(n uint64) {
	if s.err != nil {
		return
	}
	if n > uint64(ggufShimMaxBytes-s.used) {
		s.fail(fmt.Errorf("metadata entry too large (%d bytes)", n))
		return
	}
	if _, err := s.f.Seek(int64(n), io.SeekCurrent); err != nil {
		s.fail(err)
		return
	}
	s.off += int64(n)
	s.used += int64(n)
}
