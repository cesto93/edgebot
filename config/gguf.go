package config

import (
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"os"
	"regexp"
	"strings"
)

// ggufDecisionKeySuffix matches the self-describing marker of System One
// decision models: their GGUF metadata holds a "<arch>.decision.type" key
// (openjev, lev, laya, kev, ...) plus a "systemone" chat template.
const ggufDecisionKeySuffix = ".decision.type"

// ocrNameRe matches "OCR" as a word in a model name (e.g. "MiniCPM OCR 1.0",
// "MiniCPM-OCR", "mistral-ocr"). Used to tell task-specific OCR models apart
// from general multimodal LLMs, which look identical by input modality.
var ocrNameRe = regexp.MustCompile(`(?i)(^|[^a-z])ocr([^a-z]|$)`)

// ocrTemplateMarkers are task prompts embedded in some OCR chat templates.
// MiniCPM-OCR does not embed them (its marker lives in general.name), so this
// is only a secondary signal.
var ocrTemplateMarkers = []string{
	"Text Recognition:",
	"Table Recognition:",
	"Formula Recognition:",
}

// IsOCRModelFile reports whether the GGUF file at path is a task-specific OCR
// model. OCR models are multimodal too, so input modality cannot distinguish
// them from a general-purpose vision LLM; instead the self-describing metadata
// is probed: the model name/basename contains "OCR" (verified on
// openbmb/MiniCPM-OCR-1.0-gguf: general.name="MiniCPM OCR 1.0",
// general.basename="MiniCPM-OCR"), or its chat template carries OCR task
// prompts. Any read/parse failure returns false so an unknown file is listed
// as a plain LLM.
func IsOCRModelFile(path string) bool {
	values, err := ggufStringValues(path,
		"general.name", "general.basename", "tokenizer.chat_template")
	if err != nil {
		slog.Debug("ocr metadata scan failed, treating as LLM", "path", path, "err", err)
		return false
	}
	for _, key := range []string{"general.name", "general.basename"} {
		if ocrNameRe.MatchString(values[key]) {
			return true
		}
	}
	tmpl := values["tokenizer.chat_template"]
	for _, marker := range ocrTemplateMarkers {
		if strings.Contains(tmpl, marker) {
			return true
		}
	}
	return false
}

// ggufMaxKV / ggufMaxBytes bound the metadata scan so a corrupt file cannot
// stall the models list. Real GGUFs hold hundreds of keys in a few MiB.
const ggufMaxKV = 100000
const ggufMaxBytes = 64 << 20

// GGUF metadata value types (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
const (
	ggufTypeString = 8
	ggufTypeArray  = 9
)

// IsDecisionModelFile reports whether the GGUF file at path carries System
// One decision metadata, i.e. it is a decision model rather than a text LLM.
// Only the metadata keys at the start of the file are read; weights are
// never touched. Any read or parse failure returns false so an unknown
// file is listed as a plain LLM.
func IsDecisionModelFile(path string) bool {
	found, err := ggufHasKeySuffix(path, ggufDecisionKeySuffix)
	if err != nil {
		slog.Debug("decision metadata scan failed, treating as LLM", "path", path, "err", err)
		return false
	}
	return found
}

// ggufStringValues reads the string values of the given metadata keys from a
// GGUF file, in a single metadata pass. Missing keys or non-string values are
// returned as empty strings. This keeps the OCR probe cheap: tokenizer arrays
// and tensor data are skipped, never loaded.
func ggufStringValues(path string, keys ...string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	wanted := make(map[string]bool, len(keys))
	for _, k := range keys {
		wanted[k] = true
	}
	out := make(map[string]string, len(keys))

	r := &ggufReader{r: f}
	if magic := r.uint32(); magic != 0x46554747 { // "GGUF" little-endian
		return nil, fmt.Errorf("not a GGUF file")
	}
	_ = r.uint32() // version
	_ = r.uint64() // tensor count
	nKV := r.uint64()
	if r.err != nil {
		return nil, r.err
	}
	if nKV > ggufMaxKV {
		return nil, fmt.Errorf("implausible metadata count %d", nKV)
	}

	for range nKV {
		key := r.str()
		if r.err != nil {
			return nil, r.err
		}
		typ := r.uint32()
		if r.err != nil {
			return nil, r.err
		}
		if wanted[key] && typ == ggufTypeString {
			out[key] = r.str()
			if r.err != nil {
				return nil, r.err
			}
			continue
		}
		r.skipValueOfType(typ)
		if r.err != nil {
			return nil, r.err
		}
	}
	return out, nil
}

func ggufHasKeySuffix(path, suffix string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()

	r := &ggufReader{r: f}
	if magic := r.uint32(); magic != 0x46554747 { // "GGUF" little-endian
		return false, fmt.Errorf("not a GGUF file")
	}
	_ = r.uint32() // version
	_ = r.uint64() // tensor count
	nKV := r.uint64()
	if r.err != nil {
		return false, r.err
	}
	if nKV > ggufMaxKV {
		return false, fmt.Errorf("implausible metadata count %d", nKV)
	}

	for range nKV {
		key := r.str()
		if r.err != nil {
			return false, r.err
		}
		if strings.HasSuffix(key, suffix) {
			return true, nil
		}
		r.skipValue()
		if r.err != nil {
			return false, r.err
		}
	}
	return false, nil
}

// ggufReader reads GGUF primitives from the start of a file, skipping values
// via Seek so large arrays (e.g. tokenizer lists) cost no memory.
type ggufReader struct {
	r   *os.File
	err error
	// used counts every byte consumed or skipped, bounding corrupt files.
	used uint64
}

func (g *ggufReader) fail(err error) {
	if g.err == nil {
		g.err = err
	}
}

func (g *ggufReader) uint32() uint32 {
	if g.err != nil {
		return 0
	}
	var v uint32
	if err := binary.Read(g.r, binary.LittleEndian, &v); err != nil {
		g.fail(err)
		return 0
	}
	g.used += 4
	return v
}

func (g *ggufReader) uint64() uint64 {
	if g.err != nil {
		return 0
	}
	var v uint64
	if err := binary.Read(g.r, binary.LittleEndian, &v); err != nil {
		g.fail(err)
		return 0
	}
	g.used += 8
	return v
}

func (g *ggufReader) bytes(n uint64) []byte {
	if g.err != nil {
		return nil
	}
	if n > ggufMaxBytes-g.used {
		g.fail(fmt.Errorf("metadata entry too large (%d bytes)", n))
		return nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		g.fail(err)
		return nil
	}
	g.used += n
	return buf
}

// skip discards n bytes via Seek without allocating.
func (g *ggufReader) skip(n uint64) {
	if g.err != nil {
		return
	}
	if n > ggufMaxBytes-g.used {
		g.fail(fmt.Errorf("metadata entry too large (%d bytes)", n))
		return
	}
	if _, err := g.r.Seek(int64(n), io.SeekCurrent); err != nil {
		g.fail(err)
		return
	}
	g.used += n
}

func (g *ggufReader) str() string {
	return string(g.bytes(g.uint64()))
}

// skipValue discards one metadata value of the type that follows the key.
func (g *ggufReader) skipValue() {
	g.skipValueOfType(g.uint32())
}

// skipValueOfType discards one metadata value of the given GGUF type.
func (g *ggufReader) skipValueOfType(typ uint32) {
	switch typ {
	case 0, 1, 7: // uint8, int8, bool
		g.skip(1)
	case 2, 3: // uint16, int16
		g.skip(2)
	case 4, 5, 6: // uint32, int32, float32
		g.skip(4)
	case 10, 11, 12: // uint64, int64, float64
		g.skip(8)
	case ggufTypeString:
		g.skip(g.uint64())
	case ggufTypeArray:
		elem := g.uint32()
		count := g.uint64()
		g.skipArrayElems(elem, count)
	default:
		g.fail(fmt.Errorf("unknown metadata type %d", typ))
	}
}

func (g *ggufReader) skipArrayElems(elem uint32, count uint64) {
	if g.err != nil {
		return
	}
	switch elem {
	case 0, 1, 7:
		g.skip(count)
	case 2, 3:
		g.skip(count * 2)
	case 4, 5, 6:
		g.skip(count * 4)
	case 10, 11, 12:
		g.skip(count * 8)
	case ggufTypeString:
		for range count {
			g.skip(g.uint64())
			if g.err != nil {
				return
			}
		}
	case ggufTypeArray:
		g.fail(fmt.Errorf("nested arrays are not valid GGUF metadata"))
	default:
		g.fail(fmt.Errorf("unknown array element type %d", elem))
	}
}
