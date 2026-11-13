package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// mmprojLinkFile is the registry (inside the llamacpp models dir) that couples
// a model to its vision projector. The coupling is written when both files are
// downloaded (and by `edgebot models link`); it is never inferred from a
// filename at runtime.
const mmprojLinkFile = "mmproj.json"

// mmprojLinksMu guards read-modify-write cycles on the registry.
var mmprojLinksMu sync.Mutex

// mmprojLinks is the on-disk shape of the registry: model base name (no
// ".gguf") -> vision projector base name (no ".gguf").
type mmprojLinks struct {
	Links map[string]string `json:"links"`
}

// normMMProjEntry validates a model/projector name and normalizes it to a GGUF
// base name. A ".gguf" suffix is accepted and trimmed.
func normMMProjEntry(name string) (string, error) {
	name = strings.TrimSpace(name)
	lower := strings.ToLower(name)
	if strings.HasSuffix(lower, ".gguf") {
		name = name[:len(name)-len(".gguf")]
	}
	if name == "" || name == "." || strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
		return "", fmt.Errorf("invalid model or projector name %q", name)
	}
	return name, nil
}

// loadMMProjLinks reads the registry from dir. A missing file yields an empty
// map.
func loadMMProjLinks(dir string) (map[string]string, error) {
	path := filepath.Join(dir, mmprojLinkFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]string{}, nil
		}
		return nil, fmt.Errorf("read vision projector links: %w", err)
	}
	var reg mmprojLinks
	if err := json.Unmarshal(data, &reg); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if reg.Links == nil {
		reg.Links = map[string]string{}
	}
	return reg.Links, nil
}

// saveMMProjLinks atomically writes the registry to dir.
func saveMMProjLinks(dir string, links map[string]string) error {
	data, err := json.MarshalIndent(mmprojLinks{Links: links}, "", "  ")
	if err != nil {
		return fmt.Errorf("encode vision projector links: %w", err)
	}
	data = append(data, '\n')

	tmp, err := os.CreateTemp(dir, ".mmproj-*.tmp")
	if err != nil {
		return fmt.Errorf("create vision projector links file: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("write vision projector links: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("close vision projector links: %w", err)
	}
	if err := os.Chmod(tmpName, 0o644); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("chmod vision projector links: %w", err)
	}
	if err := os.Rename(tmpName, filepath.Join(dir, mmprojLinkFile)); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("save vision projector links: %w", err)
	}
	return nil
}

// SaveMMProjLink records that mmprojName is the vision projector of modelName.
// Both files must exist in the llamacpp models dir so a link never points at a
// missing projector. Names may include the ".gguf" suffix.
func SaveMMProjLink(modelName, mmprojName string) error {
	modelName, err := normMMProjEntry(modelName)
	if err != nil {
		return err
	}
	mmprojName, err = normMMProjEntry(mmprojName)
	if err != nil {
		return err
	}
	dir, err := ModelsDir("llamacpp")
	if err != nil {
		return err
	}
	for _, name := range []string{modelName, mmprojName} {
		if _, err := os.Stat(filepath.Join(dir, name+".gguf")); err != nil {
			return fmt.Errorf("file %s.gguf not found in %s", name, dir)
		}
	}

	mmprojLinksMu.Lock()
	defer mmprojLinksMu.Unlock()

	links, err := loadMMProjLinks(dir)
	if err != nil {
		return err
	}
	links[modelName] = mmprojName
	return saveMMProjLinks(dir, links)
}

// deleteMMProjLink removes the link for modelName (no-op when absent) and
// returns the projector that was linked.
func deleteMMProjLink(dir, modelName string) (string, error) {
	modelName, err := normMMProjEntry(modelName)
	if err != nil {
		return "", err
	}

	mmprojLinksMu.Lock()
	defer mmprojLinksMu.Unlock()

	links, err := loadMMProjLinks(dir)
	if err != nil {
		return "", err
	}
	proj, ok := links[modelName]
	if !ok {
		return "", nil
	}
	delete(links, modelName)
	if err := saveMMProjLinks(dir, links); err != nil {
		return "", err
	}
	return proj, nil
}

// linkedMMProjName returns the projector linked to modelName, "" when unlinked.
func linkedMMProjName(links map[string]string, modelName string) string {
	name, err := normMMProjEntry(modelName)
	if err != nil {
		return ""
	}
	return links[name]
}

// MMProjLinksFile returns the path of the coupling registry (exposed for
// messages and docs).
func MMProjLinksFile() (string, error) {
	dir, err := ModelsDir("llamacpp")
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, mmprojLinkFile), nil
}
