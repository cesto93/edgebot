package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"unicode/utf8"

	"edgebot/config"
	"edgebot/llm"

	"github.com/spf13/cobra"
	"golang.org/x/term"
)

// skillsHomeDir is swappable for tests.
var skillsHomeDir = os.UserHomeDir

const skillFileName = "SKILL.md"

var skillsPullRepo string

var skillsCmd = &cobra.Command{
	Use:   "skills",
	Short: "List all available skills",
	RunE: func(cmd *cobra.Command, args []string) error {
		if skillsPullRepo != "" {
			if err := pullSkills(skillsPullRepo); err != nil {
				return err
			}
			fmt.Println()
		}
		return runSkills()
	},
}

func init() {
	skillsCmd.Flags().StringVar(&skillsPullRepo, "pull", "",
		"install skills from a git repository before listing\n"+
			"(shallow-clones <repo>, copies every directory containing a\n"+
			"SKILL.md into ~/.agents/skills; existing skills are updated)")
	rootCmd.AddCommand(skillsCmd)
}

func runSkills() error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	if !cfg.Skills {
		fmt.Println("Note: skills is disabled in config; skills are not sent to the agent.")
	}

	skills := llm.GetSkills(cfg.Skills)
	if len(skills) == 0 {
		fmt.Println("No skills found (~/.agents/skills, ./skills).")
		return nil
	}

	nameW := utf8.RuneCountInString("NAME")
	for _, s := range skills {
		if l := utf8.RuneCountInString(s.Name); l > nameW {
			nameW = l
		}
	}
	nameW += 3

	width := 80
	if w, _, err := term.GetSize(int(os.Stdout.Fd())); err == nil && w > nameW+10 {
		width = w
	}
	descW := width - nameW

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w, "NAME\tDESCRIPTION")
	fmt.Fprintln(w, "----\t-----------")
	for _, s := range skills {
		desc := s.Description
		if desc == "" {
			desc = "-"
		}
		for i, line := range wrapText(desc, descW) {
			name := ""
			if i == 0 {
				name = s.Name
			}
			fmt.Fprintf(w, "%s\t%s\n", name, line)
		}
		for _, line := range wrapText(s.Path, descW) {
			fmt.Fprintf(w, "\t%s\n", line)
		}
		fmt.Fprintln(w)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("failed to write skills table: %w", err)
	}

	return nil
}

// pullSkills shallow-clones the git repo and installs every directory that
// contains a SKILL.md file into the global ~/.agents/skills directory. The
// install dir is named after the skill directory in the repo; an existing
// skill with the same name is replaced so repeated pulls update in place.
func pullSkills(repo string) error {
	destRoot, err := globalSkillsDir()
	if err != nil {
		return fmt.Errorf("cannot determine skills directory: %w", err)
	}
	if err := os.MkdirAll(destRoot, 0755); err != nil {
		return fmt.Errorf("failed to create skills directory: %w", err)
	}

	tmp, err := os.MkdirTemp("", "edgebot-skills-")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmp)

	fmt.Printf("Cloning %s ...\n", repo)
	if out, err := execCommand("git", "clone", "--depth", "1", repo, tmp).CombinedOutput(); err != nil {
		return fmt.Errorf("git clone failed: %v: %s", err, strings.TrimSpace(string(out)))
	}

	skillDirs, err := findSkillDirs(tmp)
	if err != nil {
		return fmt.Errorf("failed to scan cloned repo: %w", err)
	}
	if len(skillDirs) == 0 {
		return fmt.Errorf("no skills found in %s (no SKILL.md files)", repo)
	}

	for _, src := range skillDirs {
		name := filepath.Base(src)
		dst := filepath.Join(destRoot, name)
		_, statErr := os.Stat(dst)
		if err := os.RemoveAll(dst); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to replace %s: %v\n", name, err)
			continue
		}
		if err := copyDir(src, dst); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to install %s: %v\n", name, err)
			continue
		}
		action := "Installed"
		if statErr == nil {
			action = "Updated"
		}
		fmt.Printf("%s %s -> %s\n", action, name, dst)
	}
	return nil
}

// globalSkillsDir returns ~/.agents/skills.
func globalSkillsDir() (string, error) {
	home, err := skillsHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".agents", "skills"), nil
}

// findSkillDirs walks root and returns the directories containing a SKILL.md
// file, sorted by path. Directories holding a SKILL.md are not descended
// into, and .git directories are skipped entirely.
func findSkillDirs(root string) ([]string, error) {
	var dirs []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if path == root && os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if !d.IsDir() {
			return nil
		}
		if d.Name() == ".git" {
			return filepath.SkipDir
		}
		if _, err := os.Stat(filepath.Join(path, skillFileName)); err == nil {
			dirs = append(dirs, path)
			return filepath.SkipDir
		}
		return nil
	})
	sort.Strings(dirs)
	return dirs, err
}

// copyDir recursively copies src to dst, preserving file modes and creating
// dst as needed.
func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		info, err := d.Info()
		if err != nil {
			return err
		}
		if d.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
}

// wrapText wraps text to the given display width, breaking on spaces and
// hard-splitting words that are longer than the width itself.
func wrapText(text string, width int) []string {
	var lines []string
	line := ""
	for _, word := range strings.Fields(text) {
		for utf8.RuneCountInString(word) > width {
			if line != "" {
				lines = append(lines, line)
				line = ""
			}
			r := []rune(word)
			lines = append(lines, string(r[:width]))
			word = string(r[width:])
		}
		switch {
		case line == "":
			line = word
		case utf8.RuneCountInString(line)+1+utf8.RuneCountInString(word) <= width:
			line += " " + word
		default:
			lines = append(lines, line)
			line = word
		}
	}
	if line != "" {
		lines = append(lines, line)
	}
	return lines
}
