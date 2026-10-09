package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func resetDecideFlags() {
	decideModel = ""
	decideRequest = ""
	decideState = ""
	decideStateFile = ""
	decideChoice = ""
	decideScore = ""
	decideNoul = ""
	decideOutput = ""
	decideBothOrders = false
}

func TestDecideQuestionFromFlags(t *testing.T) {
	tests := []struct {
		name     string
		choice   string
		score    string
		noul     string
		wantType string
		wantOpts int
		wantErr  bool
	}{
		{"choice", "Who?|a=desc a|b", "", "", "choice", 2, false},
		{"score", "", "How?|low|high", "", "score", 2, false},
		{"noul", "", "", "Holds?|no|yes", "noul", 2, false},
		{"none", "", "", "", "", 0, true},
		{"multiple", "Who?|a|b", "How?|low|high", "", "", 0, true},
		{"bad choice", "no options here", "", "", "", 0, true},
		{"bad noul", "", "", "q|false", "", 0, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resetDecideFlags()
			defer resetDecideFlags()
			decideChoice, decideScore, decideNoul = tc.choice, tc.score, tc.noul
			q, err := decideQuestionFromFlags()
			if tc.wantErr != (err != nil) {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if string(q.Type) != tc.wantType || len(q.Options) != tc.wantOpts {
				t.Errorf("got type %q with %d options, want %q with %d",
					q.Type, len(q.Options), tc.wantType, tc.wantOpts)
			}
		})
	}
}

func TestDecideStateFromFlags(t *testing.T) {
	t.Run("inline state", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideState = "some state"
		got, err := decideStateFromFlags()
		if err != nil || got != "some state" {
			t.Errorf("got %q, %v", got, err)
		}
	})

	t.Run("missing state", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		if _, err := decideStateFromFlags(); err == nil {
			t.Error("succeeded, want error")
		}
	})

	t.Run("both state sources", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideState = "x"
		decideStateFile = "y"
		if _, err := decideStateFromFlags(); err == nil {
			t.Error("succeeded, want error")
		}
	})

	t.Run("state file", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		path := filepath.Join(t.TempDir(), "state.txt")
		if err := os.WriteFile(path, []byte("file state"), 0644); err != nil {
			t.Fatal(err)
		}
		decideStateFile = path
		got, err := decideStateFromFlags()
		if err != nil || got != "file state" {
			t.Errorf("got %q, %v", got, err)
		}
	})

	t.Run("missing state file", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideStateFile = filepath.Join(t.TempDir(), "absent.txt")
		if _, err := decideStateFromFlags(); err == nil {
			t.Error("succeeded, want error")
		}
	})
}

const decideTestRequest = `{
	"state": {"ticket": "charged twice"},
	"questions": {
		"team": {"type": "choice", "instructions": "Who handles this?",
			"criteria": {"billing": "payments", "technical": "bugs"}}
	}
}`

func TestDecideRequestFromFlags(t *testing.T) {
	t.Run("request file", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		path := filepath.Join(t.TempDir(), "req.json")
		if err := os.WriteFile(path, []byte(decideTestRequest), 0644); err != nil {
			t.Fatal(err)
		}
		decideRequest = path
		req, err := decideRequestFromFlags()
		if err != nil {
			t.Fatal(err)
		}
		if len(req.Questions) != 1 || req.Questions[0].ID != "team" {
			t.Errorf("unexpected request: %+v", req)
		}
	})

	t.Run("request rejects single-question flags", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideRequest = "req.json"
		decideState = "x"
		if _, err := decideRequestFromFlags(); err == nil {
			t.Error("succeeded, want error")
		}
	})

	t.Run("bad request file", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideRequest = filepath.Join(t.TempDir(), "absent.json")
		if _, err := decideRequestFromFlags(); err == nil {
			t.Error("succeeded, want error")
		}
	})

	t.Run("single question", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideState = "the film was excellent."
		decideScore = "Sentiment?|negative|positive"
		req, err := decideRequestFromFlags()
		if err != nil {
			t.Fatal(err)
		}
		if len(req.Questions) != 1 || string(req.Questions[0].Type) != "score" {
			t.Errorf("unexpected request: %+v", req)
		}
		if s, ok := req.State.(string); !ok || s != "the film was excellent." {
			t.Errorf("unexpected state: %#v", req.State)
		}
	})

	t.Run("json state stays ordered", func(t *testing.T) {
		resetDecideFlags()
		defer resetDecideFlags()
		decideState = `{"b": 1, "a": 2}`
		decideNoul = "Holds?|no|yes"
		req, err := decideRequestFromFlags()
		if err != nil {
			t.Fatal(err)
		}
		if raw, ok := req.State.(json.RawMessage); !ok || string(raw) != `{"b": 1, "a": 2}` {
			t.Errorf("unexpected state: %#v", req.State)
		}
	})
}
