package stats

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"path/filepath"
	"sort"
	"time"

	"edgebot/config"

	"go.etcd.io/bbolt"
)

// Usage is the token usage reported for a single LLM call.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int
	ReasoningTokens  int
	Cost             float64
}

// Entry is the usage aggregated across all calls for one model on one provider.
type Entry struct {
	Provider         string  `json:"provider"`
	Model            string  `json:"model"`
	Calls            int     `json:"calls"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	CachedTokens     int     `json:"cached_tokens"`
	ReasoningTokens  int     `json:"reasoning_tokens"`
	Cost             float64 `json:"cost"`
}

var dbPathFunc = getDBPath

func getDBPath() (string, error) {
	configPath, err := config.ConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(configPath, "usage.db"), nil
}

func getDB() (*bbolt.DB, error) {
	dbPath, err := dbPathFunc()
	if err != nil {
		return nil, err
	}
	return bbolt.Open(dbPath, 0600, &bbolt.Options{Timeout: time.Second})
}

func entryKey(provider, model string) []byte {
	return []byte(provider + "\x00" + model)
}

// RecordUsage accumulates a single call's usage into the persistent store.
// Failures are logged rather than returned so LLM calls never break because
// stats persistence is unavailable.
func RecordUsage(provider, model string, u Usage) {
	db, err := getDB()
	if err != nil {
		slog.Warn("stats: failed to open usage database", "err", err)
		return
	}
	defer db.Close()

	err = db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("usage"))
		if err != nil {
			return err
		}
		key := entryKey(provider, model)
		var e Entry
		if v := b.Get(key); v != nil {
			if err := json.Unmarshal(v, &e); err != nil {
				return err
			}
		}
		e.Provider = provider
		e.Model = model
		e.Calls++
		e.PromptTokens += u.PromptTokens
		e.CompletionTokens += u.CompletionTokens
		e.TotalTokens += u.TotalTokens
		e.CachedTokens += u.CachedTokens
		e.ReasoningTokens += u.ReasoningTokens
		e.Cost += u.Cost
		data, err := json.Marshal(e)
		if err != nil {
			return err
		}
		return b.Put(key, data)
	})
	if err != nil {
		slog.Warn("stats: failed to record usage", "err", err)
	}
}

// GetStats returns all recorded usage aggregated per provider and model,
// sorted by provider then model name.
func GetStats() ([]Entry, error) {
	db, err := getDB()
	if err != nil {
		return nil, err
	}
	defer db.Close()

	var entries []Entry
	err = db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("usage"))
		if b == nil {
			return nil
		}
		return b.ForEach(func(k, v []byte) error {
			var e Entry
			if err := json.Unmarshal(v, &e); err != nil {
				return err
			}
			entries = append(entries, e)
			return nil
		})
	})
	if err != nil {
		return nil, fmt.Errorf("failed to read usage stats: %w", err)
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Provider != entries[j].Provider {
			return entries[i].Provider < entries[j].Provider
		}
		return entries[i].Model < entries[j].Model
	})
	return entries, nil
}

// Reset wipes all recorded usage stats. It is a no-op when no stats exist.
func Reset() error {
	db, err := getDB()
	if err != nil {
		return err
	}
	defer db.Close()
	return db.Update(func(tx *bbolt.Tx) error {
		if err := tx.DeleteBucket([]byte("usage")); err == bbolt.ErrBucketNotFound {
			return nil
		} else {
			return err
		}
	})
}
