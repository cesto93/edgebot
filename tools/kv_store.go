package tools

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"edgebot/config"

	"go.etcd.io/bbolt"
)

var dbPathOverride string

func getDBPath() (string, error) {
	if dbPathOverride != "" {
		return dbPathOverride, nil
	}
	configPath, err := config.ConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(configPath, "kv_store.db"), nil
}

func getDB() (*bbolt.DB, error) {
	dbPath, err := getDBPath()
	if err != nil {
		return nil, err
	}
	return bbolt.Open(dbPath, 0600, &bbolt.Options{Timeout: time.Second})
}

func KVSet(key, value string) (string, error) {
	db, err := getDB()
	if err != nil {
		return "", fmt.Errorf("failed to open database: %w", err)
	}
	defer db.Close()

	err = db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("kv"))
		if err != nil {
			return err
		}
		return b.Put([]byte(key), []byte(value))
	})

	if err != nil {
		return "", fmt.Errorf("failed to save key: %w", err)
	}
	return fmt.Sprintf("Successfully saved key '%s'", key), nil
}

func KVGet(key string) (string, error) {
	db, err := getDB()
	if err != nil {
		return "", fmt.Errorf("failed to open database: %w", err)
	}
	defer db.Close()

	var value string
	err = db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("kv"))
		if b == nil {
			return fmt.Errorf("key '%s' not found", key)
		}
		v := b.Get([]byte(key))
		if v == nil {
			return fmt.Errorf("key '%s' not found", key)
		}
		value = string(v)
		return nil
	})

	if err != nil {
		return "", err
	}
	return value, nil
}

func KVList() (string, error) {
	db, err := getDB()
	if err != nil {
		return "", fmt.Errorf("failed to open database: %w", err)
	}
	defer db.Close()

	var keys []string
	err = db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte("kv"))
		if b == nil {
			return nil
		}
		return b.ForEach(func(k, v []byte) error {
			keys = append(keys, string(k))
			return nil
		})
	})

	if err != nil {
		return "", fmt.Errorf("failed to list keys: %w", err)
	}

	if len(keys) == 0 {
		return "No keys found in KV store", nil
	}

	sort.Strings(keys)
	return "Keys in KV store:\n- " + strings.Join(keys, "\n- "), nil
}
