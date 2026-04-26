package bm25

import (
	"path/filepath"
	"testing"
)

var testSalt = []byte("unit-test-salt-do-not-reuse")

func TestHashTokensStable(t *testing.T) {
	a := HashTokens("refactor the fetcher", testSalt)
	b := HashTokens("refactor the fetcher", testSalt)
	if len(a) != len(b) {
		t.Fatalf("len mismatch")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("hash %d differs: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestHashTokensSaltSensitive(t *testing.T) {
	a := HashTokens("refactor fetcher", []byte("salt-a"))
	b := HashTokens("refactor fetcher", []byte("salt-b"))
	if len(a) == 0 || len(b) == 0 {
		t.Fatal("empty tokens")
	}
	// Every hash should differ under a different salt.
	for i := range a {
		if a[i] == b[i] {
			t.Errorf("salt should change hash: %d vs %d", a[i], b[i])
		}
	}
}

func TestHashTokensDropsStopwords(t *testing.T) {
	// "the" is a stopword, "a" is single-char — both dropped.
	got := HashTokens("refactor the a fetcher", testSalt)
	if len(got) != 2 {
		t.Errorf("len = %d, want 2 (refactor, fetcher)", len(got))
	}
}

func TestIndexRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "idx.gob")

	idx := New()
	idx.Add(HashTokens("refactor Fetcher to Client", testSalt), Doc{
		ActiveSeconds: 500, WallSeconds: 2000,
		TaskType: "refactor", SizeBucket: "large",
		ToolCount: 91, FilesTouched: 11,
	})
	idx.Add(HashTokens("fix flake in auth tests", testSalt), Doc{
		ActiveSeconds: 42, WallSeconds: 108,
		TaskType: "debug", SizeBucket: "small",
		ToolCount: 4, FilesTouched: 1,
	})
	idx.Add(HashTokens("add retry logic to fetcher", testSalt), Doc{
		ActiveSeconds: 130, WallSeconds: 380,
		TaskType: "feature", SizeBucket: "medium",
		ToolCount: 8, FilesTouched: 3,
	})

	if err := idx.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded.Docs) != 3 {
		t.Fatalf("loaded.Docs: want 3, got %d", len(loaded.Docs))
	}

	// High-signal query matches doc 0 (refactor fetcher).
	matches := loaded.TopK(HashTokens("refactor fetcher to client", testSalt), 3)
	if len(matches) == 0 {
		t.Fatal("no matches for high-signal query")
	}
	if matches[0].Doc.ID != 0 {
		t.Errorf("top match should be doc 0, got %d", matches[0].Doc.ID)
	}
	if matches[0].Sim <= 0 || matches[0].Sim > 1 {
		t.Errorf("top sim out of [0,1]: %f", matches[0].Sim)
	}
	// The match carries no prompt text — privacy invariant.
	if matches[0].Doc.TaskType != "refactor" {
		t.Errorf("doc metadata corrupted: task=%q", matches[0].Doc.TaskType)
	}

	// Retry query should match doc 2.
	m2 := loaded.TopK(HashTokens("retry logic for fetcher", testSalt), 3)
	if len(m2) == 0 || m2[0].Doc.ID != 2 {
		t.Errorf("retry query: want top=doc 2, got %+v", m2)
	}

	// Unrelated query returns no matches.
	m3 := loaded.TopK(HashTokens("kubernetes helm chart", testSalt), 3)
	if len(m3) != 0 {
		t.Errorf("unrelated query should return no matches, got %d", len(m3))
	}
}

func TestLoadMissingFile(t *testing.T) {
	dir := t.TempDir()
	idx, err := Load(filepath.Join(dir, "nope.gob"))
	if err != nil {
		t.Fatalf("want empty on missing, got %v", err)
	}
	if len(idx.Docs) != 0 {
		t.Errorf("expected empty Docs, got %d", len(idx.Docs))
	}
	idx.Add(HashTokens("hello world test", testSalt), Doc{})
	if len(idx.Docs) != 1 {
		t.Errorf("Add to empty loaded index failed")
	}
}

func TestTopKEmpty(t *testing.T) {
	idx := New()
	if m := idx.TopK(HashTokens("anything", testSalt), 5); m != nil {
		t.Errorf("TopK on empty index should return nil, got %v", m)
	}
}
