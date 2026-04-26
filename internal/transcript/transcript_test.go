package transcript

import (
	"path/filepath"
	"testing"
	"time"
)

func TestParseFile(t *testing.T) {
	path := filepath.Join("testdata", "sample.jsonl")
	turns, err := ParseFile(path, time.Time{})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(turns) != 2 {
		t.Fatalf("want 2 turns, got %d", len(turns))
	}

	t1 := turns[0]
	if t1.PromptText != "add retry logic to fetcher" {
		t.Errorf("turn1 prompt = %q", t1.PromptText)
	}
	if t1.Model != "claude-opus-4-7" {
		t.Errorf("turn1 model = %q", t1.Model)
	}
	// Tool calls: 1 Read + 1 Edit + 1 Bash = 3 distinct tools.
	if got := t1.ToolCalls["Read"]; got != 1 {
		t.Errorf("turn1 Read = %d", got)
	}
	if got := t1.ToolCalls["Edit"]; got != 1 {
		t.Errorf("turn1 Edit = %d", got)
	}
	if got := t1.ToolCalls["Bash"]; got != 1 {
		t.Errorf("turn1 Bash = %d", got)
	}
	// Files: fetcher.go is touched by Read and Edit — should dedupe to 1.
	if t1.FilesTouched != 1 {
		t.Errorf("turn1 files_touched = %d, want 1", t1.FilesTouched)
	}
	// Wall: 10:00:00 → 10:05:06 = 306s.
	if w := t1.WallSeconds(); w != 306 {
		t.Errorf("turn1 wall = %d, want 306", w)
	}
	// Active: gaps between tool_uses, clamped at 60s each.
	//   tu1 10:00:04 (first, no gap)
	//   tu2 10:00:10 gap 6s
	//   tu3 10:05:00 gap 290s clamped to 60s
	// Total active = 0 + 6 + 60 = 66s.
	if a := t1.ActiveSeconds; a != 66 {
		t.Errorf("turn1 active = %d, want 66 (6 + 60 clamp)", a)
	}

	t2 := turns[1]
	if t2.PromptText != "fix the test flake in auth" {
		t.Errorf("turn2 prompt = %q", t2.PromptText)
	}
	if t2.FilesTouched != 1 {
		t.Errorf("turn2 files_touched = %d", t2.FilesTouched)
	}
	if w := t2.WallSeconds(); w != 45 {
		t.Errorf("turn2 wall = %d, want 45", w)
	}
	// Subagent entry between turns must NOT contaminate turn2 counts.
	for name := range t2.ToolCalls {
		if name == "SubagentTask" {
			t.Errorf("subagent tool leaked into turn2: %v", t2.ToolCalls)
		}
	}
}

func TestParseFileSince(t *testing.T) {
	path := filepath.Join("testdata", "sample.jsonl")
	cutoff, _ := time.Parse(time.RFC3339, "2026-04-19T10:09:00Z")
	turns, err := ParseFile(path, cutoff)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(turns) != 1 {
		t.Fatalf("want 1 turn after cutoff, got %d", len(turns))
	}
	if turns[0].PromptText != "fix the test flake in auth" {
		t.Errorf("filtered turn prompt = %q", turns[0].PromptText)
	}
}

func TestParseTail(t *testing.T) {
	path := filepath.Join("testdata", "sample.jsonl")
	// Small tailBytes → only the tail of the file is read; earlier turn
	// should be dropped because its user-prompt line is not in the tail.
	turns, err := ParseTail(path, 512, time.Time{})
	if err != nil {
		t.Fatalf("parse tail: %v", err)
	}
	// With 512 bytes of tail, we may land partway through turn2 events.
	// Behaviour we assert: no panic, no crash, and any returned turns are
	// valid (have a non-empty prompt).
	for _, tr := range turns {
		if tr.PromptText == "" {
			t.Errorf("tail returned turn with empty prompt: %+v", tr)
		}
	}
}
