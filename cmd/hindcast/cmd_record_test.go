package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/justinstimatze/hindcast/internal/store"
)

// fixtureTranscript builds a minimal user→assistant transcript pair
// with timestamps relative to "now" so the 2-hour `since` window in
// doRecordFallback always picks them up regardless of when the test
// runs. promptText is interpolated raw — caller must avoid JSON-breaking
// characters.
func fixtureTranscript(sessionID, cwd, promptText string) string {
	now := time.Now().UTC()
	userTS := now.Add(-1 * time.Minute).Format(time.RFC3339Nano)
	asstTS := now.Add(-30 * time.Second).Format(time.RFC3339Nano)
	return fmt.Sprintf(
		`{"parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":%q},"uuid":"u1","timestamp":%q,"sessionId":%q,"cwd":%q,"version":"2.1.0"}
{"parentUuid":"u1","isSidechain":false,"type":"assistant","message":{"role":"assistant","model":"claude-opus-4-7","content":[{"type":"text","text":"ack"}]},"uuid":"a1","timestamp":%q,"sessionId":%q}
`,
		promptText, userTS, sessionID, cwd, asstTS, sessionID,
	)
}

// TestFallbackRecordContainsNoPromptText is the fallback-path mirror of
// store.TestRecordContainsNoPromptText. The transcript fallback parses
// raw prompts in plaintext (it has to — that's the whole point of the
// fallback) and must hash them before any disk write. If a refactor
// ever lands prompt text in a Record field, this catches it.
func TestFallbackRecordContainsNoPromptText(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	// A transcript with a uniquely identifiable prompt.
	const sentinel = "FALLBACK_TEST_SENTINEL_PROMPT_TEXT"
	transcriptDir := filepath.Join(home, "transcripts", "p1")
	if err := os.MkdirAll(transcriptDir, 0700); err != nil {
		t.Fatal(err)
	}
	transcriptPath := filepath.Join(transcriptDir, "t.jsonl")
	transcript := fixtureTranscript("sess-fb-test", transcriptDir, sentinel)
	if err := os.WriteFile(transcriptPath, []byte(transcript), 0600); err != nil {
		t.Fatal(err)
	}

	in := recordInput{
		SessionID:      "sess-fb-test",
		CWD:            transcriptDir,
		TranscriptPath: transcriptPath,
		PermissionMode: "bypassPermissions",
	}
	doRecordFallback(in)

	// Sanity: the fallback should have produced at least one record;
	// otherwise the privacy assertion below is trivially satisfied.
	hash := store.ProjectHash(store.ResolveProject(transcriptDir))
	logPath, _ := store.ProjectLogPath(hash)
	if data, err := os.ReadFile(logPath); err != nil || len(data) == 0 {
		t.Fatalf("fallback produced no record (privacy assertion would be trivial): %v", err)
	}

	// Walk every file under HindcastDir and assert the sentinel is absent.
	hcDir, err := store.HindcastDir()
	if err != nil {
		t.Fatal(err)
	}
	var leakPaths []string
	err = filepath.WalkDir(hcDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		data, _ := os.ReadFile(path)
		if strings.Contains(string(data), sentinel) {
			leakPaths = append(leakPaths, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(leakPaths) > 0 {
		t.Errorf("fallback path leaked plaintext prompt to disk: %v", leakPaths)
	}
}

// TestFallbackMarkerAdvances confirms the fallback writes a marker after
// recording, and a subsequent fallback call respects it (no duplicate
// record). Without this, restart-induced double-fallbacks could double-
// count turns and pollute the predictor.
func TestFallbackMarkerAdvances(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	transcriptDir := filepath.Join(home, "transcripts", "p1")
	if err := os.MkdirAll(transcriptDir, 0700); err != nil {
		t.Fatal(err)
	}
	transcriptPath := filepath.Join(transcriptDir, "t.jsonl")
	transcript := fixtureTranscript("sess-marker", transcriptDir, "first prompt")
	if err := os.WriteFile(transcriptPath, []byte(transcript), 0600); err != nil {
		t.Fatal(err)
	}

	in := recordInput{
		SessionID:      "sess-marker",
		CWD:            transcriptDir,
		TranscriptPath: transcriptPath,
		PermissionMode: "bypassPermissions",
	}

	doRecordFallback(in)
	doRecordFallback(in) // second call should be a no-op (marker advanced)

	hash := store.ProjectHash(store.ResolveProject(transcriptDir))
	logPath, err := store.ProjectLogPath(hash)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	count := strings.Count(string(data), "\n")
	if count != 1 {
		t.Errorf("expected 1 record after two fallback calls (marker should suppress duplicate), got %d records:\n%s", count, data)
	}
}
