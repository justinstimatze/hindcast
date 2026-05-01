package store

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestProjectHashStable(t *testing.T) {
	// Stability across the binary's lifetime: the hash is the on-disk
	// directory key for a project; if the formula drifts every install
	// loses access to its own historical records. Pin to a literal so a
	// silent change to the hash function or truncation length fails the
	// test rather than passing a tautological self-comparison.
	if got := ProjectHash("hindcast"); got != "1283c759" {
		t.Errorf("ProjectHash(\"hindcast\") = %q; want 1283c759 (stability lock)", got)
	}
	if len(ProjectHash("hindcast")) != 8 {
		t.Error("hash length != 8")
	}
	if ProjectHash("hindcast") == ProjectHash("other") {
		t.Error("different inputs should hash differently")
	}
}

func TestResolveProjectOverride(t *testing.T) {
	dir := t.TempDir()
	if got := ResolveProject(dir); got != filepath.Base(dir) {
		t.Errorf("fallback = %q, want basename %q", got, filepath.Base(dir))
	}
	if err := os.WriteFile(filepath.Join(dir, ".hindcast-project"), []byte("custom-name\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if got := ResolveProject(dir); got != "custom-name" {
		t.Errorf("override = %q, want custom-name", got)
	}
}

func TestLockAcquireReleaseExclusive(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "lock")

	l1, err := AcquireLock(path)
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if _, err := AcquireLock(path); err == nil {
		t.Error("second acquire should fail while lock is held by live pid")
	}
	if err := l1.Release(); err != nil {
		t.Errorf("release: %v", err)
	}
	l2, err := AcquireLock(path)
	if err != nil {
		t.Fatalf("post-release acquire: %v", err)
	}
	l2.Release()
}

func TestLockStaleDetection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "lock")
	if err := os.WriteFile(path, []byte("99999999"), 0600); err != nil {
		t.Fatal(err)
	}
	l, err := AcquireLock(path)
	if err != nil {
		t.Fatalf("stale-lock acquire failed: %v", err)
	}
	l.Release()
}

func TestLockMalformedPID(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "lock")
	if err := os.WriteFile(path, []byte("not a number"), 0600); err != nil {
		t.Fatal(err)
	}
	l, err := AcquireLock(path)
	if err != nil {
		t.Fatalf("malformed-pid acquire failed: %v", err)
	}
	l.Release()
}

func TestAppendAndReadRecords(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.jsonl")
	for i := 0; i < 3; i++ {
		r := Record{
			TS:          time.Now().UTC(),
			SessionID:   "s1",
			ProjectHash: "abc12345",
			TaskType:    "feature",
			WallSeconds: i * 10,
		}
		if err := AppendRecord(path, r); err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
	}
	out, err := ReadRecentRecords(path, 10)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	for i, r := range out {
		if r.SchemaVersion != SchemaVersion {
			t.Errorf("record %d SchemaVersion = %d", i, r.SchemaVersion)
		}
	}
	// Newest first — ascending WallSeconds input → descending on read.
	if out[0].WallSeconds != 20 {
		t.Errorf("newest should have wall=20, got %d", out[0].WallSeconds)
	}
}

func TestAppendOversizedTruncates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.jsonl")
	huge := make(map[string]int)
	for i := 0; i < 500; i++ {
		huge[fmt.Sprintf("Tool%d", i)] = i
	}
	r := Record{ToolCalls: huge}
	if err := AppendRecord(path, r); err != nil {
		t.Fatalf("append: %v", err)
	}
	data, _ := os.ReadFile(path)
	if len(data) > MaxRecordSize {
		t.Errorf("record exceeded cap: %d > %d", len(data), MaxRecordSize)
	}
}

func TestSketchPercentiles(t *testing.T) {
	s := &Sketch{MaxSize: 20}
	for i := 1; i <= 10; i++ {
		s.Add(i*10, i)
	}
	wallMed, wallP90, activeMed, activeP90, n := s.Percentiles()
	if n != 10 {
		t.Errorf("n = %d, want 10", n)
	}
	if wallMed != 55.0 {
		t.Errorf("wallMed = %v, want 55", wallMed)
	}
	if wallP90 != 91.0 {
		t.Errorf("wallP90 = %v, want 91", wallP90)
	}
	if activeMed != 5.5 {
		t.Errorf("activeMed = %v, want 5.5", activeMed)
	}
	if activeP90 < 9.0 || activeP90 > 9.2 {
		t.Errorf("activeP90 = %v, want ~9.1", activeP90)
	}
}

func TestSketchRollingWindow(t *testing.T) {
	s := &Sketch{MaxSize: 5}
	for i := 1; i <= 10; i++ {
		s.Add(i, i)
	}
	if len(s.Wall) != 5 {
		t.Errorf("len(Wall) = %d, want 5 (rolling window)", len(s.Wall))
	}
	if s.Wall[0] != 6 || s.Wall[4] != 10 {
		t.Errorf("rolling window contents wrong: %v", s.Wall)
	}
}

func TestPendingRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pending.json")
	p := PendingTurn{
		SessionID:      "s-42",
		StartTS:        time.Now().UTC().Truncate(time.Second),
		TaskType:       "feature",
		PromptTokens:   []uint64{1, 2, 3, 4},
		PromptChars:    100,
		PermissionMode: "auto-accept",
		ProjectHash:    "abc12345",
		CWD:            "/tmp/p1",
	}
	if err := WritePending(path, p); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadPending(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got.SessionID != p.SessionID || got.TaskType != p.TaskType {
		t.Errorf("roundtrip mismatch: got %+v, want %+v", got, p)
	}
	if len(got.PromptTokens) != 4 {
		t.Errorf("tokens len = %d, want 4", len(got.PromptTokens))
	}
	if !got.StartTS.Equal(p.StartTS) {
		t.Errorf("StartTS: got %v, want %v", got.StartTS, p.StartTS)
	}
}

// v0.6.2: PendingPath embeds StartTS in the filename so multiple
// in-flight turns in one session don't collide. Zero startTS returns
// the legacy single-file shape (read-side compat for old binaries
// still in flight at upgrade time).
func TestPendingPathTimestamped(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir())
	t1 := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)
	t2 := time.Date(2026, 4, 30, 12, 0, 0, 1, time.UTC)
	p1, err := PendingPath("sess", t1)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := PendingPath("sess", t2)
	if err != nil {
		t.Fatal(err)
	}
	if p1 == p2 {
		t.Errorf("two startTS values produced same path: %s", p1)
	}
	legacy, err := PendingPath("sess", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(legacy, "pending-sess.json") {
		t.Errorf("legacy path lost shape: %s", legacy)
	}
}

// TestListPendingForSessionSorts confirms multiple pending files in
// the same session are returned oldest-first. This is the contract
// the Stop hook depends on: pick the pending whose StartTS is closest
// to the just-completed turn's PromptTS.
func TestListPendingForSessionSorts(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir())
	sess := "sess-multi"
	now := time.Now().UTC().Truncate(time.Second)
	stamps := []time.Time{
		now.Add(-30 * time.Second),
		now.Add(-10 * time.Second),
		now.Add(-20 * time.Second),
	}
	for i, ts := range stamps {
		path, err := PendingPath(sess, ts)
		if err != nil {
			t.Fatal(err)
		}
		if err := WritePending(path, PendingTurn{SessionID: sess, StartTS: ts, TaskType: fmt.Sprintf("t%d", i)}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := ListPendingForSession(sess)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d pendings, want 3", len(got))
	}
	for i := 1; i < len(got); i++ {
		if got[i].StartTS.Before(got[i-1].StartTS) {
			t.Errorf("not sorted oldest-first: %v before %v", got[i-1].StartTS, got[i].StartTS)
		}
	}
	// Oldest of the three was -30s.
	if got[0].StartTS != stamps[0] {
		t.Errorf("oldest mismatch: got %v, want %v", got[0].StartTS, stamps[0])
	}
}

// TestListPendingForSessionPicksUpLegacy verifies the glob captures
// both pending-<session>.json (pre-v0.6.2) and pending-<session>-*.json
// (v0.6.2). Sessions started under an older binary that haven't yet
// completed when the upgrade lands need their in-flight pending file
// to still be readable.
func TestListPendingForSessionPicksUpLegacy(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir())
	sess := "sess-mix"
	now := time.Now().UTC()
	// Write a legacy single-file pending.
	legacyPath, _ := PendingPath(sess, time.Time{})
	if err := WritePending(legacyPath, PendingTurn{SessionID: sess, StartTS: now.Add(-1 * time.Minute)}); err != nil {
		t.Fatal(err)
	}
	// Write a v0.6.2 timestamped pending.
	newPath, _ := PendingPath(sess, now)
	if err := WritePending(newPath, PendingTurn{SessionID: sess, StartTS: now}); err != nil {
		t.Fatal(err)
	}
	got, err := ListPendingForSession(sess)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d pendings, want 2 (legacy + new)", len(got))
	}
}

// v0.6.1: PendingTurn carries band fields (P25/P75, MaxSim, VarianceGated)
// so the Stop hook can fold them into the accuracy log for band-hit-rate.
// Verify they round-trip through WritePending/ReadPending.
func TestPendingBandFieldsRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pending.json")
	p := PendingTurn{
		SessionID:        "s-band-42",
		StartTS:          time.Now().UTC().Truncate(time.Second),
		TaskType:         "feature",
		ProjectHash:      "abc12345",
		PredictedWall:    180,
		PredictedActive:  60,
		PredictionSrc:    "knn",
		PredictedWallP25: 60,
		PredictedWallP75: 600,
		PredictedMaxSim:  0.71,
		VarianceGated:    true,
	}
	if err := WritePending(path, p); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadPending(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got.PredictedWallP25 != 60 || got.PredictedWallP75 != 600 {
		t.Errorf("band p25/p75 lost in roundtrip: got %d/%d", got.PredictedWallP25, got.PredictedWallP75)
	}
	if got.PredictedMaxSim != 0.71 {
		t.Errorf("max sim lost in roundtrip: got %v", got.PredictedMaxSim)
	}
	if !got.VarianceGated {
		t.Errorf("variance_gated lost in roundtrip")
	}
}

// Privacy invariant check: the persisted Record JSON must not contain
// any prompt_first_line field. If someone reintroduces it, this fails.
func TestRecordContainsNoPromptText(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.jsonl")
	r := Record{
		SessionID:   "s",
		TaskType:    "feature",
		WallSeconds: 10,
	}
	if err := AppendRecord(path, r); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	if strings.Contains(string(data), "prompt_first_line") {
		t.Errorf("persisted record leaks prompt_first_line: %s", data)
	}
}

func TestIsAbandonedTurn(t *testing.T) {
	cases := []struct {
		name          string
		wall, active  int
		wantAbandoned bool
	}{
		{"short turn", 60, 30, false},
		{"30min boundary all-text", 1800, 0, false},
		{"31min text-only", 1860, 0, true},
		{"31min idle-ratio just under 10%", 1801, 100, true},
		{"31min idle-ratio at 10% boundary", 1801, 181, false},
		{"1hr autonomous loop", 3600, 1200, false},
		{"9hr abandoned", 32400, 30, true},
		{"degenerate zero", 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := IsAbandonedTurn(tc.wall, tc.active)
			if got != tc.wantAbandoned {
				t.Errorf("IsAbandonedTurn(%d, %d) = %v, want %v",
					tc.wall, tc.active, got, tc.wantAbandoned)
			}
		})
	}
}
