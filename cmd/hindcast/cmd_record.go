package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/health"
	"github.com/justinstimatze/hindcast/internal/hook"
	"github.com/justinstimatze/hindcast/internal/regressor"
	"github.com/justinstimatze/hindcast/internal/sizes"
	"github.com/justinstimatze/hindcast/internal/store"
	"github.com/justinstimatze/hindcast/internal/tags"
	"github.com/justinstimatze/hindcast/internal/transcript"
)

type recordInput struct {
	SessionID      string `json:"session_id"`
	CWD            string `json:"cwd"`
	TranscriptPath string `json:"transcript_path"`
	PermissionMode string `json:"permission_mode"`
	HookEventName  string `json:"hook_event_name"`
}

const recordTimeout = 30 * time.Second

// cmdRecord dispatches: parent forwards stdin to a temp file and spawns
// a detached worker; worker reads the temp file and does the work.
// HINDCAST_WORKER=1 marks the worker invocation.
func cmdRecord() {
	if os.Getenv("HINDCAST_SKIP") == "1" {
		return
	}
	if os.Getenv("HINDCAST_WORKER") == "1" {
		recordWorker()
		return
	}
	recordParent()
}

func recordParent() {
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, os.Stdin); err != nil {
		return
	}
	if buf.Len() == 0 {
		return
	}

	tmpDir, err := store.TmpDir()
	if err != nil {
		return
	}
	tmpf, err := os.CreateTemp(tmpDir, "record-*.json")
	if err != nil {
		return
	}
	tmpPath := tmpf.Name()
	if _, err := tmpf.Write(buf.Bytes()); err != nil {
		tmpf.Close()
		os.Remove(tmpPath)
		return
	}
	tmpf.Close()

	exe, err := os.Executable()
	if err != nil {
		os.Remove(tmpPath)
		return
	}
	cmd := exec.Command(exe, "record", tmpPath)
	cmd.Env = append(os.Environ(), "HINDCAST_WORKER=1")
	cmd.Stdin = nil
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.SysProcAttr = detachAttrs()
	if err := cmd.Start(); err != nil {
		os.Remove(tmpPath)
		return
	}
}

func recordWorker() {
	if len(os.Args) < 3 {
		return
	}
	inputPath := os.Args[2]
	defer os.Remove(inputPath)

	done := make(chan struct{})
	go func() {
		defer func() {
			if r := recover(); r != nil {
				hook.Logf("record", "worker goroutine PANIC %v", r)
			}
			close(done)
		}()
		data, err := os.ReadFile(inputPath)
		if err != nil {
			return
		}
		var in recordInput
		if err := json.Unmarshal(data, &in); err != nil {
			hook.Logf("record", "worker decode: %s", err)
			return
		}
		doRecord(in)
	}()
	select {
	case <-done:
	case <-time.After(recordTimeout):
		hook.Logf("record", "worker exceeded %s timeout", recordTimeout)
	}
}

func doRecord(in recordInput) {
	if in.SessionID == "" {
		return
	}

	_ = store.SweepPending(6 * time.Hour)
	_ = store.SweepSessionMomentum(24 * time.Hour)

	// v0.6.2: Stop fires once per completed turn. Find the latest
	// completed turn in the transcript, then match it to the pending
	// whose StartTS is closest to the turn's prompt timestamp. This
	// replaces the single-pending-per-session design where rapid-fire
	// follow-up prompts overwrote the previous turn's pending before
	// Stop could read it, silently dropping completed turns.
	turns, err := transcript.ParseTail(in.TranscriptPath, 2*1024*1024, time.Now().Add(-6*time.Hour))
	if err != nil {
		hook.Logf("record", "parse transcript: %s", err)
		return
	}
	// Find the most recent completed turn (has both prompt and assistant).
	var turn transcript.Turn
	found := false
	for i := len(turns) - 1; i >= 0; i-- {
		if turns[i].WallSeconds() > 0 {
			turn = turns[i]
			found = true
			break
		}
	}
	if !found {
		// No completed turn — nothing to record. (Common: interrupted
		// prompts, escape-canceled responses, multi-stop turns.)
		return
	}

	// Match this turn to one of the session's outstanding pendings.
	// Pick the pending whose StartTS is closest to (and not after) the
	// turn's PromptTS, within a generous tolerance for clock skew
	// between the hook's time.Now() and CC's transcript timestamp.
	pendings, _ := store.ListPendingForSession(in.SessionID)
	var matched *store.PendingFile
	const matchToleranceSec = 10
	for i := range pendings {
		delta := turn.PromptTS.Sub(pendings[i].StartTS).Seconds()
		// pending's StartTS should be at-or-before the turn's PromptTS
		// (the pending is written by UserPromptSubmit just before CC
		// stamps the user message). Allow a small negative slop for
		// clock skew. Reject pendings that are far in the future.
		if delta < -matchToleranceSec || delta > 6*60*60 {
			continue
		}
		if matched == nil || math.Abs(delta) < math.Abs(turn.PromptTS.Sub(matched.StartTS).Seconds()) {
			matched = &pendings[i]
		}
	}

	if matched == nil {
		// No pending matches this turn — fall through to the same
		// transcript-only fallback used when a project shadows the
		// global hook. The fallback also sweeps stale pendings via
		// the marker mechanism.
		doRecordFallback(in)
		return
	}

	pend, err := store.ReadPending(matched.Path)
	if err != nil {
		hook.Logf("record", "read pending: %s", err)
		return
	}
	// Consume the matched pending. Any older unmatched pendings stay
	// in place — they correspond to canceled/interrupted prompts and
	// will get cleaned up by the SweepPending TTL (6h).
	defer os.Remove(matched.Path)

	// Claude's self-reported estimate (if any) — written by the
	// hindcast_estimate MCP tool to a per-session file. Read and unlink.
	estWall, estActive := 0, 0
	if estPath, err := store.EstimatePath(pend.SessionID); err == nil {
		if data, err := os.ReadFile(estPath); err == nil {
			var est struct {
				WallSeconds   int `json:"wall_seconds"`
				ActiveSeconds int `json:"active_seconds"`
			}
			if err := json.Unmarshal(data, &est); err == nil {
				estWall = est.WallSeconds
				estActive = est.ActiveSeconds
			}
			_ = os.Remove(estPath)
		}
	}

	toolCount := 0
	for _, n := range turn.ToolCalls {
		toolCount += n
	}

	r := store.Record{
		TS:                   time.Now().UTC(),
		SessionID:            pend.SessionID,
		ProjectHash:          pend.ProjectHash,
		Model:                turn.Model,
		PermissionMode:       pend.PermissionMode,
		TaskType:             pend.TaskType,
		SizeBucket:           string(sizes.Classify(turn.FilesTouched, toolCount)),
		WallSeconds:          turn.WallSeconds(),
		ClaudeActiveSeconds:  turn.ActiveSeconds,
		PromptChars:          pend.PromptChars,
		PromptTokens:         pend.PromptTokens,
		ResponseChars:        turn.ResponseChars,
		ToolCalls:            turn.ToolCalls,
		FilesTouched:         turn.FilesTouched,
		Arm:                  pend.Arm,
		ClaudeEstimateWall:   estWall,
		ClaudeEstimateActive: estActive,
	}

	lockPath, err := store.LockPath(pend.ProjectHash)
	if err != nil {
		hook.Logf("record", "lock path: %s", err)
		return
	}
	lock, err := store.AcquireLock(lockPath)
	if err != nil {
		hook.Logf("record", "acquire lock: %s", err)
		return
	}
	defer lock.Release()

	logPath, err := store.ProjectLogPath(pend.ProjectHash)
	if err != nil {
		hook.Logf("record", "log path: %s", err)
		return
	}
	if err := store.AppendRecord(logPath, r); err != nil {
		hook.Logf("record", "append: %s", err)
		return
	}

	// After appending, check whether the log rotated and if so, rebuild
	// the BM25 index from the records still in the active log (which
	// includes the just-appended record) so index and log stay in sync.
	// When rotation rebuilt, the just-appended record is already in the
	// index — skip the subsequent updateBM25 call to avoid double-
	// counting the turn.
	rotated, err := rotateBM25IfNeeded(logPath, pend.ProjectHash)
	if err != nil {
		hook.Logf("record", "bm25 rotate: %s", err)
	}

	if !rotated {
		if err := updateBM25(pend.ProjectHash, pend.PromptTokens, r, toolCount); err != nil {
			hook.Logf("record", "bm25 update: %s", err)
		}
	}
	if err := updateSketch(r); err != nil {
		hook.Logf("record", "sketch update: %s", err)
	}

	hook.Logf("record",
		"recorded: session=%s project=%s task=%s size=%s wall=%ds active=%ds tools=%d files=%d",
		r.SessionID, r.ProjectHash, r.TaskType, r.SizeBucket,
		r.WallSeconds, r.ClaudeActiveSeconds, toolCount, r.FilesTouched)

	// Reconciliation log: one line per turn with predicted vs actual.
	// This is the primary quality signal post-pivot — replaces eval-api
	// as "does hindcast's predictor help?" because it measures the
	// predictor directly without involving Claude's estimation loop.
	// Reconcile only when both sides are non-zero. Skipped sessions
	// (HINDCAST_SKIP=1), interrupts before any tool ran, and other
	// pathological turns produce actual_wall=0 — counting them in the
	// accuracy log adds noise without signal.
	if (pend.PredictedWall > 0 || pend.PredictedActive > 0) && r.WallSeconds > 0 {
		appendAccuracyLine(pend.ProjectHash, pend, r)
	}

	// Auto-tune: refresh predictor health if stale. Cheap (~500ms),
	// runs at most once per hour. Replaces "user remembers to run
	// hindcast tune" with automatic background refresh from the Stop
	// hook (which is already async and detached). The tuned threshold
	// can then be read by future UserPromptSubmit hooks for gating.
	if shouldRetune() {
		go func() {
			defer func() {
				if rec := recover(); rec != nil {
					hook.Logf("record", "auto-tune PANIC %v", rec)
				}
			}()
			byProject, err := loadRecordsByProjectSimple()
			if err != nil {
				return
			}
			sk, _ := store.LoadSketch()
			h := health.Compute(byProject, sk, 20)
			// Refresh persisted regressor models on the same cadence so
			// that when h.RegressorWinner is non-"none" the file the
			// predict path loads is current. Failures are non-fatal —
			// predict.computePrediction soft-falls-through to the ladder.
			if m, err := regressor.Train(byProject, 20); err == nil {
				_ = m.Save()
			}
			if lm, err := regressor.TrainLinearFromRecords(byProject, 20, 1.0); err == nil {
				_ = lm.Save()
			}
			_ = h.Save()
			hook.Logf("record", "auto-tune: threshold=%.2f knn-malr=%.2f winner=%s lift=%.2fx n=%d",
				h.TunedSimThreshold, h.KNNMALRAtThreshold, h.RegressorWinner, h.RegressorLiftVsLadder, h.NPredictions)
		}()
	}

	// Update per-session momentum tracker so next UserPromptSubmit in
	// the same session can use recent-turn context as an additional
	// predictor alongside the project×task bucket.
	if sm, err := store.LoadSessionMomentum(r.SessionID); err == nil {
		sm.AppendTurn(r.WallSeconds, r.ClaudeActiveSeconds)
		_ = sm.Save()
	}
}

// appendAccuracyLine writes one JSONL line per turn to the project's
// accuracy.jsonl so `hindcast show --accuracy` can compute rolling
// predictor MALR. Best-effort — soft-fails on IO errors.
func appendAccuracyLine(hash string, pend store.PendingTurn, r store.Record) {
	path, err := store.AccuracyLogPath(hash)
	if err != nil {
		return
	}
	row := map[string]any{
		"ts":               r.TS.Format(time.RFC3339),
		"session_id":       r.SessionID,
		"task_type":        r.TaskType,
		"size_bucket":      r.SizeBucket,
		"predicted_wall":   pend.PredictedWall,
		"predicted_active": pend.PredictedActive,
		"actual_wall":      r.WallSeconds,
		"actual_active":    r.ClaudeActiveSeconds,
		"source":           pend.PredictionSrc,
	}
	// v0.6.1: capture band fields so band-hit-rate is computable
	// post-hoc. Only include when populated, to keep older readers
	// tolerant of the schema change.
	if pend.PredictedWallP25 > 0 {
		row["predicted_wall_p25"] = pend.PredictedWallP25
	}
	if pend.PredictedWallP75 > 0 {
		row["predicted_wall_p75"] = pend.PredictedWallP75
	}
	if pend.PredictedWallP10 > 0 {
		row["predicted_wall_p10"] = pend.PredictedWallP10
	}
	if pend.PredictedWallP90 > 0 {
		row["predicted_wall_p90"] = pend.PredictedWallP90
	}
	if pend.PredictedMaxSim > 0 {
		row["predicted_max_sim"] = pend.PredictedMaxSim
	}
	if pend.VarianceGated {
		row["variance_gated"] = true
	}
	data, err := json.Marshal(row)
	if err != nil {
		return
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintln(f, string(data))
}

func updateBM25(hash string, promptHashes []uint64, r store.Record, toolCount int) error {
	if len(promptHashes) == 0 {
		return nil
	}
	path, err := store.ProjectBM25Path(hash)
	if err != nil {
		return err
	}
	// Auto-heal corrupt indexes: if Load returns a decode error (e.g. a
	// crash mid-write left a partial gob, or the schema bumped past
	// what's persisted), start fresh. Without this the index never
	// recovers — every subsequent updateBM25 returns the same decode
	// error and the kNN tier silently goes dark for that project.
	idx, err := bm25.Load(path)
	if err != nil {
		hook.Logf("record", "bm25 load %s failed (%s); starting fresh index", path, err)
		idx = bm25.New()
	}
	idx.Add(promptHashes, bm25.Doc{
		ActiveSeconds: r.ClaudeActiveSeconds,
		WallSeconds:   r.WallSeconds,
		TaskType:      r.TaskType,
		SizeBucket:    r.SizeBucket,
		ToolCount:     toolCount,
		FilesTouched:  r.FilesTouched,
		TS:            r.TS,
	})
	return idx.Save(path)
}

// updateSketch takes a global sketch lock around the load-modify-save
// sequence. The per-project lock doesn't cover cross-project races — two
// Stops on different projects would both read old sketch and both write
// new sketch, losing one sample.
// rotateBM25IfNeeded checks if the active log was just rotated out (file
// size back to near-zero after AppendRecord) and if so, rebuilds the
// BM25 index from the (now trimmed) record set, which already includes
// the just-appended record. Returns (true, nil) when a rebuild
// happened — the caller should NOT then call updateBM25 with the same
// record, or the index would double-count it.
//
// Returns (false, nil) on the cheap fast-path (no rotation detected,
// no rebuild needed).
func rotateBM25IfNeeded(logPath, hash string) (bool, error) {
	stat, err := os.Stat(logPath)
	if err != nil {
		return false, nil
	}
	// Rotation happened if the rotated segment (.1) exists AND is
	// younger than the active log's modification time. Heuristic: if
	// .1 exists and active log is small, we just rotated.
	rotatedPath := logPath + ".1"
	rotStat, err := os.Stat(rotatedPath)
	if err != nil || rotStat.Size() < store.LogRotateSize {
		return false, nil
	}
	if stat.Size() > store.MaxRecordSize*2 {
		return false, nil // active log not freshly rotated
	}
	// Rebuild BM25 from current active log's records.
	records, err := store.ReadRecentRecords(logPath, 500)
	if err != nil {
		return false, err
	}
	bm25Path, err := store.ProjectBM25Path(hash)
	if err != nil {
		return false, err
	}
	// Check if rebuild is actually warranted: compare index's doc count
	// to active log's record count. Only rebuild on significant drift.
	idx, _ := bm25.Load(bm25Path)
	if idx != nil && len(idx.Docs) <= len(records)*2 {
		return false, nil
	}
	fresh := bm25.New()
	for _, r := range records {
		if len(r.PromptTokens) == 0 {
			continue
		}
		toolCount := 0
		for _, c := range r.ToolCalls {
			toolCount += c
		}
		fresh.Add(r.PromptTokens, bm25.Doc{
			ActiveSeconds: r.ClaudeActiveSeconds,
			WallSeconds:   r.WallSeconds,
			TaskType:      r.TaskType,
			SizeBucket:    r.SizeBucket,
			ToolCount:     toolCount,
			FilesTouched:  r.FilesTouched,
			TS:            r.TS,
		})
	}
	if err := fresh.Save(bm25Path); err != nil {
		return false, err
	}
	return true, nil
}

func updateSketch(r store.Record) error {
	lockPath, err := store.GlobalSketchLockPath()
	if err != nil {
		return err
	}
	lock, err := store.AcquireLock(lockPath)
	if err != nil {
		return err
	}
	defer lock.Release()

	s, err := store.LoadSketch()
	if err != nil {
		return err
	}
	s.Add(r.WallSeconds, r.ClaudeActiveSeconds)
	return s.Save()
}

// shouldRetune returns true if health.json is missing, malformed, or
// older than the staleness threshold. Cheap stat-only check.
func shouldRetune() bool {
	const tuneStaleness = time.Hour
	path, err := health.HealthPath()
	if err != nil {
		return false
	}
	stat, err := os.Stat(path)
	if err != nil {
		return true // missing → tune
	}
	return time.Since(stat.ModTime()) > tuneStaleness
}

// loadRecordsByProjectSimple is the same as loadRecordsByProject from
// cmd_verify but pulled inline so the auto-tune path doesn't depend on
// flag-parsing code. Reads all per-project JSONLs into memory.
func loadRecordsByProjectSimple() (map[string][]store.Record, error) {
	projDir, err := store.ProjectsDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(projDir)
	if err != nil {
		return nil, err
	}
	out := map[string][]store.Record{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if len(name) < 7 || name[len(name)-6:] != ".jsonl" {
			continue
		}
		path := projDir + "/" + name
		recs, err := store.ReadRecentRecords(path, 100000)
		if err != nil {
			continue
		}
		hash := name[:len(name)-6]
		out[hash] = recs
	}
	return out, nil
}

// doRecordFallback is called when no pending file exists, which happens
// when the project's local .claude/settings.json defines UserPromptSubmit
// hooks and shadows the global hindcast pending hook. It reconstructs
// what it needs from the transcript and the record input directly.
//
// No accuracy-log entry is written: there's no prediction to compare
// against because hindcast pending never ran for this turn.
func doRecordFallback(in recordInput) {
	if in.SessionID == "" || in.TranscriptPath == "" {
		return
	}

	_ = store.SweepPending(6 * time.Hour)
	_ = store.SweepSessionMomentum(24 * time.Hour)

	project := store.ResolveProject(in.CWD)
	hash := store.ProjectHash(project)
	arm := store.SessionArm(in.SessionID, store.ControlPctFromEnv())

	// Use the fallback marker to find where we left off in this session,
	// so we don't re-record a turn that a prior Stop already captured.
	since := time.Now().Add(-2 * time.Hour)
	markerPath := fallbackMarkerPath(in.SessionID)
	if markerPath != "" {
		if data, err := os.ReadFile(markerPath); err == nil {
			if ts, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(string(data))); err == nil {
				since = ts.Add(time.Millisecond) // strictly after last recorded
			}
		}
	}

	turns, err := transcript.ParseTail(in.TranscriptPath, 4*1024*1024, since)
	if err != nil {
		hook.Logf("record", "fallback parse transcript: %s", err)
		return
	}
	if len(turns) == 0 {
		return
	}

	turn := turns[len(turns)-1]

	// Skip degenerate turns (no wall time means the assistant never responded).
	if turn.WallSeconds() == 0 {
		return
	}

	var promptTokens []uint64
	if turn.PromptText != "" {
		if salt, err := store.GetSalt(); err == nil {
			promptTokens = bm25.HashTokens(turn.PromptText, salt)
		}
	}

	toolCount := 0
	for _, n := range turn.ToolCalls {
		toolCount += n
	}

	r := store.Record{
		TS:                  time.Now().UTC(),
		SessionID:           in.SessionID,
		ProjectHash:         hash,
		Model:               turn.Model,
		PermissionMode:      in.PermissionMode,
		TaskType:            string(tags.Classify(firstLine(turn.PromptText))),
		SizeBucket:          string(sizes.Classify(turn.FilesTouched, toolCount)),
		WallSeconds:         turn.WallSeconds(),
		ClaudeActiveSeconds: turn.ActiveSeconds,
		PromptChars:         len(turn.PromptText),
		PromptTokens:        promptTokens,
		ResponseChars:       turn.ResponseChars,
		ToolCalls:           turn.ToolCalls,
		FilesTouched:        turn.FilesTouched,
		Arm:                 arm,
	}

	lockPath, err := store.LockPath(hash)
	if err != nil {
		hook.Logf("record", "fallback lock path: %s", err)
		return
	}
	lock, err := store.AcquireLock(lockPath)
	if err != nil {
		hook.Logf("record", "fallback acquire lock: %s", err)
		return
	}
	defer lock.Release()

	// Advance the marker BEFORE the record append. Failure modes:
	//   1. marker write fails → we exit; next call retries this turn.
	//   2. marker write succeeds, append fails → we lose this record but
	//      never duplicate it (next call's `since.Add(1ms)` filter
	//      excludes the same turn).
	//   3. all succeed → normal case.
	// If we instead wrote marker AFTER append, a crash between append
	// and marker write would re-record the turn next time, polluting
	// the predictor with a duplicate. Lost-record is preferable to
	// double-record because records are read-only inputs to a median —
	// duplicates double-weight one observation, while a miss is just
	// one fewer.
	if markerPath != "" {
		if err := os.WriteFile(markerPath, []byte(turn.PromptTS.Format(time.RFC3339Nano)), 0600); err != nil {
			hook.Logf("record", "fallback marker write: %s", err)
			return
		}
	}

	logPath, err := store.ProjectLogPath(hash)
	if err != nil {
		hook.Logf("record", "fallback log path: %s", err)
		return
	}
	if err := store.AppendRecord(logPath, r); err != nil {
		hook.Logf("record", "fallback append: %s", err)
		return
	}

	rotated, err := rotateBM25IfNeeded(logPath, hash)
	if err != nil {
		hook.Logf("record", "fallback bm25 rotate: %s", err)
	}
	if !rotated {
		if err := updateBM25(hash, promptTokens, r, toolCount); err != nil {
			hook.Logf("record", "fallback bm25 update: %s", err)
		}
	}
	if err := updateSketch(r); err != nil {
		hook.Logf("record", "fallback sketch update: %s", err)
	}

	hook.Logf("record",
		"fallback recorded: session=%s project=%s task=%s wall=%ds active=%ds tools=%d files=%d",
		r.SessionID, r.ProjectHash, r.TaskType,
		r.WallSeconds, r.ClaudeActiveSeconds, toolCount, r.FilesTouched)

	if sm, err := store.LoadSessionMomentum(r.SessionID); err == nil {
		sm.AppendTurn(r.WallSeconds, r.ClaudeActiveSeconds)
		_ = sm.Save()
	}

	if shouldRetune() {
		go func() {
			defer func() {
				if rec := recover(); rec != nil {
					hook.Logf("record", "fallback auto-tune PANIC %v", rec)
				}
			}()
			byProject, err := loadRecordsByProjectSimple()
			if err != nil {
				return
			}
			sk, _ := store.LoadSketch()
			h := health.Compute(byProject, sk, 20)
			if m, err := regressor.Train(byProject, 20); err == nil {
				_ = m.Save()
			}
			if lm, err := regressor.TrainLinearFromRecords(byProject, 20, 1.0); err == nil {
				_ = lm.Save()
			}
			_ = h.Save()
		}()
	}
}

// fallbackMarkerPath returns the path to the per-session marker file
// that tracks the PromptTS of the last turn recorded via doRecordFallback.
// Returns "" on error (caller treats as no prior marker).
func fallbackMarkerPath(sessionID string) string {
	dir, err := store.SessionDirPath(sessionID)
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "fallback-marker")
}
