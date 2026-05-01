// Package store owns the on-disk layout: per-project JSONL append
// (4KB truncation, O_APPEND atomicity), global numeric sketch
// (write-temp-then-atomic-rename), per-project advisory lock with
// stale-PID detection, 10MB log rotation, and the per-install BM25
// salt.
//
// Privacy invariant: Records carry no prompt text. PendingTurn carries
// salted token hashes (never plaintext) across the UserPromptSubmit →
// Stop gap. The on-disk surface area contains only numeric stats,
// deterministic-derived labels, and opaque hashes.
package store

import (
	"bufio"
	"crypto/md5"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Record is the persisted shape of one UserPromptSubmit → Stop turn.
// Contains no prompt text — only stats, deterministic labels, and
// salt-hashed prompt tokens (same privacy invariant as the BM25 index).
type Record struct {
	SchemaVersion       int            `json:"schema_version"`
	TS                  time.Time      `json:"ts"`
	SessionID           string         `json:"session_id"`
	ProjectHash         string         `json:"project_hash"`
	Model               string         `json:"model"`
	PermissionMode      string         `json:"permission_mode"`
	TaskType            string         `json:"task_type"`
	SizeBucket          string         `json:"size_bucket"`
	WallSeconds         int            `json:"wall_seconds"`
	ClaudeActiveSeconds int            `json:"claude_active_seconds"`
	PromptChars         int            `json:"prompt_chars"`
	PromptTokens        []uint64       `json:"prompt_tokens,omitempty"`
	ResponseChars       int            `json:"response_chars"`
	ToolCalls           map[string]int `json:"tool_calls,omitempty"`
	FilesTouched        int            `json:"files_touched"`
	Interrupted         bool           `json:"interrupted,omitempty"`
	Partial             bool           `json:"partial,omitempty"`

	// A/B calibration fields (tier 3).
	Arm                  string `json:"arm,omitempty"`                    // "control" | "treatment"
	ClaudeEstimateWall   int    `json:"claude_estimate_wall,omitempty"`   // seconds, parsed from <!-- hindcast-wall: ... -->
	ClaudeEstimateActive int    `json:"claude_estimate_active,omitempty"` // seconds, parsed from <!-- hindcast-active: ... -->
}

const (
	SchemaVersion   = 1
	MaxRecordSize   = 4096
	LogRotateSize   = 10 * 1024 * 1024
	SketchMaxWindow = 1000
	SaltSize        = 32
)

// --- Paths ---

func HindcastDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".claude", "hindcast"), nil
}

func ProjectsDir() (string, error) {
	root, err := HindcastDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "projects")
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

func ProjectHash(project string) string {
	sum := md5.Sum([]byte(project))
	return hex.EncodeToString(sum[:])[:8]
}

func ResolveProject(cwd string) string {
	if data, err := os.ReadFile(filepath.Join(cwd, ".hindcast-project")); err == nil {
		if name := strings.TrimSpace(string(data)); name != "" {
			return name
		}
	}
	return filepath.Base(cwd)
}

func ProjectLogPath(hash string) (string, error) {
	dir, err := ProjectsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, hash+".jsonl"), nil
}

func ProjectBM25Path(hash string) (string, error) {
	dir, err := ProjectsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, hash+".bm25.gob"), nil
}

func TmpDir() (string, error) {
	dir := filepath.Join(os.TempDir(), "hindcast")
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

// safeIdent returns s if it contains only safe filename characters,
// otherwise empty. Session IDs and project hashes become filename
// components and must not contain path separators or traversal sequences.
func safeIdent(s string) string {
	if s == "" {
		return ""
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '-' || r == '_':
		default:
			return ""
		}
	}
	return s
}

// PendingPath returns the per-turn pending file path. v0.6.2: filename
// now embeds the start timestamp's nanosecond stamp so multiple in-flight
// turns in the same session don't overwrite each other. If startTS is
// zero, the legacy single-file shape is returned (read-side compat).
func PendingPath(sessionID string, startTS time.Time) (string, error) {
	clean := safeIdent(sessionID)
	if clean == "" {
		return "", fmt.Errorf("invalid session id")
	}
	dir, err := TmpDir()
	if err != nil {
		return "", err
	}
	if startTS.IsZero() {
		return filepath.Join(dir, "pending-"+clean+".json"), nil
	}
	return filepath.Join(dir, fmt.Sprintf("pending-%s-%d.json", clean, startTS.UnixNano())), nil
}

// PendingFile pairs a pending file's path with the StartTS read from it,
// so callers can sort and consume pendings in chronological order.
type PendingFile struct {
	Path    string
	StartTS time.Time
}

// ListPendingForSession returns all pending files for the given session
// (both legacy single-file and v0.6.2 timestamped form), parsed and
// sorted oldest-first by StartTS. Corrupt files are skipped silently —
// the SweepPending TTL cleans them up.
//
// The Stop hook uses this to match each just-completed turn to the
// pending whose StartTS is closest to the turn's prompt timestamp,
// rather than relying on a single-file race-prone design where the
// next prompt's UserPromptSubmit overwrites the prior turn's pending
// before Stop can consume it.
func ListPendingForSession(sessionID string) ([]PendingFile, error) {
	clean := safeIdent(sessionID)
	if clean == "" {
		return nil, fmt.Errorf("invalid session id")
	}
	dir, err := TmpDir()
	if err != nil {
		return nil, err
	}
	matches, err := filepath.Glob(filepath.Join(dir, "pending-"+clean+"*.json"))
	if err != nil {
		return nil, err
	}
	out := make([]PendingFile, 0, len(matches))
	for _, m := range matches {
		p, err := ReadPending(m)
		if err != nil {
			continue
		}
		out = append(out, PendingFile{Path: m, StartTS: p.StartTS})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].StartTS.Before(out[j].StartTS) })
	return out, nil
}

func LockPath(hash string) (string, error) {
	clean := safeIdent(hash)
	if clean == "" {
		return "", fmt.Errorf("invalid project hash")
	}
	dir, err := TmpDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "lock-"+clean), nil
}

// EstimatePath is where the hindcast_estimate MCP tool stashes Claude's
// self-reported estimate for a session. Stop hook reads and unlinks.
func EstimatePath(sessionID string) (string, error) {
	clean := safeIdent(sessionID)
	if clean == "" {
		return "", fmt.Errorf("invalid session id")
	}
	dir, err := TmpDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "estimate-"+clean+".json"), nil
}

// SessionDirPath returns the per-session directory under
// HindcastDir/sessions/<id>/, creating it if missing. Used by the
// Stop-hook fallback marker.
func SessionDirPath(sessionID string) (string, error) {
	clean := safeIdent(sessionID)
	if clean == "" {
		return "", fmt.Errorf("invalid session id")
	}
	root, err := HindcastDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "sessions", clean)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

// AccuracyLogPath is the per-project reconciliation log where the Stop
// hook writes (predicted wall, actual wall, source) lines so we can
// compute predictor MALR over time.
func AccuracyLogPath(hash string) (string, error) {
	clean := safeIdent(hash)
	if clean == "" {
		return "", fmt.Errorf("invalid project hash")
	}
	root, err := HindcastDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "projects", clean)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return filepath.Join(dir, "accuracy.jsonl"), nil
}

func GlobalSketchPath() (string, error) {
	root, err := HindcastDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", err
	}
	return filepath.Join(root, "global-sketch.json"), nil
}

// GlobalSketchLockPath returns a lock file path that guards
// load-modify-save sequences on the global sketch against races between
// Stop hooks firing concurrently on different projects (the per-project
// lock doesn't cover cross-project global state).
func GlobalSketchLockPath() (string, error) {
	dir, err := TmpDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "sketch-lock"), nil
}

func SaltPath() (string, error) {
	root, err := HindcastDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", err
	}
	return filepath.Join(root, "salt"), nil
}

// GetSalt returns the per-install BM25 salt, generating one on first use.
// A lock file guards generation against concurrent first-run hooks; the
// atomic rename ensures a crash mid-init cannot leave a short salt file
// (which would silently orphan every BM25 index).
func GetSalt() ([]byte, error) {
	path, err := SaltPath()
	if err != nil {
		return nil, err
	}
	if data, err := os.ReadFile(path); err == nil && len(data) == SaltSize {
		return data, nil
	}

	// Take the generation lock so concurrent first-run hooks don't race.
	lockPath := path + ".lock"
	lock, err := AcquireLock(lockPath)
	if err == nil {
		defer lock.Release()
		// Re-check under the lock: another process may have generated it.
		if data, err := os.ReadFile(path); err == nil && len(data) == SaltSize {
			return data, nil
		}
	}

	salt := make([]byte, SaltSize)
	if _, err := rand.Read(salt); err != nil {
		return nil, err
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".salt-*")
	if err != nil {
		return nil, err
	}
	tmpName := tmp.Name()
	cleanup := func() {
		tmp.Close()
		os.Remove(tmpName)
	}
	if err := tmp.Chmod(0600); err != nil {
		cleanup()
		return nil, err
	}
	if _, err := tmp.Write(salt); err != nil {
		cleanup()
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return nil, err
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return nil, err
	}
	return salt, nil
}

// DeleteSalt removes the salt file so the next GetSalt call regenerates
// one. Used by `hindcast rotate-salt`. Caller is responsible for
// invalidating downstream state (BM25 indexes keyed on the old salt
// become unreadable and should be deleted by the same rotation flow).
func DeleteSalt() error {
	path, err := SaltPath()
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// --- Lock ---

var ErrLocked = errors.New("store: project lock is held by a live process")

type Lock struct{ path string }

func AcquireLock(path string) (*Lock, error) {
	if l, err := tryLock(path); err == nil {
		return l, nil
	}
	if !isStaleLock(path) {
		return nil, ErrLocked
	}
	_ = os.Remove(path)
	return tryLock(path)
}

func tryLock(path string) (*Lock, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	// Format: "<pid>:<start_time>". start_time is 0 if unavailable
	// (e.g., non-Linux platforms). When the same PID later runs a new
	// process, its start_time will differ and we can tell it's not us.
	fmt.Fprintf(f, "%d:%d", os.Getpid(), processStartTime(os.Getpid()))
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return nil, err
	}
	return &Lock{path: path}, nil
}

func isStaleLock(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return true
	}
	raw := strings.TrimSpace(string(data))
	pidStr, stStr, hasST := strings.Cut(raw, ":")
	pid, err := strconv.Atoi(pidStr)
	if err != nil {
		return true
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return true
	}
	if process.Signal(syscall.Signal(0)) != nil {
		return true
	}
	// Defend against PID recycling: if our recorded start_time differs
	// from the current process's start_time, the PID now names a
	// different process than the one that took the lock.
	if hasST {
		recorded, err := strconv.ParseInt(stStr, 10, 64)
		if err == nil && recorded > 0 {
			current := processStartTime(pid)
			if current > 0 && current != recorded {
				return true
			}
		}
	}
	return false
}

// processStartTime returns the target process's start time in platform
// units, or 0 if unavailable. On Linux, parses field 22 of /proc/PID/stat
// (starttime in clock ticks since boot). On macOS / other platforms,
// returns 0 and the caller falls back to PID-only liveness — see
// SECURITY.md "Known limitations" for the bounded risk of PID-recycle
// stealing a still-live lock on darwin.
func processStartTime(pid int) int64 {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0
	}
	s := string(data)
	// /proc/PID/stat field 2 is comm in parens. comm can contain spaces
	// and parens, but the last ")" in the file is always the end of comm.
	rparen := strings.LastIndex(s, ")")
	if rparen < 0 {
		return 0
	}
	fields := strings.Fields(s[rparen+1:])
	// After comm: state, ppid, pgrp, session, tty, tpgid, flags, minflt,
	// cminflt, majflt, cmajflt, utime, stime, cutime, cstime, priority,
	// nice, num_threads, itrealvalue, starttime ... → starttime is index 19
	if len(fields) < 20 {
		return 0
	}
	v, err := strconv.ParseInt(fields[19], 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func (l *Lock) Release() error {
	if l == nil {
		return nil
	}
	return os.Remove(l.path)
}

// --- JSONL append + read ---

func AppendRecord(path string, r Record) error {
	r.SchemaVersion = SchemaVersion
	data, err := marshalCapped(r)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return maybeRotate(path, f)
}

func marshalCapped(r Record) ([]byte, error) {
	data, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if len(data) < MaxRecordSize-1 {
		return data, nil
	}
	// Over budget — drop tool_calls (largest optional field).
	r.ToolCalls = nil
	data, err = json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if len(data) < MaxRecordSize-1 {
		return data, nil
	}
	// Still over — drop prompt_tokens (next largest optional field).
	r.PromptTokens = nil
	data, err = json.Marshal(r)
	if err != nil {
		return nil, err
	}
	if len(data) < MaxRecordSize-1 {
		return data, nil
	}
	// Pathological case (Model + TaskType + SizeBucket strings overflow
	// alone is implausible, but defend anyway). Strip Model — preserves
	// the numeric fields the predictor actually uses.
	r.Model = ""
	return json.Marshal(r)
}

func maybeRotate(path string, f *os.File) error {
	stat, err := f.Stat()
	f.Close()
	if err != nil || stat.Size() < LogRotateSize {
		return nil
	}
	for i := 5; i >= 1; i-- {
		src := fmt.Sprintf("%s.%d", path, i)
		if i == 5 {
			_ = os.Remove(src)
			continue
		}
		dst := fmt.Sprintf("%s.%d", path, i+1)
		_ = os.Rename(src, dst)
	}
	return os.Rename(path, path+".1")
}

func ReadRecentRecords(path string, n int) ([]Record, error) {
	if n <= 0 {
		return nil, nil
	}
	var records []Record
	paths := append([]string{path}, rotatedPaths(path)...)
	for _, p := range paths {
		segment, err := readAllRecords(p)
		if err != nil {
			return nil, err
		}
		records = append(segment, records...)
		if len(records) >= n {
			break
		}
	}
	if len(records) > n {
		records = records[len(records)-n:]
	}
	for i, j := 0, len(records)-1; i < j; i, j = i+1, j-1 {
		records[i], records[j] = records[j], records[i]
	}
	return records, nil
}

func rotatedPaths(base string) []string {
	out := make([]string, 0, 5)
	for i := 1; i <= 5; i++ {
		out = append(out, fmt.Sprintf("%s.%d", base, i))
	}
	return out
}

func readAllRecords(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), MaxRecordSize*2)
	var out []Record
	for sc.Scan() {
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			continue
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// --- Global sketch ---

type Sketch struct {
	Wall    []int `json:"wall_seconds"`
	Active  []int `json:"active_seconds"`
	MaxSize int   `json:"max_size"`
}

func LoadSketch() (*Sketch, error) {
	path, err := GlobalSketchPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &Sketch{MaxSize: SketchMaxWindow}, nil
		}
		return nil, err
	}
	s := &Sketch{}
	if err := json.Unmarshal(data, s); err != nil {
		return nil, err
	}
	if s.MaxSize == 0 {
		s.MaxSize = SketchMaxWindow
	}
	return s, nil
}

func (s *Sketch) Add(wall, active int) {
	if s.MaxSize == 0 {
		s.MaxSize = SketchMaxWindow
	}
	s.Wall = appendBounded(s.Wall, wall, s.MaxSize)
	s.Active = appendBounded(s.Active, active, s.MaxSize)
}

func appendBounded(xs []int, v, cap int) []int {
	xs = append(xs, v)
	if len(xs) > cap {
		xs = xs[len(xs)-cap:]
	}
	return xs
}

func (s *Sketch) Save() error {
	path, err := GlobalSketchPath()
	if err != nil {
		return err
	}
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".sketch-*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() {
		tmp.Close()
		os.Remove(tmpName)
	}
	if err := tmp.Chmod(0600); err != nil {
		cleanup()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

func (s *Sketch) Percentiles() (wallMed, wallP90, activeMed, activeP90 float64, n int) {
	n = len(s.Wall)
	wallMed, wallP90 = Quantiles(s.Wall)
	activeMed, activeP90 = Quantiles(s.Active)
	return
}

// Quantiles returns median and p90 of xs via linear interpolation.
// Empty input returns (0, 0).
func Quantiles(xs []int) (median, p90 float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	sorted := make([]int, len(xs))
	copy(sorted, xs)
	sort.Ints(sorted)
	return QuantileAt(sorted, 0.5), QuantileAt(sorted, 0.9)
}

// QuantilesWide returns median, p75, and p90 — used by the SessionStart
// injection so Claude sees the shape of the right tail, not just the
// midpoint. Duration distributions are right-skewed; a wider picture
// corrects for the natural under-anchor on the median alone.
func QuantilesWide(xs []int) (median, p75, p90 float64) {
	if len(xs) == 0 {
		return 0, 0, 0
	}
	sorted := make([]int, len(xs))
	copy(sorted, xs)
	sort.Ints(sorted)
	return QuantileAt(sorted, 0.5), QuantileAt(sorted, 0.75), QuantileAt(sorted, 0.9)
}

// QuantileAt returns the q-th quantile of a pre-sorted slice.
func QuantileAt(sorted []int, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	pos := q * float64(len(sorted)-1)
	lo := int(pos)
	if lo >= len(sorted)-1 {
		return float64(sorted[len(sorted)-1])
	}
	frac := pos - float64(lo)
	return float64(sorted[lo])*(1-frac) + float64(sorted[lo+1])*frac
}

// ComputeBiasFactor returns the median wall and active
// claude_estimate/actual ratios from records matching model. Prefers
// control-arm records when available (those measure Claude's raw bias
// with no correction applied, which is the right thing to divide by
// for treatment recommendations). Falls back to all arms if too few
// control-arm samples.
//
// Used by the live bias-correction loop: hindcast_prior emits a
// bias-corrected recommendation so Claude doesn't have to do the
// math itself.
func ComputeBiasFactor(records []Record, model string) (wallFactor, activeFactor float64, n int) {
	var (
		controlWall, controlActive []float64
		allWall, allActive         []float64
	)
	for _, r := range records {
		if model != "" && r.Model != model {
			continue
		}
		if r.ClaudeEstimateWall > 0 && r.WallSeconds > 0 {
			ratio := float64(r.ClaudeEstimateWall) / float64(r.WallSeconds)
			allWall = append(allWall, ratio)
			if r.Arm == "control" {
				controlWall = append(controlWall, ratio)
			}
		}
		if r.ClaudeEstimateActive > 0 && r.ClaudeActiveSeconds > 0 {
			ratio := float64(r.ClaudeEstimateActive) / float64(r.ClaudeActiveSeconds)
			allActive = append(allActive, ratio)
			if r.Arm == "control" {
				controlActive = append(controlActive, ratio)
			}
		}
	}
	// Prefer control arm if we have enough: those are raw-bias samples,
	// not post-correction residuals, so they're the right factor to use.
	if len(controlWall) >= 10 {
		sort.Float64s(controlWall)
		sort.Float64s(controlActive)
		return median64(controlWall), median64(controlActive), len(controlWall)
	}
	sort.Float64s(allWall)
	sort.Float64s(allActive)
	return median64(allWall), median64(allActive), len(allWall)
}

func median64(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	mid := len(xs) / 2
	if len(xs)%2 == 0 {
		return (xs[mid-1] + xs[mid]) / 2
	}
	return xs[mid]
}

// --- Pending file ---

// PendingTurn bridges UserPromptSubmit → Stop. Carries the task-type
// classification (computed in-memory at submit time from the prompt),
// salted token hashes (for the Stop hook to add to the BM25 index),
// and session metadata. No plaintext prompt ever.
type PendingTurn struct {
	SessionID      string    `json:"session_id"`
	StartTS        time.Time `json:"start_ts"`
	TaskType       string    `json:"task_type"`
	PromptTokens   []uint64  `json:"prompt_tokens,omitempty"`
	PromptChars    int       `json:"prompt_chars"`
	PermissionMode string    `json:"permission_mode"`
	ProjectHash    string    `json:"project_hash"`
	CWD            string    `json:"cwd"`
	Arm            string    `json:"arm,omitempty"`

	// Prediction captured at UserPromptSubmit time so the Stop hook can
	// reconcile predicted vs actual without re-running the predictor.
	PredictedWall   int    `json:"predicted_wall,omitempty"`
	PredictedActive int    `json:"predicted_active,omitempty"`
	PredictionSrc   string `json:"prediction_src,omitempty"`

	// Band fields so the accuracy log can compute a band-hit-rate metric
	// alongside point MALR. Without these, accuracy.jsonl only measures
	// the point estimate — which the inject suppresses behind the
	// variance gate when the band is wide. The band is what Claude
	// actually sees, so it's the more meaningful target.
	PredictedWallP25 int `json:"predicted_wall_p25,omitempty"`
	PredictedWallP75 int `json:"predicted_wall_p75,omitempty"`
	// v0.6.3: wider quantiles, rendered as the band when variance gate
	// trips. Hit rate against [P10, P90] should track ~80% on calibrated
	// kNN distributions; against [P25, P75] tracks 50%.
	PredictedWallP10 int     `json:"predicted_wall_p10,omitempty"`
	PredictedWallP90 int     `json:"predicted_wall_p90,omitempty"`
	PredictedMaxSim  float64 `json:"predicted_max_sim,omitempty"`
	VarianceGated    bool    `json:"variance_gated,omitempty"` // true when inject rendered the band as headline (no point shown)
}

func WritePending(path string, p PendingTurn) error {
	data, err := json.Marshal(p)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func ReadPending(path string) (PendingTurn, error) {
	var p PendingTurn
	data, err := os.ReadFile(path)
	if err != nil {
		return p, err
	}
	err = json.Unmarshal(data, &p)
	return p, err
}

// --- A/B arm assignment ---

const (
	ArmControl        = "control"
	ArmTreatment      = "treatment"
	DefaultControlPct = 10
)

// SessionArm maps a session id to an arm via stable hash. controlPct
// is 0-100; <=0 forces everyone to treatment, >=100 forces control.
func SessionArm(sessionID string, controlPct int) string {
	if controlPct <= 0 {
		return ArmTreatment
	}
	if controlPct >= 100 {
		return ArmControl
	}
	sum := md5.Sum([]byte(sessionID))
	// Use first two bytes for 16-bit granularity; % 100 gives a bucket.
	bucket := (int(sum[0])<<8 | int(sum[1])) % 100
	if bucket < controlPct {
		return ArmControl
	}
	return ArmTreatment
}

// ControlPctFromEnv reads HINDCAST_CONTROL_PCT, defaulting to
// DefaultControlPct (10%). Out-of-range values fall back to the default.
func ControlPctFromEnv() int {
	v := os.Getenv("HINDCAST_CONTROL_PCT")
	if v == "" {
		return DefaultControlPct
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 || n > 100 {
		return DefaultControlPct
	}
	return n
}

func SweepPending(maxAge time.Duration) error {
	dir, err := TmpDir()
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	cutoff := time.Now().Add(-maxAge)
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "pending-") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			_ = os.Remove(filepath.Join(dir, e.Name()))
		}
	}
	return nil
}
