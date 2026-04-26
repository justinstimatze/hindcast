package store

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"time"
)

// SessionMomentum tracks recent turn durations within a CC session.
// Sessions have inertia: short turns cluster with short turns, long
// with long. Within-session correlation is a predictor the
// project×task×size buckets don't capture.
type SessionMomentum struct {
	SessionID string    `json:"session_id"`
	Updated   time.Time `json:"updated"`
	WallTurns []int     `json:"wall_turns"`   // last N wall_seconds
	ActiveTurns []int   `json:"active_turns"` // parallel
}

// momentumMaxTurns caps the rolling window. Five is plenty for
// "current session vibe" without amplifying stale turns.
const momentumMaxTurns = 5

func SessionMomentumPath(sessionID string) (string, error) {
	clean := safeIdent(sessionID)
	if clean == "" {
		return "", nil
	}
	dir, err := TmpDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "session-"+clean+".json"), nil
}

func LoadSessionMomentum(sessionID string) (*SessionMomentum, error) {
	path, err := SessionMomentumPath(sessionID)
	if err != nil || path == "" {
		return &SessionMomentum{SessionID: sessionID}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &SessionMomentum{SessionID: sessionID}, nil
		}
		return nil, err
	}
	var sm SessionMomentum
	if err := json.Unmarshal(data, &sm); err != nil {
		return &SessionMomentum{SessionID: sessionID}, nil
	}
	return &sm, nil
}

// AppendTurn updates session momentum after a Stop, capping the window
// at momentumMaxTurns. Writes via temp+rename so a crashed record
// subprocess can't leave a partial file.
func (sm *SessionMomentum) AppendTurn(wall, active int) {
	sm.WallTurns = append(sm.WallTurns, wall)
	sm.ActiveTurns = append(sm.ActiveTurns, active)
	if len(sm.WallTurns) > momentumMaxTurns {
		sm.WallTurns = sm.WallTurns[len(sm.WallTurns)-momentumMaxTurns:]
	}
	if len(sm.ActiveTurns) > momentumMaxTurns {
		sm.ActiveTurns = sm.ActiveTurns[len(sm.ActiveTurns)-momentumMaxTurns:]
	}
	sm.Updated = time.Now().UTC()
}

func (sm *SessionMomentum) Save() error {
	path, err := SessionMomentumPath(sm.SessionID)
	if err != nil || path == "" {
		return nil
	}
	data, err := json.Marshal(sm)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".session-*.json")
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

// WallMedian returns the median of recent turn wall seconds; 0 if
// no history.
func (sm *SessionMomentum) WallMedian() int {
	return intMedian(sm.WallTurns)
}

func (sm *SessionMomentum) ActiveMedian() int {
	return intMedian(sm.ActiveTurns)
}

func intMedian(xs []int) int {
	if len(xs) == 0 {
		return 0
	}
	copy := append([]int(nil), xs...)
	sortInts(copy)
	mid := len(copy) / 2
	if len(copy)%2 == 0 {
		return (copy[mid-1] + copy[mid]) / 2
	}
	return copy[mid]
}

func sortInts(xs []int) {
	// Simple insertion sort — input is always <= momentumMaxTurns
	// so O(n²) is fine and avoids pulling in sort.Ints noise.
	for i := 1; i < len(xs); i++ {
		for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
			xs[j-1], xs[j] = xs[j], xs[j-1]
		}
	}
}

// SweepSessionMomentum deletes session files older than maxAge. Called
// alongside pending-file TTL sweep.
func SweepSessionMomentum(maxAge time.Duration) error {
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
		if e.IsDir() {
			continue
		}
		if !hasPrefix(e.Name(), "session-") {
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

func hasPrefix(s, p string) bool {
	if len(s) < len(p) {
		return false
	}
	return s[:len(p)] == p
}
