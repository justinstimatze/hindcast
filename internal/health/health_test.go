package health

import (
	"math"
	"testing"
	"time"

	"github.com/justinstimatze/hindcast/internal/store"
)

// Empty input should produce a no-data verdict with TunedSimThreshold
// at +Inf — the gate stays closed and the predict path falls through.
// Catches a regression where Compute returns the zero Health value
// (TunedSimThreshold=0) which would silently open the gate.
func TestComputeEmpty(t *testing.T) {
	h := Compute(map[string][]store.Record{}, nil, 5)
	if h == nil {
		t.Fatal("Compute returned nil on empty input")
	}
	if !math.IsInf(h.TunedSimThreshold, 1) {
		t.Errorf("expected +Inf threshold on empty data, got %v", h.TunedSimThreshold)
	}
	if h.Verdict == "" {
		t.Error("Verdict must be non-empty even on empty input")
	}
}

// On synthetic data where every project's durations are flat (no learnable
// signal beyond the bucket median), the regressor cannot beat the ladder.
// RegressorWinner must be "none". Catches a regression where the lift
// gate flips inequality direction or drops the floor.
func TestComputeFlatDataKeepsRegressorDormant(t *testing.T) {
	// Need to clear regressor.MinTrainRecords (100) on the train half:
	// 7 projects × 40 records = 280; first-half = 140; minus 7×warmup = 105.
	by := map[string][]store.Record{}
	for proj := 0; proj < 7; proj++ {
		var recs []store.Record
		for i := 0; i < 40; i++ {
			recs = append(recs, store.Record{
				TS:                  time.Date(2026, 1, 1, 0, 0, i, 0, time.UTC),
				TaskType:            "refactor",
				PromptChars:         100,
				PromptTokens:        []uint64{uint64(proj*1000 + i)},
				WallSeconds:         100, // flat: nothing to learn
				ClaudeActiveSeconds: 50,
				ToolCalls:           map[string]int{"Edit": 1},
			})
		}
		by["p"+string(rune('A'+proj))] = recs
	}
	h := Compute(by, nil, 5)
	if h.RegressorWinner != "none" {
		t.Errorf("expected RegressorWinner=none on flat data, got %q (lift=%.2f)", h.RegressorWinner, h.RegressorLiftVsLadder)
	}
	if h.NHeldOut == 0 {
		t.Error("expected NHeldOut > 0 with 120 records across 3 projects")
	}
}
