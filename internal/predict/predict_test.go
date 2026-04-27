package predict

import (
	"testing"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/store"
)

// kNN tier fires when ≥3 neighbors are above the noise floor and returns
// the weighted median. The hot path depends on this exact threshold: if
// it changes silently the bucket fallback would absorb traffic and MALR
// in show --accuracy would shift mysteriously.
func TestPredictKNNTierFires(t *testing.T) {
	idx := bm25.New()
	tokens := []uint64{1, 2, 3}
	for i := 0; i < 5; i++ {
		idx.Add(tokens, bm25.Doc{WallSeconds: 100, ActiveSeconds: 50, TaskType: "refactor"})
	}
	p := Predict(tokens, idx, nil, nil, "refactor")
	if p.Source != SourceKNN {
		t.Fatalf("expected SourceKNN with 5 perfect matches, got %s", p.Source)
	}
	if p.WallSeconds != 100 {
		t.Errorf("expected wall=100 (median of equal docs), got %d", p.WallSeconds)
	}
	if p.N < 3 {
		t.Errorf("expected ≥3 neighbors, got %d", p.N)
	}
}

// When BM25 sim is below the floor, kNN must NOT fire — the bucket tier
// catches it. Catches the regression where someone lowers knnMinSim
// thinking it'll help kNN coverage but actually feeds noise to predict.
func TestPredictFallsThroughToBucket(t *testing.T) {
	idx := bm25.New()
	idx.Add([]uint64{99, 100}, bm25.Doc{WallSeconds: 1, TaskType: "refactor"})
	records := []store.Record{
		{TaskType: "refactor", WallSeconds: 200, ClaudeActiveSeconds: 100},
		{TaskType: "refactor", WallSeconds: 220, ClaudeActiveSeconds: 110},
		{TaskType: "refactor", WallSeconds: 180, ClaudeActiveSeconds: 90},
		{TaskType: "refactor", WallSeconds: 210, ClaudeActiveSeconds: 105},
	}
	p := Predict([]uint64{1, 2, 3}, idx, records, nil, "refactor")
	if p.Source != SourceBucket {
		t.Fatalf("expected SourceBucket fallback, got %s (wall=%d)", p.Source, p.WallSeconds)
	}
}

// No data anywhere → SourceNone. The status line treats this as "no data
// yet" and renders an unobtrusive message; downstream code must be safe
// against zero-valued WallSeconds.
func TestPredictNoneWhenEmpty(t *testing.T) {
	p := Predict(nil, nil, nil, nil, "")
	if p.Source != SourceNone {
		t.Fatalf("expected SourceNone, got %s", p.Source)
	}
	if p.WallSeconds != 0 {
		t.Errorf("SourceNone must return zero wall, got %d", p.WallSeconds)
	}
}

// Project tier fires when bucket has < 4 records but project total ≥ 4.
// Catches a swap of the two if-blocks.
func TestPredictProjectTierWhenBucketThin(t *testing.T) {
	records := []store.Record{
		{TaskType: "test", WallSeconds: 30, ClaudeActiveSeconds: 15},
		{TaskType: "test", WallSeconds: 40, ClaudeActiveSeconds: 20},
		{TaskType: "feature", WallSeconds: 300, ClaudeActiveSeconds: 150},
		{TaskType: "feature", WallSeconds: 310, ClaudeActiveSeconds: 155},
		{TaskType: "feature", WallSeconds: 290, ClaudeActiveSeconds: 145},
		{TaskType: "feature", WallSeconds: 305, ClaudeActiveSeconds: 152},
	}
	// "debug" has zero records; bucket fails; project (n=6) fires.
	p := Predict(nil, nil, records, nil, "debug")
	if p.Source != SourceProject {
		t.Fatalf("expected SourceProject, got %s", p.Source)
	}
	if p.N != len(records) {
		t.Errorf("expected n=%d, got %d", len(records), p.N)
	}
}
