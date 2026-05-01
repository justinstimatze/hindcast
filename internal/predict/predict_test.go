package predict

import (
	"math"
	"testing"
	"time"

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

// v0.6.3: kNN populates WallP10/WallP90 alongside P25/P75. The variance
// gate uses these to render a wider band when uncertainty is high.
// A spread of neighbors should produce P10 < P25 ≤ median ≤ P75 < P90.
func TestPredictKNNComputesWideQuantiles(t *testing.T) {
	idx := bm25.New()
	tokens := []uint64{1, 2, 3}
	// Spread of wall durations so quantiles are distinct.
	walls := []int{30, 60, 90, 120, 180, 240, 360, 480}
	for _, w := range walls {
		idx.Add(tokens, bm25.Doc{WallSeconds: w, ActiveSeconds: w / 2, TaskType: "feature"})
	}
	p := Predict(tokens, idx, nil, nil, "feature")
	if p.Source != SourceKNN {
		t.Fatalf("expected SourceKNN, got %s", p.Source)
	}
	if !(p.WallP10 < p.WallP25 && p.WallP25 <= p.WallSeconds && p.WallSeconds <= p.WallP75 && p.WallP75 < p.WallP90) {
		t.Errorf("expected ordered quantiles P10<P25<=med<=P75<P90, got P10=%d P25=%d med=%d P75=%d P90=%d",
			p.WallP10, p.WallP25, p.WallSeconds, p.WallP75, p.WallP90)
	}
}

// v0.6.3: recencyWeight returns 1.0 for zero TS (back-compat), 1.0 for
// today's record, ~0.5 for one half-life ago, ~0.25 for two half-lives.
func TestRecencyWeight(t *testing.T) {
	now := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name string
		ts   time.Time
		want float64
		tol  float64
	}{
		{"zero TS → neutral", time.Time{}, 1.0, 0.001},
		{"now → 1.0", now, 1.0, 0.001},
		{"future → 1.0 (clock skew safety)", now.Add(time.Hour), 1.0, 0.001},
		{"60d ago → 0.5", now.AddDate(0, 0, -60), 0.5, 0.01},
		{"120d ago → 0.25", now.AddDate(0, 0, -120), 0.25, 0.01},
		{"7d ago → ~0.92", now.AddDate(0, 0, -7), 0.923, 0.01},
	}
	for _, c := range cases {
		got := recencyWeight(c.ts, now)
		if math.Abs(got-c.want) > c.tol {
			t.Errorf("%s: got %.4f, want %.4f±%.3f", c.name, got, c.want, c.tol)
		}
	}
}

// v0.6.3: with mixed-age neighbors, the freshness-weighted median
// should shift toward recent values. Test: 5 neighbors, half from
// long ago at wall=60s, half recent at wall=300s. Pre-v0.6.3 (no
// freshness): unweighted median ~180s. Post-v0.6.3: median pulled
// toward 300s by recency weighting.
func TestPredictKNNFreshnessShiftsMedianTowardRecent(t *testing.T) {
	idx := bm25.New()
	tokens := []uint64{1, 2, 3}
	now := time.Now().UTC()
	old := now.AddDate(0, 0, -180) // 3 half-lives = ~12.5% weight
	// 4 old short turns
	for i := 0; i < 4; i++ {
		idx.Add(tokens, bm25.Doc{WallSeconds: 60, ActiveSeconds: 30, TaskType: "feature", TS: old})
	}
	// 4 recent long turns
	for i := 0; i < 4; i++ {
		idx.Add(tokens, bm25.Doc{WallSeconds: 300, ActiveSeconds: 150, TaskType: "feature", TS: now})
	}
	p := Predict(tokens, idx, nil, nil, "feature")
	if p.Source != SourceKNN {
		t.Fatalf("expected SourceKNN, got %s", p.Source)
	}
	// With freshness weighting, the recent (heavy) cluster should
	// dominate. Median should be closer to 300 than to the unweighted
	// midpoint of 180.
	if p.WallSeconds < 240 {
		t.Errorf("freshness should pull median toward recent (300s), got %ds; expect ≥240", p.WallSeconds)
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
