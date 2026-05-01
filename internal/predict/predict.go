// Package predict is hindcast's local duration predictor. Output is
// emitted by the UserPromptSubmit hook as a calibrated band that the
// formatter (cmd_pending.formatClaudeInjection) gates by source tier
// and variance before showing it to Claude. Claude is instructed to
// cite the band, override only on structural scope mismatch, and not
// pad the override out of caution.
//
// Architecture: a BM25-weighted k-nearest-neighbors regressor over the
// project's own past turn records. Aggregation = weighted median plus
// P10/P25/P75/P90 of the top-k neighbors' wall and active seconds,
// weighted by BM25 similarity × an exponential recency factor (60-day
// half-life by default; HINDCAST_FRESHNESS_HALFLIFE_DAYS overrides).
// Empirical quantiles are widened via small-sample shrinkage — at n=7
// neighbors the band widens by ~19%, asymptotically 1.0 as n→∞.
//
// Fallback ladder when kNN sim or sample is insufficient: task-type
// bucket → overall project → global sketch → none. Bucket and project
// tiers are populated; global is computed but the inject path
// suppresses it (biased short on new projects).
//
// The render-vs-suppress decision (anchoring trade-off, per Lou
// et al. 2024) is handled at the rendering layer in
// cmd_pending.formatClaudeInjection — this package is purely numeric.
package predict

import (
	"math"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/store"
)

// Source names the record tier the prediction came from. Carried
// through the inject so Claude (and downstream readers like
// `hindcast show --accuracy`) can judge how grounded each prediction is.
type Source string

const (
	SourceKNN       Source = "knn"       // BM25 top-k neighbors, similarity above threshold
	SourceBucket    Source = "bucket"    // task-type bucket within project
	SourceProject   Source = "project"   // overall project aggregate
	SourceGlobal    Source = "global"    // cross-project sketch
	SourceRegressor Source = "regressor" // learned regressor (gbdt/linear), gated by health
	SourceNone      Source = "none"      // no data at all
)

// Prediction is the local kNN forecast for a turn. All durations in seconds.
type Prediction struct {
	WallSeconds   int `json:"wall_seconds"`
	ActiveSeconds int `json:"active_seconds"`
	WallP25       int `json:"wall_p25"`
	WallP75       int `json:"wall_p75"`
	ActiveP25     int `json:"active_p25"`
	ActiveP75     int `json:"active_p75"`
	// v0.6.3: wider quantiles. Used as the rendered band when the
	// variance gate trips — a calibrated [P10, P90] captures 80% of
	// actual durations vs 50% for [P25, P75], which matches "useful
	// without overfitting": wider but still actionable, and a strict
	// superset of the existing distribution so there's no model to
	// overfit. Populated by the kNN tier only; bucket / project tiers
	// leave them zero.
	WallP10      int     `json:"wall_p10,omitempty"`
	WallP90      int     `json:"wall_p90,omitempty"`
	ActiveP10    int     `json:"active_p10,omitempty"`
	ActiveP90    int     `json:"active_p90,omitempty"`
	N            int     `json:"n"`                 // neighbors used
	MaxSim       float64 `json:"max_sim,omitempty"` // top neighbor similarity (knn or regressor's bm25 feature)
	Source       Source  `json:"source"`
	SourceDetail string  `json:"source_detail,omitempty"` // e.g. regressor variant ("gbdt"|"linear")
	TaskType     string  `json:"task_type,omitempty"`
	SessionID    string  `json:"session_id,omitempty"`
}

// kNN threshold: a top neighbor below this is treated as insufficient
// signal and we fall back to bucket stats. Chosen generously (0.15) so
// BM25 drives on any reasonable lexical overlap; the fallback floor
// catches cold-start and topic-drift cases.
const knnMinSim = 0.15

// Default k. Small corpora dominate — median over 3–7 is typical.
const defaultK = 7

// freshnessHalfLifeDays is the half-life used to decay older records'
// weight in the kNN median. Conservative on purpose: a 60-day half-life
// means a 60-day-old turn carries half the influence of a turn from
// today, but week-old turns still carry ~92% — enough to track
// genuine drift without overfitting one weird week.
//
// Override via HINDCAST_FRESHNESS_HALFLIFE_DAYS for experimentation;
// 0 or negative disables freshness weighting entirely (pre-v0.6.3
// behavior, useful for A/B vs the freshness-on default).
const freshnessHalfLifeDays = 60.0

// recencyWeight returns a multiplicative weight in (0, 1] based on the
// document's age. Zero TS (pre-v0.6.3 records) returns 1.0 — neutral
// — so existing indexes don't degrade until they refresh.
func recencyWeight(docTS time.Time, now time.Time) float64 {
	if docTS.IsZero() {
		return 1.0
	}
	halfLife := freshnessHalfLifeDays
	if v := os.Getenv("HINDCAST_FRESHNESS_HALFLIFE_DAYS"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			halfLife = f
		}
	}
	if halfLife <= 0 {
		return 1.0
	}
	ageDays := now.Sub(docTS).Hours() / 24.0
	if ageDays <= 0 {
		return 1.0
	}
	return math.Pow(2, -ageDays/halfLife)
}

// Predict computes a Prediction from the given signals. `queryHashes`
// is the BM25-tokenized prompt; `idx` may be nil (disables kNN path).
// `records` is the project's recent records; `sk` is the global sketch
// (optional cross-project fallback). `taskType` is the classified bucket.
//
// Semantically: try kNN → task-type bucket → overall project → global
// sketch → none. Return the first tier with enough signal.
func Predict(queryHashes []uint64, idx *bm25.Index, records []store.Record, sk *store.Sketch, taskType string) Prediction {
	// --- kNN ---
	if idx != nil && len(idx.Docs) > 0 && len(queryHashes) > 0 {
		matches := idx.TopK(queryHashes, defaultK)
		// Keep only matches above the noise floor.
		var good []bm25.Match
		for _, m := range matches {
			if m.Sim >= knnMinSim {
				good = append(good, m)
			}
		}
		if len(good) >= 3 {
			walls := make([]weighted, 0, len(good))
			actives := make([]weighted, 0, len(good))
			now := time.Now().UTC()
			for _, m := range good {
				// v0.6.3: weight = sim × recency. Recency decays older
				// records so genuine project drift (a project's recent
				// turns getting longer) shifts the median toward what's
				// actually happening. Conservative half-life (60d) keeps
				// stable projects stable.
				w := m.Sim * recencyWeight(m.Doc.TS, now)
				walls = append(walls, weighted{float64(m.Doc.WallSeconds), w})
				actives = append(actives, weighted{float64(m.Doc.ActiveSeconds), w})
			}
			n := len(good)
			medianWall := weightedMedian(walls)
			medianActive := weightedMedian(actives)
			// v0.6.4: small-sample shrinkage correction. Empirical
			// quantiles from n≈7 neighbors are systematically biased
			// toward the median compared to the true quantiles —
			// the v0.6.3 production data showed point-rendered band
			// hit rate at 31% (target 50%), implying P25/P75 were too
			// narrow. We widen each quantile away from the median by
			// (1 + alpha/sqrt(n)). Alpha=0.5 → ~19% widening at n=7,
			// ~11% at n=20, asymptotically 1.0 (no widening) as n→∞.
			factor := 1.0 + 0.5/math.Sqrt(float64(n))
			// Shrinkage widens quantiles away from the median; for low
			// quantiles (P10 especially) this can push the value below
			// zero on small samples. Clamp at 1s — the floor of any
			// real turn — so the inject doesn't render "0s" or worse,
			// negative seconds, which would be meaningless to Claude.
			clamp := func(x float64) float64 {
				if x < 1 {
					return 1
				}
				return x
			}
			shrinkWall := func(q float64) float64 { return clamp(medianWall + (q-medianWall)*factor) }
			shrinkActive := func(q float64) float64 { return clamp(medianActive + (q-medianActive)*factor) }
			return Prediction{
				WallSeconds:   int(medianWall + 0.5),
				ActiveSeconds: int(medianActive + 0.5),
				WallP25:       int(shrinkWall(weightedQuantile(walls, 0.25)) + 0.5),
				WallP75:       int(shrinkWall(weightedQuantile(walls, 0.75)) + 0.5),
				WallP10:       int(shrinkWall(weightedQuantile(walls, 0.10)) + 0.5),
				WallP90:       int(shrinkWall(weightedQuantile(walls, 0.90)) + 0.5),
				ActiveP25:     int(shrinkActive(weightedQuantile(actives, 0.25)) + 0.5),
				ActiveP75:     int(shrinkActive(weightedQuantile(actives, 0.75)) + 0.5),
				ActiveP10:     int(shrinkActive(weightedQuantile(actives, 0.10)) + 0.5),
				ActiveP90:     int(shrinkActive(weightedQuantile(actives, 0.90)) + 0.5),
				N:             n,
				MaxSim:        good[0].Sim,
				Source:        SourceKNN,
				TaskType:      taskType,
			}
		}
	}

	// --- task-type bucket ---
	var bucketWalls, bucketActives []float64
	for _, r := range records {
		if r.TaskType == taskType {
			bucketWalls = append(bucketWalls, float64(r.WallSeconds))
			bucketActives = append(bucketActives, float64(r.ClaudeActiveSeconds))
		}
	}
	if len(bucketWalls) >= 4 {
		return quantilePrediction(bucketWalls, bucketActives, SourceBucket, taskType)
	}

	// --- overall project ---
	if len(records) >= 4 {
		allWalls := make([]float64, 0, len(records))
		allActives := make([]float64, 0, len(records))
		for _, r := range records {
			allWalls = append(allWalls, float64(r.WallSeconds))
			allActives = append(allActives, float64(r.ClaudeActiveSeconds))
		}
		return quantilePrediction(allWalls, allActives, SourceProject, taskType)
	}

	// --- global sketch ---
	if sk != nil && len(sk.Wall) >= 4 {
		walls := make([]float64, len(sk.Wall))
		actives := make([]float64, len(sk.Active))
		for i, v := range sk.Wall {
			walls[i] = float64(v)
		}
		for i, v := range sk.Active {
			actives[i] = float64(v)
		}
		return quantilePrediction(walls, actives, SourceGlobal, taskType)
	}

	// SourceNone: every tier refused. Pass len(records) through as N so
	// CLI readers (`hindcast predict`, `hindcast show --accuracy`) can
	// render a useful cold-start message instead of a bare "no data
	// yet" — the project tier needs ≥4 records, so N tells the reader
	// how close they are. The injection path treats SourceNone as
	// suppressed regardless.
	return Prediction{Source: SourceNone, TaskType: taskType, N: len(records)}
}

type weighted struct {
	value, weight float64
}

// weightedQuantile returns the weighted q-quantile of samples.
// Sorts by value, walks cumulative weight, interpolates at the target.
func weightedQuantile(ws []weighted, q float64) float64 {
	if len(ws) == 0 {
		return 0
	}
	sorted := append([]weighted(nil), ws...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].value < sorted[j].value })
	total := 0.0
	for _, w := range sorted {
		total += w.weight
	}
	if total <= 0 {
		// Equal-weight fallback.
		i := int(math.Round(q * float64(len(sorted)-1)))
		return sorted[i].value
	}
	target := q * total
	cum := 0.0
	for _, w := range sorted {
		cum += w.weight
		if cum >= target {
			return w.value
		}
	}
	return sorted[len(sorted)-1].value
}

func weightedMedian(ws []weighted) float64 {
	return weightedQuantile(ws, 0.5)
}

func quantilePrediction(walls, actives []float64, src Source, taskType string) Prediction {
	wm := unweightedQuantile(walls, 0.5)
	wp25 := unweightedQuantile(walls, 0.25)
	wp75 := unweightedQuantile(walls, 0.75)
	am := unweightedQuantile(actives, 0.5)
	ap25 := unweightedQuantile(actives, 0.25)
	ap75 := unweightedQuantile(actives, 0.75)
	return Prediction{
		WallSeconds:   int(wm + 0.5),
		ActiveSeconds: int(am + 0.5),
		WallP25:       int(wp25 + 0.5),
		WallP75:       int(wp75 + 0.5),
		ActiveP25:     int(ap25 + 0.5),
		ActiveP75:     int(ap75 + 0.5),
		N:             len(walls),
		Source:        src,
		TaskType:      taskType,
	}
}

func unweightedQuantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	i := int(math.Round(q * float64(len(sorted)-1)))
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}
