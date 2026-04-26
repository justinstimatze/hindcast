// Package predict is hindcast's human-facing duration predictor.
// Replaces the earlier "inject priors into Claude's context" mechanism
// (which exploited LLM anchoring per Lou et al. 2024) with a local
// numeric predictor whose output is surfaced to the human via Claude
// Code's status line — Claude never sees it, so anchoring can't fire.
//
// The predictor is a BM25-weighted k-nearest-neighbors regressor over
// the project's own past turn records. Aggregation: weighted median +
// p25/p75 of the top-k neighbors' wall/active seconds, weighted by
// similarity. Falls back to task-type bucket stats, then overall
// project, then global sketch, then "unknown."
package predict

import (
	"math"
	"sort"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/store"
)

// Source names the record tier the prediction came from. Lets the
// status-line formatter show provenance so the human can judge trust.
type Source string

const (
	SourceKNN       Source = "knn"       // BM25 top-k neighbors, similarity above threshold
	SourceBucket    Source = "bucket"    // task-type bucket within project
	SourceProject   Source = "project"   // overall project aggregate
	SourceGlobal    Source = "global"    // cross-project sketch
	SourceRegressor Source = "regressor" // learned regressor (gbdt/linear), gated by health
	SourceNone      Source = "none"      // no data at all
)

// Prediction is what gets written to the last-prediction file for the
// status line to consume. All durations in seconds.
type Prediction struct {
	WallSeconds    int     `json:"wall_seconds"`
	ActiveSeconds  int     `json:"active_seconds"`
	WallP25        int     `json:"wall_p25"`
	WallP75        int     `json:"wall_p75"`
	ActiveP25      int     `json:"active_p25"`
	ActiveP75      int     `json:"active_p75"`
	N              int     `json:"n"`                  // neighbors used
	MaxSim         float64 `json:"max_sim,omitempty"`  // top neighbor similarity, knn only
	Source         Source  `json:"source"`
	TaskType       string  `json:"task_type,omitempty"`
	SessionID      string  `json:"session_id,omitempty"`
}

// kNN threshold: a top neighbor below this is treated as insufficient
// signal and we fall back to bucket stats. Chosen generously (0.15) so
// BM25 drives on any reasonable lexical overlap; the fallback floor
// catches cold-start and topic-drift cases.
const knnMinSim = 0.15

// Default k. Small corpora dominate — median over 3–7 is typical.
const defaultK = 7

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
			for _, m := range good {
				walls = append(walls, weighted{float64(m.Doc.WallSeconds), m.Sim})
				actives = append(actives, weighted{float64(m.Doc.ActiveSeconds), m.Sim})
			}
			return Prediction{
				WallSeconds:   int(weightedMedian(walls) + 0.5),
				ActiveSeconds: int(weightedMedian(actives) + 0.5),
				WallP25:       int(weightedQuantile(walls, 0.25) + 0.5),
				WallP75:       int(weightedQuantile(walls, 0.75) + 0.5),
				ActiveP25:     int(weightedQuantile(actives, 0.25) + 0.5),
				ActiveP75:     int(weightedQuantile(actives, 0.75) + 0.5),
				N:             len(good),
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

	return Prediction{Source: SourceNone, TaskType: taskType}
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
		i := int(math.Round(q*float64(len(sorted)-1)))
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
