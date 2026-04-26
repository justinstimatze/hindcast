// Package regressor is hindcast's universal-features predictor tier.
// A per-user gradient-boosted regression tree model trained on features
// derivable at UserPromptSubmit time (no leakage from the unobserved
// post-turn ToolCalls/FilesTouched/ResponseChars). Sits above the BM25
// kNN tier in the predict ladder when sufficient training data exists.
//
// Privacy invariant: the trained model is opaque numeric weights — no
// prompt text, no file paths, no project names. Trained from store.Record
// which already enforces the no-prompt-text constraint.
package regressor

import (
	"math"
	"sort"

	"github.com/justinstimatze/hindcast/internal/store"
)

// FeatureNames is the canonical column order. Used for both training and
// inference — feeding the model in a different order than it trained on
// would silently corrupt predictions.
//
// NOTE: SizeBucket was removed in v0.4.1. It's `sizes.Classify(filesTouched,
// toolCount)` — both post-turn observations not available at UserPromptSubmit
// time. Earlier v0.4 trained with it populated from completed records and
// would have served predictions with it always empty, a silent train/predict
// distribution shift. All features below MUST be derivable at UserPromptSubmit.
var FeatureNames = []string{
	"log_prompt_chars",    // 0
	"task_refactor",       // 1
	"task_debug",          // 2
	"task_feature",        // 3
	"task_test",           // 4
	"task_docs",           // 5
	"task_other",          // 6
	"recent_log_p50_wall", // 7
	"recent_log_p75_wall", // 8
	"recent_p50_tools",    // 9
	"recent_frac_bash",    // 10
	"recent_frac_edit",    // 11
	"recent_frac_write",   // 12
	"recent_frac_grep",    // 13
	"recent_frac_read",    // 14
	"bm25_max_sim",        // 15
	"bm25_log_knn_pred",   // 16, 0 if no kNN signal
	"bm25_has_signal",     // 17, 1 if kNN tier fired with ≥3 good neighbors
}

// Context is the prompt-time state needed to derive features for one
// prediction. History is the project's records BEFORE the turn being
// predicted (training: prefix; inference: full project history).
type Context struct {
	PromptChars  int
	TaskType     string
	History      []store.Record
	BM25MaxSim   float64
	BM25PredWall int // kNN tier prediction in seconds; 0 if not available
}

// recentWindow caps how many trailing records contribute to the
// "recent project" features. Long enough to smooth noise, short enough
// to track regime shifts (project gets more complex over time, etc.).
const recentWindow = 20

// Extract converts Context into the canonical feature vector. Length
// matches FeatureNames.
func Extract(ctx Context) []float64 {
	feats := make([]float64, len(FeatureNames))

	// Log-prompt-chars (right-skewed → log-transform stabilizes).
	feats[0] = math.Log(float64(ctx.PromptChars) + 1)

	// Task-type one-hot. Default bucket catches unknown task types so
	// a future TaskType value still produces a valid feature row.
	switch ctx.TaskType {
	case "refactor":
		feats[1] = 1
	case "debug":
		feats[2] = 1
	case "feature":
		feats[3] = 1
	case "test":
		feats[4] = 1
	case "docs":
		feats[5] = 1
	default:
		feats[6] = 1
	}

	// Recent-history features. Take the trailing window of the project's
	// history; if the project is new, the features stay at zero and the
	// model learns to weight the prompt-time + BM25 features instead.
	recent := ctx.History
	if len(recent) > recentWindow {
		recent = recent[len(recent)-recentWindow:]
	}
	if len(recent) > 0 {
		walls := make([]float64, 0, len(recent))
		toolCounts := make([]float64, 0, len(recent))
		var bashHits, editHits, writeHits, grepHits, readHits int
		for _, r := range recent {
			if r.WallSeconds > 0 {
				walls = append(walls, math.Log(float64(r.WallSeconds)))
			}
			tc := 0
			for _, c := range r.ToolCalls {
				tc += c
			}
			toolCounts = append(toolCounts, float64(tc))
			if r.ToolCalls["Bash"] > 0 {
				bashHits++
			}
			if r.ToolCalls["Edit"] > 0 {
				editHits++
			}
			if r.ToolCalls["Write"] > 0 {
				writeHits++
			}
			if r.ToolCalls["Grep"] > 0 {
				grepHits++
			}
			if r.ToolCalls["Read"] > 0 {
				readHits++
			}
		}
		if len(walls) > 0 {
			feats[7] = quantile(walls, 0.5)
			feats[8] = quantile(walls, 0.75)
		}
		if len(toolCounts) > 0 {
			feats[9] = quantile(toolCounts, 0.5)
		}
		n := float64(len(recent))
		feats[10] = float64(bashHits) / n
		feats[11] = float64(editHits) / n
		feats[12] = float64(writeHits) / n
		feats[13] = float64(grepHits) / n
		feats[14] = float64(readHits) / n
	}

	// BM25 features. has_signal lets the model learn a different policy
	// when the kNN tier didn't fire (it can't trust bm25_log_knn_pred=0
	// as a real prediction).
	feats[15] = ctx.BM25MaxSim
	if ctx.BM25PredWall > 0 {
		feats[16] = math.Log(float64(ctx.BM25PredWall))
		feats[17] = 1
	}

	return feats
}

func quantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	pos := q * float64(len(sorted)-1)
	lo := int(pos)
	if lo >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[lo+1]*frac
}
