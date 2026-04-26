// Package health computes per-user predictor health from the user's own
// records via prefix-LOO. The output is the empirical sim cliff: the
// threshold above which the BM25-kNN tier reliably beats the bucket
// tier on this user's data.
//
// Why this exists: the cross-corpus benchmark in v0.3 showed that the
// sim cliff observed in the maintainer's data (~0.5) does not generalize
// to all corpora. Users whose Claude Code work patterns differ will have
// different cliffs (or none). We tune from each user's own data instead
// of shipping a global constant.
package health

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/predict"
	"github.com/justinstimatze/hindcast/internal/regressor"
	"github.com/justinstimatze/hindcast/internal/store"
)

// prefRow is a single prefix-LOO prediction. Pulled out of Compute so
// helpers (malrOf, etc.) can take it as a typed parameter.
type prefRow struct {
	source predict.Source
	sim    float64
	pred   int
	actual int
}

// Health captures the empirical state of the user's predictor. Persisted
// to ~/.claude/hindcast/health.json so the UserPromptSubmit hook can
// read it without recomputation.
type Health struct {
	// Tuning result.
	TunedSimThreshold float64 `json:"tuned_sim_threshold"` // gate sim ≥ this; ∞ = never inject
	KNNMALRAtThreshold float64 `json:"knn_malr_at_threshold"`
	BucketMALR         float64 `json:"bucket_malr"`
	GroupMALR          float64 `json:"group_malr"` // overall project tier
	GlobalMALR         float64 `json:"global_malr"`
	NPredictions       int     `json:"n_predictions"`
	NKNN               int     `json:"n_knn"`
	NBucket            int     `json:"n_bucket"`
	LastTunedAt        string  `json:"last_tuned_at"` // RFC3339 UTC

	// Verdict — single-line summary the status line and `show --health` use.
	Verdict string `json:"verdict"`

	// Per-sim-bucket detail for debugging.
	SimBuckets []SimBucketStat `json:"sim_buckets,omitempty"`

	// 50/50 chronological held-out eval. Separate methodology from the
	// prefix-LOO numbers above — this exists to pick a per-user winner
	// among {ladder, gbdt, linear}. Apples-to-apples on coverage: every
	// counted test record has a prediction from each variant.
	NHeldOut          int     `json:"n_held_out"`
	LadderMALRHeldOut float64 `json:"ladder_malr_held_out"`
	GBDTMALRHeldOut   float64 `json:"gbdt_malr_held_out"`
	LinearMALRHeldOut float64 `json:"linear_malr_held_out"`
	// RegressorWinner is "gbdt" | "linear" | "none". When "none", the
	// existing ladder is the locally-winning strategy and predict.Predict
	// keeps full control. When "gbdt"/"linear", the named regressor goes
	// first in the predict path.
	RegressorWinner       string  `json:"regressor_winner"`
	RegressorLiftVsLadder float64 `json:"regressor_lift_vs_ladder"` // ladder_malr / winner_malr; >1 = winner better
}

type SimBucketStat struct {
	LoSim   float64 `json:"lo_sim"`
	HiSim   float64 `json:"hi_sim"`
	N       int     `json:"n"`
	WallMALR float64 `json:"wall_malr"`
}

// HealthPath returns the location of the per-install health.json.
func HealthPath() (string, error) {
	root, err := store.HindcastDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", err
	}
	return filepath.Join(root, "health.json"), nil
}

// Save persists the health to disk atomically.
func (h *Health) Save() error {
	path, err := HealthPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(h, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Load reads the persisted health, returning nil + error if absent.
func Load() (*Health, error) {
	path, err := HealthPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var h Health
	if err := json.Unmarshal(data, &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// Compute runs prefix-LOO across the user's records grouped by project,
// finds the empirical sim cliff, and returns a populated Health.
//
// Algorithm:
//  1. For each project, sort records chronologically. Build BM25 index
//     incrementally. For each record past warmup, predict using prior
//     records and capture (source, max_sim, predicted, actual).
//  2. Aggregate predictions by source. Compute MALR per tier.
//  3. For the kNN tier, scan candidate thresholds [0.15, 0.20, ..., 0.95].
//     For each, compute "kNN MALR among predictions where max_sim ≥ t".
//     The tuned threshold is the smallest t where:
//        knn_malr_at_t ≤ 0.85 × bucket_malr
//     i.e. kNN beats bucket fallback by at least 15% at this gate.
//     If no such t exists, tuned threshold = +Inf (gate never opens).
//
// Records are grouped by project hash (caller responsibility) — this
// function works on a flat record slice and uses ProjectHash to
// segment internally.
func Compute(byProject map[string][]store.Record, sk *store.Sketch, warmup int) *Health {
	var rows []prefRow

	for _, recs := range byProject {
		if len(recs) < warmup+5 {
			continue
		}
		sort.Slice(recs, func(i, j int) bool { return recs[i].TS.Before(recs[j].TS) })
		idx := bm25.New()
		for i, r := range recs {
			if i >= warmup && r.WallSeconds > 0 && len(r.PromptTokens) > 0 {
				p := predict.Predict(r.PromptTokens, idx, recs[:i], sk, r.TaskType)
				if p.Source != predict.SourceNone {
					rows = append(rows, prefRow{
						source: p.Source, sim: p.MaxSim,
						pred: p.WallSeconds, actual: r.WallSeconds,
					})
				}
			}
			tc := 0
			for _, c := range r.ToolCalls {
				tc += c
			}
			idx.Add(r.PromptTokens, bm25.Doc{
				WallSeconds:   r.WallSeconds,
				ActiveSeconds: r.ClaudeActiveSeconds,
				TaskType:      r.TaskType,
				SizeBucket:    r.SizeBucket,
				ToolCount:     tc,
				FilesTouched:  r.FilesTouched,
			})
		}
	}

	h := &Health{LastTunedAt: time.Now().UTC().Format(time.RFC3339)}
	if len(rows) == 0 {
		h.TunedSimThreshold = math.Inf(1)
		h.Verdict = "no data — backfill or use Claude Code for a few sessions"
		return h
	}

	// Per-source MALR.
	bySrc := map[predict.Source][]prefRow{}
	for _, r := range rows {
		bySrc[r.source] = append(bySrc[r.source], r)
	}
	h.NPredictions = len(rows)
	h.NKNN = len(bySrc[predict.SourceKNN])
	h.NBucket = len(bySrc[predict.SourceBucket])
	h.BucketMALR = malrOf(bySrc[predict.SourceBucket])
	h.GroupMALR = malrOf(bySrc[predict.SourceProject])
	h.GlobalMALR = malrOf(bySrc[predict.SourceGlobal])

	// Sim-bucket detail for the kNN tier.
	bucketEdges := []float64{0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}
	for i := 0; i < len(bucketEdges)-1; i++ {
		lo, hi := bucketEdges[i], bucketEdges[i+1]
		var picked []prefRow
		for _, r := range bySrc[predict.SourceKNN] {
			if r.sim >= lo && r.sim < hi {
				picked = append(picked, r)
			}
		}
		h.SimBuckets = append(h.SimBuckets, SimBucketStat{
			LoSim: lo, HiSim: hi, N: len(picked), WallMALR: malrOf(picked),
		})
	}

	// Find the tuned threshold: the cliff at which kNN starts to beat
	// the bucket fallback. Algorithm walks sim thresholds bottom-up;
	// for each candidate t, picks predictions with sim in [t, t+0.05]
	// (a narrow per-bucket window, NOT aggregate above t — that would
	// hide local cliffs). Returns the smallest t where the per-bucket
	// MALR is ≤ 0.70 × bucket-tier MALR AND the aggregate-above-t MALR
	// continues to beat bucket. The narrow-window check finds where
	// kNN starts winning; the aggregate check ensures it stays winning.
	//
	// If bucket has too few samples, anchor on global MALR instead.
	target := h.BucketMALR
	if h.NBucket < 20 || target == 0 {
		target = h.GlobalMALR
	}
	if target == 0 {
		h.TunedSimThreshold = 0.7
		h.Verdict = "insufficient baseline; defaulting to sim≥0.70"
		return h
	}
	threshold := math.Inf(1)
	candMALR := 0.0
	candidates := []float64{0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
	for _, t := range candidates {
		var window []prefRow
		var aboveT []prefRow
		for _, r := range bySrc[predict.SourceKNN] {
			if r.sim >= t && r.sim < t+0.10 {
				window = append(window, r)
			}
			if r.sim >= t {
				aboveT = append(aboveT, r)
			}
		}
		if len(window) < 30 || len(aboveT) < 30 {
			continue
		}
		windowMALR := malrOf(window)
		aboveMALR := malrOf(aboveT)
		// Cliff: per-window MALR meaningfully better AND aggregate stays better.
		if windowMALR > 0 && windowMALR <= 0.70*target && aboveMALR <= 0.85*target {
			threshold = t
			candMALR = aboveMALR
			break
		}
	}
	h.TunedSimThreshold = threshold
	h.KNNMALRAtThreshold = candMALR
	switch {
	case math.IsInf(threshold, 1):
		h.Verdict = "kNN does not beat bucket on this user's data"
	default:
		h.Verdict = "kNN earns its keep at sim≥" + formatFloat(threshold)
	}

	// Per-user adaptive tier selection. Run a 50/50 chronological split
	// per project, train both regressor variants on the first half, and
	// compare ladder/gbdt/linear MALR on the same held-out test set.
	// Soft-fails (zero MALRs, RegressorWinner="none") if data is thin.
	evalAdaptive(h, byProject, sk, warmup)
	return h
}

// evalAdaptive populates the 50/50 fields on h. Apples-to-apples: every
// counted test record has a ladder pred AND a gbdt pred AND a linear pred.
// If either regressor fails to train (insufficient data), winner="none"
// and the ladder remains primary.
func evalAdaptive(h *Health, byProject map[string][]store.Record, sk *store.Sketch, warmup int) {
	trainRecs := map[string][]store.Record{}
	testRecs := map[string][]store.Record{}
	for k, recs := range byProject {
		if len(recs) < 10 {
			continue
		}
		sorted := make([]store.Record, len(recs))
		copy(sorted, recs)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].TS.Before(sorted[j].TS) })
		cut := len(sorted) / 2
		trainRecs[k] = sorted[:cut]
		testRecs[k] = sorted[cut:]
	}

	gbdt, gErr := regressor.Train(trainRecs, warmup)
	lin, lErr := regressor.TrainLinearFromRecords(trainRecs, warmup, 1.0)
	if gErr != nil && lErr != nil {
		h.RegressorWinner = "none"
		return
	}

	type triple struct{ ladder, gbdt, lin float64 }
	var trios []triple
	for k, test := range testRecs {
		hist := append([]store.Record(nil), trainRecs[k]...)
		idx := bm25.New()
		for _, r := range hist {
			regressor.AddRecordToIndex(idx, r)
		}
		for _, r := range test {
			if r.WallSeconds <= 0 {
				continue
			}
			lp := predict.Predict(r.PromptTokens, idx, hist, sk, r.TaskType)
			if lp.Source == predict.SourceNone || lp.WallSeconds <= 0 {
				hist = append(hist, r)
				regressor.AddRecordToIndex(idx, r)
				continue
			}
			ctx := regressor.MakeContext(r.PromptChars, r.TaskType, hist, r.PromptTokens, idx)
			gPred := 0
			if gbdt != nil {
				gPred = gbdt.PredictWall(ctx)
			}
			lPred := 0
			if lin != nil {
				lPred = lin.PredictWall(ctx)
			}
			if gPred > 0 && lPred > 0 {
				trios = append(trios, triple{
					ladder: math.Abs(math.Log(float64(lp.WallSeconds) / float64(r.WallSeconds))),
					gbdt:   math.Abs(math.Log(float64(gPred) / float64(r.WallSeconds))),
					lin:    math.Abs(math.Log(float64(lPred) / float64(r.WallSeconds))),
				})
			}
			hist = append(hist, r)
			regressor.AddRecordToIndex(idx, r)
		}
	}
	if len(trios) < 50 {
		h.RegressorWinner = "none"
		return
	}

	ladderArr := make([]float64, len(trios))
	gbdtArr := make([]float64, len(trios))
	linArr := make([]float64, len(trios))
	for i, t := range trios {
		ladderArr[i] = t.ladder
		gbdtArr[i] = t.gbdt
		linArr[i] = t.lin
	}
	h.NHeldOut = len(trios)
	h.LadderMALRHeldOut = medianExp(ladderArr)
	h.GBDTMALRHeldOut = medianExp(gbdtArr)
	h.LinearMALRHeldOut = medianExp(linArr)

	// Pick winner: lowest MALR among the regressor variants, but only if
	// it beats the ladder by ≥15% lift. The 0.85× threshold mirrors the
	// kNN/bucket gate above for consistency.
	var winnerName string
	var winnerMALR float64
	if h.GBDTMALRHeldOut > 0 && (winnerMALR == 0 || h.GBDTMALRHeldOut < winnerMALR) {
		winnerName, winnerMALR = "gbdt", h.GBDTMALRHeldOut
	}
	if h.LinearMALRHeldOut > 0 && (winnerMALR == 0 || h.LinearMALRHeldOut < winnerMALR) {
		winnerName, winnerMALR = "linear", h.LinearMALRHeldOut
	}
	if winnerMALR > 0 && h.LadderMALRHeldOut > 0 && winnerMALR <= 0.85*h.LadderMALRHeldOut {
		h.RegressorWinner = winnerName
		h.RegressorLiftVsLadder = h.LadderMALRHeldOut / winnerMALR
	} else {
		h.RegressorWinner = "none"
		if h.LadderMALRHeldOut > 0 && winnerMALR > 0 {
			h.RegressorLiftVsLadder = h.LadderMALRHeldOut / winnerMALR
		}
	}
}

func medianExp(absLogs []float64) float64 {
	if len(absLogs) == 0 {
		return 0
	}
	sort.Float64s(absLogs)
	return math.Exp(absLogs[len(absLogs)/2])
}

func malrOf(rs []prefRow) float64 {
	abs := make([]float64, 0, len(rs))
	for _, r := range rs {
		if r.pred <= 0 || r.actual <= 0 {
			continue
		}
		abs = append(abs, math.Abs(math.Log(float64(r.pred)/float64(r.actual))))
	}
	if len(abs) == 0 {
		return 0
	}
	sort.Float64s(abs)
	return math.Exp(abs[len(abs)/2])
}

func formatFloat(f float64) string {
	if math.IsInf(f, 1) {
		return "+Inf"
	}
	return strconv.FormatFloat(f, 'f', 2, 64)
}
