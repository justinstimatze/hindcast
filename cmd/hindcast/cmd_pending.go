package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/health"
	"github.com/justinstimatze/hindcast/internal/hook"
	"github.com/justinstimatze/hindcast/internal/predict"
	"github.com/justinstimatze/hindcast/internal/regressor"
	"github.com/justinstimatze/hindcast/internal/store"
	"github.com/justinstimatze/hindcast/internal/tags"
	"github.com/justinstimatze/hindcast/internal/transcript"
)

type pendingInput struct {
	SessionID      string `json:"session_id"`
	CWD            string `json:"cwd"`
	PermissionMode string `json:"permission_mode"`
	Prompt         string `json:"prompt"`
	HookEventName  string `json:"hook_event_name"`
	TranscriptPath string `json:"transcript_path"`
}

// cmdPending runs as the UserPromptSubmit hook. It classifies the prompt
// in memory, hashes its tokens with the per-install salt, and records
// those (plus session metadata) in a pending file for the Stop hook to
// consume. It also prints a BM25 top-match line to stdout for CC to
// inject into the current turn. Plaintext prompt never touches disk.
func cmdPending() {
	if os.Getenv("HINDCAST_SKIP") == "1" {
		return
	}
	var in pendingInput
	if err := hook.Decode("pending", &in); err != nil {
		return
	}
	if in.CWD == "" || in.SessionID == "" {
		return
	}

	project := store.ResolveProject(in.CWD)
	hash := store.ProjectHash(project)
	arm := store.SessionArm(in.SessionID, store.ControlPctFromEnv())

	// Classification runs on the user prompt directly. BM25 retrieval
	// gets the wider rolling window so short prompts like "do it" still
	// find relevant past turns via surrounding context. These used to be
	// conflated — firstLine(effective) picked up the oldest prior line,
	// not the user's actual request, and every post-rolling-window turn
	// was classifying on whatever assistant text happened to head the
	// window. See cmd_eval_api.go + cmd_mcp.go for the matching fix.
	effective := transcript.ComposeEffectiveTask(in.Prompt, in.TranscriptPath)

	taskType := string(tags.Classify(firstLine(in.Prompt)))

	salt, err := store.GetSalt()
	var tokens []uint64
	if err != nil {
		hook.Logf("pending", "salt: %s", err)
	} else {
		tokens = bm25.HashTokens(effective, salt)
	}

	pred := computePrediction(hash, tokens, taskType, in.SessionID, len(in.Prompt))

	// Mirror what formatClaudeInjection actually surfaces. variance_gated
	// means *the inject was rendered AND it showed the band as headline*
	// — false when the inject was suppressed by the tier gate (global /
	// none) or the WallSeconds<=0 check. Without this tier-aware gating,
	// global-tier high-variance predictions would count toward the
	// "variance-gated band hit rate," which would be incoherent because
	// Claude never saw them.
	varianceGated := false
	switch pred.Source {
	case predict.SourceRegressor, predict.SourceKNN, predict.SourceBucket, predict.SourceProject:
		if pred.WallSeconds > 0 && pred.WallP25 > 0 && pred.WallP75 > 0 {
			// Dynamic threshold: small-n / low-sim kNN matches trip more easily.
			th := 3.0
			if pred.Source == predict.SourceKNN && (pred.N < 5 || pred.MaxSim < 0.4) {
				th = 2.0
			}
			if float64(pred.WallP75)/float64(pred.WallP25) > th {
				varianceGated = true
			}
		}
	}

	startTS := time.Now().UTC()
	if path, err := store.PendingPath(in.SessionID, startTS); err == nil {
		p := store.PendingTurn{
			SessionID:        in.SessionID,
			StartTS:          startTS,
			TaskType:         taskType,
			PromptTokens:     tokens,
			PromptChars:      len(in.Prompt),
			PermissionMode:   in.PermissionMode,
			ProjectHash:      hash,
			CWD:              in.CWD,
			Arm:              arm,
			PredictedWall:    pred.WallSeconds,
			PredictedActive:  pred.ActiveSeconds,
			PredictionSrc:    string(pred.Source),
			PredictedWallP25: pred.WallP25,
			PredictedWallP75: pred.WallP75,
			PredictedWallP10: pred.WallP10,
			PredictedWallP90: pred.WallP90,
			PredictedMaxSim:  pred.MaxSim,
			VarianceGated:    varianceGated,
		}
		if err := store.WritePending(path, p); err != nil {
			hook.Logf("pending", "write pending: %s", err)
		}
	} else {
		hook.Logf("pending", "path: %s", err)
	}

	// Legacy v0.1 inject (full bucket-table) preserved for eval-api A/B.
	if arm == store.ArmTreatment && os.Getenv("HINDCAST_LEGACY_INJECT") == "1" {
		fmt.Print(hindcastPrior(in.Prompt, effective, in.CWD, in.SessionID))
		return
	}

	// v0.6 minimum-viable inject. Decision recorded 2026-04-30: stock
	// estimates were wildly inflated (anecdotal but consistent across a
	// week of use); status-line-only didn't reach Claude; the predictor
	// was calibrated but invisible. Re-introducing a small block — gated
	// to non-global tiers and to non-pathological variance — trades a
	// bounded anchoring risk for actually moving Claude's prior.
	//
	// Tier gate: skip global / none. Global is biased short on new
	// projects (sketch dominated by maintainer/test sessions), so an
	// anchor would flip the failure mode from "wildly over" to "wildly
	// under" — same magnitude, opposite sign.
	//
	// Variance gate: if P75/P25 > 3, replace the point with the band as
	// the headline. A high-spread interval is honest where a precise-
	// looking point would falsely anchor.
	//
	// HINDCAST_INJECT=0 disables, default on.
	if arm == store.ArmTreatment && os.Getenv("HINDCAST_INJECT") != "0" {
		if s := formatClaudeInjection(pred); s != "" {
			fmt.Println(s)
		}
	}
}

// formatClaudeInjection renders the prediction as a Claude-facing block
// for UserPromptSubmit additionalContext. Returns "" when the tier or
// variance gate trips.
//
// Tier gate notes:
//   - regressor: per-prediction confidence is NOT gated — we rely on the
//     global gate in health.json (regressor only fires when it beats the
//     ladder by ≥15% on the user's held-out split). This means a specific
//     prompt could still be off; the variance gate is the per-prediction
//     backstop.
//   - knn: predictor itself enforces sim ≥ 0.45 via tune'd threshold.
//   - bucket / project: no per-prediction gate; rely on n ≥ 4 floor in
//     the predictor and the variance gate here.
func formatClaudeInjection(p predict.Prediction) string {
	switch p.Source {
	case predict.SourceRegressor, predict.SourceKNN, predict.SourceBucket, predict.SourceProject:
	default:
		return ""
	}
	if p.WallSeconds <= 0 {
		return ""
	}

	src := string(p.Source)
	switch p.Source {
	case predict.SourceKNN:
		src = fmt.Sprintf("knn sim=%.2f", p.MaxSim)
	case predict.SourceRegressor:
		if p.SourceDetail != "" {
			src = "regressor:" + p.SourceDetail
		}
	}

	var b strings.Builder
	b.WriteString("Hindcast prediction for this turn (calibrated from your past sessions on similar prompts):\n")

	// Active phrase: drop when zero. Records sometimes have active=0
	// from transcript-parse edge cases; surfacing "~0s active" is a
	// data-quality artifact, not a useful signal for Claude.
	activePhrase := ""
	if p.ActiveSeconds > 0 {
		activePhrase = fmt.Sprintf(" · active ~%s", humanDuration(p.ActiveSeconds))
	}

	// v0.6.3: dynamic variance gate threshold. The base threshold is
	// 3.0 (P75/P25 > 3 means a wide-enough spread that a point estimate
	// would be falsely precise). Small-sample or low-confidence kNN
	// matches drop the threshold to 2.0 — the underlying signal is
	// inherently noisier at low n / low sim, so the inject should
	// surface the band more readily rather than commit to a point.
	threshold := 3.0
	if p.Source == predict.SourceKNN && (p.N < 5 || p.MaxSim < 0.4) {
		threshold = 2.0
	}
	highVar := p.WallP25 > 0 && p.WallP75 > 0 && float64(p.WallP75)/float64(p.WallP25) > threshold
	if highVar {
		// v0.6.3: render the wider [P10, P90] when the variance gate
		// trips (when present — only kNN populates the wide quantiles).
		// A calibrated [P10, P90] hits ~80% of actuals vs ~50% for
		// [P25, P75], which is the "useful without overfitting" target.
		// Falls back to [P25, P75] when wide quantiles aren't populated.
		lo, hi := p.WallP25, p.WallP75
		bandLabel := "P25–P75"
		if p.WallP10 > 0 && p.WallP90 > 0 {
			lo, hi = p.WallP10, p.WallP90
			bandLabel = "P10–P90"
		}
		fmt.Fprintf(&b, "  wall %s–%s (high uncertainty, no point estimate; %s band)%s\n",
			humanDuration(lo), humanDuration(hi), bandLabel, activePhrase)
	} else {
		band := ""
		if p.WallP25 > 0 && p.WallP75 > 0 && p.WallP75 >= p.WallP25 {
			band = fmt.Sprintf(" (P25–P75: %s–%s)", humanDuration(p.WallP25), humanDuration(p.WallP75))
		}
		fmt.Fprintf(&b, "  wall ~%s%s%s\n",
			humanDuration(p.WallSeconds), band, activePhrase)
	}
	fmt.Fprintf(&b, "  source: %s · n=%d", src, p.N)
	if p.TaskType != "" {
		fmt.Fprintf(&b, " · task=%s", p.TaskType)
	}
	b.WriteString("\n\n")
	b.WriteString("If you give a wall-clock estimate this turn, cite this number in the form:\n")
	b.WriteString("  \"~Xm wall (P25–P75: a–b, source n=N)\"\n")
	b.WriteString("Use it as your baseline. Override only if the prompt has a structural reason\n")
	b.WriteString("the predictor cannot see (much larger scope, blocked on long external\n")
	b.WriteString("process, etc.) — say so explicitly. Do NOT pad the override out of caution;\n")
	b.WriteString("decompose into verify-existing vs implement-fresh and estimate each tightly.")
	return b.String()
}

// deliverBM25 prints one line of retrieval context for CC to inject.
// Shows only stats of the closest past match — no prompt text, since
// that's what the privacy-first design guarantees.
func deliverBM25(queryHashes []uint64, hash string) {
	if len(queryHashes) == 0 {
		return
	}
	bm25Path, err := store.ProjectBM25Path(hash)
	if err != nil {
		return
	}
	idx, err := bm25.Load(bm25Path)
	if err != nil || idx == nil || len(idx.Docs) == 0 {
		return
	}
	matches := idx.TopK(queryHashes, 1)
	if len(matches) == 0 || matches[0].Sim < 0.10 {
		return
	}
	m := matches[0]
	activeMin := float64(m.Doc.ActiveSeconds) / 60.0
	wallMin := float64(m.Doc.WallSeconds) / 60.0
	bucket := m.Doc.TaskType
	if m.Doc.SizeBucket != "" {
		bucket = fmt.Sprintf("%s-%s", m.Doc.TaskType, m.Doc.SizeBucket)
	}
	fmt.Printf("Closest past turn (BM25 sim=%.2f): %s, %d tools, %d files — active %.1fm / wall %.1fm\n",
		m.Sim, bucket, m.Doc.ToolCount, m.Doc.FilesTouched, activeMin, wallMin)
}

// computePrediction returns a Prediction for the current turn. The tier
// order is health-driven: if `hindcast tune` measured a learned regressor
// to beat the ladder by ≥15% on this user's 50/50 held-out split, the
// regressor goes first. Otherwise the existing kNN→bucket→project→global
// ladder runs unchanged. Soft-fails to SourceNone on any IO error.
func computePrediction(hash string, tokens []uint64, taskType, sessionID string, promptChars int) predict.Prediction {
	var idx *bm25.Index
	if bm25Path, err := store.ProjectBM25Path(hash); err == nil {
		if loaded, err := bm25.Load(bm25Path); err == nil {
			idx = loaded
		}
	}
	var records []store.Record
	if logPath, err := store.ProjectLogPath(hash); err == nil {
		records, _ = store.ReadRecentRecords(logPath, 500)
	}
	var sk *store.Sketch
	if loaded, err := store.LoadSketch(); err == nil {
		sk = loaded
	}

	if pred, ok := tryRegressorPrediction(tokens, taskType, promptChars, records, idx); ok {
		pred.SessionID = sessionID
		return pred
	}

	p := predict.Predict(tokens, idx, records, sk, taskType)
	p.SessionID = sessionID
	return p
}

// tryRegressorPrediction consults health.json's RegressorWinner and, if
// non-empty, loads the named model and returns its prediction. Any
// failure (no health, winner=none, model load error, prediction <= 0)
// returns ok=false so the caller falls through to the ladder.
func tryRegressorPrediction(tokens []uint64, taskType string, promptChars int, records []store.Record, idx *bm25.Index) (predict.Prediction, bool) {
	h, err := health.Load()
	if err != nil || h == nil {
		return predict.Prediction{}, false
	}
	if h.RegressorWinner != "gbdt" && h.RegressorWinner != "linear" {
		return predict.Prediction{}, false
	}
	ctx := regressor.MakeContext(promptChars, taskType, records, tokens, idx)
	var wall int
	var resP25, resP75 float64
	switch h.RegressorWinner {
	case "gbdt":
		m, err := regressor.Load()
		if err != nil || m == nil {
			return predict.Prediction{}, false
		}
		wall = m.PredictWall(ctx)
		resP25, resP75 = m.TrainResidualP25, m.TrainResidualP75
	case "linear":
		m, err := regressor.LoadLinear()
		if err != nil || m == nil {
			return predict.Prediction{}, false
		}
		wall = m.PredictWall(ctx)
		resP25, resP75 = m.TrainResidualP25, m.TrainResidualP75
	}
	if wall <= 0 {
		return predict.Prediction{}, false
	}
	// Band: residuals are (pred - actual) in log-space. p25 (often <0) means
	// model undershot; map to upper band. p75 (often >0) means model
	// overshot; map to lower band. In-sample residuals understate held-out
	// spread but beat a degenerate point band.
	wallLow, wallHigh := bandFromResiduals(wall, resP25, resP75)
	// Active seconds: scale by the same active/wall ratio kNN observed in
	// nearest neighbors when available, else fall back to wall as a floor.
	// Avoids the regressor being mute on active when status line wants it.
	active := wall
	if idx != nil && len(idx.Docs) > 0 && len(tokens) > 0 {
		matches := idx.TopK(tokens, 3)
		var num, den int
		for _, m := range matches {
			if m.Sim < 0.15 || m.Doc.WallSeconds <= 0 {
				continue
			}
			num += m.Doc.ActiveSeconds
			den += m.Doc.WallSeconds
		}
		if den > 0 {
			active = wall * num / den
			if active < 1 {
				active = 1
			}
		}
	}
	activeLow, activeHigh := wallLow, wallHigh
	if wall > 0 {
		activeLow = wallLow * active / wall
		activeHigh = wallHigh * active / wall
	}
	return predict.Prediction{
		WallSeconds:   wall,
		ActiveSeconds: active,
		WallP25:       wallLow,
		WallP75:       wallHigh,
		ActiveP25:     activeLow,
		ActiveP75:     activeHigh,
		N:             h.NHeldOut,
		MaxSim:        ctx.BM25MaxSim,
		Source:        predict.SourceRegressor,
		SourceDetail:  h.RegressorWinner,
		TaskType:      taskType,
	}, true
}

// bandFromResiduals turns log-space residual percentiles into a wall-second
// band around pred. Residual = log(pred) - log(actual); positive means
// overshoot. P25/P75 of residuals therefore become divisors that produce
// wallLow (from upper residual percentile) and wallHigh (from lower).
func bandFromResiduals(pred int, resP25, resP75 float64) (int, int) {
	if pred <= 0 || (resP25 == 0 && resP75 == 0) {
		return pred, pred
	}
	low := float64(pred) * math.Exp(-resP75)
	high := float64(pred) * math.Exp(-resP25)
	if low < 1 {
		low = 1
	}
	if high < low {
		high = low
	}
	return int(low + 0.5), int(high + 0.5)
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}
