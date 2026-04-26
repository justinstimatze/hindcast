package main

import (
	"encoding/json"
	"fmt"
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

	// Compute prediction for the human via status line. Claude never sees
	// this — it's written to disk for `hindcast statusline` to render.
	pred := computePrediction(hash, tokens, taskType, in.SessionID, len(in.Prompt))
	writeLastPrediction(in.SessionID, pred)

	if path, err := store.PendingPath(in.SessionID); err == nil {
		p := store.PendingTurn{
			SessionID:       in.SessionID,
			StartTS:         time.Now().UTC(),
			TaskType:        taskType,
			PromptTokens:    tokens,
			PromptChars:     len(in.Prompt),
			PermissionMode:  in.PermissionMode,
			ProjectHash:     hash,
			CWD:             in.CWD,
			Arm:             arm,
			PredictedWall:   pred.WallSeconds,
			PredictedActive: pred.ActiveSeconds,
			PredictionSrc:   string(pred.Source),
		}
		if err := store.WritePending(path, p); err != nil {
			hook.Logf("pending", "write pending: %s", err)
		}
	} else {
		hook.Logf("pending", "path: %s", err)
	}

	// Default post-v0.2: emit nothing to Claude. The lit review
	// (Lou et al. 2024 on LLM anchoring; Vacareanu 2024 on retrieval
	// vs regression) showed that our "inject a number, tell Claude to
	// use it" mechanism is anchoring, not calibration — if the bucket
	// is wrong, we confidently mislead. Product pivots to surfacing
	// predictions to the human via status line, not into Claude's
	// context. Legacy anchoring behavior stays available behind the
	// HINDCAST_LEGACY_INJECT=1 flag so eval-api can A/B old-vs-new.
	if arm == store.ArmTreatment && os.Getenv("HINDCAST_LEGACY_INJECT") == "1" {
		fmt.Print(hindcastPrior(in.Prompt, effective, in.CWD, in.SessionID))
	}
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
	switch h.RegressorWinner {
	case "gbdt":
		m, err := regressor.Load()
		if err != nil || m == nil {
			return predict.Prediction{}, false
		}
		wall = m.PredictWall(ctx)
	case "linear":
		m, err := regressor.LoadLinear()
		if err != nil || m == nil {
			return predict.Prediction{}, false
		}
		wall = m.PredictWall(ctx)
	}
	if wall <= 0 {
		return predict.Prediction{}, false
	}
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
	return predict.Prediction{
		WallSeconds:   wall,
		ActiveSeconds: active,
		WallP25:       wall, // regressor returns a point estimate; band reuses it
		WallP75:       wall,
		ActiveP25:     active,
		ActiveP75:     active,
		N:             h.NHeldOut,
		MaxSim:        ctx.BM25MaxSim,
		Source:        predict.SourceRegressor,
		TaskType:      taskType,
	}, true
}

// writeLastPrediction persists the current prediction to both a
// per-session file and the active-session pointer so `hindcast
// statusline` can find the latest value without a session arg.
func writeLastPrediction(sessionID string, p predict.Prediction) {
	path, err := store.LastPredictionPath(sessionID)
	if err != nil {
		hook.Logf("pending", "last-prediction path: %s", err)
		return
	}
	data, err := json.Marshal(p)
	if err != nil {
		return
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		hook.Logf("pending", "last-prediction write: %s", err)
		return
	}
	if ptr, err := store.CurrentSessionPointerPath(); err == nil {
		_ = os.WriteFile(ptr, []byte(sessionID), 0600)
	}
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}
