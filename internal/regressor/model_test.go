package regressor

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/justinstimatze/hindcast/internal/store"
)

// Synthetic byProject big enough to clear MinTrainRecords. Two projects,
// 80 records each, so sample size = 160 - warmup*2 well above the floor.
// Wall durations correlate with PromptChars so even a depth-3 GBDT
// should learn a non-trivial relationship.
func makeSynthByProject(t *testing.T) map[string][]store.Record {
	t.Helper()
	out := map[string][]store.Record{}
	for proj := 0; proj < 2; proj++ {
		var recs []store.Record
		for i := 0; i < 80; i++ {
			tt := []string{"refactor", "debug", "feature"}[i%3]
			chars := 100 + i*10
			wall := 30 + i*2 // monotonic in i, so chars predicts wall
			recs = append(recs, store.Record{
				TS:                  time.Date(2026, 1, 1, 0, 0, i, 0, time.UTC),
				TaskType:            tt,
				PromptChars:         chars,
				WallSeconds:         wall,
				ClaudeActiveSeconds: wall / 2,
				ToolCalls:           map[string]int{"Edit": 1, "Read": 2},
			})
		}
		out["proj-"+string(rune('A'+proj))] = recs
	}
	return out
}

// Training on the synthetic data must succeed and produce a model whose
// in-sample MALR is meaningfully better than 1.0× (perfect) but worse
// than chance. The point is to catch a regression where Train returns
// without learning — e.g., trees produce zero residuals or BaseValue
// drifts away from log(median).
func TestTrainGBDTSucceedsOnSynthetic(t *testing.T) {
	by := makeSynthByProject(t)
	m, err := Train(by, 5)
	if err != nil {
		t.Fatalf("Train failed: %s", err)
	}
	if m.NTrain < MinTrainRecords {
		t.Fatalf("NTrain=%d below floor", m.NTrain)
	}
	if len(m.Trees) != 100 {
		t.Errorf("expected 100 trees, got %d", len(m.Trees))
	}
	if m.TrainMALR <= 1.0 {
		t.Errorf("in-sample MALR=%.2f looks too perfect (suspect leak)", m.TrainMALR)
	}
}

// Save/Load roundtrip: a freshly trained model must produce identical
// predictions before and after persistence. Catches gob schema drift if
// someone reorders fields in Model.
func TestGBDTSaveLoadRoundtrip(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	if err := os.MkdirAll(filepath.Join(tmp, ".claude", "hindcast"), 0700); err != nil {
		t.Fatalf("mkdir: %s", err)
	}

	m, err := Train(makeSynthByProject(t), 5)
	if err != nil {
		t.Fatalf("Train: %s", err)
	}
	if err := m.Save(); err != nil {
		t.Fatalf("Save: %s", err)
	}
	loaded, err := Load()
	if err != nil {
		t.Fatalf("Load: %s", err)
	}
	ctx := Context{TaskType: "refactor", PromptChars: 250}
	a := m.PredictWall(ctx)
	b := loaded.PredictWall(ctx)
	if a != b {
		t.Fatalf("predictions diverged after save/load: %d vs %d", a, b)
	}
}

// Same roundtrip for the linear model — separate path on disk, separate
// gob encoding, can independently regress.
func TestLinearSaveLoadRoundtrip(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	if err := os.MkdirAll(filepath.Join(tmp, ".claude", "hindcast"), 0700); err != nil {
		t.Fatalf("mkdir: %s", err)
	}

	lm, err := TrainLinearFromRecords(makeSynthByProject(t), 5, 1.0)
	if err != nil {
		t.Fatalf("TrainLinear: %s", err)
	}
	if err := lm.Save(); err != nil {
		t.Fatalf("Save: %s", err)
	}
	loaded, err := LoadLinear()
	if err != nil {
		t.Fatalf("LoadLinear: %s", err)
	}
	ctx := Context{TaskType: "feature", PromptChars: 600}
	a := lm.PredictWall(ctx)
	b := loaded.PredictWall(ctx)
	if a != b {
		t.Fatalf("predictions diverged after save/load: %d vs %d", a, b)
	}
}

// Insufficient data must surface as IsInsufficient(err) so callers can
// distinguish "not ready yet" from real failures (file IO, malformed
// records). cmd_train.go relies on this distinction for its exit code.
func TestTrainReturnsInsufficientOnSparseData(t *testing.T) {
	sparse := map[string][]store.Record{
		"tiny": {{TS: time.Now(), WallSeconds: 10, TaskType: "refactor"}},
	}
	_, err := Train(sparse, 5)
	if err == nil {
		t.Fatal("expected error on sparse data, got nil")
	}
	if !IsInsufficient(err) {
		t.Errorf("expected errInsufficient, got %v", err)
	}
}
