package regressor

import (
	"encoding/gob"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/store"
)

// MinTrainRecords is the floor for activating the regressor tier. Below
// this the model overfits noise; we fall through to kNN/bucket. Picked
// to roughly match where bench-cross MALR stabilizes in practice.
const MinTrainRecords = 100

var (
	errEmpty       = errors.New("regressor: no training data")
	errInsufficient = errors.New("regressor: insufficient training data")
)

// Model is the persisted regressor: the boosted tree ensemble plus
// metadata for cold-start checks (when was it trained, on how many rows,
// how did it score on its own training data).
type Model struct {
	BaseValue    float64
	LR           float64
	Trees        []Tree
	FeatureNames []string

	// Metadata for diagnostics + staleness.
	TrainedAt    time.Time
	NTrain       int
	TrainMALR    float64 // in-sample MALR (sanity check, not held-out)
}

// Predict returns the model's prediction in log-space. Use PredictWall
// for the seconds-domain caller-friendly version.
func (m *Model) Predict(feats []float64) float64 {
	pred := m.BaseValue
	for i := range m.Trees {
		pred += m.LR * m.Trees[i].Predict(feats)
	}
	return pred
}

// PredictWall returns the predicted wall_seconds for the given Context.
func (m *Model) PredictWall(ctx Context) int {
	feats := Extract(ctx)
	logPred := m.Predict(feats)
	v := math.Exp(logPred)
	if v < 1 {
		v = 1
	}
	return int(v + 0.5)
}

// GBDTModelPath is where the per-install GBDT model lives.
func GBDTModelPath() (string, error) {
	root, err := store.HindcastDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", err
	}
	return filepath.Join(root, "regressor.gbdt.gob"), nil
}

// Save writes the GBDT model atomically. Caller is responsible for path safety.
func (m *Model) Save() error {
	path, err := GBDTModelPath()
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".regressor-*.gob")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() {
		tmp.Close()
		os.Remove(tmpName)
	}
	if err := tmp.Chmod(0600); err != nil {
		cleanup()
		return err
	}
	if err := gob.NewEncoder(tmp).Encode(m); err != nil {
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// Load returns the persisted GBDT model, or nil + error if absent / unreadable.
func Load() (*Model, error) {
	path, err := GBDTModelPath()
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var m Model
	if err := gob.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

// Train builds a model from the user's records grouped by project. For
// each record past warmup it computes prefix features (history = records
// before this one in the same project) and the BM25 kNN signal as it
// would have looked at prediction time. Trains on log(wall_seconds).
//
// Returns nil + errInsufficient if fewer than MinTrainRecords usable rows
// emerge across all projects — at that point the regressor isn't ready
// to displace the kNN tier and the caller should fall through.
func Train(byProject map[string][]store.Record, warmup int) (*Model, error) {
	X, y := buildTrainingMatrix(byProject, warmup)
	if len(X) < MinTrainRecords {
		return nil, errInsufficient
	}

	m, err := trainGBDT(X, y, DefaultParams())
	if err != nil {
		return nil, err
	}
	m.TrainedAt = time.Now().UTC()
	m.NTrain = len(X)

	preds := make([]float64, len(X))
	for i := range X {
		preds[i] = m.Predict(X[i])
	}
	m.TrainMALR = MAD(preds, y)

	return m, nil
}

// IsInsufficient reports whether err is the "not enough data yet" signal.
// Lets callers distinguish "regressor isn't ready" from a real failure.
func IsInsufficient(err error) bool {
	return errors.Is(err, errInsufficient)
}

// Ensemble blends regressor and kNN predictions in log-space weighted by
// max_sim. Motivation: cross-corpus eval showed regressor wins when
// max_sim is low (no good lexical match → universal features add signal)
// and kNN wins when max_sim is high (close lexical match → past durations
// are strongly predictive). Linear log-space blend is the simplest
// continuous interpolation between the two regimes.
//
// At maxSim=0, returns regressor; at maxSim=1, returns kNN. If either
// input is non-positive, returns the other (graceful degradation when
// one tier didn't fire).
func Ensemble(regressorWall, knnWall int, maxSim float64) int {
	if regressorWall <= 0 && knnWall <= 0 {
		return 0
	}
	if knnWall <= 0 {
		return regressorWall
	}
	if regressorWall <= 0 {
		return knnWall
	}
	w := maxSim
	if w < 0 {
		w = 0
	} else if w > 1 {
		w = 1
	}
	logBlend := w*math.Log(float64(knnWall)) + (1-w)*math.Log(float64(regressorWall))
	v := math.Exp(logBlend)
	if v < 1 {
		v = 1
	}
	return int(v + 0.5)
}

// weightedMedianWall mirrors the kNN tier's aggregation so the "BM25 kNN
// pred" feature matches what predict.Predict would output. Kept local so
// the regressor doesn't depend on the predict package (would create a
// cycle once predict imports regressor).
func weightedMedianWall(ms []bm25.Match) float64 {
	type wv struct{ value, weight float64 }
	ws := make([]wv, 0, len(ms))
	for _, m := range ms {
		ws = append(ws, wv{float64(m.Doc.WallSeconds), m.Sim})
	}
	sort.Slice(ws, func(i, j int) bool { return ws[i].value < ws[j].value })
	total := 0.0
	for _, w := range ws {
		total += w.weight
	}
	if total <= 0 {
		return ws[len(ws)/2].value
	}
	target := 0.5 * total
	cum := 0.0
	for _, w := range ws {
		cum += w.weight
		if cum >= target {
			return w.value
		}
	}
	return ws[len(ws)-1].value
}
