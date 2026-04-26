package regressor

import (
	"errors"
	"math"
	"sort"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/store"
)

// LinearModel is ridge regression in log-target space over the canonical
// feature set. Lives alongside Model (GBDT) so we can compare model
// classes empirically — if the GBDT is overfitting at our sample sizes,
// linear will outperform on held-out evaluation.
//
// Predicts in log-space: y_log = bias + Σ w_j × (x_j - mean_j) / std_j.
// Standardization is stored on the model so inference uses the same
// scaling the model was trained under.
type LinearModel struct {
	Weights      []float64
	Bias         float64 // mean(y) in training
	FeatureMeans []float64
	FeatureStds  []float64 // 1.0 substituted for zero-variance columns
	FeatureNames []string

	TrainedAt time.Time
	NTrain    int
	TrainMALR float64
	Lambda    float64
}

// TrainLinear fits a ridge regression on (X, y) under squared loss.
// lambda is the L2 penalty; 1.0 is a sensible default for standardized
// features at our sample sizes (a few thousand rows × 21 features).
func TrainLinear(X [][]float64, y []float64, lambda float64) (*LinearModel, error) {
	if len(X) == 0 || len(X) != len(y) {
		return nil, errEmpty
	}
	n := len(X)
	d := len(X[0])

	yMean := 0.0
	for _, v := range y {
		yMean += v
	}
	yMean /= float64(n)

	means := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			means[j] += X[i][j]
		}
	}
	for j := range means {
		means[j] /= float64(n)
	}

	stds := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			diff := X[i][j] - means[j]
			stds[j] += diff * diff
		}
	}
	for j := range stds {
		stds[j] = math.Sqrt(stds[j] / float64(n))
		if stds[j] < 1e-9 {
			stds[j] = 1.0 // zero-variance feature → leave at zero post-centering, no scaling
		}
	}

	// Build X^T X and X^T y on standardized X with centered y.
	XtX := make([][]float64, d)
	for j := range XtX {
		XtX[j] = make([]float64, d)
	}
	Xty := make([]float64, d)
	xs := make([]float64, d) // scratch row
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			xs[j] = (X[i][j] - means[j]) / stds[j]
		}
		yc := y[i] - yMean
		for j := 0; j < d; j++ {
			Xty[j] += xs[j] * yc
			for k := j; k < d; k++ {
				XtX[j][k] += xs[j] * xs[k]
			}
		}
	}
	for j := 0; j < d; j++ {
		for k := 0; k < j; k++ {
			XtX[j][k] = XtX[k][j]
		}
		XtX[j][j] += lambda
	}

	w, err := solveLinear(XtX, Xty)
	if err != nil {
		return nil, err
	}

	m := &LinearModel{
		Weights:      w,
		Bias:         yMean,
		FeatureMeans: means,
		FeatureStds:  stds,
		FeatureNames: append([]string(nil), FeatureNames...),
		TrainedAt:    time.Now().UTC(),
		NTrain:       n,
		Lambda:       lambda,
	}
	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		preds[i] = m.Predict(X[i])
	}
	m.TrainMALR = MAD(preds, y)
	return m, nil
}

// Predict returns log-space prediction for one feature row.
func (m *LinearModel) Predict(feats []float64) float64 {
	pred := m.Bias
	for j := range feats {
		x := (feats[j] - m.FeatureMeans[j]) / m.FeatureStds[j]
		pred += x * m.Weights[j]
	}
	return pred
}

// PredictWall returns predicted wall_seconds for the given Context.
func (m *LinearModel) PredictWall(ctx Context) int {
	feats := Extract(ctx)
	logPred := m.Predict(feats)
	v := math.Exp(logPred)
	if v < 1 {
		v = 1
	}
	return int(v + 0.5)
}

// solveLinear solves Aw = b via Gauss-Jordan with partial pivoting. A is
// destructively modified.
func solveLinear(A [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	if len(A) != n {
		return nil, errors.New("regressor: dimension mismatch")
	}
	// Augmented matrix [A | b].
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, n+1)
		copy(aug[i], A[i])
		aug[i][n] = b[i]
	}
	for i := 0; i < n; i++ {
		// Partial pivot.
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(aug[k][i]) > math.Abs(aug[maxRow][i]) {
				maxRow = k
			}
		}
		aug[i], aug[maxRow] = aug[maxRow], aug[i]
		if math.Abs(aug[i][i]) < 1e-12 {
			return nil, errors.New("regressor: singular matrix")
		}
		// Eliminate column i from all other rows.
		for k := 0; k < n; k++ {
			if k == i {
				continue
			}
			factor := aug[k][i] / aug[i][i]
			for j := i; j <= n; j++ {
				aug[k][j] -= factor * aug[i][j]
			}
		}
	}
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = aug[i][n] / aug[i][i]
	}
	return w, nil
}

// TrainLinearFromRecords mirrors Train but fits a LinearModel. Same
// feature pipeline, different model class. Returns nil + errInsufficient
// when fewer than MinTrainRecords usable rows.
func TrainLinearFromRecords(byProject map[string][]store.Record, warmup int, lambda float64) (*LinearModel, error) {
	X, y := buildTrainingMatrix(byProject, warmup)
	if len(X) < MinTrainRecords {
		return nil, errInsufficient
	}
	return TrainLinear(X, y, lambda)
}

// buildTrainingMatrix is the shared feature-extraction pipeline used by
// both Train (GBDT) and TrainLinearFromRecords. Pulled out so the two
// model classes train on identical inputs.
func buildTrainingMatrix(byProject map[string][]store.Record, warmup int) ([][]float64, []float64) {
	var X [][]float64
	var y []float64

	for _, recs := range byProject {
		if len(recs) < warmup+5 {
			continue
		}
		sorted := make([]store.Record, len(recs))
		copy(sorted, recs)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].TS.Before(sorted[j].TS) })

		idx := bm25.New()
		for i, r := range sorted {
			if i >= warmup && r.WallSeconds > 0 {
				ctx := Context{
					PromptChars: r.PromptChars,
					TaskType:    r.TaskType,
					SizeBucket:  r.SizeBucket,
					History:     sorted[:i],
				}
				if len(r.PromptTokens) > 0 && len(idx.Docs) > 0 {
					matches := idx.TopK(r.PromptTokens, 7)
					var good []bm25.Match
					for _, m := range matches {
						if m.Sim >= 0.15 {
							good = append(good, m)
						}
					}
					if len(matches) > 0 {
						ctx.BM25MaxSim = matches[0].Sim
					}
					if len(good) >= 3 {
						ctx.BM25PredWall = int(weightedMedianWall(good) + 0.5)
					}
				}
				X = append(X, Extract(ctx))
				y = append(y, math.Log(float64(r.WallSeconds)))
			}
			if len(r.PromptTokens) > 0 {
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
	}
	return X, y
}
