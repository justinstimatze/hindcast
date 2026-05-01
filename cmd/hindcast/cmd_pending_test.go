package main

import (
	"strings"
	"testing"

	"github.com/justinstimatze/hindcast/internal/predict"
)

// TestFormatClaudeInjectionTierGate exercises the tier gate. Only
// regressor / knn / bucket / project should emit; global / none must
// stay silent (returning ""), since global p50 is biased short on new
// projects and would flip Claude's failure mode from "wildly over" to
// "wildly under."
func TestFormatClaudeInjectionTierGate(t *testing.T) {
	cases := []struct {
		src     predict.Source
		wantOut bool
	}{
		{predict.SourceRegressor, true},
		{predict.SourceKNN, true},
		{predict.SourceBucket, true},
		{predict.SourceProject, true},
		{predict.SourceGlobal, false},
		{predict.SourceNone, false},
	}
	for _, tc := range cases {
		t.Run(string(tc.src), func(t *testing.T) {
			p := predict.Prediction{
				Source:        tc.src,
				WallSeconds:   60,
				ActiveSeconds: 30,
				WallP25:       45,
				WallP75:       90,
				N:             5,
				MaxSim:        0.8,
				TaskType:      "feature",
			}
			got := formatClaudeInjection(p)
			if (got != "") != tc.wantOut {
				t.Errorf("source=%s: emitted=%v want=%v\n%s", tc.src, got != "", tc.wantOut, got)
			}
		})
	}
}

// TestFormatClaudeInjectionVarianceGate verifies that when P75/P25 > 3
// the band is rendered as the headline. The variance gate is the
// per-prediction backstop against falsely-precise point estimates.
func TestFormatClaudeInjectionVarianceGate(t *testing.T) {
	tight := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 120,
		WallP25: 60, WallP75: 180, // ratio 3.0 — boundary, not high-var
		N: 5, MaxSim: 0.8,
	}
	wide := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 120,
		WallP25: 30, WallP75: 600, // ratio 20 — high variance
		N: 5, MaxSim: 0.8,
	}

	tightOut := formatClaudeInjection(tight)
	if strings.Contains(tightOut, "high uncertainty") {
		t.Errorf("ratio=3.0 should NOT trip variance gate, got high-uncertainty output:\n%s", tightOut)
	}
	if !strings.Contains(tightOut, "wall ~2m") {
		t.Errorf("tight band should render point estimate, got:\n%s", tightOut)
	}

	wideOut := formatClaudeInjection(wide)
	if !strings.Contains(wideOut, "high uncertainty") {
		t.Errorf("ratio=20 should trip variance gate, got:\n%s", wideOut)
	}
	if strings.Contains(wideOut, "wall ~2m ") {
		t.Errorf("wide band should NOT render point estimate, got:\n%s", wideOut)
	}
}

// TestFormatClaudeInjectionActiveZeroSuppressed confirms the active
// phrase is dropped when ActiveSeconds == 0. This protects against
// surfacing a data-quality artifact as if it were signal.
func TestFormatClaudeInjectionActiveZeroSuppressed(t *testing.T) {
	p := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 60,
		ActiveSeconds: 0,
		WallP25:       45, WallP75: 90,
		N: 5, MaxSim: 0.8,
	}
	got := formatClaudeInjection(p)
	if strings.Contains(got, "active") {
		t.Errorf("active=0 should suppress phrase, got:\n%s", got)
	}

	p.ActiveSeconds = 30
	got = formatClaudeInjection(p)
	if !strings.Contains(got, "active ~30s") {
		t.Errorf("active>0 should render phrase, got:\n%s", got)
	}
}

// TestFormatClaudeInjectionDontPadGuidance is a regression test: the
// "don't pad on override" rule must appear in the per-turn injection,
// not just CLAUDE.md. CLAUDE.md is loaded once at session start and
// competes with every other section for attention; the per-turn block
// is where the rule actually fires.
func TestFormatClaudeInjectionDontPadGuidance(t *testing.T) {
	p := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 60,
		WallP25: 45, WallP75: 90, N: 5, MaxSim: 0.8,
	}
	got := formatClaudeInjection(p)
	if !strings.Contains(strings.ToLower(got), "do not pad") &&
		!strings.Contains(strings.ToLower(got), "don't pad") &&
		!strings.Contains(strings.ToLower(got), "do not pad the override") {
		t.Errorf("injection must include don't-pad guidance, got:\n%s", got)
	}
	if !strings.Contains(got, "verify-existing") {
		t.Errorf("injection must include decompose-into-verify-vs-implement guidance, got:\n%s", got)
	}
}

// TestFormatClaudeInjectionSourceRendering verifies per-source label,
// including the regressor-without-SourceDetail fallback (we want the
// label to degrade to "regressor", not blank).
func TestFormatClaudeInjectionSourceRendering(t *testing.T) {
	cases := []struct {
		name string
		p    predict.Prediction
		want string
	}{
		{
			"knn shows similarity",
			predict.Prediction{
				Source: predict.SourceKNN, WallSeconds: 60,
				WallP25: 45, WallP75: 90, N: 5, MaxSim: 0.71,
			},
			"knn sim=0.71",
		},
		{
			"regressor with detail",
			predict.Prediction{
				Source: predict.SourceRegressor, SourceDetail: "linear",
				WallSeconds: 60, WallP25: 45, WallP75: 90, N: 5,
			},
			"regressor:linear",
		},
		{
			"regressor without detail degrades cleanly",
			predict.Prediction{
				Source:      predict.SourceRegressor, // no SourceDetail
				WallSeconds: 60, WallP25: 45, WallP75: 90, N: 5,
			},
			"source: regressor",
		},
		{
			"bucket",
			predict.Prediction{
				Source: predict.SourceBucket, WallSeconds: 60,
				WallP25: 45, WallP75: 90, N: 5,
			},
			"bucket",
		},
		{
			"project",
			predict.Prediction{
				Source: predict.SourceProject, WallSeconds: 60,
				WallP25: 45, WallP75: 90, N: 5,
			},
			"project",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := formatClaudeInjection(tc.p)
			if !strings.Contains(got, tc.want) {
				t.Errorf("expected %q in output, got:\n%s", tc.want, got)
			}
		})
	}
}

// TestFormatClaudeInjectionZeroWallSuppressed: a prediction with a
// non-positive WallSeconds is degenerate (predictor produced nothing
// usable). Suppress entirely rather than emit a "wall ~0s" anchor.
func TestFormatClaudeInjectionZeroWallSuppressed(t *testing.T) {
	cases := []int{0, -1}
	for _, w := range cases {
		p := predict.Prediction{
			Source: predict.SourceKNN, WallSeconds: w,
			WallP25: 45, WallP75: 90, N: 5, MaxSim: 0.8,
		}
		if got := formatClaudeInjection(p); got != "" {
			t.Errorf("WallSeconds=%d should suppress, got:\n%s", w, got)
		}
	}
}

// TestFormatClaudeInjectionVarianceBoundary locks in the boundary
// behavior at exactly ratio=3.0. Code uses `> 3.0`; ratio==3 should
// render a point estimate, ratio just above should render the band.
// If someone changes the operator to `>=`, this catches it.
func TestFormatClaudeInjectionVarianceBoundary(t *testing.T) {
	atBoundary := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 60,
		WallP25: 60, WallP75: 180, // ratio = 3.0 exactly
		N: 5, MaxSim: 0.8,
	}
	if got := formatClaudeInjection(atBoundary); strings.Contains(got, "high uncertainty") {
		t.Errorf("ratio=3.0 (boundary) should NOT trip variance gate, got:\n%s", got)
	}

	justOver := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 60,
		WallP25: 60, WallP75: 181, // ratio just over 3
		N: 5, MaxSim: 0.8,
	}
	if got := formatClaudeInjection(justOver); !strings.Contains(got, "high uncertainty") {
		t.Errorf("ratio=181/60 (just over 3) should trip variance gate, got:\n%s", got)
	}
}

// v0.6.3: dynamic variance gate threshold. The base threshold is
// P75/P25 > 3.0; small-n (<5) or low-sim (<0.4) kNN matches drop to 2.0.
// This expresses "the predictor is uncertain about this turn anyway, so
// surface the band rather than commit to a point."
func TestFormatClaudeInjectionDynamicVarianceThreshold(t *testing.T) {
	// Ratio = 2.5 (P75/P25 = 100/40). Above the small-n threshold (2.0)
	// but below the base threshold (3.0). Behavior depends on n / MaxSim.
	cases := []struct {
		name      string
		p         predict.Prediction
		wantPoint bool // true: point estimate rendered (no high-uncertainty)
	}{
		{
			"large-n, high-sim → point rendered (ratio 2.5 < base 3.0)",
			predict.Prediction{
				Source: predict.SourceKNN, WallSeconds: 60,
				WallP25: 40, WallP75: 100, N: 10, MaxSim: 0.7,
			},
			true,
		},
		{
			"small-n trips lowered threshold (ratio 2.5 > 2.0)",
			predict.Prediction{
				Source: predict.SourceKNN, WallSeconds: 60,
				WallP25: 40, WallP75: 100, N: 3, MaxSim: 0.7,
			},
			false,
		},
		{
			"low-sim trips lowered threshold (ratio 2.5 > 2.0)",
			predict.Prediction{
				Source: predict.SourceKNN, WallSeconds: 60,
				WallP25: 40, WallP75: 100, N: 10, MaxSim: 0.3,
			},
			false,
		},
		{
			"non-kNN tier uses base threshold regardless of n",
			predict.Prediction{
				Source: predict.SourceBucket, WallSeconds: 60,
				WallP25: 40, WallP75: 100, N: 3,
			},
			true, // bucket tier doesn't get the small-n drop
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := formatClaudeInjection(c.p)
			highVar := strings.Contains(got, "high uncertainty")
			if c.wantPoint && highVar {
				t.Errorf("expected point estimate, got high-uncertainty band:\n%s", got)
			}
			if !c.wantPoint && !highVar {
				t.Errorf("expected variance gate to trip, got point estimate:\n%s", got)
			}
		})
	}
}

// v0.6.3: when the variance gate trips and WallP10/WallP90 are
// populated, the inject renders the wider [P10, P90] band as the
// headline. Falls back to [P25, P75] if wide quantiles aren't there
// (e.g., bucket / project tiers, or pre-v0.6.3 readers).
func TestFormatClaudeInjectionRendersWidePercentilesWhenAvailable(t *testing.T) {
	withWide := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 120,
		WallP25: 60, WallP75: 200, // ratio 3.33 — trips base
		WallP10: 30, WallP90: 480,
		N: 7, MaxSim: 0.5,
	}
	got := formatClaudeInjection(withWide)
	if !strings.Contains(got, "P10–P90 band") {
		t.Errorf("expected P10–P90 label in headline, got:\n%s", got)
	}
	// And the rendered numbers are P10/P90, not P25/P75.
	if !strings.Contains(got, "30s") || !strings.Contains(got, "8m") {
		t.Errorf("expected 30s and 8m (P10/P90 boundaries) in band, got:\n%s", got)
	}

	withoutWide := predict.Prediction{
		Source: predict.SourceBucket, WallSeconds: 120,
		WallP25: 60, WallP75: 200, // ratio 3.33
		// No P10/P90 (bucket tier)
		N: 8,
	}
	got = formatClaudeInjection(withoutWide)
	if !strings.Contains(got, "P25–P75 band") {
		t.Errorf("expected P25–P75 fallback label when wide quantiles unavailable, got:\n%s", got)
	}
}

// TestFormatClaudeInjectionCitationFormSpec: the injection must
// instruct Claude to cite in the literal form `~Xm wall (P25–P75: a–b,
// source n=N)`. If the format string drifts, downstream parsing of
// Claude's responses (or future eval-api work) breaks silently.
func TestFormatClaudeInjectionCitationFormSpec(t *testing.T) {
	p := predict.Prediction{
		Source: predict.SourceKNN, WallSeconds: 60,
		WallP25: 45, WallP75: 90, N: 5, MaxSim: 0.8,
	}
	got := formatClaudeInjection(p)
	if !strings.Contains(got, "~Xm wall (P25–P75: a–b, source n=N)") {
		t.Errorf("citation form spec drifted; expected literal form in output, got:\n%s", got)
	}
}
