// Shipped bias-correction defaults by model, used when a project has
// too few live samples to compute its own. Values are measured where
// possible and rough where not.
//
// Regenerate via `hindcast eval-api --n 200` for a specific model,
// then record the treatment-arm bias column here.
package seed

type BiasDefault struct {
	WallFactor   float64 // claude_estimate_wall / actual_wall, measured at median
	ActiveFactor float64 // same for active_seconds
	Source       string  // provenance — "eval-api v3 n=117 seed=99" etc.
}

// shippedBias is the measured baseline, compounded across eval-api
// iterations. v3 measured treatment bias 1.70× (after imperative + few-
// shot + task-matched priors); v4 applied that 1.70× correction and
// measured residual treatment bias 1.46× — the correction undershot.
// Compound: 1.70 × 1.46 = 2.48× is the full effective correction.
//
// Models not listed fall back to a conservative 2.0× — still stronger
// than the pre-iteration 1.50× guess.
var shippedBias = map[string]BiasDefault{
	"claude-sonnet-4-6": {
		WallFactor:   2.48,
		ActiveFactor: 2.48,
		Source:       "eval-api v3+v4 compound n=117 seed=99",
	},
}

func BiasDefaultFor(model string) BiasDefault {
	if v, ok := shippedBias[model]; ok {
		return v
	}
	return BiasDefault{
		WallFactor:   2.00,
		ActiveFactor: 2.00,
		Source:       "default (unmeasured model)",
	}
}
