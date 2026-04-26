// Package seed ships default numeric priors embedded in the binary so
// a fresh `hindcast install` on a machine with no CC history still has
// something to show Claude at SessionStart. Numeric-only — sample
// vectors of wall/active durations from the maintainer's own backfilled
// CC sessions. No prompt text, no project identity, no hashes.
//
// Regenerate via:  hindcast export-seed > internal/seed/priors.json
package seed

import (
	_ "embed"
	"encoding/json"
)

//go:embed priors.json
var priorsBytes []byte

type Priors struct {
	Wall   []int `json:"wall_seconds"`
	Active []int `json:"active_seconds"`
}

// Default returns the embedded priors. Empty arrays mean "no seed shipped."
func Default() Priors {
	var p Priors
	_ = json.Unmarshal(priorsBytes, &p)
	return p
}
