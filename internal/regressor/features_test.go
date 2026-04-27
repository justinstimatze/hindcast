package regressor

import "testing"

// Extract MUST return exactly len(FeatureNames) values. The bug class is:
// add a name to FeatureNames but forget to set its column in Extract (or
// vice versa). Either way, models train and predict but features get
// silently misaligned. Length parity is the cheapest possible canary.
func TestExtractLengthMatchesFeatureNames(t *testing.T) {
	feats := Extract(Context{TaskType: "refactor", PromptChars: 100})
	if len(feats) != len(FeatureNames) {
		t.Fatalf("Extract returned %d features but FeatureNames has %d", len(feats), len(FeatureNames))
	}
}

// Task one-hot must be mutually exclusive: exactly one of task_* is 1
// per row. Catches duplicated assignments after a refactor.
func TestExtractTaskOneHotExclusive(t *testing.T) {
	taskCols := []int{1, 2, 3, 4, 5, 6}
	cases := map[string]int{"refactor": 1, "debug": 2, "feature": 3, "test": 4, "docs": 5, "unknown": 6}
	for tt, want := range cases {
		feats := Extract(Context{TaskType: tt})
		var hot []int
		for _, c := range taskCols {
			if feats[c] == 1 {
				hot = append(hot, c)
			}
		}
		if len(hot) != 1 || hot[0] != want {
			t.Errorf("task=%q: expected only column %d hot, got %v", tt, want, hot)
		}
	}
}

// SizeBucket was removed in v0.5 (post-turn data leak). Guard against
// reintroduction by name — Context has no SizeBucket field, and
// FeatureNames must not contain a size_* column.
func TestNoSizeFeaturesInSchema(t *testing.T) {
	for _, n := range FeatureNames {
		if len(n) >= 5 && n[:5] == "size_" {
			t.Fatalf("size_* feature reintroduced into FeatureNames: %q — this was the v0.4 train/predict leak", n)
		}
	}
}
