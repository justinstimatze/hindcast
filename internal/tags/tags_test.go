package tags

import "testing"

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want TaskType
	}{
		{"empty", "", Other},
		{"whitespace", "   ", Other},
		{"question non-task", "what's the latency impact here", Other},

		{"debug fix flake", "fix flake in auth", Debug},
		{"debug crash", "app crashes on startup", Debug},
		{"debug regression", "fix the regression from yesterday's merge", Debug},
		{"debug leak", "memory leak in image decoder", Debug},

		{"refactor rename", "rename Fetcher to Client", Refactor},
		{"refactor rewrite", "rewrite auth module", Refactor},
		{"refactor migrate", "migrate payment service to new API", Refactor},
		{"refactor extract", "extract retry logic into helper", Refactor},

		{"feature add", "add retry logic to fetcher", Feature},
		{"feature implement", "implement pagination", Feature},
		{"feature wire", "wire the new endpoint into the dashboard", Feature},

		{"test add", "add a test for retry logic", Test},
		{"test coverage", "increase test coverage for payments", Test},

		{"docs readme", "update the readme", Docs},
		{"docs comment", "add comments to payment handler", Docs},

		{"case insensitive", "FIX BUG", Debug},

		// Precedence: a prompt with both "fix" and "test" should be Debug,
		// not Test, because debugging a test is the salient activity.
		{"debug beats test", "fix test flake in auth", Debug},

		{"refactor beats feature", "add support for rename across packages", Refactor},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.in)
			if got != tc.want {
				t.Errorf("Classify(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
