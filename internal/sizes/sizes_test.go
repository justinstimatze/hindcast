package sizes

import "testing"

func TestClassify(t *testing.T) {
	cases := []struct {
		name  string
		files int
		tools int
		want  Bucket
	}{
		{"zero", 0, 0, Small},
		{"tiny debug", 1, 4, Small},                 // score 3.2
		{"medium feature", 3, 8, Medium},            // score 8.4
		{"large refactor", 11, 91, Large},           // score 49.3
		{"just under medium", 2, 3, Small},          // score 4.9
		{"just medium", 2, 4, Medium},               // score 5.2
		{"large files only", 15, 0, Large},          // score 30
		{"large tools only", 0, 90, Large},          // score 27
		{"medium heavy tools", 0, 50, Medium},       // score 15
		{"boundary at medium/large", 12, 3, Medium}, // score 24.9
		{"just large", 12, 4, Large},                // score 25.2
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.files, tc.tools)
			if got != tc.want {
				t.Errorf("Classify(files=%d, tools=%d) = %q, want %q (score=%.2f)",
					tc.files, tc.tools, got, tc.want, score(tc.files, tc.tools))
			}
		})
	}
}
