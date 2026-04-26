// Package sizes buckets a completed turn into small/medium/large via
// hand-drawn thresholds on files_touched + tool_count. Splits bimodal
// distributions within a task_type so medians aren't misleading.
package sizes

type Bucket string

const (
	Small  Bucket = "small"
	Medium Bucket = "medium"
	Large  Bucket = "large"
)

// Weights are hand-picked so the BOOTSTRAP example turns land in their
// intended buckets: (3 files, 8 tools) → medium, (11 files, 91 tools) →
// large, (1 file, 4 tools) → small. Re-tune once real data is in.
const (
	filesWeight = 2.0
	toolsWeight = 0.3

	smallMediumCutoff = 5.0
	mediumLargeCutoff = 25.0
)

func score(filesTouched, toolCount int) float64 {
	return float64(filesTouched)*filesWeight + float64(toolCount)*toolsWeight
}

func Classify(filesTouched, toolCount int) Bucket {
	s := score(filesTouched, toolCount)
	switch {
	case s < smallMediumCutoff:
		return Small
	case s < mediumLargeCutoff:
		return Medium
	default:
		return Large
	}
}
