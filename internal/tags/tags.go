// Package tags classifies a prompt's first line into a task-type bucket
// via deterministic keyword regex. Pure, no dependencies, table-driven.
// Buckets: bug, feature, refactor, test, docs, other.
package tags

import (
	"regexp"
	"strings"
)

type TaskType string

const (
	Refactor TaskType = "refactor"
	Debug    TaskType = "debug"
	Feature  TaskType = "feature"
	Test     TaskType = "test"
	Docs     TaskType = "docs"
	Other    TaskType = "other"
)

// Order is load-bearing: first match wins, so more-specific rules come
// before more-general ones. Debug precedes Refactor because "fix test
// flake" should be debug, not test. Feature is last because "add" is
// weakly informative.
// Each stem covers common English verb inflections so "crash",
// "crashes", "crashing", "crashed" all classify the same.
var rules = []struct {
	tag TaskType
	re  *regexp.Regexp
}{
	{Debug, regexp.MustCompile(`\b(fix(?:es|ed|ing)?|bugs?|broken|breaks?|broke|flakes?|flaky|hangs?|hung|crash(?:es|ed|ing)?|errors?|regressions?|leaks?|leaking|leaked|fails?|failing|failed)\b`)},
	{Refactor, regexp.MustCompile(`\b(refactors?|refactor(?:ed|ing)|rewrites?|rewriting|rewrote|rewritten|migrate[sd]?|migrating|migration|restructures?|restructur(?:ed|ing)|renames?|renam(?:ed|ing)|extracts?|extract(?:ed|ing)|inlines?|inlin(?:ed|ing))\b`)},
	{Test, regexp.MustCompile(`\b(tests?|testing|specs?|coverage|e2e)\b`)},
	{Docs, regexp.MustCompile(`\b(docs?|documentation|comments?|comment(?:ed|ing)|readme|changelog|docstrings?)\b`)},
	{Feature, regexp.MustCompile(`\b(adds?|adding|added|implements?|implement(?:ed|ing|ation)|builds?|building|built|introduces?|introduc(?:ed|ing)|creates?|creat(?:ed|ing)|supports?|support(?:ed|ing)|wires?|wir(?:ed|ing)|ships?|shipped|shipping|enables?|enabl(?:ed|ing))\b`)},
}

func Classify(line string) TaskType {
	lower := strings.ToLower(line)
	for _, r := range rules {
		if r.re.MatchString(lower) {
			return r.tag
		}
	}
	return Other
}
