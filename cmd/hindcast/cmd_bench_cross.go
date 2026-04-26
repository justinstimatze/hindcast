package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/justinstimatze/hindcast/internal/bm25"
)

// cmdBenchCross runs hindcast's predictor logic against METR HCAST agent
// run data — a public corpus of real coding-agent wall-clock traces. The
// goal is to answer one question: does our predictor's algorithm work on
// data that did not come from this user's machine?
//
// Method: prefix-LOO over each (model, task_family) group of runs, sorted
// chronologically. For each run after warmup, four predictors compete:
//
//   - global_median:   median of ALL prior Claude runs across all groups
//                      (the dumbest possible baseline)
//   - group_median:    median of prior runs in the same (model, family)
//                      (the bucket/project tier)
//   - task_id_median:  median of prior runs with the same task_id
//                      (the within-task baseline)
//   - knn_taskid:      BM25-weighted kNN over task_id tokens
//                      (the closest analog to hindcast's kNN tier)
//
// MALR per predictor tells us whether each tier earns its keep on
// cross-corpus data. Lift over the dumb baseline is the question.
//
// Data: METR HCAST runs.jsonl, cached at ~/.cache/hindcast-bench/.
// Schema: task_id, task_family, model, started_at, completed_at,
// score_binarized, task_source. Task_id is used as a prompt proxy
// (the corpus has no prompt text in this file).
func cmdBenchCross(args []string) {
	fl := flag.NewFlagSet("bench-cross", flag.ExitOnError)
	corpus := fl.String("corpus", "metr", "corpus: metr | openhands")
	dataPath := fl.String("data", "", "path to corpus jsonl (default depends on corpus)")
	modelPrefix := fl.String("model-prefix", "", "filter to model slugs starting with this (default 'claude_' for metr, 'claude' for openhands)")
	warmup := fl.Int("warmup", 5, "skip the first N runs per (model, group) group")
	minGroupSize := fl.Int("min-group", 10, "skip groups with fewer total runs")
	source := fl.String("task-source", "", "metr only: filter to HCAST / SWAA / RE-Bench")
	verbose := fl.Bool("v", false, "per-model breakdown")
	_ = fl.Parse(args)

	if *modelPrefix == "" {
		if *corpus == "metr" {
			*modelPrefix = "claude_"
		} else {
			*modelPrefix = "claude"
		}
	}
	if *dataPath == "" {
		*dataPath = defaultCorpusPath(*corpus)
	}

	var runs []benchRun
	var err error
	switch *corpus {
	case "metr":
		runs, err = loadMetrCorpus(*dataPath, *modelPrefix, *source)
	case "openhands":
		runs, err = loadOpenHandsCorpus(*dataPath, *modelPrefix)
	default:
		fmt.Fprintf(os.Stderr, "hindcast bench-cross: unknown corpus %q (want metr|openhands)\n", *corpus)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast bench-cross: %s\n", err)
		os.Exit(1)
	}
	if len(runs) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast bench-cross: no runs match filters")
		os.Exit(1)
	}

	// Global baseline: median of all runs in corpus (filtered).
	allWalls := make([]float64, 0, len(runs))
	for _, r := range runs {
		allWalls = append(allWalls, r.WallSeconds)
	}
	sort.Float64s(allWalls)
	globalMedian := allWalls[len(allWalls)/2]

	// Group by (model, group_key); sort each group chronologically.
	groups := map[string][]benchRun{}
	for _, r := range runs {
		k := r.Model + "\x00" + r.GroupKey
		groups[k] = append(groups[k], r)
	}

	var rows []benchPredRow
	skippedGroups := 0
	knnLabel := "knn_prompt"
	if *corpus == "metr" {
		knnLabel = "knn_taskid"
	}

	type knnDetail struct{ sim float64 }
	knnDetails := []knnDetail{}

	for _, group := range groups {
		if len(group) < *minGroupSize {
			skippedGroups++
			continue
		}
		sort.Slice(group, func(i, j int) bool { return group[i].StartedAtMS < group[j].StartedAtMS })

		idx := bm25.New()
		var priorWalls []float64
		subWalls := map[string][]float64{}

		for i, r := range group {
			if i >= *warmup {
				preds := map[string]float64{
					"global_median": globalMedian,
				}
				if len(priorWalls) >= 4 {
					preds["group_median"] = quantile(priorWalls, 0.5)
				}
				if tw := subWalls[r.SubKey]; len(tw) >= 2 {
					preds["task_id_median"] = quantile(tw, 0.5)
				}
				if len(r.Tokens) > 0 {
					matches := idx.TopK(r.Tokens, 7)
					var good []bm25.Match
					maxSim := 0.0
					for _, m := range matches {
						if m.Sim >= 0.15 {
							good = append(good, m)
						}
						if m.Sim > maxSim {
							maxSim = m.Sim
						}
					}
					if len(good) >= 3 {
						preds[knnLabel] = weightedMedianMatches(good)
						knnDetails = append(knnDetails, knnDetail{sim: maxSim})
					}
				}
				rows = append(rows, benchPredRow{
					model: r.Model, family: r.GroupKey, taskID: r.SubKey,
					actual: r.WallSeconds, preds: preds,
				})
			}
			// Update state with this run.
			priorWalls = append(priorWalls, r.WallSeconds)
			subWalls[r.SubKey] = append(subWalls[r.SubKey], r.WallSeconds)
			idx.Add(r.Tokens, bm25.Doc{
				WallSeconds: int(r.WallSeconds + 0.5),
			})
		}
	}

	// Per-sim-bucket MALR for kNN (the gate-design question).
	if len(knnDetails) > 0 {
		// Build parallel rows indexed to knnDetails order.
		// For each pred row that has a knn pred, we know its sim from the index.
		// Re-walk rows pulling knn preds; sim aligns to knnDetails order.
		simBuckets := []struct {
			label  string
			lo, hi float64
		}{
			{"0.15-0.30", 0.15, 0.30}, {"0.30-0.50", 0.30, 0.50},
			{"0.50-0.70", 0.50, 0.70}, {"0.70+", 0.70, 999},
		}
		// rebuild knn-only row+sim list
		var knnRows []benchPredRow
		idxKD := 0
		for _, r := range rows {
			if _, ok := r.preds[knnLabel]; ok {
				knnRows = append(knnRows, r)
				idxKD++
			}
		}
		fmt.Printf("\n%s MALR by max_sim bucket:\n", knnLabel)
		fmt.Printf("  %-12s  %6s  %10s  %8s\n", "sim", "n", "wall MALR", "bias")
		for _, b := range simBuckets {
			var picked []benchPredRow
			for j, r := range knnRows {
				if j >= len(knnDetails) {
					break
				}
				s := knnDetails[j].sim
				if s >= b.lo && s < b.hi {
					picked = append(picked, r)
				}
			}
			if len(picked) == 0 {
				fmt.Printf("  %-12s  %6d  %10s  %8s\n", b.label, 0, "-", "-")
				continue
			}
			n, malr, bias := scorePredictor(picked, knnLabel)
			fmt.Printf("  %-12s  %6d  %9.2fx  %7.2fx\n", b.label, n, malr, bias)
		}
	}

	if len(rows) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast bench-cross: no evaluable predictions")
		os.Exit(1)
	}

	fmt.Printf("hindcast bench-cross: METR HCAST corpus\n")
	fmt.Printf("  filter: model-prefix=%q  task-source=%q\n", *modelPrefix, *source)
	fmt.Printf("  groups: %d evaluable, %d skipped (<%d runs)\n",
		len(groups)-skippedGroups, skippedGroups, *minGroupSize)
	fmt.Printf("  predictions: %d  warmup: %d per group\n\n", len(rows), *warmup)

	predictors := []string{"global_median", "group_median", "task_id_median", knnLabel}
	fmt.Printf("%-16s  %6s  %10s  %8s  %8s\n", "predictor", "n", "wall MALR", "bias", "vs base")
	for _, p := range predictors {
		n, malr, bias := scorePredictor(rows, p)
		lift := 0.0
		if p != "global_median" {
			_, baseMALR, _ := scorePredictor(rows, "global_median")
			if malr > 0 {
				lift = baseMALR / malr
			}
		}
		liftStr := "-"
		if lift > 0 {
			liftStr = fmt.Sprintf("%.2fx", lift)
		}
		if n == 0 {
			fmt.Printf("%-16s  %6d  %10s  %8s  %8s\n", p, 0, "-", "-", "-")
			continue
		}
		fmt.Printf("%-16s  %6d  %9.2fx  %7.2fx  %8s\n", p, n, malr, bias, liftStr)
	}

	if *verbose {
		fmt.Printf("\nper-model breakdown (%s only):\n", knnLabel)
		modelRows := map[string][]benchPredRow{}
		for _, r := range rows {
			modelRows[r.model] = append(modelRows[r.model], r)
		}
		var models []string
		for m := range modelRows {
			models = append(models, m)
		}
		sort.Strings(models)
		for _, m := range models {
			n, malr, _ := scorePredictor(modelRows[m], knnLabel)
			if n == 0 {
				continue
			}
			fmt.Printf("  %-40s  n=%5d  knn MALR %.2fx\n", m, n, malr)
		}
	}

	// Verdict.
	_, knnMALR, _ := scorePredictor(rows, knnLabel)
	_, baseMALR, _ := scorePredictor(rows, "global_median")
	fmt.Println()
	if knnMALR > 0 && baseMALR > 0 {
		lift := baseMALR / knnMALR
		switch {
		case lift >= 1.5:
			fmt.Printf("VERDICT: kNN beats global baseline by %.2fx — predictor approach generalizes.\n", lift)
		case lift >= 1.1:
			fmt.Printf("VERDICT: kNN beats global baseline by %.2fx — modest cross-corpus signal.\n", lift)
		default:
			fmt.Printf("VERDICT: kNN lift only %.2fx — predictor barely better than corpus median.\n", lift)
		}
	}
}

// benchRun is the corpus-agnostic row used by the prefix-LOO loop.
// GroupKey is the project-equivalent (task_family for METR, repo for OH).
// SubKey is the within-group bucket (task_id for METR, instance_id for OH).
// Tokens are BM25-tokenized prompt content (real prompt for OH, task_id
// fallback for METR which has no prompt text in runs.jsonl).
type benchRun struct {
	Model       string
	GroupKey    string
	SubKey      string
	StartedAtMS float64
	WallSeconds float64
	Tokens      []uint64
}

func loadMetrCorpus(path, modelPrefix, source string) ([]benchRun, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s (try `curl -fsSL -o %s https://raw.githubusercontent.com/METR/eval-analysis-public/main/reports/time-horizon-1-1/data/raw/runs.jsonl`): %w", path, path, err)
	}
	defer f.Close()

	type metrRow struct {
		TaskID      string  `json:"task_id"`
		Family      string  `json:"task_family"`
		Model       string  `json:"model"`
		StartedAtMS float64 `json:"started_at"`
		CompletedMS float64 `json:"completed_at"`
		TaskSource  string  `json:"task_source"`
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	var out []benchRun
	for scanner.Scan() {
		var r metrRow
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			continue
		}
		if r.Model == "" || r.Model == "human" {
			continue
		}
		if modelPrefix != "" && !strings.HasPrefix(r.Model, modelPrefix) {
			continue
		}
		if source != "" && r.TaskSource != source {
			continue
		}
		if r.StartedAtMS == 0 || r.CompletedMS == 0 {
			continue
		}
		wall := (r.CompletedMS - r.StartedAtMS) / 1000.0
		if wall <= 0 {
			continue
		}
		out = append(out, benchRun{
			Model: r.Model, GroupKey: r.Family, SubKey: r.TaskID,
			StartedAtMS: r.StartedAtMS, WallSeconds: wall,
			Tokens: tokenizeTaskID(r.TaskID),
		})
	}
	return out, scanner.Err()
}

func loadOpenHandsCorpus(path, modelPrefix string) ([]benchRun, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s (run scripts/openhands_extract.py first): %w", path, err)
	}
	defer f.Close()

	type ohRow struct {
		InstanceID  string  `json:"instance_id"`
		Repo        string  `json:"repo"`
		Model       string  `json:"model"`
		Prompt      string  `json:"prompt"`
		WallSeconds float64 `json:"wall_seconds"`
		StartedAtMS int64   `json:"started_at_ms"`
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	var out []benchRun
	for scanner.Scan() {
		var r ohRow
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			continue
		}
		if modelPrefix != "" && !strings.HasPrefix(r.Model, modelPrefix) {
			continue
		}
		if r.WallSeconds <= 0 || r.Repo == "" || r.Prompt == "" {
			continue
		}
		out = append(out, benchRun{
			Model: r.Model, GroupKey: r.Repo, SubKey: r.InstanceID,
			StartedAtMS: float64(r.StartedAtMS), WallSeconds: r.WallSeconds,
			Tokens: bm25.HashTokens(r.Prompt, nil),
		})
	}
	return out, scanner.Err()
}

// tokenizeTaskID splits "acdc_bug/fix_checkpointing" into hash tokens.
// Stand-in for prompt tokens since runs.jsonl has no prompt text.
func tokenizeTaskID(id string) []uint64 {
	parts := strings.FieldsFunc(id, func(r rune) bool {
		return r == '/' || r == '_' || r == '-' || r == '.'
	})
	out := make([]uint64, 0, len(parts))
	for _, p := range parts {
		if p == "" {
			continue
		}
		out = append(out, fnv1a(strings.ToLower(p)))
	}
	return out
}

func fnv1a(s string) uint64 {
	const offset = 14695981039346656037
	const prime = 1099511628211
	h := uint64(offset)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime
	}
	return h
}

func defaultCorpusPath(corpus string) string {
	home, _ := os.UserHomeDir()
	switch corpus {
	case "openhands":
		return filepath.Join(home, ".cache", "hindcast-bench", "openhands.jsonl")
	default:
		return filepath.Join(home, ".cache", "hindcast-bench", "metr-runs.jsonl")
	}
}

func quantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	i := int(math.Round(q * float64(len(sorted)-1)))
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

func weightedMedianMatches(ms []bm25.Match) float64 {
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

type benchPredRow struct {
	model  string
	family string
	taskID string
	actual float64
	preds  map[string]float64
}

func scorePredictor(rows []benchPredRow, name string) (n int, malr, bias float64) {
	abs := []float64{}
	signed := []float64{}
	for _, r := range rows {
		p, ok := r.preds[name]
		if !ok || p <= 0 || r.actual <= 0 {
			continue
		}
		abs = append(abs, math.Abs(math.Log(p/r.actual)))
		signed = append(signed, math.Log(p/r.actual))
	}
	if len(abs) == 0 {
		return 0, 0, 0
	}
	sort.Float64s(abs)
	sort.Float64s(signed)
	return len(abs), math.Exp(abs[len(abs)/2]), math.Exp(signed[len(signed)/2])
}
