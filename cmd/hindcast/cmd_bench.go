package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/store"
)

// cmdBench runs the offline duration-prediction benchmark across all
// backfilled records. Walks records chronologically; at each turn i,
// uses turns [0..i-1] as prior corpus and predicts duration via each
// method, comparing to actual. Reports median absolute log-ratio (MALR)
// as a multiplicative error factor — 1.0x is perfect, 2.0x is 2× off.
func cmdBench(args []string) {
	fl := flag.NewFlagSet("bench", flag.ExitOnError)
	metric := fl.String("metric", "wall", "metric to predict: wall|active")
	warmup := fl.Int("warmup", 20, "skip the first N records (need prior corpus)")
	_ = fl.Parse(args)

	records, err := loadAllRecords()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast bench: %s\n", err)
		os.Exit(1)
	}
	if len(records) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast bench: no records — run `hindcast backfill` first")
		os.Exit(1)
	}

	// Chronological order is how priors accumulate in real use.
	sort.Slice(records, func(i, j int) bool { return records[i].TS.Before(records[j].TS) })

	var getter func(store.Record) int
	label := *metric
	switch label {
	case "wall":
		getter = func(r store.Record) int { return r.WallSeconds }
	case "active":
		getter = func(r store.Record) int { return r.ClaudeActiveSeconds }
	default:
		fmt.Fprintf(os.Stderr, "unknown --metric %q (want wall|active)\n", label)
		os.Exit(1)
	}

	if len(records) < *warmup+10 {
		fmt.Fprintf(os.Stderr, "bench: too few records (%d) — need at least %d\n",
			len(records), *warmup+10)
		os.Exit(1)
	}

	methods := []struct {
		name    string
		predict func(prior []store.Record, target store.Record, get func(store.Record) int) (int, bool, bool)
	}{
		{"global-median", globalMedian},
		{"tags-only", tagsOnly},
		{"project+tags", projectTags},
		{"project+tags+size", projectTagsSize},
	}

	errs := make([][]float64, len(methods))
	rawErrs := make([][]float64, len(methods))
	signed := make([][]float64, len(methods))
	for i := *warmup; i < len(records); i++ {
		target := records[i]
		actual := getter(target)
		if actual <= 0 {
			continue
		}
		prior := records[:i]
		for mi, m := range methods {
			pred, ok, raw := m.predict(prior, target, getter)
			if !ok || pred <= 0 {
				continue
			}
			lr := math.Log(float64(pred) / float64(actual))
			errs[mi] = append(errs[mi], math.Abs(lr))
			signed[mi] = append(signed[mi], lr)
			if raw {
				rawErrs[mi] = append(rawErrs[mi], math.Abs(lr))
			}
		}
	}

	fmt.Printf("hindcast bench — predicting %s_seconds over %d records (warmup=%d)\n\n",
		label, len(records), *warmup)
	fmt.Printf("%-20s %6s %6s  %8s  %8s  %8s  %10s\n",
		"method", "n", "n_raw", "MALR", "p90", "worst", "bias")
	fmt.Println(strings.Repeat("-", 76))
	for mi, m := range methods {
		if len(errs[mi]) == 0 {
			continue
		}
		sort.Float64s(errs[mi])
		malrX := math.Exp(quantile64(errs[mi], 0.50))
		p90X := math.Exp(quantile64(errs[mi], 0.90))
		maxX := math.Exp(errs[mi][len(errs[mi])-1])
		biasX := math.Exp(median64Sorted(append([]float64(nil), signed[mi]...)))
		fmt.Printf("%-20s %6d %6d  %6.2fx  %6.2fx  %6.2fx  %9.2fx\n",
			m.name, len(errs[mi]), len(rawErrs[mi]), malrX, p90X, maxX, biasX)
	}

	// Full stack: bucket median blended with BM25 top-1 close match.
	bm25Errs, bm25Signed := runBM25Bench(records, *warmup, getter)
	if len(bm25Errs) > 0 {
		sort.Float64s(bm25Errs)
		malrX := math.Exp(quantile64(bm25Errs, 0.50))
		p90X := math.Exp(quantile64(bm25Errs, 0.90))
		maxX := math.Exp(bm25Errs[len(bm25Errs)-1])
		biasX := math.Exp(median64Sorted(bm25Signed))
		fmt.Printf("%-20s %6d %6s  %6.2fx  %6.2fx  %6.2fx  %9.2fx\n",
			"full (+bm25)", len(bm25Errs), "—", malrX, p90X, maxX, biasX)
	}

	fmt.Println()
	fmt.Println("MALR / p90 / worst = error multipliers (lower better; 1.00x = perfect).")
	fmt.Println("bias = exp(median signed log-ratio). >1 means the method systematically overestimates;")
	fmt.Println("       <1 means it systematically underestimates. 1.00x = unbiased.")
	fmt.Println("n_raw = predictions where the method's own bucket had sufficient priors (no fallback).")
	fmt.Println("n     = total predictions including fallback to broader methods when n_raw was too low.")
	fmt.Println()
	fmt.Println("full (+bm25) blends project+tags+size median with BM25 top-1 when sim >= 0.30.")
}

// median64Sorted sorts in place and returns the median. Keeps callers
// from polluting the unsorted slice when we only need a single quantile.
func median64Sorted(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sort.Float64s(xs)
	return quantile64(xs, 0.5)
}

// runBM25Bench walks records chronologically, maintaining an in-memory
// BM25 index. At each target turn, predicts via project+tags+size bucket
// median blended with the current index's top-1 match (weighted by sim).
// Pre-warm with the first `warmup` records so early queries have priors.
func runBM25Bench(records []store.Record, warmup int, getter func(store.Record) int) (abs []float64, signed []float64) {
	idx := bm25.New()
	for i := 0; i < warmup && i < len(records); i++ {
		addToIndex(idx, records[i])
	}
	for i := warmup; i < len(records); i++ {
		target := records[i]
		actual := getter(target)
		if actual <= 0 {
			addToIndex(idx, target)
			continue
		}
		prior := records[:i]

		base, ok, _ := projectTagsSize(prior, target, getter)
		if !ok || base <= 0 {
			addToIndex(idx, target)
			continue
		}
		pred := base

		if len(target.PromptTokens) > 0 && len(idx.Docs) > 0 {
			matches := idx.TopK(target.PromptTokens, 1)
			if len(matches) > 0 && matches[0].Sim >= 0.30 {
				m := matches[0]
				bm25Pred := getter(store.Record{
					WallSeconds:         m.Doc.WallSeconds,
					ClaudeActiveSeconds: m.Doc.ActiveSeconds,
				})
				if bm25Pred > 0 {
					sim := matches[0].Sim
					pred = int(sim*float64(bm25Pred) + (1-sim)*float64(base) + 0.5)
				}
			}
		}

		if pred > 0 {
			lr := math.Log(float64(pred) / float64(actual))
			abs = append(abs, math.Abs(lr))
			signed = append(signed, lr)
		}
		addToIndex(idx, target)
	}
	return
}

func addToIndex(idx *bm25.Index, r store.Record) {
	if len(r.PromptTokens) == 0 {
		return
	}
	toolCount := 0
	for _, c := range r.ToolCalls {
		toolCount += c
	}
	idx.Add(r.PromptTokens, bm25.Doc{
		ActiveSeconds: r.ClaudeActiveSeconds,
		WallSeconds:   r.WallSeconds,
		TaskType:      r.TaskType,
		SizeBucket:    r.SizeBucket,
		ToolCount:     toolCount,
		FilesTouched:  r.FilesTouched,
	})
}

// Each prediction function returns (value, ok, rawHit). rawHit is true
// when the method's own criteria had ≥3 priors — i.e., this wasn't a
// fallback to a broader method. Tracked separately so reported n
// reflects the method's actual discriminating power, not just what the
// fallback chain caught.

func globalMedian(prior []store.Record, _ store.Record, get func(store.Record) int) (int, bool, bool) {
	values := make([]int, 0, len(prior))
	for _, r := range prior {
		if v := get(r); v > 0 {
			values = append(values, v)
		}
	}
	if len(values) == 0 {
		return 0, false, false
	}
	med, _ := store.Quantiles(values)
	return int(med + 0.5), true, true
}

func tagsOnly(prior []store.Record, target store.Record, get func(store.Record) int) (int, bool, bool) {
	values := make([]int, 0, len(prior))
	for _, r := range prior {
		if r.TaskType != target.TaskType {
			continue
		}
		if v := get(r); v > 0 {
			values = append(values, v)
		}
	}
	if len(values) < 3 {
		v, ok, _ := globalMedian(prior, target, get)
		return v, ok, false
	}
	med, _ := store.Quantiles(values)
	return int(med + 0.5), true, true
}

func projectTags(prior []store.Record, target store.Record, get func(store.Record) int) (int, bool, bool) {
	values := make([]int, 0, len(prior))
	for _, r := range prior {
		if r.ProjectHash != target.ProjectHash {
			continue
		}
		if r.TaskType != target.TaskType {
			continue
		}
		if v := get(r); v > 0 {
			values = append(values, v)
		}
	}
	if len(values) < 3 {
		v, ok, _ := tagsOnly(prior, target, get)
		return v, ok, false
	}
	med, _ := store.Quantiles(values)
	return int(med + 0.5), true, true
}

func projectTagsSize(prior []store.Record, target store.Record, get func(store.Record) int) (int, bool, bool) {
	values := make([]int, 0, len(prior))
	for _, r := range prior {
		if r.ProjectHash != target.ProjectHash {
			continue
		}
		if r.TaskType != target.TaskType {
			continue
		}
		if r.SizeBucket != target.SizeBucket {
			continue
		}
		if v := get(r); v > 0 {
			values = append(values, v)
		}
	}
	if len(values) < 3 {
		v, ok, _ := projectTags(prior, target, get)
		return v, ok, false
	}
	med, _ := store.Quantiles(values)
	return int(med + 0.5), true, true
}

func loadAllRecords() ([]store.Record, error) {
	dir, err := store.ProjectsDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var all []store.Record
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if filepath.Ext(e.Name()) != ".jsonl" {
			continue
		}
		rs, err := store.ReadRecentRecords(filepath.Join(dir, e.Name()), 1<<20)
		if err != nil {
			continue
		}
		all = append(all, rs...)
	}
	return all, nil
}

func quantile64(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	pos := q * float64(len(sorted)-1)
	lo := int(pos)
	if lo >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[lo+1]*frac
}
