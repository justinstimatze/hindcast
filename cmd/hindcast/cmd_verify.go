package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/predict"
	"github.com/justinstimatze/hindcast/internal/store"
)

// verifyRow is one prefix-LOO prediction result.
type verifyRow struct {
	predictedWall   int
	predictedActive int
	actualWall      int
	actualActive    int
	source          predict.Source
}

// cmdVerify runs a prefix-based leave-one-out evaluation of the live
// predict.Predict function over all backfilled records. For each record
// r_i it calls the predictor against records[0..i-1], compares the
// prediction to the actual duration, and aggregates MALR by source tier.
//
// Emits a yes/no verdict and a non-zero exit code if the kNN tier
// doesn't beat the global baseline by --lift-min. This is the
// self-evaluating quality gate that replaces "dogfood for a week and
// check hindcast show --accuracy."
//
// No API calls; uses your own data. Safe to re-run anytime.
func cmdVerify(args []string) {
	fl := flag.NewFlagSet("verify", flag.ExitOnError)
	liftMin := fl.Float64("lift-min", 1.10, "kNN wall-MALR must beat global by at least this factor")
	warmup := fl.Int("warmup", 20, "skip the first N records per project (kNN needs density)")
	perProjectMax := fl.Int("max-per-project", 1000, "cap records per project to bound runtime")
	verbose := fl.Bool("v", false, "print per-project breakdown")
	_ = fl.Parse(args)

	byProject, err := loadRecordsByProject()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast verify: %s\n", err)
		os.Exit(1)
	}
	if len(byProject) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast verify: no records — run `hindcast backfill` first")
		os.Exit(1)
	}

	sk, _ := store.LoadSketch()

	var allRows []verifyRow
	perProject := map[string][]verifyRow{}

	evaluated := 0
	for hash, recs := range byProject {
		// Chronological order mirrors real use: you always know only past.
		sort.Slice(recs, func(i, j int) bool { return recs[i].TS.Before(recs[j].TS) })
		if len(recs) < *warmup+10 {
			continue
		}
		if len(recs) > *perProjectMax {
			recs = recs[len(recs)-*perProjectMax:]
		}

		// Incremental BM25 index: start empty, query with tokens[i] against
		// index built from records[0..i-1], then add record i.
		idx := bm25.New()
		for i, r := range recs {
			if i >= *warmup && r.WallSeconds > 0 && len(r.PromptTokens) > 0 {
				prefixRecords := recs[:i]
				p := predict.Predict(r.PromptTokens, idx, prefixRecords, sk, r.TaskType)
				if p.Source != predict.SourceNone {
					vr := verifyRow{
						predictedWall:   p.WallSeconds,
						predictedActive: p.ActiveSeconds,
						actualWall:      r.WallSeconds,
						actualActive:    r.ClaudeActiveSeconds,
						source:          p.Source,
					}
					allRows = append(allRows, vr)
					perProject[hash] = append(perProject[hash], vr)
					evaluated++
				}
			}
			// Add this record to the index for future iterations.
			toolCount := 0
			for _, c := range r.ToolCalls {
				toolCount += c
			}
			idx.Add(r.PromptTokens, bm25.Doc{
				WallSeconds:   r.WallSeconds,
				ActiveSeconds: r.ClaudeActiveSeconds,
				TaskType:      r.TaskType,
				SizeBucket:    r.SizeBucket,
				ToolCount:     toolCount,
				FilesTouched:  r.FilesTouched,
			})
		}
	}

	if evaluated == 0 {
		fmt.Fprintln(os.Stderr, "hindcast verify: no records evaluable (need >warmup+10 per project)")
		os.Exit(1)
	}

	fmt.Printf("hindcast verify: %d predictions across %d project(s)\n\n", evaluated, len(perProject))

	// Overall breakdown by source tier.
	bySrc := map[predict.Source][]verifyRow{}
	for _, r := range allRows {
		bySrc[r.source] = append(bySrc[r.source], r)
	}

	srcOrder := []predict.Source{
		predict.SourceKNN, predict.SourceBucket,
		predict.SourceProject, predict.SourceGlobal,
	}
	fmt.Printf("%-8s  %6s  %10s  %10s  %8s\n", "source", "n", "wall MALR", "active MALR", "bias")
	for _, s := range srcOrder {
		rs := bySrc[s]
		if len(rs) == 0 {
			fmt.Printf("%-8s  %6d  %10s  %10s  %8s\n", s, 0, "-", "-", "-")
			continue
		}
		wm := verifyMALR(rs, true)
		am := verifyMALR(rs, false)
		b := verifyBias(rs)
		fmt.Printf("%-8s  %6d  %9.2fx  %10.2fx  %7.2fx\n", s, len(rs), wm, am, b)
	}

	if *verbose {
		fmt.Println("\nper project (wall MALR, overall):")
		var hashes []string
		for h := range perProject {
			hashes = append(hashes, h)
		}
		sort.Strings(hashes)
		for _, h := range hashes {
			rs := perProject[h]
			fmt.Printf("  %s  n=%4d   wall MALR %.2fx\n", h, len(rs), verifyMALR(rs, true))
		}
	}

	// Verdict.
	knnRows := bySrc[predict.SourceKNN]
	globalRows := bySrc[predict.SourceGlobal]
	// Fall back to bucket if global is too thin for a fair comparison.
	baselineLabel := "global"
	baselineRows := globalRows
	if len(baselineRows) < 10 {
		baselineRows = bySrc[predict.SourceBucket]
		baselineLabel = "bucket"
	}

	fmt.Println()
	switch {
	case len(knnRows) < 10:
		fmt.Printf("VERDICT: UNDETERMINED — only %d kNN predictions; need more history.\n", len(knnRows))
		os.Exit(2)
	case len(baselineRows) < 10:
		fmt.Println("VERDICT: UNDETERMINED — not enough baseline predictions for a fair comparison.")
		os.Exit(2)
	default:
		knnMALR := verifyMALR(knnRows, true)
		baselineMALR := verifyMALR(baselineRows, true)
		lift := baselineMALR / knnMALR
		if lift >= *liftMin {
			fmt.Printf("VERDICT: PASS — kNN MALR %.2fx beats %s MALR %.2fx (lift %.2fx ≥ %.2fx).\n",
				knnMALR, baselineLabel, baselineMALR, lift, *liftMin)
			os.Exit(0)
		}
		fmt.Printf("VERDICT: FAIL — kNN MALR %.2fx vs %s MALR %.2fx (lift %.2fx < %.2fx).\n",
			knnMALR, baselineLabel, baselineMALR, lift, *liftMin)
		fmt.Println("         Predictor's kNN tier is not earning its keep.")
		fmt.Println("         Options: raise knnMinSim, drop the kNN tier, or widen --lift-min.")
		os.Exit(1)
	}
}

// loadRecordsByProject groups all backfilled records by project hash.
func loadRecordsByProject() (map[string][]store.Record, error) {
	records, err := loadAllRecords()
	if err != nil {
		return nil, err
	}
	out := map[string][]store.Record{}
	for _, r := range records {
		out[r.ProjectHash] = append(out[r.ProjectHash], r)
	}
	return out, nil
}

func verifyMALR(rs []verifyRow, wall bool) float64 {
	abs := make([]float64, 0, len(rs))
	for _, r := range rs {
		var p, a int
		if wall {
			p, a = r.predictedWall, r.actualWall
		} else {
			p, a = r.predictedActive, r.actualActive
		}
		if p <= 0 || a <= 0 {
			continue
		}
		abs = append(abs, math.Abs(math.Log(float64(p)/float64(a))))
	}
	if len(abs) == 0 {
		return 0
	}
	sort.Float64s(abs)
	return math.Exp(abs[len(abs)/2])
}

func verifyBias(rs []verifyRow) float64 {
	signed := make([]float64, 0, len(rs))
	for _, r := range rs {
		if r.predictedWall <= 0 || r.actualWall <= 0 {
			continue
		}
		signed = append(signed, math.Log(float64(r.predictedWall)/float64(r.actualWall)))
	}
	if len(signed) == 0 {
		return 0
	}
	sort.Float64s(signed)
	return math.Exp(signed[len(signed)/2])
}
