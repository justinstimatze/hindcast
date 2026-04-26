package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/justinstimatze/hindcast/internal/regressor"
	"github.com/justinstimatze/hindcast/internal/store"
)

// cmdTrain trains the per-user regressor over backfilled records and
// persists it to ~/.claude/hindcast/regressor.gob. Mirrors cmd_tune in
// shape: cheap enough (~1-3s on a thousand records), idempotent, safe to
// run from the Stop hook periodically.
//
// Returns nonzero exit if there is not yet enough data — the kNN tier
// keeps serving until the regressor earns its keep on the user's corpus.
func cmdTrain(args []string) {
	fl := flag.NewFlagSet("train", flag.ExitOnError)
	warmup := fl.Int("warmup", 5, "skip the first N records per project (avoid cold-start noise)")
	verbose := fl.Bool("v", false, "print feature names + model size")
	_ = fl.Parse(args)

	byProject, err := loadRecordsByProject()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast train: %s\n", err)
		os.Exit(1)
	}
	if len(byProject) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast train: no records — run `hindcast backfill` first")
		os.Exit(1)
	}

	m, err := regressor.Train(byProject, *warmup)
	if err != nil {
		if regressor.IsInsufficient(err) {
			fmt.Fprintf(os.Stderr,
				"hindcast train: not yet enough data (need ≥%d usable records). kNN tier remains active.\n",
				regressor.MinTrainRecords)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stderr, "hindcast train: %s\n", err)
		os.Exit(1)
	}
	if err := m.Save(); err != nil {
		fmt.Fprintf(os.Stderr, "hindcast train: save gbdt: %s\n", err)
		os.Exit(1)
	}

	gbdtPath, _ := regressor.GBDTModelPath()
	fmt.Printf("hindcast train: trained regressor on %d records (%d project(s))\n",
		m.NTrain, len(byProject))
	fmt.Printf("  GBDT:   %d trees (depth ≤3, lr=%.2f)  in-sample MALR=%.2fx\n",
		len(m.Trees), m.LR, m.TrainMALR)

	// Also fit and persist a linear model. On user data with rich
	// within-project history, GBDT can win; on cross-corpus unique-instance
	// regimes, linear wins. Saving both lets the per-user winner pick at
	// `hindcast tune` time route to whichever earns its keep.
	if lm, err := regressor.TrainLinearFromRecords(byProject, *warmup, 1.0); err == nil {
		fmt.Printf("  linear: ridge λ=%.1f                in-sample MALR=%.2fx\n",
			lm.Lambda, lm.TrainMALR)
		if err := lm.Save(); err != nil {
			fmt.Fprintf(os.Stderr, "hindcast train: save linear: %s\n", err)
		}
	}
	fmt.Printf("  saved:  %s (+ linear sibling)\n", gbdtPath)

	if *verbose {
		fmt.Println("\n  features:")
		for i, name := range regressor.FeatureNames {
			fmt.Printf("    %2d  %s\n", i, name)
		}
	}

	// Held-out comparison: chronological 50/50 split per project. Trains
	// both models on the first half and reports held-out MALR on the
	// second half — the honest "would these predictions help" signal.
	heldoutEval(byProject, *warmup)
}

// heldoutEval runs a chronological 50/50 split per project and reports
// held-out MALR for both GBDT and linear, alongside the kNN baseline.
// In-sample MALR overstates real performance; held-out is what matters.
func heldoutEval(byProject map[string][]store.Record, warmup int) {
	trainRecs := map[string][]store.Record{}
	testRecs := map[string][]store.Record{}
	for k, recs := range byProject {
		if len(recs) < 10 {
			continue
		}
		sorted := make([]store.Record, len(recs))
		copy(sorted, recs)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].TS.Before(sorted[j].TS) })
		cut := len(sorted) / 2
		trainRecs[k] = sorted[:cut]
		testRecs[k] = sorted[cut:]
	}

	gbdt, err := regressor.Train(trainRecs, warmup)
	if err != nil {
		fmt.Printf("\nheld-out eval: skipped — %s\n", err)
		return
	}
	lin, linErr := regressor.TrainLinearFromRecords(trainRecs, warmup, 1.0)
	if linErr != nil {
		lin = nil
	}

	var gbdtAbs, linAbs []float64
	for k, test := range testRecs {
		hist := append([]store.Record(nil), trainRecs[k]...)
		for _, r := range test {
			if r.WallSeconds <= 0 {
				continue
			}
			ctx := regressor.Context{
				PromptChars: r.PromptChars,
				TaskType:    r.TaskType,
				History:     hist,
			}
			gPred := gbdt.PredictWall(ctx)
			if gPred > 0 {
				gbdtAbs = append(gbdtAbs, math.Abs(math.Log(float64(gPred)/float64(r.WallSeconds))))
			}
			if lin != nil {
				lPred := lin.PredictWall(ctx)
				if lPred > 0 {
					linAbs = append(linAbs, math.Abs(math.Log(float64(lPred)/float64(r.WallSeconds))))
				}
			}
			hist = append(hist, r)
		}
	}

	fmt.Println("\n  held-out (50/50 chronological per-project split):")
	if len(gbdtAbs) > 0 {
		sort.Float64s(gbdtAbs)
		fmt.Printf("    GBDT     n=%d  wall MALR=%.2fx\n", len(gbdtAbs), math.Exp(gbdtAbs[len(gbdtAbs)/2]))
	}
	if len(linAbs) > 0 {
		sort.Float64s(linAbs)
		fmt.Printf("    linear   n=%d  wall MALR=%.2fx\n", len(linAbs), math.Exp(linAbs[len(linAbs)/2]))
	}
}
