package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/justinstimatze/hindcast/internal/health"
	"github.com/justinstimatze/hindcast/internal/store"
)

// cmdTune runs prefix-LOO across the user's backfilled records, finds
// the empirical sim cliff (the threshold above which kNN beats bucket
// fallback by ≥15% on this user's data), and persists the result to
// health.json.
//
// Replaces the previously-shipped global `sim ≥ 0.5` constant. The
// cross-corpus benchmark in v0.3 showed that the cliff varies by use
// pattern — multi-turn evolving project work has a clean cliff around
// 0.5, independent-task work has none. Per-user tuning fits each user.
//
// Cheap (~500ms over thousands of records) and idempotent. Safe to run
// from the Stop hook periodically.
func cmdTune(args []string) {
	fl := flag.NewFlagSet("tune", flag.ExitOnError)
	warmup := fl.Int("warmup", 20, "skip the first N records per project")
	verbose := fl.Bool("v", false, "print per-sim-bucket breakdown")
	_ = fl.Parse(args)

	byProject, err := loadRecordsByProject()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast tune: %s\n", err)
		os.Exit(1)
	}
	if len(byProject) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast tune: no records — run `hindcast backfill` first")
		os.Exit(1)
	}

	sk, _ := store.LoadSketch()
	h := health.Compute(byProject, sk, *warmup)
	if err := h.Save(); err != nil {
		fmt.Fprintf(os.Stderr, "hindcast tune: save: %s\n", err)
		os.Exit(1)
	}

	fmt.Printf("hindcast tune: %d predictions across %d project(s)\n\n", h.NPredictions, len(byProject))
	fmt.Printf("  bucket MALR:  %.2fx  (n=%d)\n", h.BucketMALR, h.NBucket)
	fmt.Printf("  project MALR: %.2fx\n", h.GroupMALR)
	fmt.Printf("  global MALR:  %.2fx\n", h.GlobalMALR)
	fmt.Printf("  kNN tier:     n=%d\n", h.NKNN)
	fmt.Println()
	if math.IsInf(h.TunedSimThreshold, 1) {
		fmt.Printf("  tuned threshold: never inject (kNN does not beat bucket on your data)\n")
	} else {
		fmt.Printf("  tuned threshold: sim ≥ %.2f  →  kNN MALR %.2fx (vs bucket %.2fx)\n",
			h.TunedSimThreshold, h.KNNMALRAtThreshold, h.BucketMALR)
	}
	fmt.Printf("  verdict: %s\n", h.Verdict)

	if *verbose {
		fmt.Println("\n  kNN MALR by sim bucket:")
		fmt.Printf("    %-12s  %6s  %10s\n", "sim", "n", "wall MALR")
		for _, b := range h.SimBuckets {
			if b.N == 0 {
				fmt.Printf("    %.2f-%.2f    %6d  %10s\n", b.LoSim, b.HiSim, 0, "-")
				continue
			}
			fmt.Printf("    %.2f-%.2f    %6d  %9.2fx\n", b.LoSim, b.HiSim, b.N, b.WallMALR)
		}
	}

	if path, err := health.HealthPath(); err == nil {
		fmt.Printf("\n  saved: %s\n", path)
	}
}
