package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/justinstimatze/hindcast/internal/store"
)

// bootstrapIters — number of resamples when computing confidence
// intervals. 1000 is enough for 95% CI on percentiles with reasonable
// stability; runs in <100ms for arms with <10k samples.
const bootstrapIters = 1000

// cmdCalibrate is the online A/B analysis: partitions records by arm
// (control vs treatment), computes MALR of Claude's emitted estimate
// vs actual duration, and reports the lift.
//
// A record participates only if it has Arm set AND a parseable
// ClaudeEstimate*. Legacy backfilled records (no arm, no estimate) are
// skipped — they're the bench's territory.
func cmdCalibrate(args []string) {
	fl := flag.NewFlagSet("calibrate", flag.ExitOnError)
	metric := fl.String("metric", "wall", "metric: wall|active")
	_ = fl.Parse(args)

	records, err := loadAllRecords()
	if err != nil {
		fmt.Fprintf(os.Stderr, "calibrate: %s\n", err)
		os.Exit(1)
	}

	var getter func(store.Record) (actual, estimate int)
	switch *metric {
	case "wall":
		getter = func(r store.Record) (int, int) { return r.WallSeconds, r.ClaudeEstimateWall }
	case "active":
		getter = func(r store.Record) (int, int) { return r.ClaudeActiveSeconds, r.ClaudeEstimateActive }
	default:
		fmt.Fprintf(os.Stderr, "unknown --metric %q (wall|active)\n", *metric)
		os.Exit(1)
	}

	// Eligibility: arm set + estimate > 0 + actual > 0.
	byArm := map[string][]float64{}       // abs log ratios
	signedByArm := map[string][]float64{} // signed log ratios (for bias)
	total := map[string]int{}
	tagged := map[string]int{}
	for _, r := range records {
		if r.Arm == "" {
			continue
		}
		total[r.Arm]++
		actual, est := getter(r)
		if est <= 0 {
			continue
		}
		tagged[r.Arm]++
		if actual <= 0 {
			continue
		}
		lr := math.Log(float64(est) / float64(actual))
		byArm[r.Arm] = append(byArm[r.Arm], math.Abs(lr))
		signedByArm[r.Arm] = append(signedByArm[r.Arm], lr)
	}

	fmt.Printf("hindcast calibrate — %s metric\n\n", *metric)
	fmt.Printf("%-10s %6s %9s %8s %14s %8s %9s\n",
		"arm", "n", "tag rate", "MALR", "95% CI", "p90", "bias")
	fmt.Println("--------------------------------------------------------------------------------")

	armResults := map[string]struct{ malr, p90 float64 }{}
	armLogRatios := map[string][]float64{} // for lift CI
	for _, arm := range []string{store.ArmControl, store.ArmTreatment} {
		errs := byArm[arm]
		if total[arm] == 0 {
			fmt.Printf("%-10s %6s %9s\n", arm, "0", "—")
			continue
		}
		tagRate := float64(tagged[arm]) / float64(total[arm])
		if len(errs) == 0 {
			fmt.Printf("%-10s %6d %8.1f%%  (no parseable estimates yet)\n", arm, total[arm], tagRate*100)
			continue
		}
		sort.Float64s(errs)
		malr := math.Exp(quantile64(errs, 0.5))
		p90 := math.Exp(quantile64(errs, 0.9))
		signed := append([]float64(nil), signedByArm[arm]...)
		sort.Float64s(signed)
		bias := math.Exp(quantile64(signed, 0.5))
		malrLo, malrHi := bootstrapMALRCI(errs)
		fmt.Printf("%-10s %6d %8.1f%%  %6.2fx  [%.2f–%.2f]  %6.2fx  %8.2fx\n",
			arm, total[arm], tagRate*100, malr, malrLo, malrHi, p90, bias)
		armResults[arm] = struct{ malr, p90 float64 }{malr, p90}
		armLogRatios[arm] = errs
	}

	c, ok1 := armResults[store.ArmControl]
	t, ok2 := armResults[store.ArmTreatment]
	if ok1 && ok2 {
		lift := c.malr / t.malr
		liftLo, liftHi := bootstrapLiftCI(armLogRatios[store.ArmControl], armLogRatios[store.ArmTreatment])
		fmt.Printf("\nLift (control MALR / treatment MALR) = %.2fx  [95%% CI %.2f–%.2f]\n", lift, liftLo, liftHi)
		switch {
		case liftLo > 1.0:
			fmt.Printf("  → treatment reduces error with statistical confidence.\n")
		case liftHi < 1.0:
			fmt.Printf("  → treatment INCREASES error with statistical confidence (priors are hurting).\n")
		default:
			fmt.Printf("  → CI crosses 1.0 — direction uncertain; keep collecting samples.\n")
		}

		// Tag-rate divergence check: if Claude emits the estimate tag at
		// very different rates across arms, the MALR comparison is a
		// biased sample and the reported lift is untrustworthy.
		cTotal, tTotal := total[store.ArmControl], total[store.ArmTreatment]
		cTagged, tTagged := tagged[store.ArmControl], tagged[store.ArmTreatment]
		if cTotal > 0 && tTotal > 0 {
			cRate := float64(cTagged) / float64(cTotal)
			tRate := float64(tTagged) / float64(tTotal)
			diff := cRate - tRate
			if diff < 0 {
				diff = -diff
			}
			if diff > 0.10 {
				fmt.Printf("\n⚠  Tag-rate divergence between arms: control=%.1f%%, treatment=%.1f%% (Δ=%.1fpp).\n",
					cRate*100, tRate*100, diff*100)
				fmt.Println("   Lift number is biased — selection effect. Raise tag rate on the underrepresented arm before trusting the comparison.")
			}
		}
	} else {
		fmt.Println("\nLift: need samples in BOTH arms to compute (re-run after more sessions).")
	}

	fmt.Println("\nbias = exp(median signed log-ratio). >1 means Claude overestimates; <1 underestimates.")
	fmt.Println("95% CI = bootstrap over 1000 resamples; wider CI = fewer samples / more variance.")
	fmt.Println("Note: requires Claude emitting <!-- hindcast-wall: Nm --> tags per MCP instructions.")
	fmt.Println("Tag rate = fraction of arm's turns where Claude emitted a parseable estimate.")
}

// bootstrapMALRCI returns the 2.5th and 97.5th percentile of the
// bootstrapped MALR distribution — a 95% confidence interval on MALR
// for this arm. errs is a slice of |log(est/actual)| values.
func bootstrapMALRCI(errs []float64) (lo, hi float64) {
	if len(errs) < 2 {
		return math.Exp(errs[0]), math.Exp(errs[0])
	}
	rng := rand.New(rand.NewSource(42)) // deterministic
	malrs := make([]float64, 0, bootstrapIters)
	sample := make([]float64, len(errs))
	for i := 0; i < bootstrapIters; i++ {
		for j := range sample {
			sample[j] = errs[rng.Intn(len(errs))]
		}
		sort.Float64s(sample)
		malrs = append(malrs, quantile64(sample, 0.5))
	}
	sort.Float64s(malrs)
	return math.Exp(quantile64(malrs, 0.025)), math.Exp(quantile64(malrs, 0.975))
}

// bootstrapLiftCI returns the 95% CI on lift = control_MALR / treatment_MALR.
// Each resample draws independently from both arms; 1000 iterations.
func bootstrapLiftCI(control, treatment []float64) (lo, hi float64) {
	if len(control) < 2 || len(treatment) < 2 {
		return 0, 0
	}
	rng := rand.New(rand.NewSource(42))
	lifts := make([]float64, 0, bootstrapIters)
	cs := make([]float64, len(control))
	ts := make([]float64, len(treatment))
	for i := 0; i < bootstrapIters; i++ {
		for j := range cs {
			cs[j] = control[rng.Intn(len(control))]
		}
		for j := range ts {
			ts[j] = treatment[rng.Intn(len(treatment))]
		}
		sort.Float64s(cs)
		sort.Float64s(ts)
		cM := math.Exp(quantile64(cs, 0.5))
		tM := math.Exp(quantile64(ts, 0.5))
		if tM > 0 {
			lifts = append(lifts, cM/tM)
		}
	}
	if len(lifts) < 2 {
		return 0, 0
	}
	sort.Float64s(lifts)
	return quantile64(lifts, 0.025), quantile64(lifts, 0.975)
}
