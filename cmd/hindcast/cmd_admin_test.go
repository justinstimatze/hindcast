package main

import (
	"testing"
)

// TestWithBandRows confirms the filter excludes:
//   - pre-v0.6.1 entries (no band fields populated)
//   - entries with no usable actual
//   - tiers the inject doesn't surface (global / none / unknown) —
//     measuring their band hit rate would be measuring something Claude
//     never saw, which is incoherent.
func TestWithBandRows(t *testing.T) {
	rows := []accuracyRow{
		{Source: "knn", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},                      // included
		{Source: "bucket", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},                   // included
		{Source: "project", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},                  // included
		{Source: "regressor", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},                // included
		{Source: "global", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},                   // EXCLUDED — global never injects
		{Source: "none", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},                     // EXCLUDED
		{Source: "knn", ActualWall: 60, PredictedWallP25: 0, PredictedWallP75: 90},                       // EXCLUDED — pre-v0.6.1
		{Source: "knn", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 0},                       // EXCLUDED — pre-v0.6.1
		{Source: "knn", ActualWall: 0, PredictedWallP25: 30, PredictedWallP75: 90},                       // EXCLUDED — no actual
		{Source: "knn", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90, VarianceGated: true}, // included
	}
	got := withBandRows(rows)
	if len(got) != 5 {
		t.Errorf("expected 5 usable rows, got %d", len(got))
	}
	for _, r := range got {
		if r.Source == "global" || r.Source == "none" {
			t.Errorf("global/none should be filtered out, got source=%s", r.Source)
		}
	}
}

// TestBandHitRateComputation verifies the in-band check matches the
// inclusive-bracket spec used in showAccuracy. The test exercises a
// synthetic mix of point-rendered and variance-gated entries — all
// from inject-eligible source tiers (knn here).
func TestBandHitRateComputation(t *testing.T) {
	rows := []accuracyRow{
		// Point-rendered (variance_gated=false): 2/3 hit
		{Source: "knn", ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90},  // hit (60 in [30,90])
		{Source: "knn", ActualWall: 30, PredictedWallP25: 30, PredictedWallP75: 90},  // hit (boundary inclusive)
		{Source: "knn", ActualWall: 200, PredictedWallP25: 30, PredictedWallP75: 90}, // miss (above)
		// Variance-gated: 1/2 hit
		{Source: "knn", ActualWall: 5, PredictedWallP25: 1, PredictedWallP75: 10, VarianceGated: true},      // hit
		{Source: "knn", ActualWall: 1000, PredictedWallP25: 60, PredictedWallP75: 600, VarianceGated: true}, // miss
	}
	band := withBandRows(rows)
	if len(band) != 5 {
		t.Fatalf("filter dropped rows: got %d, want 5", len(band))
	}
	hits, ptHits, ptN, vgHits, vgN := 0, 0, 0, 0, 0
	for _, r := range band {
		inBand := r.ActualWall >= r.PredictedWallP25 && r.ActualWall <= r.PredictedWallP75
		if inBand {
			hits++
		}
		if r.VarianceGated {
			vgN++
			if inBand {
				vgHits++
			}
		} else {
			ptN++
			if inBand {
				ptHits++
			}
		}
	}
	if hits != 3 {
		t.Errorf("total hits: got %d, want 3", hits)
	}
	if ptHits != 2 || ptN != 3 {
		t.Errorf("point: got %d/%d, want 2/3", ptHits, ptN)
	}
	if vgHits != 1 || vgN != 2 {
		t.Errorf("variance-gated: got %d/%d, want 1/2", vgHits, vgN)
	}
}
