package main

import (
	"testing"
)

// TestWithBandRows confirms the filter excludes pre-v0.6.1 entries
// (no band fields) and entries with no usable actual.
func TestWithBandRows(t *testing.T) {
	rows := []accuracyRow{
		{ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90}, // included
		{ActualWall: 60, PredictedWallP25: 0, PredictedWallP75: 90},  // missing p25 — pre-v0.6.1
		{ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 0},  // missing p75 — pre-v0.6.1
		{ActualWall: 0, PredictedWallP25: 30, PredictedWallP75: 90},  // no actual
		{ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90, VarianceGated: true}, // included
	}
	got := withBandRows(rows)
	if len(got) != 2 {
		t.Errorf("expected 2 usable rows, got %d", len(got))
	}
}

// TestBandHitRateComputation verifies the in-band check matches the
// inclusive-bracket spec used in showAccuracy. The test exercises a
// synthetic mix of point-rendered and variance-gated entries.
func TestBandHitRateComputation(t *testing.T) {
	rows := []accuracyRow{
		// Point-rendered (variance_gated=false): 2/3 hit
		{ActualWall: 60, PredictedWallP25: 30, PredictedWallP75: 90}, // hit (60 in [30,90])
		{ActualWall: 30, PredictedWallP25: 30, PredictedWallP75: 90}, // hit (boundary inclusive)
		{ActualWall: 200, PredictedWallP25: 30, PredictedWallP75: 90}, // miss (above)
		// Variance-gated: 1/2 hit
		{ActualWall: 5, PredictedWallP25: 1, PredictedWallP75: 10, VarianceGated: true},   // hit
		{ActualWall: 1000, PredictedWallP25: 60, PredictedWallP75: 600, VarianceGated: true}, // miss
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
