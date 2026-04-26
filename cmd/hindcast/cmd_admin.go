package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/health"
	"github.com/justinstimatze/hindcast/internal/hook"
	"github.com/justinstimatze/hindcast/internal/store"
)

// parseWithUsage adds -h/--help support to commands that don't take
// arguments otherwise. Keeps `hindcast <cmd> -h` consistent across
// all user-facing commands.
func parseWithUsage(name, desc string, args []string) {
	fl := flag.NewFlagSet(name, flag.ExitOnError)
	fl.SetOutput(os.Stderr)
	fl.Usage = func() {
		fmt.Fprintf(fl.Output(), "%s — %s\n\nUsage: hindcast %s\n", name, desc, name)
	}
	_ = fl.Parse(args)
}

// Silence unused import if io ever gets pulled in for help-wrapping.
var _ = io.Discard

// cmdShow dumps what hindcast has recorded. With --project P, scopes to
// a single project; otherwise reports all projects.
func cmdShow(args []string) {
	fl := flag.NewFlagSet("show", flag.ExitOnError)
	project := fl.String("project", "", "scope to a single project (by name or hash)")
	limit := fl.Int("n", 10, "show last N records per project")
	accuracy := fl.Bool("accuracy", false, "report predictor MALR vs actual from accuracy.jsonl")
	healthFlag := fl.Bool("health", false, "report tuned predictor state from health.json")
	_ = fl.Parse(args)

	if *accuracy {
		showAccuracy(*project)
		return
	}
	if *healthFlag {
		showHealth()
		return
	}

	projectsDir, err := store.ProjectsDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast show: %s\n", err)
		os.Exit(1)
	}
	entries, err := os.ReadDir(projectsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast show: %s\n", err)
		os.Exit(1)
	}

	targetHash := ""
	if *project != "" {
		if len(*project) == 8 {
			targetHash = *project
		} else {
			targetHash = store.ProjectHash(*project)
		}
	}

	type projInfo struct {
		hash    string
		records []store.Record
	}
	var projects []projInfo
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".jsonl" {
			continue
		}
		hash := strings.TrimSuffix(e.Name(), ".jsonl")
		if targetHash != "" && hash != targetHash {
			continue
		}
		rs, err := store.ReadRecentRecords(filepath.Join(projectsDir, e.Name()), 1<<20)
		if err != nil {
			continue
		}
		projects = append(projects, projInfo{hash: hash, records: rs})
	}
	sort.Slice(projects, func(i, j int) bool {
		return len(projects[i].records) > len(projects[j].records)
	})

	total := 0
	for _, p := range projects {
		total += len(p.records)
	}
	fmt.Printf("hindcast: %d records across %d projects\n\n", total, len(projects))

	for _, p := range projects {
		fmt.Printf("## project %s    n=%d\n", p.hash, len(p.records))
		byBucketStats(p.records)
		show := p.records
		if len(show) > *limit {
			show = show[:*limit]
		}
		if len(show) > 0 {
			fmt.Printf("\n  Last %d records (newest first):\n", len(show))
			for _, r := range show {
				tools := 0
				for _, c := range r.ToolCalls {
					tools += c
				}
				fmt.Printf("    %s  %-16s  active %5ds / wall %6ds  %3d tools  %2d files  %s\n",
					r.TS.Format("2006-01-02 15:04"),
					taskSizeLabel(r.TaskType, r.SizeBucket),
					r.ClaudeActiveSeconds, r.WallSeconds,
					tools, r.FilesTouched, r.Model)
			}
		}
		fmt.Println()
	}

	if sketch, err := store.LoadSketch(); err == nil {
		_, _, activeMed, activeP90, n := sketch.Percentiles()
		wallMed, wallP90, _, _, _ := sketch.Percentiles()
		if n > 0 {
			fmt.Printf("Global sketch (n=%d, numeric-only):\n", n)
			fmt.Printf("  wall   median %.1fm / p90 %.1fm\n", wallMed/60, wallP90/60)
			fmt.Printf("  active median %.1fm / p90 %.1fm\n", activeMed/60, activeP90/60)
		}
	}
}

// accuracyRow is one reconciled turn loaded from accuracy.jsonl.
type accuracyRow struct {
	TS              string `json:"ts"`
	SessionID       string `json:"session_id"`
	TaskType        string `json:"task_type"`
	PredictedWall   int    `json:"predicted_wall"`
	PredictedActive int    `json:"predicted_active"`
	ActualWall      int    `json:"actual_wall"`
	ActualActive    int    `json:"actual_active"`
	Source          string `json:"source"`
}

// showAccuracy aggregates accuracy.jsonl across one or all projects and
// reports predictor MALR (median |log(predicted / actual)|) broken down
// by prediction source. This is the primary quality signal post-pivot.
// The number to watch: MALR by source — if SourceKNN >> SourceBucket,
// the kNN path earns its keep; if not, the stratification is noise.
func showAccuracy(projectFilter string) {
	root, err := store.HindcastDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "show --accuracy: %s\n", err)
		os.Exit(1)
	}
	projDir := filepath.Join(root, "projects")
	entries, err := os.ReadDir(projDir)
	if err != nil {
		fmt.Println("hindcast: no accuracy data yet (need turns with predictions recorded post-v0.2)")
		return
	}

	targetHash := ""
	if projectFilter != "" {
		if len(projectFilter) == 8 {
			targetHash = projectFilter
		} else {
			targetHash = store.ProjectHash(projectFilter)
		}
	}

	var rows []accuracyRow
	projectCount := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if targetHash != "" && e.Name() != targetHash {
			continue
		}
		path := filepath.Join(projDir, e.Name(), "accuracy.jsonl")
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		loaded := 0
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for sc.Scan() {
			var r accuracyRow
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				continue
			}
			rows = append(rows, r)
			loaded++
		}
		f.Close()
		if loaded > 0 {
			projectCount++
		}
	}

	if len(rows) == 0 {
		fmt.Println("hindcast: no accuracy data yet (need turns with predictions recorded post-v0.2)")
		return
	}

	fmt.Printf("hindcast predictor accuracy: %d turns across %d project(s)\n\n", len(rows), projectCount)

	// Overall MALR (wall and active).
	overallWall := malr(wallLogRatios(rows))
	overallActive := malr(activeLogRatios(rows))
	fmt.Printf("overall MALR: wall %.2fx   active %.2fx\n", overallWall, overallActive)

	// Group by source.
	bySrc := map[string][]accuracyRow{}
	for _, r := range rows {
		src := r.Source
		if src == "" {
			src = "unknown"
		}
		bySrc[src] = append(bySrc[src], r)
	}
	var srcs []string
	for k := range bySrc {
		srcs = append(srcs, k)
	}
	sort.Slice(srcs, func(i, j int) bool { return len(bySrc[srcs[i]]) > len(bySrc[srcs[j]]) })

	fmt.Printf("\n%-10s  %6s  %10s  %10s  %10s\n", "source", "n", "wall MALR", "active MALR", "bias")
	for _, s := range srcs {
		rs := bySrc[s]
		wm := malr(wallLogRatios(rs))
		am := malr(activeLogRatios(rs))
		b := biasFactor(wallLogRatios(rs))
		fmt.Printf("%-10s  %6d  %9.2fx  %10.2fx  %9.2fx\n", s, len(rs), wm, am, b)
	}
}

func wallLogRatios(rs []accuracyRow) []float64 {
	out := make([]float64, 0, len(rs))
	for _, r := range rs {
		if r.PredictedWall > 0 && r.ActualWall > 0 {
			out = append(out, math.Log(float64(r.PredictedWall)/float64(r.ActualWall)))
		}
	}
	return out
}

func activeLogRatios(rs []accuracyRow) []float64 {
	out := make([]float64, 0, len(rs))
	for _, r := range rs {
		if r.PredictedActive > 0 && r.ActualActive > 0 {
			out = append(out, math.Log(float64(r.PredictedActive)/float64(r.ActualActive)))
		}
	}
	return out
}

// malr = exp(median(|log-ratio|)). 1.0 = perfect; higher = worse.
func malr(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	abs := make([]float64, len(xs))
	for i, v := range xs {
		abs[i] = math.Abs(v)
	}
	sort.Float64s(abs)
	med := abs[len(abs)/2]
	return math.Exp(med)
}

// biasFactor = exp(median(log-ratio)). >1 = systematic over-predict,
// <1 = under-predict. Sign-preserving counterpart to MALR.
func biasFactor(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	med := sorted[len(sorted)/2]
	return math.Exp(med)
}

func byBucketStats(records []store.Record) {
	groups := map[string][]store.Record{}
	for _, r := range records {
		k := taskSizeLabel(r.TaskType, r.SizeBucket)
		groups[k] = append(groups[k], r)
	}
	var keys []string
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return len(groups[keys[j]]) < len(groups[keys[i]]) })
	fmt.Printf("  %-20s %6s  %-10s  %-10s\n", "bucket", "n", "wall med", "active med")
	for _, k := range keys {
		rs := groups[k]
		walls := make([]int, 0, len(rs))
		actives := make([]int, 0, len(rs))
		for _, r := range rs {
			walls = append(walls, r.WallSeconds)
			actives = append(actives, r.ClaudeActiveSeconds)
		}
		wmed, _ := store.Quantiles(walls)
		amed, _ := store.Quantiles(actives)
		fmt.Printf("  %-20s %6d  %7.1fm   %7.1fm\n", k, len(rs), wmed/60, amed/60)
	}
}

// cmdStatus tails ~/.claude/hindcast/hook.log and summarizes recent hook
// activity so the user can answer "is this working?" Surfaces panic
// counts prominently — a silent-failure discovery aid.
func cmdStatus(args ...string) {
	parseWithUsage("status", "hook health check and recent activity summary", args)
	logPath, err := hook.LogPath()
	if err != nil {
		fmt.Fprintf(os.Stderr, "status: %s\n", err)
		os.Exit(1)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("hindcast: no hook.log yet — run a CC turn to trigger hooks")
			return
		}
		fmt.Fprintf(os.Stderr, "status: %s\n", err)
		os.Exit(1)
	}

	fmt.Printf("hook.log: %s (%d bytes)\n\n", logPath, len(data))
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")

	cutoff := time.Now().Add(-24 * time.Hour)
	counts := map[string]int{}
	panicCount24h := 0
	var recentPanics []string
	for _, l := range lines {
		if i := strings.Index(l, "["); i >= 0 {
			ts, err := time.Parse(time.RFC3339, strings.TrimSpace(l[:i]))
			if err != nil || ts.Before(cutoff) {
				continue
			}
			rest := l[i:]
			if j := strings.Index(rest, "]"); j > 1 {
				counts[rest[1:j]]++
			}
			if strings.Contains(l, "PANIC") {
				panicCount24h++
				if len(recentPanics) < 5 {
					recentPanics = append(recentPanics, l)
				}
			}
		}
	}

	if panicCount24h > 0 {
		fmt.Printf("⚠  %d panic(s) in the last 24h — hindcast caught them but the hook that panicked produced no record:\n", panicCount24h)
		for _, l := range recentPanics {
			fmt.Printf("  %s\n", l)
		}
		fmt.Println()
	}

	// Show current shell's project + recent sessions' A/B arms so the
	// user can answer "why does hindcast inject sometimes?"
	if cwd, err := os.Getwd(); err == nil {
		project := store.ResolveProject(cwd)
		hash := store.ProjectHash(project)
		fmt.Printf("current cwd: %s\n  project=%s (hash=%s)\n", cwd, project, hash)
		if logPath, err := store.ProjectLogPath(hash); err == nil {
			if recs, err := store.ReadRecentRecords(logPath, 10); err == nil && len(recs) > 0 {
				seen := map[string]string{}
				order := []string{}
				for _, r := range recs {
					if r.SessionID == "" {
						continue
					}
					if _, ok := seen[r.SessionID]; !ok {
						arm := r.Arm
						if arm == "" {
							arm = "legacy(backfill)"
						}
						seen[r.SessionID] = arm
						order = append(order, r.SessionID)
					}
					if len(order) >= 5 {
						break
					}
				}
				if len(order) > 0 {
					fmt.Printf("  recent sessions (arm): ")
					for i, sid := range order {
						if i > 0 {
							fmt.Print(", ")
						}
						short := sid
						if len(sid) > 8 {
							short = sid[:8]
						}
						fmt.Printf("%s=%s", short, seen[sid])
					}
					fmt.Println()
				}
			}
		}
		fmt.Println()
	}

	n := 20
	if len(lines) < n {
		n = len(lines)
	}
	fmt.Printf("last %d entries:\n", n)
	for _, l := range lines[len(lines)-n:] {
		fmt.Printf("  %s\n", l)
	}

	if len(counts) > 0 {
		fmt.Printf("\nlast 24h by hook: ")
		var kinds []string
		for k := range counts {
			kinds = append(kinds, k)
		}
		sort.Strings(kinds)
		for _, k := range kinds {
			fmt.Printf("%s=%d  ", k, counts[k])
		}
		fmt.Println()
	}
}

// cmdRotateSalt generates a new per-install salt and deletes all
// existing BM25 indexes. Existing indexes were hashed under the old
// salt and are unreadable without it; since records don't store
// plaintext tokens, the indexes cannot be re-keyed — they rebuild
// naturally from new turns going forward. Records themselves are kept
// (stats-only, not salt-dependent).
func cmdRotateSalt(args []string) {
	parseWithUsage("rotate-salt", "regenerate BM25 salt and clear all indexes (records kept)", args)
	projectsDir, err := store.ProjectsDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "rotate-salt: %s\n", err)
		os.Exit(1)
	}
	// Delete all per-project BM25 indexes.
	entries, _ := os.ReadDir(projectsDir)
	deleted := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(e.Name(), ".bm25.gob") {
			if err := os.Remove(filepath.Join(projectsDir, e.Name())); err == nil {
				deleted++
			}
		}
	}
	// Delete the salt itself. Next GetSalt() regenerates.
	if err := store.DeleteSalt(); err != nil {
		fmt.Fprintf(os.Stderr, "rotate-salt: %s\n", err)
		os.Exit(1)
	}
	// Touch a fresh salt so the file exists at the expected path.
	if _, err := store.GetSalt(); err != nil {
		fmt.Fprintf(os.Stderr, "rotate-salt: regenerate: %s\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "rotate-salt: new salt generated; %d BM25 index(es) cleared.\n", deleted)
	fmt.Fprintln(os.Stderr, "  Records (stats-only) are retained. Indexes rebuild from next turns onward.")
}

// cmdForget deletes one project's log + index, then rebuilds the global
// sketch from remaining projects.
func cmdForget(args []string) {
	fl := flag.NewFlagSet("forget", flag.ExitOnError)
	fl.Usage = func() {
		fmt.Fprintln(os.Stderr, "forget — delete a project's data, rebuild global sketch")
		fmt.Fprintln(os.Stderr, "\nUsage: hindcast forget <project-name|project-hash>")
	}
	_ = fl.Parse(args)
	if fl.NArg() < 1 {
		fl.Usage()
		os.Exit(1)
	}
	arg := fl.Arg(0)
	hash := arg
	if len(arg) != 8 {
		hash = store.ProjectHash(arg)
	}

	projectsDir, err := store.ProjectsDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "forget: %s\n", err)
		os.Exit(1)
	}

	// Delete primary + rotated JSONL segments + BM25 index for this hash.
	for _, name := range []string{hash + ".jsonl", hash + ".bm25.gob"} {
		_ = os.Remove(filepath.Join(projectsDir, name))
	}
	for i := 1; i <= 5; i++ {
		_ = os.Remove(filepath.Join(projectsDir, fmt.Sprintf("%s.jsonl.%d", hash, i)))
	}

	// Rebuild global sketch from remaining projects.
	sketch := &store.Sketch{MaxSize: store.SketchMaxWindow}
	entries, _ := os.ReadDir(projectsDir)
	rebuilt := 0
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".jsonl" {
			continue
		}
		rs, err := store.ReadRecentRecords(filepath.Join(projectsDir, e.Name()), 1<<20)
		if err != nil {
			continue
		}
		for _, r := range rs {
			sketch.Add(r.WallSeconds, r.ClaudeActiveSeconds)
			rebuilt++
		}
	}
	if err := sketch.Save(); err != nil {
		fmt.Fprintf(os.Stderr, "forget: sketch rebuild: %s\n", err)
	}

	fmt.Fprintf(os.Stderr, "forgot project %s; global sketch rebuilt from %d remaining records\n",
		hash, rebuilt)
}


// showHealth prints the persisted tuned predictor state — the result of
// the last `hindcast tune` run. If never tuned, suggests running it.
func showHealth() {
	h, err := health.Load()
	if err != nil {
		fmt.Println("hindcast: no health data yet — run `hindcast tune` to compute the empirical sim cliff for your data")
		return
	}
	fmt.Printf("hindcast predictor health (last tuned %s)\n\n", h.LastTunedAt)
	fmt.Printf("  predictions analyzed: %d  (kNN n=%d, bucket n=%d)\n", h.NPredictions, h.NKNN, h.NBucket)
	fmt.Printf("  bucket MALR:  %.2fx\n", h.BucketMALR)
	fmt.Printf("  project MALR: %.2fx\n", h.GroupMALR)
	if h.GlobalMALR > 0 {
		fmt.Printf("  global MALR:  %.2fx\n", h.GlobalMALR)
	}
	fmt.Println()
	if math.IsInf(h.TunedSimThreshold, 1) {
		fmt.Println("  tuned threshold: never inject (kNN does not beat bucket on your data)")
	} else {
		fmt.Printf("  tuned threshold: sim ≥ %.2f  →  kNN MALR %.2fx\n",
			h.TunedSimThreshold, h.KNNMALRAtThreshold)
	}
	fmt.Printf("  verdict: %s\n", h.Verdict)
	if len(h.SimBuckets) > 0 {
		fmt.Println("\n  kNN MALR by sim bucket:")
		fmt.Printf("    %-12s  %6s  %10s\n", "sim", "n", "wall MALR")
		for _, b := range h.SimBuckets {
			if b.N == 0 {
				continue
			}
			fmt.Printf("    %.2f-%.2f    %6d  %9.2fx\n", b.LoSim, b.HiSim, b.N, b.WallMALR)
		}
	}
}
