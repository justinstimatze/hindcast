package main

import (
	"bufio"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	iofs "io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/sizes"
	"github.com/justinstimatze/hindcast/internal/store"
	"github.com/justinstimatze/hindcast/internal/tags"
	"github.com/justinstimatze/hindcast/internal/transcript"
)

// cmdBackfill walks ~/.claude/projects/*/*.jsonl, converts historical
// turns into Records, and populates per-project logs + BM25 indexes +
// global sketch. Uses a done-file per transcript for safe restart.
//
// Plaintext prompts are processed in-memory only (classification +
// token hashing). They never land on disk — same privacy guarantees
// as the live hook path.
func cmdBackfill(args []string) {
	fl := flag.NewFlagSet("backfill", flag.ExitOnError)
	verbose := fl.Bool("v", false, "verbose per-transcript output")
	rebuild := fl.Bool("rebuild", false, "clear done-files and reprocess all transcripts (wipes project data first)")
	_ = fl.Parse(args)

	if *rebuild {
		if err := wipeBackfillState(); err != nil {
			fmt.Fprintf(os.Stderr, "hindcast backfill: rebuild wipe: %s\n", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "hindcast backfill: rebuild — cleared existing project data and done-files")
	}

	projectsRoot, err := findCCProjectsRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast backfill: cannot find CC projects dir: %s\n", err)
		os.Exit(1)
	}

	transcripts := collectTranscripts(projectsRoot)
	fmt.Fprintf(os.Stderr, "hindcast backfill: scanning %d transcript files from %s\n",
		len(transcripts), projectsRoot)

	salt, err := store.GetSalt()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast backfill: salt init: %s\n", err)
		os.Exit(1)
	}

	doneDir, err := backfillDoneDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast backfill: done dir: %s\n", err)
		os.Exit(1)
	}

	sketch, err := store.LoadSketch()
	if err != nil {
		sketch = &store.Sketch{}
	}

	var (
		totalTurns, totalTranscripts, skipped int
		perProject                            = map[string]int{}
		start                                 = time.Now()
	)

	for _, path := range transcripts {
		doneFile := filepath.Join(doneDir, transcriptKey(path))
		if _, err := os.Stat(doneFile); err == nil {
			skipped++
			continue
		}

		cwd, sessionID := sniffMeta(path)
		if cwd == "" {
			if *verbose {
				fmt.Fprintf(os.Stderr, "  skip (no cwd): %s\n", path)
			}
			continue
		}

		project := store.ResolveProject(cwd)
		hash := store.ProjectHash(project)

		turns, err := transcript.ParseFile(path, time.Time{})
		if err != nil {
			if *verbose {
				fmt.Fprintf(os.Stderr, "  parse error %s: %s\n", path, err)
			}
			continue
		}
		if len(turns) == 0 {
			_ = markDone(doneFile)
			continue
		}

		logPath, err := store.ProjectLogPath(hash)
		if err != nil {
			continue
		}
		bm25Path, err := store.ProjectBM25Path(hash)
		if err != nil {
			continue
		}
		idx, err := bm25.Load(bm25Path)
		if err != nil {
			idx = bm25.New()
		}

		turnsWritten := 0
		for _, t := range turns {
			toolCount := 0
			for _, c := range t.ToolCalls {
				toolCount += c
			}

			// Classification runs on first-line only to match runtime
			// cmdPending behavior. Full prompt feeds BM25 since BM25
			// benefits from more tokens.
			taskType := string(tags.Classify(firstLine(t.PromptText)))
			sizeBucket := string(sizes.Classify(t.FilesTouched, toolCount))

			var tokens []uint64
			if len(t.PromptText) > 0 {
				tokens = bm25.HashTokens(t.PromptText, salt)
			}
			r := store.Record{
				TS:                  t.StopTS.UTC(),
				SessionID:           sessionID,
				ProjectHash:         hash,
				Model:               t.Model,
				PermissionMode:      "", // not recoverable from old transcripts
				TaskType:            taskType,
				SizeBucket:          sizeBucket,
				WallSeconds:         t.WallSeconds(),
				ClaudeActiveSeconds: t.ActiveSeconds,
				PromptChars:         len(t.PromptText),
				PromptTokens:        tokens,
				ResponseChars:       t.ResponseChars,
				ToolCalls:           t.ToolCalls,
				FilesTouched:        t.FilesTouched,
			}
			// Skip turns with zero wall — usually means an incomplete turn
			// (interrupted or malformed). They pollute medians.
			if r.WallSeconds <= 0 {
				continue
			}
			// Skip abandoned/sleep-resumed turns (wall huge, active tiny).
			// Common in historical transcripts; they break the lognormal
			// assumption the predictor relies on.
			if store.IsAbandonedTurn(r.WallSeconds, r.ClaudeActiveSeconds) {
				continue
			}
			if err := store.AppendRecord(logPath, r); err != nil {
				continue
			}
			if len(tokens) > 0 {
				idx.Add(tokens, bm25.Doc{
					ActiveSeconds: r.ClaudeActiveSeconds,
					WallSeconds:   r.WallSeconds,
					TaskType:      r.TaskType,
					SizeBucket:    r.SizeBucket,
					ToolCount:     toolCount,
					FilesTouched:  r.FilesTouched,
					TS:            r.TS,
				})
			}
			sketch.Add(r.WallSeconds, r.ClaudeActiveSeconds)
			turnsWritten++
			totalTurns++
			perProject[project]++
		}

		_ = idx.Save(bm25Path)
		_ = markDone(doneFile)
		totalTranscripts++

		if *verbose {
			fmt.Fprintf(os.Stderr, "  %-50s  %3d turns -> %s\n",
				filepath.Base(path), turnsWritten, project)
		}
	}

	if err := sketch.Save(); err != nil {
		fmt.Fprintf(os.Stderr, "hindcast backfill: sketch save: %s\n", err)
	}

	elapsed := time.Since(start).Round(time.Millisecond)
	fmt.Fprintf(os.Stderr, "\nbackfilled %d turns from %d transcripts across %d projects (skipped %d already-done) in %s\n",
		totalTurns, totalTranscripts, len(perProject), skipped, elapsed)

	// Top 10 projects by turn count.
	type pc struct {
		name string
		n    int
	}
	var list []pc
	for n, c := range perProject {
		list = append(list, pc{n, c})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].n > list[j].n })
	top := 10
	if len(list) < top {
		top = len(list)
	}
	if top > 0 {
		fmt.Fprintf(os.Stderr, "\nTop projects by turn count:\n")
		for i := 0; i < top; i++ {
			fmt.Fprintf(os.Stderr, "  %4d  %s\n", list[i].n, list[i].name)
		}
	}
}

func findCCProjectsRoot() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	path := filepath.Join(home, ".claude", "projects")
	if _, err := os.Stat(path); err != nil {
		return "", err
	}
	return path, nil
}

func collectTranscripts(root string) []string {
	var paths []string
	_ = filepath.WalkDir(root, func(p string, d iofs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			// Skip subagent transcripts — hindcast only tracks main sessions.
			if d.Name() == "subagents" {
				return iofs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".jsonl") {
			return nil
		}
		paths = append(paths, p)
		return nil
	})
	sort.Strings(paths)
	return paths
}

// sniffMeta reads the first few entries of a transcript to discover its
// cwd and sessionId. Both are stable across a transcript; we only need
// one hit each.
func sniffMeta(path string) (cwd, sessionID string) {
	f, err := os.Open(path)
	if err != nil {
		return "", ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for sc.Scan() {
		var e struct {
			CWD       string `json:"cwd"`
			SessionID string `json:"sessionId"`
		}
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			continue
		}
		if cwd == "" {
			cwd = e.CWD
		}
		if sessionID == "" {
			sessionID = e.SessionID
		}
		if cwd != "" && sessionID != "" {
			return
		}
	}
	return
}

func transcriptKey(path string) string {
	sum := md5.Sum([]byte(path))
	return hex.EncodeToString(sum[:])[:16]
}

func backfillDoneDir() (string, error) {
	root, err := store.HindcastDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "backfill-done")
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

func markDone(path string) error {
	return os.WriteFile(path, []byte(time.Now().UTC().Format(time.RFC3339)), 0600)
}

// wipeBackfillState clears done-files, per-project logs, BM25 indexes,
// and the global sketch so that `backfill --rebuild` starts from zero.
func wipeBackfillState() error {
	done, err := backfillDoneDir()
	if err == nil {
		_ = os.RemoveAll(done)
	}
	projects, err := store.ProjectsDir()
	if err == nil {
		_ = os.RemoveAll(projects)
	}
	if sketch, err := store.GlobalSketchPath(); err == nil {
		_ = os.Remove(sketch)
	}
	return nil
}
