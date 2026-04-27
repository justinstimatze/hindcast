package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/predict"
	"github.com/justinstimatze/hindcast/internal/store"
	"github.com/justinstimatze/hindcast/internal/tags"
)

// cmdPredict is the user-facing CLI predictor. Takes a prompt on argv
// or stdin, runs the same kNN the UserPromptSubmit hook runs, prints
// the prediction as a single human-readable line. Useful for "is this
// going to take 5 minutes or an hour?" before deciding whether to ask.
func cmdPredict(args []string) {
	fs := flag.NewFlagSet("predict", flag.ExitOnError)
	project := fs.String("project", "", "project name or path (default: cwd)")
	jsonOut := fs.Bool("json", false, "emit the full Prediction as JSON")
	_ = fs.Parse(args)

	prompt := strings.Join(fs.Args(), " ")
	if prompt == "" {
		data, _ := io.ReadAll(os.Stdin)
		prompt = strings.TrimSpace(string(data))
	}
	if prompt == "" {
		fmt.Fprintln(os.Stderr, "hindcast predict: no prompt (pass as args or on stdin)")
		os.Exit(2)
	}

	resolved := *project
	if resolved == "" {
		if cwd, err := os.Getwd(); err == nil {
			resolved = cwd
		}
	}
	resolved = store.ResolveProject(resolved)
	hash := store.ProjectHash(resolved)
	taskType := string(tags.Classify(firstLine(prompt)))

	var tokens []uint64
	if salt, err := store.GetSalt(); err == nil {
		tokens = bm25.HashTokens(prompt, salt)
	}

	p := computePrediction(hash, tokens, taskType, "", len(prompt))

	if *jsonOut {
		data, _ := json.MarshalIndent(p, "", "  ")
		fmt.Println(string(data))
		return
	}
	fmt.Println(formatPrediction(p))
}

// cmdStatusline reads the active session's latest prediction and prints
// one line suitable for the Claude Code status line. Silently exits if
// no prediction exists — empty output is the correct status line when
// hindcast has nothing to say.
func cmdStatusline(args []string) {
	fs := flag.NewFlagSet("statusline", flag.ExitOnError)
	session := fs.String("session", "", "session id (default: current-session pointer)")
	_ = fs.Parse(args)

	sessionID := *session
	if sessionID == "" {
		ptr, err := store.CurrentSessionPointerPath()
		if err != nil {
			return
		}
		data, err := os.ReadFile(ptr)
		if err != nil {
			return
		}
		sessionID = strings.TrimSpace(string(data))
	}
	if sessionID == "" {
		return
	}

	path, err := store.LastPredictionPath(sessionID)
	if err != nil {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var p predict.Prediction
	if err := json.Unmarshal(data, &p); err != nil {
		return
	}
	fmt.Println(formatPrediction(p))
}

// formatPrediction produces a compact single-line rendering of a
// Prediction suitable for a status bar. No color codes — Claude Code
// handles its own styling.
func formatPrediction(p predict.Prediction) string {
	if p.Source == predict.SourceNone {
		switch {
		case p.N == 0:
			return "hindcast: calibrating · 0 records — predictions land after a few turns"
		case p.N < 4:
			return fmt.Sprintf("hindcast: calibrating · %d/4 records in this project", p.N)
		default:
			return fmt.Sprintf("hindcast: insufficient signal · %d records but no tier fired", p.N)
		}
	}
	wall := humanDuration(p.WallSeconds)
	active := humanDuration(p.ActiveSeconds)
	band := ""
	if p.WallP25 > 0 && p.WallP75 > 0 && p.WallP75 > p.WallP25 {
		band = fmt.Sprintf(" [%s–%s]", humanDuration(p.WallP25), humanDuration(p.WallP75))
	}
	tag := p.TaskType
	if tag == "" {
		tag = "other"
	}
	src := string(p.Source)
	switch p.Source {
	case predict.SourceKNN:
		src = fmt.Sprintf("knn sim=%.2f", p.MaxSim)
	case predict.SourceRegressor:
		if p.SourceDetail != "" {
			src = fmt.Sprintf("regressor:%s", p.SourceDetail)
		}
	}
	return fmt.Sprintf("hindcast: ~%s wall / %s active%s · %s n=%d · %s",
		wall, active, band, src, p.N, tag)
}

// humanDuration renders seconds as a compact minutes/seconds string.
// Keeps the status line short: "45s", "3m", "1h12m".
func humanDuration(s int) string {
	if s <= 0 {
		return "0s"
	}
	if s < 60 {
		return fmt.Sprintf("%ds", s)
	}
	if s < 3600 {
		return fmt.Sprintf("%dm", (s+30)/60)
	}
	h := s / 3600
	m := (s % 3600) / 60
	if m == 0 {
		return fmt.Sprintf("%dh", h)
	}
	return fmt.Sprintf("%dh%dm", h, m)
}
