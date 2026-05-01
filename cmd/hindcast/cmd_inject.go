package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/hook"
	"github.com/justinstimatze/hindcast/internal/store"
)

type injectInput struct {
	SessionID      string `json:"session_id"`
	CWD            string `json:"cwd"`
	PermissionMode string `json:"permission_mode"`
	HookEventName  string `json:"hook_event_name"`
}

// Minimum records in a bucket before we display it. Below this, the
// bucket is too noisy to cite; fall back to overall.
const bucketMinN = 4

// Recent-turn count for the eyeball-retrieval block.
const recentN = 3

func cmdInject() {
	if os.Getenv("HINDCAST_SKIP") == "1" {
		return
	}
	var in injectInput
	if err := hook.Decode("inject", &in); err != nil {
		return
	}
	if in.CWD == "" {
		return
	}

	// Opportunistic TTL sweep of orphaned pending files.
	_ = store.SweepPending(6 * time.Hour)
	_ = store.SweepSessionMomentum(24 * time.Hour)

	// SessionStart is silent on the default path. v0.6 moved active
	// injection to UserPromptSubmit (cmd_pending.go formatClaudeInjection)
	// so it can carry the per-prompt prediction. The v0.1 ungated
	// session-scoped bucket-table inject below stays available behind
	// HINDCAST_LEGACY_INJECT=1 for eval-api A/B research.
	if os.Getenv("HINDCAST_LEGACY_INJECT") != "1" {
		return
	}

	// Control-arm sessions get no injection — measurement floor for lift.
	if store.SessionArm(in.SessionID, store.ControlPctFromEnv()) == store.ArmControl {
		hook.Logf("inject", "session %s in control arm — no injection", in.SessionID)
		return
	}

	project := store.ResolveProject(in.CWD)
	hash := store.ProjectHash(project)

	logPath, err := store.ProjectLogPath(hash)
	if err != nil {
		hook.Logf("inject", "log path: %s", err)
		return
	}
	records, err := store.ReadRecentRecords(logPath, 500)
	if err != nil {
		hook.Logf("inject", "read records: %s", err)
		records = nil
	}
	sketch, err := store.LoadSketch()
	if err != nil {
		hook.Logf("inject", "load sketch: %s", err)
		sketch = &store.Sketch{}
	}

	context := formatInjection(project, in.SessionID, in.PermissionMode, records, sketch)
	emit(context)
}

// emit writes the SessionStart hook's JSON response on stdout. CC injects
// additionalContext as a user-invisible system prompt for the session.
func emit(context string) {
	resp := map[string]any{
		"hookSpecificOutput": map[string]any{
			"hookEventName":     "SessionStart",
			"additionalContext": context,
		},
	}
	data, _ := json.Marshal(resp)
	fmt.Println(string(data))
}

func formatInjection(project, sessionID, mode string, records []store.Record, sketch *store.Sketch) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Wall-clock priors for this project (Claude-minutes)\n")

	// Records matching the current permission_mode are the most relevant;
	// pooling across modes produces misleading wall-clock aggregates.
	scoped := filterByMode(records, mode)
	fmt.Fprintf(&b, "Project: %s    session_id: %s    n=%d    permission_mode=%s\n\n",
		project, sessionID, len(scoped), fallback(mode, "unknown"))

	if len(scoped) > 0 {
		wm, wP75, wP90, am, aP75, aP90 := aggregate(scoped)
		fmt.Fprintf(&b, "Overall:\n")
		fmt.Fprintf(&b, "  active med %sm / p75 %sm / p90 %sm   |   wall med %sm / p75 %sm / p90 %sm\n\n",
			fmtMin(am), fmtMin(aP75), fmtMin(aP90), fmtMin(wm), fmtMin(wP75), fmtMin(wP90))

		buckets := byBucket(scoped)
		if len(buckets) > 0 {
			fmt.Fprintf(&b, "By task type × size bucket (self-tag, cite the matching row):\n")
			for _, bk := range buckets {
				fmt.Fprintf(&b, "  %-16s (n=%d) active med %s/p75 %s/p90 %s | wall med %s/p75 %s/p90 %s\n",
					bk.Label, bk.N,
					fmtMin(bk.ActiveMed), fmtMin(bk.ActiveP75), fmtMin(bk.ActiveP90),
					fmtMin(bk.WallMed), fmtMin(bk.WallP75), fmtMin(bk.WallP90))
			}
			b.WriteString("\n")
		}

		models := byModel(scoped, 10)
		if len(models) >= 2 {
			fmt.Fprintf(&b, "By model (n>=10 each):\n")
			for _, m := range models {
				fmt.Fprintf(&b, "  %-28s (n=%d) active med %s/p75 %s | wall med %s/p75 %s\n",
					m.Label, m.N,
					fmtMin(m.ActiveMed), fmtMin(m.ActiveP75),
					fmtMin(m.WallMed), fmtMin(m.WallP75))
			}
			b.WriteString("\n")
		}
	} else {
		fmt.Fprintf(&b, "No per-project records yet under this permission mode.\n")
		if len(records) > 0 {
			fmt.Fprintf(&b, "(%d records exist under other modes — not cited to avoid cross-mode pollution.)\n", len(records))
		}
		b.WriteString("\n")
	}

	if _, _, _, _, gn := sketch.Percentiles(); gn > 0 {
		_, _, ga, gap, _ := sketch.Percentiles()
		fmt.Fprintf(&b, "Global (all projects, numeric-only, n=%d):\n", gn)
		fmt.Fprintf(&b, "  active median %sm / p90 %sm\n\n", fmtMin(ga), fmtMin(gap))
	}

	// Recent: newest first, already in reverse-chron order from ReadRecentRecords.
	recent := records
	if len(recent) > recentN {
		recent = recent[:recentN]
	}
	if len(recent) > 0 {
		fmt.Fprintf(&b, "Recent %d in this project (stats-only — no prompt text persists):\n", len(recent))
		for _, r := range recent {
			tools := 0
			for _, c := range r.ToolCalls {
				tools += c
			}
			label := taskSizeLabel(r.TaskType, r.SizeBucket)
			fmt.Fprintf(&b, "- %-16s — active %sm / wall %sm — %d tools, %d files\n",
				label,
				fmtMin(float64(r.ClaudeActiveSeconds)),
				fmtMin(float64(r.WallSeconds)),
				tools, r.FilesTouched)
		}
		b.WriteString("\n")
	}

	b.WriteString(`When estimating wall-clock:
  1. Tag the current prompt mentally (task_type + size).
  2. Cite the matching bucket if n>=4; else fall back to overall project; else global.
  3. State metric (active vs wall), bucket, n. Small-sample: say so.
  4. A BM25 close-match line may arrive separately at UserPromptSubmit — cross-reference it.
  5. Call the hindcast_estimate MCP tool with session_id above + your wall/active guess.
`)

	s := b.String()
	if len(s) > maxInjectionChars {
		hook.Logf("inject", "payload %d chars > %d — truncating", len(s), maxInjectionChars)
		cut := maxInjectionChars
		for cut > 0 && s[cut] != '\n' {
			cut--
		}
		if cut == 0 {
			cut = maxInjectionChars
		}
		s = s[:cut] + "\n[hindcast: truncated at token budget]\n"
	}
	return s
}

// maxInjectionChars keeps the SessionStart payload well under 500 tokens
// (roughly 4 chars/token). Runs on every session, so the budget matters.
const maxInjectionChars = 2000

func filterByMode(records []store.Record, mode string) []store.Record {
	if mode == "" {
		return records
	}
	out := make([]store.Record, 0, len(records))
	for _, r := range records {
		if r.PermissionMode == mode || r.PermissionMode == "" {
			out = append(out, r)
		}
	}
	return out
}

func aggregate(records []store.Record) (wallMed, wallP75, wallP90, activeMed, activeP75, activeP90 float64) {
	walls := make([]int, 0, len(records))
	actives := make([]int, 0, len(records))
	for _, r := range records {
		walls = append(walls, r.WallSeconds)
		actives = append(actives, r.ClaudeActiveSeconds)
	}
	wallMed, wallP75, wallP90 = store.QuantilesWide(walls)
	activeMed, activeP75, activeP90 = store.QuantilesWide(actives)
	return
}

type bucketRow struct {
	Label                           string
	N                               int
	WallMed, WallP75, WallP90       float64
	ActiveMed, ActiveP75, ActiveP90 float64
}

func byBucket(records []store.Record) []bucketRow {
	groups := map[string][]store.Record{}
	for _, r := range records {
		label := taskSizeLabel(r.TaskType, r.SizeBucket)
		groups[label] = append(groups[label], r)
	}
	var rows []bucketRow
	for label, rs := range groups {
		if len(rs) < bucketMinN {
			continue
		}
		wm, wP75, wP90, am, aP75, aP90 := aggregate(rs)
		rows = append(rows, bucketRow{
			Label: label, N: len(rs),
			WallMed: wm, WallP75: wP75, WallP90: wP90,
			ActiveMed: am, ActiveP75: aP75, ActiveP90: aP90,
		})
	}
	sort.Slice(rows, func(a, b int) bool {
		if rows[a].N != rows[b].N {
			return rows[a].N > rows[b].N
		}
		return rows[a].Label < rows[b].Label
	})
	return rows
}

func byModel(records []store.Record, minN int) []bucketRow {
	groups := map[string][]store.Record{}
	for _, r := range records {
		if r.Model == "" {
			continue
		}
		groups[r.Model] = append(groups[r.Model], r)
	}
	var rows []bucketRow
	for label, rs := range groups {
		if len(rs) < minN {
			continue
		}
		wm, wP75, wP90, am, aP75, aP90 := aggregate(rs)
		rows = append(rows, bucketRow{
			Label: label, N: len(rs),
			WallMed: wm, WallP75: wP75, WallP90: wP90,
			ActiveMed: am, ActiveP75: aP75, ActiveP90: aP90,
		})
	}
	sort.Slice(rows, func(a, b int) bool { return rows[a].N > rows[b].N })
	return rows
}

func taskSizeLabel(task, size string) string {
	if task == "" {
		task = "other"
	}
	if size == "" {
		return task
	}
	return task + "-" + size
}

// fmtMin formats a seconds value as minutes: "2.1" for <10m, "38" for
// bigger values. Empty / zero shows as "0.0".
func fmtMin(sec float64) string {
	m := sec / 60.0
	if m < 10 {
		return fmt.Sprintf("%.1f", m)
	}
	return fmt.Sprintf("%.0f", m)
}

func fallback(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
