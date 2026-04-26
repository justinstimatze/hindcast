package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	iofs "io/fs"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/seed"
	"github.com/justinstimatze/hindcast/internal/sizes"
	"github.com/justinstimatze/hindcast/internal/store"
	"github.com/justinstimatze/hindcast/internal/tags"
	"github.com/justinstimatze/hindcast/internal/transcript"
)

// currentEvalModel is populated at cmdEvalAPI entry and read by
// buildPriorsBlockForProject so bias factors match the model being
// evaluated. Module-level state is slightly ugly but keeps the
// per-sample call site terse.
var currentEvalModel = "claude-sonnet-4-6"

// cmdEvalAPI runs the offline A/B against the Claude API. Samples N
// historical turns from local transcripts (known prompts, known actual
// durations), queries Claude twice per sample — once with hindcast's
// priors injected, once without — via a forced hindcast_estimate tool
// call. Parses the tool input for Claude's self-reported estimate,
// compares to actual, reports MALR per arm.
//
// This is the evidence that backs the README's before/after story.
// Requires ANTHROPIC_API_KEY (from shell env or .env in cwd).
func cmdEvalAPI(args []string) {
	fl := flag.NewFlagSet("eval-api", flag.ExitOnError)
	n := fl.Int("n", 50, "number of historical turns to evaluate")
	model := fl.String("model", "claude-sonnet-4-6", "Claude model to query")
	seed := fl.Int64("seed", 42, "random seed for sample selection")
	maxChars := fl.Int("max-prompt-chars", 4000, "skip turns whose prompt exceeds this length")
	_ = fl.Parse(args)

	apiKey := loadAPIKey()
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "eval-api: ANTHROPIC_API_KEY not found (shell env or .env in cwd)")
		os.Exit(1)
	}

	// Stash for buildPriorsBlockForProject so we resolve bias factors
	// against the eval's target model.
	currentEvalModel = *model

	fmt.Fprintf(os.Stderr, "eval-api: sampling up to %d turns (seed=%d, model=%s)\n", *n, *seed, *model)
	samples := pickAPISamples(*n, *seed, *maxChars)
	if len(samples) == 0 {
		fmt.Fprintln(os.Stderr, "eval-api: no historical transcripts found at ~/.claude/projects/")
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "eval-api: %d samples selected; running A/B against %s...\n", len(samples), *model)

	results := evalAPISamples(samples, apiKey, *model)
	reportAPIResults(results)
}

type apiSample struct {
	Prompt        string
	Project       string
	ProjectHash   string
	ActualWall    int
	ActualActive  int
	TaskType      string
	SizeBucket    string
	PriorsBlock   string
}

func pickAPISamples(n int, seed int64, maxChars int) []apiSample {
	projectsRoot, err := findCCProjectsRoot()
	if err != nil {
		return nil
	}
	transcripts := collectTranscripts(projectsRoot)
	if len(transcripts) == 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(transcripts), func(i, j int) { transcripts[i], transcripts[j] = transcripts[j], transcripts[i] })

	// Cache priors per project and cap samples per project so the A/B
	// doesn't end up dominated by one chatty codebase.
	priorsCache := map[string]string{}
	perProject := map[string]int{}
	maxPerProject := n/4 + 1

	var out []apiSample
	for _, path := range transcripts {
		if len(out) >= n {
			break
		}
		cwd, _ := sniffMeta(path)
		if cwd == "" {
			continue
		}
		project := store.ResolveProject(cwd)
		hash := store.ProjectHash(project)
		if perProject[hash] >= maxPerProject {
			continue
		}

		turns, err := transcript.ParseFile(path, time.Time{})
		if err != nil || len(turns) == 0 {
			continue
		}
		rng.Shuffle(len(turns), func(i, j int) { turns[i], turns[j] = turns[j], turns[i] })
		for _, t := range turns {
			if len(out) >= n {
				break
			}
			if t.WallSeconds() <= 0 || len(t.PromptText) == 0 {
				continue
			}
			if len(t.PromptText) > maxChars {
				continue
			}
			// Skip automation-shaped prompts (structured triage messages,
			// JSON blobs) — they skew toward terse templated content
			// that doesn't exercise Claude's estimation faculty.
			if looksAutomated(t.PromptText) {
				continue
			}
			toolCount := 0
			for _, c := range t.ToolCalls {
				toolCount += c
			}
			_ = priorsCache // kept for future caching of non-prompt-specific priors
			// What goes to the API as the user message: rolling-window
			// context + current prompt, matching what the product hook
			// feeds into Claude's attention at UserPromptSubmit time.
			// Classification runs on the raw prompt only — firstLine on
			// the concatenated window returned the oldest line, not the
			// user's request, silently degrading task-type selection for
			// every post-rolling-window turn.
			effectivePrompt := transcript.ComposeEffectiveTask(t.PromptText, path)
			pb := buildPriorsBlockForProject(project, hash, t.PromptText, currentEvalModel)
			out = append(out, apiSample{
				Prompt:       effectivePrompt,
				Project:      project,
				ProjectHash:  hash,
				ActualWall:   t.WallSeconds(),
				ActualActive: t.ActiveSeconds,
				TaskType:     string(tags.Classify(firstLine(t.PromptText))),
				SizeBucket:   string(sizes.Classify(t.FilesTouched, toolCount)),
				PriorsBlock:  pb,
			})
			perProject[hash]++
			break
		}
	}
	return out
}

// looksAutomated flags prompts that look templated / non-interactive.
// Round-4 panel found the initial filter missed `<task-notification>`,
// shell paste lines, marker phrases, and too-short prompts. Widened.
func looksAutomated(s string) bool {
	trimmed := strings.TrimSpace(s)
	if len(trimmed) < 10 {
		return true
	}
	// First line is the strongest signal for automation shape.
	firstLine := trimmed
	if i := strings.IndexByte(trimmed, '\n'); i > 0 {
		firstLine = trimmed[:i]
	}
	// XML/HTML-like tag prefix: <task-notification>, <system-reminder>, etc.
	if strings.HasPrefix(firstLine, "<") && strings.Contains(firstLine, ">") {
		return true
	}
	// Bracket-prefix automation ("[GAS TOWN] boot <- daemon • ...").
	if strings.HasPrefix(firstLine, "[") && strings.Contains(firstLine, "] ") {
		return true
	}
	// Bullet-date pattern (schedulers that log "• 2026-04-19T03:26").
	if strings.Contains(firstLine, "•") && strings.Contains(firstLine, "T") {
		return true
	}
	// Shell-command paste.
	if strings.HasPrefix(firstLine, "$ ") || strings.HasPrefix(firstLine, "sudo ") {
		return true
	}
	// Marker phrases — automation sends these as status updates.
	for _, m := range []string{
		"_DONE ", "_READY ", "_COMPLETE ", "_COMPLETED",
		"Polecat dispatched", "QUEUED NUDGE", "MERGE_READY",
		"Boot triage:", "Formula mol-",
	} {
		if strings.Contains(firstLine, m) {
			return true
		}
	}
	return false
}

// buildPriorsBlockForProject returns a task-type-matched priors block
// for the treatment arm. Leads with a bias-corrected recommendation so
// Claude has the exact wall/active integers to pass to hindcast_estimate
// without Claude-side arithmetic. This is the v0.1.1 bias-correction
// loop: bucket p75 divided by the known (model, project) Claude-bias
// factor produces a corrected number that should land near the bucket's
// actual bias (0.93×).
func buildPriorsBlockForProject(project, hash, prompt, evalModel string) string {
	logPath, err := store.ProjectLogPath(hash)
	if err != nil {
		return ""
	}
	records, _ := store.ReadRecentRecords(logPath, 500)
	if len(records) == 0 {
		return ""
	}

	// Bias factor: live measured on this model if enough samples, else
	// shipped default.
	wallBias, activeBias, biasN := store.ComputeBiasFactor(records, evalModel)
	biasSource := fmt.Sprintf("live n=%d on %s", biasN, evalModel)
	if biasN < 10 {
		def := seed.BiasDefaultFor(evalModel)
		wallBias, activeBias = def.WallFactor, def.ActiveFactor
		biasSource = fmt.Sprintf("shipped default (%s)", def.Source)
	}
	if wallBias <= 0 {
		wallBias = 1.0
	}
	if activeBias <= 0 {
		activeBias = 1.0
	}

	var b strings.Builder
	fmt.Fprintf(&b, "## Wall-clock priors for this project\nProject: %s    n=%d\n", project, len(records))

	// Overall project stats.
	walls := make([]int, 0, len(records))
	actives := make([]int, 0, len(records))
	for _, r := range records {
		walls = append(walls, r.WallSeconds)
		actives = append(actives, r.ClaudeActiveSeconds)
	}
	wm, wP75, wP90 := store.QuantilesWide(walls)
	am, aP75, aP90 := store.QuantilesWide(actives)
	fmt.Fprintf(&b, "Overall: active med %.1fm / p75 %.1fm / p90 %.1fm  |  wall med %.1fm / p75 %.1fm / p90 %.1fm\n",
		am/60, aP75/60, aP90/60, wm/60, wP75/60, wP90/60)

	// Task-type-matched bucket for THIS prompt.
	taskType := string(tags.Classify(firstLine(prompt)))
	var taskWalls, taskActives []int
	for _, r := range records {
		if r.TaskType == taskType {
			taskWalls = append(taskWalls, r.WallSeconds)
			taskActives = append(taskActives, r.ClaudeActiveSeconds)
		}
	}

	var (
		wallP75Sec, activeP75Sec int
		bucketLabel              string
	)
	if len(taskWalls) >= 3 {
		twm, twP75, twP90 := store.QuantilesWide(taskWalls)
		_, taP75, _ := store.QuantilesWide(taskActives)
		fmt.Fprintf(&b, "This prompt tagged: %s. Matching bucket (n=%d): wall med %.1fm / p75 %.1fm / p90 %.1fm\n",
			taskType, len(taskWalls), twm/60, twP75/60, twP90/60)
		wallP75Sec = int(twP75 + 0.5)
		activeP75Sec = int(taP75 + 0.5)
		bucketLabel = fmt.Sprintf("%s bucket p75", taskType)
	} else {
		fmt.Fprintf(&b, "This prompt tagged: %s. Bucket has only n=%d (below threshold) — falling back to overall project.\n",
			taskType, len(taskWalls))
		wallP75Sec = int(wP75 + 0.5)
		activeP75Sec = int(aP75 + 0.5)
		bucketLabel = "overall project wall p75"
	}

	recommendedWall := int(float64(wallP75Sec)/wallBias + 0.5)
	recommendedActive := int(float64(activeP75Sec)/activeBias + 0.5)
	fmt.Fprintf(&b, "→ Recommended hindcast_estimate (bias-corrected, %s):\n", biasSource)
	fmt.Fprintf(&b, "   wall_seconds=%d  (= %s %ds ÷ %.2f Claude-bias)\n", recommendedWall, bucketLabel, wallP75Sec, wallBias)
	fmt.Fprintf(&b, "   active_seconds=%d\n", recommendedActive)

	return b.String()
}

type apiEstimate struct {
	WallSeconds   int
	ActiveSeconds int
	Success       bool
	Reason        string
}

type apiEvalResult struct {
	Sample apiSample
	ArmA   apiEstimate // control (no priors)
	ArmB   apiEstimate // treatment (with priors)
}

func evalAPISamples(samples []apiSample, apiKey, model string) []apiEvalResult {
	client := &http.Client{Timeout: 60 * time.Second}
	var results []apiEvalResult
	for i, s := range samples {
		a := queryEstimate(client, apiKey, model, s.Prompt, "")
		b := queryEstimate(client, apiKey, model, s.Prompt, s.PriorsBlock)
		results = append(results, apiEvalResult{Sample: s, ArmA: a, ArmB: b})
		fmt.Fprintf(os.Stderr, "  [%2d/%d] actual=%4ds  A=%5ds (ok=%v)  B=%5ds (ok=%v)  %q\n",
			i+1, len(samples),
			s.ActualWall,
			a.WallSeconds, a.Success,
			b.WallSeconds, b.Success,
			truncate(s.Prompt, 60),
		)
	}
	return results
}

func queryEstimate(client *http.Client, apiKey, model, prompt, priorsBlock string) apiEstimate {
	// Levers 1 + 3: imperative framing + few-shot examples. Priors are
	// Lever 2 (task-matched) delivered via the priorsBlock built upstream.
	var system string
	if priorsBlock != "" {
		system = priorsBlock + `

You are an experienced software engineer estimating how long a task will take.

RULES:
1. Use the wall p75 from the priors block above as your estimate, exactly.
2. Do NOT adjust for perceived complexity — the bucket already encodes it.
3. Do NOT reason from the prompt's wording ("thorough", "quick", "moderate").
4. Just look up the number and call hindcast_estimate with it.

GOOD example:
  Prompt: "add retry logic to fetcher"
  Priors block says: feature bucket wall p75 = 3.2m (192s)
  → hindcast_estimate(wall_seconds=192, active_seconds=60)
  (You used the number from the priors. You did not add your own intuition.)

BAD example (anti-pattern):
  Prompt: "add retry logic to fetcher"
  Priors block says: feature bucket wall p75 = 3.2m
  Your response: "This sounds moderately complex, probably 15-30 minutes..."
  (You overrode the priors with semantic intuition. Don't do this.)

Call hindcast_estimate once at the start of your response. wall_seconds is the user-experienced duration including approvals and AFK; active_seconds is just Claude's own compute time.`
	} else {
		// Control arm — no priors, natural estimate.
		system = "You are an experienced software engineer estimating how long a task will take. Call the hindcast_estimate tool with your best guess. wall_seconds is user-experienced duration (including approvals and AFK); active_seconds is Claude's compute-only time."
	}

	reqBody := map[string]any{
		"model":      model,
		"max_tokens": 1024,
		"system":     system,
		"messages":   []any{map[string]any{"role": "user", "content": prompt}},
		"tools": []any{
			map[string]any{
				"name":        "hindcast_estimate",
				"description": "Record your estimate of how long this task will take.",
				"input_schema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"wall_seconds":   map[string]any{"type": "number"},
						"active_seconds": map[string]any{"type": "number"},
					},
					"required": []any{"wall_seconds", "active_seconds"},
				},
			},
		},
		"tool_choice": map[string]any{"type": "tool", "name": "hindcast_estimate"},
	}
	body, _ := json.Marshal(reqBody)

	// Retry on transient failures: 429 (rate limit), 529 (overloaded),
	// and any 5xx. Exponential backoff: 1s, 2s, 4s. Beyond that we give
	// up and the sample drops out — drops are visible in the per-row
	// log, so the user can tell if retries are exhausted at scale.
	const maxAttempts = 3
	var lastReason string
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<uint(attempt-1)) * time.Second)
		}
		req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
		if err != nil {
			return apiEstimate{Reason: "request build failed"}
		}
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("content-type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			lastReason = "http: " + err.Error()
			continue
		}
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 429 || resp.StatusCode == 529 || (resp.StatusCode >= 500 && resp.StatusCode < 600) {
			lastReason = fmt.Sprintf("http %d (retrying)", resp.StatusCode)
			continue
		}
		if resp.StatusCode != 200 {
			return apiEstimate{Reason: fmt.Sprintf("http %d", resp.StatusCode)}
		}

		var r struct {
			Content []struct {
				Type  string          `json:"type"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			} `json:"content"`
		}
		if err := json.Unmarshal(respBody, &r); err != nil {
			return apiEstimate{Reason: "decode: " + err.Error()}
		}
		for _, c := range r.Content {
			if c.Type == "tool_use" && c.Name == "hindcast_estimate" {
				var in struct {
					WallSeconds   float64 `json:"wall_seconds"`
					ActiveSeconds float64 `json:"active_seconds"`
				}
				if err := json.Unmarshal(c.Input, &in); err != nil {
					return apiEstimate{Reason: "tool input decode: " + err.Error()}
				}
				// Defensive validation: negative or absurd values mean
				// the model returned garbage despite the schema.
				if in.WallSeconds < 0 || in.ActiveSeconds < 0 || in.WallSeconds > 86400 {
					return apiEstimate{Reason: "out-of-range tool input"}
				}
				return apiEstimate{
					WallSeconds:   int(in.WallSeconds + 0.5),
					ActiveSeconds: int(in.ActiveSeconds + 0.5),
					Success:       true,
				}
			}
		}
		return apiEstimate{Reason: "no tool_use in response"}
	}
	return apiEstimate{Reason: "retries exhausted: " + lastReason}
}

func reportAPIResults(results []apiEvalResult) {
	var aErr, aSigned, bErr, bSigned []float64
	var aOK, bOK int
	for _, r := range results {
		if r.ArmA.Success && r.ArmA.WallSeconds > 0 && r.Sample.ActualWall > 0 {
			lr := math.Log(float64(r.ArmA.WallSeconds) / float64(r.Sample.ActualWall))
			aErr = append(aErr, math.Abs(lr))
			aSigned = append(aSigned, lr)
			aOK++
		}
		if r.ArmB.Success && r.ArmB.WallSeconds > 0 && r.Sample.ActualWall > 0 {
			lr := math.Log(float64(r.ArmB.WallSeconds) / float64(r.Sample.ActualWall))
			bErr = append(bErr, math.Abs(lr))
			bSigned = append(bSigned, lr)
			bOK++
		}
	}
	fmt.Printf("\nhindcast eval-api — Claude API A/B against %d historical turns\n\n", len(results))
	fmt.Printf("%-12s %6s %8s %10s %8s\n", "arm", "n", "MALR", "p90", "bias")
	fmt.Println("------------------------------------------------")
	printArm("control", aErr, aSigned)
	printArm("treatment", bErr, bSigned)

	if len(aErr) >= 3 && len(bErr) >= 3 {
		aSorted := append([]float64(nil), aErr...)
		bSorted := append([]float64(nil), bErr...)
		sort.Float64s(aSorted)
		sort.Float64s(bSorted)
		aMALR := math.Exp(quantile64(aSorted, 0.5))
		bMALR := math.Exp(quantile64(bSorted, 0.5))
		lift := aMALR / bMALR
		lo, hi := bootstrapLiftCI(aErr, bErr)
		fmt.Printf("\nLift (control/treatment) = %.2fx  [95%% CI %.2f–%.2f]", lift, lo, hi)
		switch {
		case lo > 1.0:
			fmt.Println("  → treatment reduces error with statistical confidence.")
		case hi < 1.0:
			fmt.Println("  → treatment INCREASES error with statistical confidence (priors hurting).")
		default:
			fmt.Println("  → CI crosses 1.0; need more samples for confidence.")
		}
	}
	fmt.Println()
	fmt.Println("MALR / p90 = error multipliers (lower = better; 1.00x = perfect).")
	fmt.Println("bias = exp(median signed log-ratio). >1 overestimates; <1 underestimates.")
}

func printArm(label string, errs, signed []float64) {
	if len(errs) == 0 {
		fmt.Printf("%-12s %6s %8s\n", label, "0", "—")
		return
	}
	sort.Float64s(errs)
	sort.Float64s(signed)
	malr := math.Exp(quantile64(errs, 0.5))
	p90 := math.Exp(quantile64(errs, 0.9))
	bias := math.Exp(quantile64(signed, 0.5))
	fmt.Printf("%-12s %6d %7.2fx %9.2fx %7.2fx\n", label, len(errs), malr, p90, bias)
}

// loadAPIKey checks the ANTHROPIC_API_KEY env var first, then walks up
// the directory tree from cwd looking for a .env file that defines it.
// Quoted values (single or double) have their quotes stripped so
// `ANTHROPIC_API_KEY="sk-..."` works the same as `ANTHROPIC_API_KEY=sk-...`.
func loadAPIKey() string {
	if v := os.Getenv("ANTHROPIC_API_KEY"); v != "" {
		return stripQuotes(v)
	}
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if v := readEnvFrom(filepath.Join(dir, ".env")); v != "" {
			return v
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

func readEnvFrom(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if v, ok := strings.CutPrefix(line, "ANTHROPIC_API_KEY="); ok {
			return stripQuotes(strings.TrimSpace(v))
		}
	}
	return ""
}

func stripQuotes(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}

// Silence unused imports on some platforms.
var _ iofs.DirEntry
var _ = filepath.Join
