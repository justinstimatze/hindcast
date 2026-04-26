package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/bm25"
	"github.com/justinstimatze/hindcast/internal/hook"
	"github.com/justinstimatze/hindcast/internal/seed"
	"github.com/justinstimatze/hindcast/internal/store"
	"github.com/justinstimatze/hindcast/internal/tags"
)

// MCP stdio server. Speaks JSON-RPC 2.0 on newline-delimited stdio per
// the Model Context Protocol spec. Exposes one tool — hindcast_prior —
// which takes a candidate prompt and returns duration priors (bucket
// stats + BM25 top match). Intended for plan-mode per-step estimation.

type mcpRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	ID      json.RawMessage `json:"id,omitempty"`
}

type mcpResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  any             `json:"result,omitempty"`
	Error   *mcpError       `json:"error,omitempty"`
	ID      json.RawMessage `json:"id"`
}

type mcpError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func cmdMCP() {
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	enc := json.NewEncoder(os.Stdout)

	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var req mcpRequest
		if err := json.Unmarshal(line, &req); err != nil {
			continue
		}
		handleMCP(enc, req)
	}
	if err := sc.Err(); err != nil {
		hook.Logf("mcp", "stdin scan: %s", err)
	}
}

func handleMCP(enc *json.Encoder, req mcpRequest) {
	isNotification := len(req.ID) == 0 || string(req.ID) == "null"

	switch req.Method {
	case "initialize":
		_ = enc.Encode(mcpResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities": map[string]any{
					"tools": map[string]any{},
				},
				"serverInfo": map[string]any{
					"name":    "hindcast",
					"version": version,
				},
				"instructions": mcpInstructions,
			},
		})

	case "notifications/initialized", "notifications/cancelled":
		// No response for notifications.

	case "tools/list":
		_ = enc.Encode(mcpResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"tools": []any{priorToolDefinition(), estimateToolDefinition()},
			},
		})

	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			_ = enc.Encode(errResp(req.ID, -32602, "invalid params"))
			return
		}
		switch p.Name {
		case "hindcast_prior":
			var args struct {
				Prompt    string `json:"prompt"`
				Project   string `json:"project"`
				SessionID string `json:"session_id"`
			}
			if err := json.Unmarshal(p.Arguments, &args); err != nil {
				_ = enc.Encode(errResp(req.ID, -32602, "invalid arguments"))
				return
			}
			// External MCP callers pass a single `prompt` arg; there's no
			// separate rolling-window retrieval context, so it serves as both.
			text := hindcastPrior(args.Prompt, args.Prompt, args.Project, args.SessionID)
			_ = enc.Encode(mcpResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result: map[string]any{
					"content": []any{map[string]any{"type": "text", "text": text}},
				},
			})
		case "hindcast_estimate":
			var args struct {
				SessionID     string  `json:"session_id"`
				WallSeconds   float64 `json:"wall_seconds"`
				ActiveSeconds float64 `json:"active_seconds"`
			}
			if err := json.Unmarshal(p.Arguments, &args); err != nil {
				_ = enc.Encode(errResp(req.ID, -32602, "invalid arguments"))
				return
			}
			text := hindcastEstimate(args.SessionID, int(args.WallSeconds+0.5), int(args.ActiveSeconds+0.5))
			_ = enc.Encode(mcpResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result: map[string]any{
					"content": []any{map[string]any{"type": "text", "text": text}},
				},
			})
		default:
			_ = enc.Encode(errResp(req.ID, -32601, "unknown tool: "+p.Name))
		}

	default:
		if !isNotification {
			_ = enc.Encode(errResp(req.ID, -32601, "method not found: "+req.Method))
		}
	}
}

// mcpInstructions is the behavioral contract delivered via the MCP
// server's initialize response. Replaces the earlier CLAUDE.md snippet
// so install stops touching ~/.claude/CLAUDE.md — cleaner install/
// uninstall, instructions travel with the tool.
const mcpInstructions = `hindcast records wall-clock and active-time
priors for CC sessions and injects them back as session context.

HOW TO ESTIMATE:

1. At UserPromptSubmit, you receive a task-matched priors block with a
   specific bucket wall p75 in seconds. USE THAT NUMBER. Do not adjust
   for perceived complexity — the bucket already encodes complexity.
   Do not reason from prompt wording like "thorough" or "quick" or
   "moderate". Just look up the bucket p75 and use it.

2. Call hindcast_estimate at the start of any response with a
   wall-clock implication:

     hindcast_estimate(
       session_id="<from the SessionStart priors block>",
       wall_seconds=<bucket wall p75>,
       active_seconds=<bucket active p75>,
     )

GOOD example:
  Prompt: "add retry logic to fetcher"
  Priors block: feature bucket wall p75 = 3.2m (192s), active p75 = 1m (60s)
  → hindcast_estimate(session_id="...", wall_seconds=192, active_seconds=60)

BAD example (anti-pattern):
  Prompt: "add retry logic to fetcher"
  Priors block: feature bucket wall p75 = 3.2m
  Your response: "This sounds moderately complex, probably 15-30 minutes..."
  (You overrode the priors with semantic intuition. Don't do this.)

FALLBACK ORDER when priors are sparse:
  - Use task-matched bucket p75 (ideal).
  - If bucket has n<3, use overall project wall p75.
  - If project has no data, use the global sketch median.

OTHER:
  - Control-arm sessions get no priors injection. Call hindcast_estimate
    anyway — that's the measurement floor for the A/B lift computation.
  - Call hindcast_prior inside plan generation to attach calibrated
    per-step wall-clock estimates.
  - "wall" is user-experienced duration (approvals + AFK); "active" is
    Claude's compute-only time.`

func priorToolDefinition() map[string]any {
	return map[string]any{
		"name":        "hindcast_prior",
		"description": "Look up wall-clock / active-time priors for a candidate prompt. Returns task type, complexity-matched bucket, BM25 top match, session momentum (if session_id provided), and a bias-corrected recommended hindcast_estimate value. Use inside plans when you want per-step duration estimates calibrated against real past data.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"prompt": map[string]any{
					"type":        "string",
					"description": "The candidate prompt text to estimate for.",
				},
				"project": map[string]any{
					"type":        "string",
					"description": "Optional: project name or path. Defaults to current working directory.",
				},
				"session_id": map[string]any{
					"type":        "string",
					"description": "Optional: current session id. If provided, the response includes recent-turn momentum from this session.",
				},
			},
			"required": []any{"prompt"},
		},
	}
}

func estimateToolDefinition() map[string]any {
	return map[string]any{
		"name":        "hindcast_estimate",
		"description": "Record your own wall-clock and active-time estimate for the current turn. Call this at the start of any response with a wall-clock implication, including in the control-arm sessions (control is the measurement floor, not a skip signal). Pass session_id from the SessionStart priors block.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"session_id": map[string]any{
					"type":        "string",
					"description": "Current session id (emitted in the SessionStart priors block).",
				},
				"wall_seconds": map[string]any{
					"type":        "number",
					"description": "Best-guess wall-clock seconds (includes user approvals and AFK time).",
				},
				"active_seconds": map[string]any{
					"type":        "number",
					"description": "Best-guess Claude-active seconds (just tool execution and reasoning).",
				},
			},
			"required": []any{"session_id", "wall_seconds", "active_seconds"},
		},
	}
}

// hindcastEstimate is the implementation of the hindcast_estimate MCP
// tool. Writes the estimate to /tmp/hindcast/estimate-<session_id>.json
// for the Stop hook to read and fold into the turn's Record.
func hindcastEstimate(sessionID string, wallSeconds, activeSeconds int) string {
	if sessionID == "" {
		return "hindcast_estimate: session_id is required (see the SessionStart priors block)"
	}
	path, err := store.EstimatePath(sessionID)
	if err != nil {
		return fmt.Sprintf("hindcast_estimate: %s", err)
	}
	payload := map[string]any{
		"wall_seconds":   wallSeconds,
		"active_seconds": activeSeconds,
		"ts":             time.Now().UTC().Format(time.RFC3339),
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Sprintf("hindcast_estimate: %s", err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Sprintf("hindcast_estimate: %s", err)
	}
	return fmt.Sprintf("recorded estimate: %ds wall / %ds active", wallSeconds, activeSeconds)
}

func errResp(id json.RawMessage, code int, msg string) mcpResponse {
	return mcpResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &mcpError{Code: code, Message: msg},
	}
}

// complexityOf returns a three-level complexity tag from the prompt's
// character length. Short prompts (< 200 chars) tend to be quick acks
// or questions; long prompts (>= 1000 chars) tend to be substantive
// task requests with embedded context. Used as an additional bucket
// stratification dimension beyond task_type × size_bucket.
func complexityOf(prompt string) string {
	n := len(prompt)
	switch {
	case n < 200:
		return "short"
	case n < 1000:
		return "medium"
	default:
		return "long"
	}
}

// hindcastPrior is the implementation of the hindcast_prior MCP tool.
// Returns a formatted text block suitable for Claude to read and reason
// about. Leads with a bias-corrected recommendation so Claude has a
// specific number to pass to hindcast_estimate without doing arithmetic.
//
// `prompt` is the user's actual request and drives classification /
// complexity bucketing. `retrievalText` is the wider rolling-window
// context used for BM25 retrieval; short prompts like "do it" benefit
// from the extra tokens without distorting task-type classification.
// External MCP callers pass the same string for both.
//
// No secrets: stats on the closest bucket, BM25 match, and the
// model-specific bias factor (from live records if available, shipped
// default otherwise). Also consults complexity-matched records and
// per-session momentum as supplementary predictors.
func hindcastPrior(prompt, retrievalText, project, sessionID string) string {
	if project == "" {
		if cwd, err := os.Getwd(); err == nil {
			project = cwd
		}
	}
	resolved := store.ResolveProject(project)
	hash := store.ProjectHash(resolved)

	taskType := string(tags.Classify(firstLine(prompt)))

	logPath, _ := store.ProjectLogPath(hash)
	records, _ := store.ReadRecentRecords(logPath, 500)

	// Infer current model from the most recent record with a model set.
	currentModel := "claude-sonnet-4-6"
	for _, r := range records {
		if r.Model != "" {
			currentModel = r.Model
			break
		}
	}

	// Live bias or shipped default.
	wallBias, activeBias, biasN := store.ComputeBiasFactor(records, currentModel)
	biasSource := fmt.Sprintf("live n=%d on %s", biasN, currentModel)
	if biasN < 10 {
		def := seed.BiasDefaultFor(currentModel)
		wallBias, activeBias = def.WallFactor, def.ActiveFactor
		biasSource = fmt.Sprintf("shipped default for %s (%s)", currentModel, def.Source)
	}
	if wallBias <= 0 {
		wallBias = 1.0
	}
	if activeBias <= 0 {
		activeBias = 1.0
	}

	// Primary: task_type match. Refine by complexity if we have enough
	// records in the (task_type × complexity) sub-bucket — this is a
	// finer stratification than task_type alone and can lower the
	// dispersion floor. Complexity is derived from prompt length, so
	// it works against existing records without schema migration.
	promptComplexity := complexityOf(prompt)
	var matching []store.Record
	var complexMatched []store.Record
	for _, r := range records {
		if r.TaskType == taskType {
			matching = append(matching, r)
			if complexityOf(strings.Repeat("x", r.PromptChars)) == promptComplexity {
				complexMatched = append(complexMatched, r)
			}
		}
	}
	usedComplexity := false
	if len(complexMatched) >= 4 {
		matching = complexMatched
		usedComplexity = true
	}

	var b strings.Builder
	fmt.Fprintf(&b, "hindcast_prior for task_type=%s (project=%s):\n", taskType, resolved)

	// Primary stats block — show med/p75/p90 so the recommendation math
	// is transparent.
	var (
		wallP75Sec, activeP75Sec int
		bucketLabel              string
		bucketN                  int
	)

	if len(matching) >= 4 {
		wm, wP75, wP90, am, aP75, aP90 := statsWide(matching)
		label := fmt.Sprintf("%s bucket", taskType)
		if usedComplexity {
			label = fmt.Sprintf("%s×%s bucket", taskType, promptComplexity)
		}
		fmt.Fprintf(&b, "  %s (n=%d): active med %.1fm / p75 %.1fm / p90 %.1fm  |  wall med %.1fm / p75 %.1fm / p90 %.1fm\n",
			label, len(matching),
			am/60, aP75/60, aP90/60,
			wm/60, wP75/60, wP90/60)
		wallP75Sec = int(wP75 + 0.5)
		activeP75Sec = int(aP75 + 0.5)
		bucketLabel = label
		bucketN = len(matching)
	} else if len(records) > 0 {
		wm, wP75, wP90, am, aP75, aP90 := statsWide(records)
		fmt.Fprintf(&b, "  %s bucket has only n=%d (below threshold); overall project (n=%d): active med %.1fm / p75 %.1fm / p90 %.1fm  |  wall med %.1fm / p75 %.1fm / p90 %.1fm\n",
			taskType, len(matching), len(records),
			am/60, aP75/60, aP90/60,
			wm/60, wP75/60, wP90/60)
		wallP75Sec = int(wP75 + 0.5)
		activeP75Sec = int(aP75 + 0.5)
		bucketLabel = "overall project"
		bucketN = len(records)
	} else if sk, err := store.LoadSketch(); err == nil {
		wm, wP75, wP90 := store.QuantilesWide(sk.Wall)
		am, aP75, aP90 := store.QuantilesWide(sk.Active)
		if len(sk.Wall) > 0 {
			fmt.Fprintf(&b, "  no project records yet; global fallback (n=%d): active med %.1fm / p75 %.1fm / p90 %.1fm  |  wall med %.1fm / p75 %.1fm / p90 %.1fm\n",
				len(sk.Wall),
				am/60, aP75/60, aP90/60,
				wm/60, wP75/60, wP90/60)
			wallP75Sec = int(wP75 + 0.5)
			activeP75Sec = int(aP75 + 0.5)
			bucketLabel = "global"
			bucketN = len(sk.Wall)
		} else {
			fmt.Fprint(&b, "  no records at all — collect a few turns first.\n")
		}
	}

	// BM25 close-match.
	if salt, err := store.GetSalt(); err == nil {
		if idx, err := bm25.Load(mustBM25Path(hash)); err == nil && len(idx.Docs) > 0 {
			matches := idx.TopK(bm25.HashTokens(retrievalText, salt), 1)
			if len(matches) > 0 && matches[0].Sim >= 0.10 {
				m := matches[0]
				bucket := m.Doc.TaskType
				if m.Doc.SizeBucket != "" {
					bucket = fmt.Sprintf("%s-%s", m.Doc.TaskType, m.Doc.SizeBucket)
				}
				fmt.Fprintf(&b, "  closest past (BM25 sim=%.2f): %s, %d tools, %d files — active %.1fm / wall %.1fm\n",
					m.Sim, bucket, m.Doc.ToolCount, m.Doc.FilesTouched,
					float64(m.Doc.ActiveSeconds)/60, float64(m.Doc.WallSeconds)/60)
			}
		}
	}

	// Session momentum — if prior turns exist in this session, surface
	// their median. Claude uses it as sanity-check on the bucket
	// recommendation (short-session recommendations land too high if
	// the bucket medians were drawn from a mixed-duration corpus).
	if sessionID != "" {
		if sm, err := store.LoadSessionMomentum(sessionID); err == nil && len(sm.WallTurns) >= 2 {
			wm := sm.WallMedian()
			am := sm.ActiveMedian()
			fmt.Fprintf(&b, "  session momentum (last %d turns): wall median %ds / active median %ds\n",
				len(sm.WallTurns), wm, am)
		}
	}

	// Bias-corrected recommendation — leads with the specific integers
	// to pass to hindcast_estimate, bypassing Claude-side arithmetic.
	if wallP75Sec > 0 {
		recommendedWall := int(float64(wallP75Sec)/wallBias + 0.5)
		recommendedActive := int(float64(activeP75Sec)/activeBias + 0.5)
		fmt.Fprintf(&b, "→ Recommended hindcast_estimate (bias-corrected, %s):\n", biasSource)
		fmt.Fprintf(&b, "   wall_seconds=%d  (= %s wall p75 %ds ÷ %.2f)\n",
			recommendedWall, bucketLabel, wallP75Sec, wallBias)
		if activeP75Sec > 0 {
			fmt.Fprintf(&b, "   active_seconds=%d  (= %s active p75 %ds ÷ %.2f)\n",
				recommendedActive, bucketLabel, activeP75Sec, activeBias)
		}
		_ = bucketN
	}

	return b.String()
}

func mustBM25Path(hash string) string {
	p, _ := store.ProjectBM25Path(hash)
	return p
}

// statsWide returns median/p75/p90 for both wall and active durations.
func statsWide(rs []store.Record) (wm, wP75, wP90, am, aP75, aP90 float64) {
	walls := make([]int, 0, len(rs))
	actives := make([]int, 0, len(rs))
	for _, r := range rs {
		walls = append(walls, r.WallSeconds)
		actives = append(actives, r.ClaudeActiveSeconds)
	}
	wm, wP75, wP90 = store.QuantilesWide(walls)
	am, aP75, aP90 = store.QuantilesWide(actives)
	return
}
