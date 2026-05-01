// Package transcript parses Claude Code session transcripts
// (~/.claude/projects/*/*.jsonl) into per-turn records.
//
// Each JSONL line is one CC event: a user message, an assistant message,
// or a streamed assistant content block. We group events into turns
// bounded by top-level user prompts, accumulating tool-call counts,
// files touched, active-time gaps, and duration.
//
// Subagent events (isSidechain: true) are excluded — hindcast scores the
// main session only. Tool-result "user" messages are also excluded from
// turn boundaries; only new-prompt user messages start a turn.
package transcript

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"strings"
	"time"
	"unicode/utf8"
)

// Turn is one recorded UserPromptSubmit → Stop span. Claude's
// self-reported estimate is NOT parsed from response text — it's
// written by the hindcast_estimate MCP tool and read separately by
// the Stop hook.
type Turn struct {
	PromptTS      time.Time
	StopTS        time.Time
	PromptText    string // raw prompt — ephemeral, never persisted
	ResponseChars int
	ToolCalls     map[string]int
	FilesTouched  int
	Model         string
	ActiveSeconds int
}

// WallSeconds is a convenience derived from PromptTS and StopTS.
func (t Turn) WallSeconds() int {
	if t.PromptTS.IsZero() || t.StopTS.IsZero() || t.StopTS.Before(t.PromptTS) {
		return 0
	}
	return int(t.StopTS.Sub(t.PromptTS).Seconds())
}

// activeGapCap clamps any inter-tool gap above this to this value —
// approximates "Claude was idle / user was AFK or approving a prompt".
const activeGapCap = 60 * time.Second

type rawEntry struct {
	ParentUUID  *string         `json:"parentUuid"`
	IsSidechain bool            `json:"isSidechain"`
	Type        string          `json:"type"`
	Message     json.RawMessage `json:"message"`
	Timestamp   time.Time       `json:"timestamp"`
}

type rawMessage struct {
	Role    string          `json:"role"`
	Model   string          `json:"model"`
	Content json.RawMessage `json:"content"`
}

type contentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

// Parse reads a transcript from r and returns all turns whose PromptTS
// is >= since. Pass time.Time{} to get every turn.
func Parse(r io.Reader, since time.Time) ([]Turn, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 32*1024*1024)

	var turns []Turn
	var curr *Turn
	var lastTool time.Time
	filePaths := map[string]bool{}

	finalize := func() {
		if curr == nil {
			return
		}
		if !curr.PromptTS.Before(since) {
			curr.FilesTouched = len(filePaths)
			turns = append(turns, *curr)
		}
	}

	for sc.Scan() {
		var e rawEntry
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			continue
		}
		if e.IsSidechain {
			continue
		}
		var m rawMessage
		if err := json.Unmarshal(e.Message, &m); err != nil {
			continue
		}

		switch e.Type {
		case "user":
			if !isNewUserPrompt(m.Content) {
				continue
			}
			finalize()
			curr = &Turn{
				PromptTS:   e.Timestamp,
				PromptText: extractUserString(m.Content),
				ToolCalls:  map[string]int{},
			}
			filePaths = map[string]bool{}
			lastTool = time.Time{}

		case "assistant":
			if curr == nil {
				continue
			}
			curr.StopTS = e.Timestamp
			if m.Model != "" && curr.Model == "" {
				curr.Model = m.Model
			}
			blocks := extractBlocks(m.Content)
			for _, b := range blocks {
				switch b.Type {
				case "text":
					curr.ResponseChars += len(b.Text)
				case "tool_use":
					curr.ToolCalls[b.Name]++
					if !lastTool.IsZero() {
						gap := e.Timestamp.Sub(lastTool)
						if gap > activeGapCap {
							gap = activeGapCap
						}
						if gap > 0 {
							curr.ActiveSeconds += int(gap.Seconds())
						}
					}
					lastTool = e.Timestamp
					if p := extractFilePath(b.Input); p != "" {
						filePaths[p] = true
					}
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return turns, err
	}
	finalize()
	return turns, nil
}

// ParseFile is Parse over an os.Open'd path.
func ParseFile(path string, since time.Time) ([]Turn, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Parse(f, since)
}

// ParseTail reads only the last tailBytes of path and parses turns whose
// PromptTS is >= since. The tail-seek trades completeness for speed on
// large transcripts — a turn whose user message is before the tail
// window will be dropped. Caller uses `since` to narrow to their turn.
func ParseTail(path string, tailBytes int64, since time.Time) ([]Turn, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var start int64
	if stat.Size() > tailBytes {
		start = stat.Size() - tailBytes
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return nil, err
	}
	// If we're not at the start of the file, the first line is probably
	// partial — discard it.
	r := bufio.NewReader(f)
	if start > 0 {
		if _, err := r.ReadBytes('\n'); err != nil && err != io.EOF {
			return nil, err
		}
	}
	return Parse(r, since)
}

// isNewUserPrompt returns true when the content is a bare string (the
// shape CC emits for a fresh user prompt) as opposed to the array form
// it uses for tool_result injections.
func isNewUserPrompt(raw json.RawMessage) bool {
	trimmed := strings.TrimLeft(string(raw), " \t\r\n")
	return strings.HasPrefix(trimmed, `"`)
}

func extractUserString(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Array form fallback — concatenate any text blocks.
	blocks := extractBlocks(raw)
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == "text" {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

func extractBlocks(raw json.RawMessage) []contentBlock {
	// Try array form first.
	var arr []contentBlock
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr
	}
	// String form → single text block.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []contentBlock{{Type: "text", Text: s}}
	}
	return nil
}

// maxEffectiveContextChars bounds how much prior conversational text
// feeds into classification/BM25 at UserPromptSubmit. ~2500 chars is
// roughly 500 tokens or 2 conversational turns on average — enough to
// capture "what's the task" without diluting signal with drift.
const maxEffectiveContextChars = 2500

// ComposeEffectiveTask returns the current user prompt prefixed with
// the most recent ~maxEffectiveContextChars of prior conversational
// text (user prompts + assistant text blocks, in chronological order).
// This is the "unit of work" hindcast classifies — so "do it" after a
// long plan gets classified on the plan's content, not the literal
// two-word prompt.
//
// Budget-based rather than message-count-based: assistant turns can
// have 1-10 text blocks depending on tool interleaving, which makes
// a fixed-N-entries window inconsistent. A char budget scales with
// available content uniformly.
//
// If the transcript is missing or has no prior content, returns the
// userPrompt alone.
func ComposeEffectiveTask(userPrompt, transcriptPath string) string {
	prior := readRecentTextsBudget(transcriptPath, maxEffectiveContextChars)
	if prior == "" {
		return userPrompt
	}
	return prior + "\n\n" + userPrompt
}

// readRecentTextsBudget tail-reads the transcript and returns the most
// recent text content fitting within budget chars. Walks forward
// through the tail window collecting text-bearing entries, then
// trims from the front to fit the budget (most-recent content wins
// when we have to cut). Joined with "\n\n" between entries.
func readRecentTextsBudget(path string, budget int) string {
	if path == "" || budget <= 0 {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return ""
	}
	var start int64
	const tailBytes = 512 * 1024
	if stat.Size() > tailBytes {
		start = stat.Size() - tailBytes
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return ""
	}

	r := bufio.NewReader(f)
	if start > 0 {
		if _, err := r.ReadBytes('\n'); err != nil && err != io.EOF {
			return ""
		}
	}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)

	var texts []string
	for sc.Scan() {
		var e rawEntry
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			continue
		}
		if e.IsSidechain {
			continue
		}
		var m rawMessage
		if err := json.Unmarshal(e.Message, &m); err != nil {
			continue
		}
		switch e.Type {
		case "user":
			if isNewUserPrompt(m.Content) {
				if t := strings.TrimSpace(extractUserString(m.Content)); t != "" {
					texts = append(texts, t)
				}
			}
		case "assistant":
			blocks := extractBlocks(m.Content)
			for _, b := range blocks {
				if b.Type == "text" {
					if t := strings.TrimSpace(b.Text); t != "" {
						texts = append(texts, t)
					}
				}
			}
		}
	}

	// Walk backward from newest text, accumulating until we hit budget.
	// Most-recent content gets included first; older content drops off
	// when we run out of budget.
	remaining := budget
	var selected []string
	for i := len(texts) - 1; i >= 0; i-- {
		t := texts[i]
		if len(t) >= remaining {
			// Tail-cap: take the final `remaining` chars of this entry
			// so we capture the most recent content within the budget.
			// Align to a UTF-8 rune boundary so we don't produce
			// fragments like \x94 at the start (continuation bytes
			// from a sliced multi-byte character).
			cut := len(t) - remaining
			for cut < len(t) && !utf8.RuneStart(t[cut]) {
				cut++
			}
			t = t[cut:]
			selected = append([]string{t}, selected...)
			break
		}
		selected = append([]string{t}, selected...)
		remaining -= len(t) + 2 // account for "\n\n" joiner
		if remaining <= 0 {
			break
		}
	}
	return strings.Join(selected, "\n\n")
}

func extractFilePath(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return ""
	}
	if fp, ok := m["file_path"].(string); ok && fp != "" {
		return fp
	}
	if np, ok := m["notebook_path"].(string); ok && np != "" {
		return np
	}
	return ""
}
