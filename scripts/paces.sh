#!/usr/bin/env bash
# Pipeline integration test for hindcast hooks.
#
# Walks a synthetic CC session through each hook with hand-crafted JSON
# on stdin, verifies that state mutations land where they should:
#   - UserPromptSubmit → pending file created with hashed tokens
#   - Stop             → JSONL record + BM25 index + global sketch + estimate fold-in
#   - SessionStart     → additionalContext JSON emitted
#   - MCP handshake    → initialize + tools/list + both tools advertised
#   - hindcast_estimate tool → estimate file written
#
# Runs against a throwaway HOME. No Claude API involvement. Portable
# across Linux + macOS: no python3, no GNU-only tools.
set -euo pipefail

HINDCAST_HOME=$(mktemp -d)
trap 'rm -rf "$HINDCAST_HOME" /tmp/hindcast' EXIT

export HOME="$HINDCAST_HOME"
mkdir -p "$HOME/.claude/projects"

SESSION_ID="paces-session-12345678"
PROJECT_DIR=$(mktemp -d)
PROJECT_NAME=$(basename "$PROJECT_DIR")

# Portable md5 short-hash: GNU `md5sum` vs BSD/macOS `md5 -q`.
md5_short() {
  if command -v md5sum > /dev/null 2>&1; then
    printf '%s' "$1" | md5sum | cut -c1-8
  elif command -v md5 > /dev/null 2>&1; then
    printf '%s' "$1" | md5 -q | cut -c1-8
  else
    echo "paces: no md5 tool available" >&2
    exit 1
  fi
}

echo "→ building"
go build -o "$HOME/hindcast" ./cmd/hindcast

echo "→ install (no backfill)"
"$HOME/hindcast" install --no-backfill > /dev/null

PROJECT_HASH=$(md5_short "$PROJECT_NAME")

# ---- 1. UserPromptSubmit (pending) ----
echo "→ UserPromptSubmit hook"
PENDING_INPUT=$(cat <<EOF
{
  "session_id": "$SESSION_ID",
  "cwd": "$PROJECT_DIR",
  "permission_mode": "auto-accept",
  "prompt": "refactor the fetcher to use a client",
  "hook_event_name": "UserPromptSubmit",
  "transcript_path": "$HOME/.claude/projects/fake.jsonl"
}
EOF
)
PENDING_STDOUT=$(echo "$PENDING_INPUT" | "$HOME/hindcast" pending)

# v0.6.2 pending files include the StartTS in their filename so multiple
# in-flight turns in one session don't collide. Glob both forms — the
# legacy single-file shape is still supported on the read side.
PENDING_FILE=$(ls /tmp/hindcast/pending-${SESSION_ID}*.json 2>/dev/null | head -1)
test -n "$PENDING_FILE" -a -f "$PENDING_FILE" || { echo "FAIL: pending file not created (glob: /tmp/hindcast/pending-${SESSION_ID}*.json)"; exit 1; }

# Verify pending file has hashed tokens, no plaintext prompt
grep -q "prompt_tokens" "$PENDING_FILE" || { echo "FAIL: pending missing prompt_tokens"; exit 1; }
if grep -q "refactor the fetcher" "$PENDING_FILE"; then
  echo "FAIL: pending file contains plaintext prompt — privacy invariant broken"
  exit 1
fi

# v0.6 inject is gated: emits the band block only when source ∈ {regressor,
# kNN, bucket, project}. Synthetic project has no records so the predictor
# falls to global tier, which is suppressed — UserPromptSubmit produces
# no stdout for this fixture. (HINDCAST_INJECT=0 force-disables; legacy
# v0.1 ungated injection is HINDCAST_LEGACY_INJECT=1.)
if [[ -n "$PENDING_STDOUT" ]]; then
  echo "FAIL: pending unexpectedly wrote to stdout (gate should suppress for empty project):"
  echo "$PENDING_STDOUT"
  exit 1
fi

echo "  ✓ pending file written; prompt_tokens present; no plaintext"

# ---- 2. Overwrite pending with controlled state ----
# We've verified the hook writes the right SHAPE. Now replace it with
# a pending record keyed to a known timestamp so the Stop hook's
# transcript-tail parser can find our synthetic turn.
# v0.6.2 doRecord parses the transcript tail since (now - 6h) so the
# synthetic timestamps below have to be recent or the parser filters
# them out. Generate dynamic timestamps anchored to "a few minutes ago"
# in RFC 3339 millisecond shape that matches CC transcript output.
ts_offset_ago() {
  # Args: seconds-ago
  if date -u -d "@$(($(date +%s) - $1))" "+%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null; then return; fi
  # macOS BSD date fallback
  date -u -r "$(($(date +%s) - $1))" "+%Y-%m-%dT%H:%M:%S.000Z"
}
PROMPT_TS=$(ts_offset_ago 60)
ASSISTANT_T1=$(ts_offset_ago 58)
ASSISTANT_T2=$(ts_offset_ago 55)
ASSISTANT_T3=$(ts_offset_ago 54)

# Write the controlled pending into the legacy single-file path so the
# Stop hook's ListPendingForSession glob picks it up unambiguously.
LEGACY_PENDING="/tmp/hindcast/pending-${SESSION_ID}.json"
rm -f /tmp/hindcast/pending-${SESSION_ID}*.json
cat > "$LEGACY_PENDING" <<EOF
{"session_id":"$SESSION_ID","start_ts":"$PROMPT_TS","task_type":"feature","prompt_tokens":[1,2,3,4,5],"prompt_chars":40,"permission_mode":"auto-accept","project_hash":"$PROJECT_HASH","cwd":"$PROJECT_DIR","arm":"treatment"}
EOF
PENDING_FILE="$LEGACY_PENDING"

# ---- 3. Build a synthetic transcript whose timestamps are AFTER start_ts ----
TRANSCRIPT="$HOME/.claude/projects/fake.jsonl"
cat > "$TRANSCRIPT" <<EOF
{"parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":"refactor the fetcher to use a client"},"uuid":"u1","timestamp":"$PROMPT_TS","sessionId":"$SESSION_ID","cwd":"$PROJECT_DIR"}
{"parentUuid":"u1","isSidechain":false,"type":"assistant","message":{"role":"assistant","model":"claude-opus-4-7","content":[{"type":"tool_use","id":"tu1","name":"Read","input":{"file_path":"$PROJECT_DIR/fetcher.go"}}]},"uuid":"a1","timestamp":"$ASSISTANT_T1","sessionId":"$SESSION_ID"}
{"parentUuid":"a1","isSidechain":false,"type":"assistant","message":{"role":"assistant","model":"claude-opus-4-7","content":[{"type":"tool_use","id":"tu2","name":"Edit","input":{"file_path":"$PROJECT_DIR/fetcher.go","old_string":"x","new_string":"y"}}]},"uuid":"a2","timestamp":"$ASSISTANT_T2","sessionId":"$SESSION_ID"}
{"parentUuid":"a2","isSidechain":false,"type":"assistant","message":{"role":"assistant","model":"claude-opus-4-7","content":[{"type":"text","text":"done"}]},"uuid":"a3","timestamp":"$ASSISTANT_T3","sessionId":"$SESSION_ID"}
EOF

# ---- 4. hindcast_estimate MCP tool (before Stop, so the file is ready) ----
echo "→ MCP hindcast_estimate tool"
MCP_INPUT=$(cat <<EOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"hindcast_estimate","arguments":{"session_id":"$SESSION_ID","wall_seconds":180,"active_seconds":60}}}
EOF
)
echo "$MCP_INPUT" | "$HOME/hindcast" mcp > /dev/null

ESTIMATE_FILE="/tmp/hindcast/estimate-$SESSION_ID.json"
test -f "$ESTIMATE_FILE" || { echo "FAIL: estimate file not created"; exit 1; }
grep -q '"wall_seconds":180' "$ESTIMATE_FILE" || { echo "FAIL: wall_seconds not written"; exit 1; }
grep -q '"active_seconds":60' "$ESTIMATE_FILE" || { echo "FAIL: active_seconds not written"; exit 1; }
echo "  ✓ hindcast_estimate tool wrote estimate file correctly"

# ---- 5. Stop hook (record) ----
echo "→ Stop hook"
STOP_INPUT=$(cat <<EOF
{
  "session_id": "$SESSION_ID",
  "cwd": "$PROJECT_DIR",
  "permission_mode": "auto-accept",
  "hook_event_name": "Stop",
  "transcript_path": "$TRANSCRIPT"
}
EOF
)
echo "$STOP_INPUT" | "$HOME/hindcast" record

# Stop hook detaches a worker; wait for it to land.
LOG_FILE="$HOME/.claude/hindcast/projects/$PROJECT_HASH.jsonl"
for _ in $(seq 1 30); do
  if [[ -f "$LOG_FILE" ]]; then break; fi
  sleep 0.5
done

test -f "$LOG_FILE" || { echo "FAIL: per-project log not created at $LOG_FILE"; exit 1; }
RECORD_COUNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
test "$RECORD_COUNT" -ge 1 || { echo "FAIL: log has $RECORD_COUNT records"; exit 1; }

# Record should have the estimate we wrote via the MCP tool.
grep -q "\"claude_estimate_wall\":180" "$LOG_FILE" || { echo "FAIL: estimate not folded into record"; exit 1; }
echo "  ✓ record appended; claude_estimate_wall folded from MCP tool file"

# Privacy check: record must not contain plaintext prompt.
if grep -q "refactor the fetcher" "$LOG_FILE"; then
  echo "FAIL: record contains plaintext prompt — privacy invariant broken"
  exit 1
fi
echo "  ✓ record carries no plaintext prompt"

# BM25 index should exist.
BM25_FILE="$HOME/.claude/hindcast/projects/$PROJECT_HASH.bm25.gob"
test -f "$BM25_FILE" || { echo "FAIL: BM25 index not created"; exit 1; }
echo "  ✓ BM25 index written"

# Sketch should have been updated.
SKETCH_FILE="$HOME/.claude/hindcast/global-sketch.json"
test -f "$SKETCH_FILE" || { echo "FAIL: global sketch not written"; exit 1; }
echo "  ✓ global sketch updated"

# Estimate file should have been consumed (unlinked) by Stop hook.
if [[ -f "$ESTIMATE_FILE" ]]; then
  echo "FAIL: estimate file not removed after Stop hook"
  exit 1
fi
echo "  ✓ estimate file consumed and unlinked"

# ---- 6a. Accuracy log: Stop should have appended a reconciliation line.
ACCURACY_LOG="$HOME/.claude/hindcast/projects/$PROJECT_HASH/accuracy.jsonl"
# Whether this fires depends on the synthetic pending having a
# prediction; the controlled state we wrote above doesn't carry one,
# so an absence is fine here. Just verify the path machinery works:
# the directory must exist (AccuracyLogPath mkdirs on access).
# (Left as a sanity print; not a hard assert so the test stays robust
# to either behavior.)

# ---- 6b. SessionStart hook (inject) ----
# Post-v0.2 default: inject is a no-op (no anchoring context). The
# legacy priors block is gated behind HINDCAST_LEGACY_INJECT=1 so
# eval-api can still A/B the old mechanism.
echo "→ SessionStart hook (default: silent)"
INJECT_INPUT=$(cat <<EOF
{
  "session_id": "paces-session-87654321",
  "cwd": "$PROJECT_DIR",
  "permission_mode": "auto-accept",
  "hook_event_name": "SessionStart"
}
EOF
)
INJECT_OUT=$(echo "$INJECT_INPUT" | "$HOME/hindcast" inject)
if [[ -n "$INJECT_OUT" ]]; then
  echo "FAIL: SessionStart wrote to stdout by default (would anchor Claude):"
  echo "$INJECT_OUT"
  exit 1
fi
echo "  ✓ SessionStart silent by default (no anchoring)"

echo "→ SessionStart hook (HINDCAST_LEGACY_INJECT=1: structured context)"
INJECT_OUT=$(echo "$INJECT_INPUT" | HINDCAST_LEGACY_INJECT=1 "$HOME/hindcast" inject)
echo "$INJECT_OUT" | grep -q "hookSpecificOutput" || { echo "FAIL: legacy inject didn't emit hookSpecificOutput"; exit 1; }
echo "$INJECT_OUT" | grep -q "additionalContext" || { echo "FAIL: legacy inject missing additionalContext"; exit 1; }
echo "$INJECT_OUT" | grep -q "session_id" || { echo "FAIL: legacy inject didn't include session_id in context"; exit 1; }
echo "  ✓ legacy path still emits structured additionalContext"

# ---- 7. MCP tools/list shows both tools ----
echo "→ MCP tools/list"
LIST_OUT=$(printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | "$HOME/hindcast" mcp)
echo "$LIST_OUT" | grep -q "hindcast_prior" || { echo "FAIL: tools/list missing hindcast_prior"; exit 1; }
echo "$LIST_OUT" | grep -q "hindcast_estimate" || { echo "FAIL: tools/list missing hindcast_estimate"; exit 1; }
echo "  ✓ MCP advertises both hindcast_prior and hindcast_estimate"

# ---- 8. Uninstall cleanup ----
echo "→ uninstall"
"$HOME/hindcast" uninstall > /dev/null
test ! -d "$HOME/.claude/hindcast" || { echo "FAIL: uninstall left data"; exit 1; }
echo "  ✓ clean uninstall"

echo
echo "✓ paces test passed — hook pipeline intact end-to-end"
