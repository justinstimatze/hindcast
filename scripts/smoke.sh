#!/usr/bin/env bash
# Smoke test for hindcast install/uninstall cycle.
# Runs against a throwaway HOME so it doesn't touch the user's real CC
# config. Verifies settings.json mutations and clean removal.
set -euo pipefail

HINDCAST_HOME="$(mktemp -d)"
trap 'rm -rf "$HINDCAST_HOME"' EXIT

export HOME="$HINDCAST_HOME"
mkdir -p "$HOME/.claude/projects"

echo "→ building"
go build -o "$HOME/hindcast" ./cmd/hindcast

echo "→ install (no backfill)"
"$HOME/hindcast" install --no-backfill

echo "→ verify settings.json has all 4 registrations"
SETTINGS="$HOME/.claude/settings.json"
test -f "$SETTINGS" || { echo "FAIL: no settings.json"; exit 1; }
for sub in pending record inject mcp; do
  if ! grep -q "hindcast $sub" "$SETTINGS" 2>/dev/null && ! grep -q "\"hindcast\"" "$SETTINGS" 2>/dev/null; then
    # pending/record/inject appear as "hindcast <sub>"; mcp appears as mcpServers.hindcast
    case "$sub" in
      mcp) grep -q '"hindcast"' "$SETTINGS" || { echo "FAIL: missing $sub"; exit 1; } ;;
      *)   grep -q "hindcast $sub" "$SETTINGS" || { echo "FAIL: missing $sub"; exit 1; } ;;
    esac
  fi
done

echo "→ verify hindcast dir was created"
test -d "$HOME/.claude/hindcast" || { echo "FAIL: no hindcast dir"; exit 1; }

echo "→ verify salt file"
test -f "$HOME/.claude/hindcast/salt" || { echo "FAIL: no salt"; exit 1; }
SALT_SIZE=$(stat -c%s "$HOME/.claude/hindcast/salt" 2>/dev/null || stat -f%z "$HOME/.claude/hindcast/salt")
test "$SALT_SIZE" = "32" || { echo "FAIL: salt size $SALT_SIZE != 32"; exit 1; }

echo "→ verify seed loaded (global sketch exists)"
test -f "$HOME/.claude/hindcast/global-sketch.json" || { echo "FAIL: no sketch (seed didn't load)"; exit 1; }

echo "→ uninstall"
"$HOME/hindcast" uninstall

echo "→ verify clean removal"
test ! -d "$HOME/.claude/hindcast" || { echo "FAIL: hindcast dir not removed"; exit 1; }
if grep -q "hindcast" "$SETTINGS" 2>/dev/null; then
  echo "FAIL: settings.json still mentions hindcast"; exit 1
fi

echo "✓ smoke test passed"
