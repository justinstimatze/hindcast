package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/justinstimatze/hindcast/internal/seed"
	"github.com/justinstimatze/hindcast/internal/store"
)

// Behavioral instructions live in the MCP server's initialize response
// (see cmd_mcp.go mcpInstructions) — install does NOT touch CLAUDE.md.
// Cleaner uninstall, instructions travel with the tool.

func cmdInstall(args []string) {
	fl := flag.NewFlagSet("install", flag.ExitOnError)
	skipBackfill := fl.Bool("no-backfill", false, "skip parsing ~/.claude/projects/*/")
	_ = fl.Parse(args)

	if runtime.GOOS == "windows" {
		fmt.Fprintln(os.Stderr, "hindcast install: Windows is build-only.")
		fmt.Fprintln(os.Stderr, "  Hooks require Unix syscalls (setsid, signal-0 liveness probe,")
		fmt.Fprintln(os.Stderr, "  /proc process-start-time). The binary compiles on Windows for")
		fmt.Fprintln(os.Stderr, "  development, but install is only supported on Linux and macOS.")
		os.Exit(1)
	}

	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast install: cannot find own path: %s\n", err)
		os.Exit(1)
	}
	if abs, err := filepath.Abs(exe); err == nil {
		exe = abs
	}

	fmt.Fprintf(os.Stderr, "hindcast install (binary: %s)\n", exe)

	if err := mergeClaudeSettings(exe); err != nil {
		fmt.Fprintf(os.Stderr, "  settings.json: %s\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "  settings.json: hooks + mcpServers merged")

	// Seed global sketch from embedded defaults if first-install.
	seedIfEmpty()

	// Materialize the salt during install so first-hook-on-first-run
	// never races on salt generation.
	if _, err := store.GetSalt(); err != nil {
		fmt.Fprintf(os.Stderr, "  salt: %s (non-fatal)\n", err)
	} else {
		fmt.Fprintln(os.Stderr, "  salt: initialized")
	}

	if !*skipBackfill {
		fmt.Fprintln(os.Stderr, "  backfill: running...")
		cmdBackfill(nil)
	}

	fmt.Fprintln(os.Stderr, "\nhindcast installed. Priors appear at your next Claude Code session.")
	fmt.Fprintln(os.Stderr, "Inspect: `hindcast show`    Health check: `hindcast status`")
}

func cmdUninstall(args []string) {
	fl := flag.NewFlagSet("uninstall", flag.ExitOnError)
	keepData := fl.Bool("keep-data", false, "keep recorded turns in ~/.claude/hindcast/")
	_ = fl.Parse(args)

	// Remove hindcast entries from settings.json (match by command path prefix).
	if err := unmergeClaudeSettings(); err != nil {
		fmt.Fprintf(os.Stderr, "  settings.json: %s\n", err)
	} else {
		fmt.Fprintln(os.Stderr, "  settings.json: hindcast hooks + mcpServers removed")
	}

	if !*keepData {
		if root, err := store.HindcastDir(); err == nil {
			if err := os.RemoveAll(root); err == nil {
				fmt.Fprintf(os.Stderr, "  data: %s removed\n", root)
			} else {
				fmt.Fprintf(os.Stderr, "  data: remove failed: %s\n", err)
			}
		}
	} else {
		fmt.Fprintln(os.Stderr, "  data: kept (use --keep-data=false to delete)")
	}

	fmt.Fprintln(os.Stderr, "\nhindcast uninstalled.")
}

func mergeClaudeSettings(exe string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	path := filepath.Join(home, ".claude", "settings.json")
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}

	// Preserve the existing file's mode rather than always writing 0644.
	// Settings often contains paths to other tools' commands and (in some
	// configurations) MCP env values; if the user wrote it 0600 we should
	// not downgrade their privacy posture on rewrite. Default is 0600 for
	// fresh-install (no existing file).
	var settingsMode os.FileMode = 0600

	var settings map[string]any
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if info, statErr := os.Stat(path); statErr == nil {
			settingsMode = info.Mode().Perm()
		}
		backupPath := path + ".hindcast-backup-" + time.Now().Format("20060102-150405")
		if err := os.WriteFile(backupPath, data, 0600); err != nil {
			return fmt.Errorf("backup: %w", err)
		}
		if err := json.Unmarshal(data, &settings); err != nil {
			return fmt.Errorf("existing settings.json is invalid JSON: %w", err)
		}
	case errors.Is(err, os.ErrNotExist):
		settings = map[string]any{}
	default:
		return err
	}
	if settings == nil {
		settings = map[string]any{}
	}

	hooks, _ := settings["hooks"].(map[string]any)
	if hooks == nil {
		hooks = map[string]any{}
	}
	addHook(hooks, "UserPromptSubmit", exe+" pending")
	addHook(hooks, "Stop", exe+" record")
	addHook(hooks, "SessionStart", exe+" inject")
	settings["hooks"] = hooks

	mcps, _ := settings["mcpServers"].(map[string]any)
	if mcps == nil {
		mcps = map[string]any{}
	}
	mcps["hindcast"] = map[string]any{
		"command": exe,
		"args":    []any{"mcp"},
		"env":     map[string]any{},
	}
	settings["mcpServers"] = mcps

	// v0.6 migration: pre-v0.6 installs wired `hindcast statusline` into
	// settings.statusLine. The statusline subcommand no longer exists, so
	// leaving the entry would break Claude Code's status bar with an
	// "unknown command" error on every prompt. Remove it if it's ours;
	// preserve any user-customized statusLine.
	if sl, ok := settings["statusLine"].(map[string]any); ok {
		if cmd, _ := sl["command"].(string); isLegacyHindcastStatusline(cmd) {
			delete(settings, "statusLine")
			fmt.Fprintln(os.Stderr, "  statusLine: removed legacy `hindcast statusline` entry (v0.6 dropped status line)")
		}
	}

	out, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(out, '\n'), settingsMode)
}

// addHook appends a hindcast hook entry unless one with the same command
// is already registered.
func addHook(hooks map[string]any, event, cmd string) {
	existing, _ := hooks[event].([]any)
	for _, entry := range existing {
		em, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		inner, _ := em["hooks"].([]any)
		for _, h := range inner {
			hm, ok := h.(map[string]any)
			if !ok {
				continue
			}
			if c, _ := hm["command"].(string); c == cmd {
				return
			}
		}
	}
	existing = append(existing, map[string]any{
		"matcher": "",
		"hooks": []any{
			map[string]any{"type": "command", "command": cmd},
		},
	})
	hooks[event] = existing
}

func unmergeClaudeSettings() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	path := filepath.Join(home, ".claude", "settings.json")
	var settingsMode os.FileMode = 0600
	if info, err := os.Stat(path); err == nil {
		settingsMode = info.Mode().Perm()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		return fmt.Errorf("settings.json is invalid JSON: %w", err)
	}

	hooks, _ := settings["hooks"].(map[string]any)
	if hooks != nil {
		for event, val := range hooks {
			list, ok := val.([]any)
			if !ok {
				continue
			}
			filtered := list[:0]
		entry:
			for _, entry := range list {
				em, ok := entry.(map[string]any)
				if !ok {
					filtered = append(filtered, entry)
					continue
				}
				inner, _ := em["hooks"].([]any)
				for _, h := range inner {
					hm, ok := h.(map[string]any)
					if !ok {
						continue
					}
					if c, _ := hm["command"].(string); isHindcastCmd(c) {
						continue entry
					}
				}
				filtered = append(filtered, entry)
			}
			if len(filtered) == 0 {
				delete(hooks, event)
			} else {
				hooks[event] = filtered
			}
		}
		if len(hooks) == 0 {
			delete(settings, "hooks")
		}
	}

	mcps, _ := settings["mcpServers"].(map[string]any)
	if mcps != nil {
		delete(mcps, "hindcast")
		if len(mcps) == 0 {
			delete(settings, "mcpServers")
		}
	}

	// Also clean up any legacy v0.5-and-earlier `hindcast statusline`
	// entry — the subcommand no longer exists.
	if sl, ok := settings["statusLine"].(map[string]any); ok {
		if cmd, _ := sl["command"].(string); isLegacyHindcastStatusline(cmd) {
			delete(settings, "statusLine")
		}
	}

	out, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(out, '\n'), settingsMode)
}

// seedIfEmpty loads the embedded priors into the global sketch if no
// sketch exists yet — gives new users meaningful priors from day 1
// instead of a 5-turn cold start. User's own data displaces the seed
// over time via the rolling window.
func seedIfEmpty() {
	s, err := store.LoadSketch()
	if err != nil {
		return
	}
	if len(s.Wall) > 0 {
		return
	}
	p := seed.Default()
	if len(p.Wall) == 0 && len(p.Active) == 0 {
		return
	}
	s.Wall = p.Wall
	s.Active = p.Active
	if s.MaxSize == 0 {
		s.MaxSize = store.SketchMaxWindow
	}
	if err := s.Save(); err == nil {
		fmt.Fprintf(os.Stderr, "  seed: loaded %d sample priors (bootstrap)\n", len(p.Wall))
	}
}

// isHindcastCmd returns true when a settings.json command string looks
// like one we registered. Matches any "... hindcast <subcommand>" shape.
func isHindcastCmd(cmd string) bool {
	fields := strings.Fields(cmd)
	if len(fields) < 2 {
		return false
	}
	base := filepath.Base(fields[0])
	return base == "hindcast" &&
		(fields[1] == "pending" || fields[1] == "record" ||
			fields[1] == "inject" || fields[1] == "mcp")
}

// isLegacyHindcastStatusline matches the pre-v0.6 status line entry so
// install / uninstall can clean it up. The statusline subcommand was
// removed in v0.6; an existing entry would error on every prompt.
func isLegacyHindcastStatusline(cmd string) bool {
	fields := strings.Fields(cmd)
	if len(fields) < 2 {
		return false
	}
	base := filepath.Base(fields[0])
	return base == "hindcast" && fields[1] == "statusline"
}
