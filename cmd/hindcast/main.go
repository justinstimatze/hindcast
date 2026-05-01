// Command hindcast is the entry point for all hindcast subcommands:
// the Claude Code hooks (pending/record/inject), the MCP stdio server,
// and the user-facing CLI (install/uninstall/show/status/forget).
//
// All subcommands share a single Go binary so settings.json only has to
// point at one executable path.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/justinstimatze/hindcast/internal/hook"
)

const version = "0.6.4-dev"

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(1)
	}
	args := os.Args[1:]
	switch args[0] {
	// Hooks — invoked by Claude Code via settings.json; never run by hand.
	case "pending":
		hook.Guard("pending", cmdPending)
	case "record":
		hook.Guard("record", cmdRecord)
	case "inject":
		hook.Guard("inject", cmdInject)
	case "mcp":
		hook.Guard("mcp", cmdMCP)

	// User-facing CLI.
	case "install":
		cmdInstall(args[1:])
	case "backfill":
		cmdBackfill(args[1:])
	case "bench":
		cmdBench(args[1:])
	case "calibrate":
		cmdCalibrate(args[1:])
	case "export-seed":
		cmdExportSeed(args[1:])
	case "eval-api":
		cmdEvalAPI(args[1:])
	case "uninstall":
		cmdUninstall(args[1:])
	case "show":
		cmdShow(args[1:])
	case "status":
		cmdStatus(args[1:]...)
	case "forget":
		cmdForget(args[1:])
	case "rotate-salt":
		cmdRotateSalt(args[1:])
	case "predict":
		cmdPredict(args[1:])
	case "verify":
		cmdVerify(args[1:])
	case "bench-cross":
		cmdBenchCross(args[1:])
	case "tune":
		cmdTune(args[1:])
	case "train":
		cmdTrain(args[1:])

	case "version", "--version", "-v":
		fmt.Println("hindcast", version)
	case "help", "--help", "-h":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "hindcast: unknown command %q\n\n", args[0])
		usage(os.Stderr)
		os.Exit(1)
	}
}

func usage(w io.Writer) {
	fmt.Fprint(w, `hindcast — wall-clock estimation priors for Claude Code

Usage:
  hindcast install               Wire hooks + MCP into ~/.claude/settings.json,
                                 append CLAUDE.md snippet, backfill from transcripts.
  hindcast uninstall             Remove hook/MCP entries and all stored data.
  hindcast show [--project P]    Dump what hindcast has recorded.
  hindcast status                Hook health check (tail of hook.log).
  hindcast forget PROJECT        Delete a project's data, rebuild global sketch.
  hindcast rotate-salt           New BM25 salt; clears all indexes (records kept).
  hindcast version               Print version.
  hindcast backfill              Parse ~/.claude/projects/* into records (safe to re-run).
  hindcast bench                 Offline prediction-accuracy benchmark on backfilled data.
  hindcast bench-cross           Cross-corpus benchmark vs METR HCAST / OpenHands traces.
  hindcast verify                Self-eval (prefix-LOO) on your own backfilled data.
  hindcast tune                  Find your empirical sim cliff; write health.json.
  hindcast train                 Train per-user GBDT regressor on backfilled records.
  hindcast show --health         Display tuned predictor state.
  hindcast calibrate             Online A/B analysis (control vs treatment, legacy).
  hindcast predict [PROMPT]      One-shot kNN prediction for a prompt (CLI).

Hooks (invoked by Claude Code; not for manual use):
  hindcast pending               UserPromptSubmit — record turn start + BM25 delivery.
  hindcast record                Stop — record completed turn.
  hindcast inject                SessionStart — inject session-scoped priors.
  hindcast mcp                   MCP stdio server — exposes hindcast_prior tool.

See BOOTSTRAP.md for the full design spec.
`)
}

