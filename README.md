# hindcast

> *"I departed London on the 2nd of October at 8:45 PM and returned on the 21st of December at 8:50 PM. Not 8:49. Not 8:51. I keep meticulous logs. You estimated an hour; it took four minutes. I find this — adjusts pocket watch — correctable."*
> — Phileas Fogg, allegedly

Claude Code's wall-clock estimates are not trustworthy for scheduling. The priors come from training — human engineers writing "refactor module = 2 days" — which is reasonable for a human and broken for an agent that finishes the same work in minutes. You can stop trusting the estimates entirely, or you can keep your own log and read it back.

hindcast is the log. Every turn's actual wall-clock and active-compute duration gets recorded. Before each turn runs, a local k-nearest-neighbors predictor matches the current prompt against past turns and renders a one-line estimate in your terminal's status bar — not in Claude's context, by deliberate design.

## Why not inject priors into Claude's context?

The earlier version of hindcast did exactly that — injected bucket p75 stats into Claude's context and told it to use the number. The mechanism worked by exploiting LLM anchoring ([Lou et al. 2024](https://arxiv.org/abs/2412.06593)): models obey injected numbers 20–60% of the time regardless of whether the number is well-chosen. When the retrieval is right, that's calibration. When the retrieval is wrong, it's confident error. For a personal tool whose classifier is coarse, the wrong-retrieval case is common enough to be disqualifying.

So hindcast v0.2 predicts numerically, locally, and surfaces the number to the **human**. Claude is never told. If you still want the anchoring mechanism, set `HINDCAST_LEGACY_INJECT=1`.

## Install

```sh
go install github.com/justinstimatze/hindcast/cmd/hindcast@latest
hindcast install
```

Two commands. Pure Go, no CGO, no network, ~5 MB binary. `install` merges three hooks + one MCP server + a `statusLine` entry into `~/.claude/settings.json` (timestamped backup), initializes a per-install salt, and backfills priors from your `~/.claude/projects/*/` history. Restart Claude Code and predictions render in the status line.

To remove: `hindcast uninstall`.

## Demo

```
> Refactor the fetcher to use the new Client type.

  [status line]  hindcast: ~3m wall / 45s active [1m–6m] · knn sim=0.42 n=5 · refactor

Claude: I'll trace the fetcher's callers, update signatures,
and run the test suite.

  [running…]

— actual: 4 minutes 12 seconds —
```

The status line tells **you** to expect about three minutes. Claude thinks whatever it thinks; hindcast doesn't interfere with Claude's own output. You use the prediction to decide whether to stay at the keyboard or start the next human task in parallel.

Fogg, *checking his pocket watch*: "Three minutes. Four minutes twelve. Satisfactory."

## The predictor

A BM25-weighted k-nearest-neighbors regressor over your own project's history. Given the current prompt's hashed tokens, it finds the top-k most similar past turns and takes a similarity-weighted median of their wall / active seconds. Tiers, in order:

1. **kNN** — at least 3 neighbors above similarity 0.15. Returns weighted median + p25/p75 band.
2. **task-type bucket** — records classified into the same task type (refactor / debug / feature / test / docs / other), n ≥ 4.
3. **overall project** — any record from the same project, n ≥ 4.
4. **global sketch** — cross-project bootstrap prior, n=1000.
5. **none** — status line says "no data yet."

Source is shown in the status line so you know what the number is grounded in.

### Measuring it — offline, in half a second

`hindcast verify` runs a prefix-based leave-one-out evaluation over all backfilled records: for each past turn it predicts using only earlier turns, compares to the actual duration, and reports MALR by source tier with a pass/fail verdict.

```
$ hindcast verify
hindcast verify: 4120 predictions across 18 project(s)

source         n   wall MALR  active MALR      bias
knn         3842       1.60x        1.60x     1.00x
bucket       262       2.45x        2.38x     1.10x
project       16       3.00x        6.18x     0.50x
global         0           -           -         -

VERDICT: PASS — kNN MALR 1.60x beats bucket MALR 2.45x (lift 1.53x ≥ 1.10x).
```

Exit code 0 on PASS, 1 on FAIL, 2 on UNDETERMINED (not enough data). Lift threshold is tunable with `--lift-min`. No API calls; uses your own history.

### Measuring it — live

`hindcast show --accuracy` reads the per-project reconciliation log (predicted vs actual on every completed turn since you installed v0.2) and reports MALR by prediction source.

```
$ hindcast show --accuracy
hindcast predictor accuracy: 142 turns across 3 project(s)

overall MALR: wall 1.84x   active 2.10x

source          n   wall MALR   active MALR        bias
knn            83      1.52x         1.73x       0.94x
bucket         41      2.10x         2.41x       1.06x
project        13      2.97x         3.15x       1.11x
global          5      4.20x         4.66x       1.40x
```

1.00× is a perfect prediction. The expected ordering is kNN < bucket < project < global — if yours is inverted, the predictor's stratification isn't earning its keep and you should tell me.

## Privacy, by construction

**No prompt text ever touches disk.** Records carry only numeric stats and deterministic-derived labels.

- `task_type` is classified in-memory at UserPromptSubmit via keyword regex; the prompt is then dropped.
- BM25 index stores salt-hashed 64-bit tokens, never plaintext. Per-install random 32-byte salt at `~/.claude/hindcast/salt`.
- Every log / index / salt file is 0600; every directory 0700.
- Zero network calls in the hook and predictor paths. The only network-using command is `hindcast eval-api` — a legacy A/B tool that evaluates the deprecated anchoring mechanism against the Claude API. Runs only when you invoke it.
- `HINDCAST_SKIP=1` opts a session out of recording entirely.
- `hindcast show` dumps everything stored. What you see is all there is.
- `hindcast rotate-salt` regenerates the salt if you believe it's been compromised; records are retained, BM25 indexes rebuild from next turns.

Caveat: salted tokens are reversible given the salt. The salt lives on the same filesystem as the index. "Privacy by construction" here means "no plaintext, ever" — not "cryptographically irreversible if the disk is compromised."

See [`SECURITY.md`](SECURITY.md) for the full threat model.

## Commands

```
hindcast install                Wire hooks + MCP + statusLine + seed + backfill. Run once.
hindcast uninstall              Reverse install. Drops data unless --keep-data.
hindcast predict [PROMPT]       One-shot kNN prediction for a prompt (CLI / stdin).
hindcast statusline             Render the active session's prediction. Wired into statusLine.
hindcast show [--project P]     Recorded records and bucket stats.
hindcast show --accuracy        Predictor MALR by source, from reconciliation log.
hindcast status                 Hook health: hook.log tail, panic count, current arm.
hindcast bench                  Offline prediction-accuracy ablation over backfilled data.
hindcast bench-cross [--corpus] Cross-corpus benchmark vs METR HCAST or OpenHands traces.
hindcast verify [--lift-min X]  Self-eval: LOO predict on your data, emit PASS/FAIL verdict.
hindcast tune                   Find your empirical sim cliff; persist to health.json.
hindcast show --health          Display tuned predictor state + sim-bucket MALR.
hindcast train [--warmup N]     Train per-user GBDT + linear regressor (offline tooling).
hindcast backfill [--rebuild]   Re-parse ~/.claude/projects/*/. Safe to re-run.
hindcast forget <project>       Delete one project's data.
hindcast rotate-salt            Regenerate BM25 salt; clears indexes (records kept).
hindcast calibrate              Legacy online A/B (evaluates deprecated anchoring path).
hindcast eval-api [--n 50]      Legacy offline A/B via Claude API (anchoring path).
hindcast export-seed            Dump global sketch as JSON (maintainer only).
```

## Configuration

hindcast is zero-config on the value path. These env vars exist for edge cases:

| var | default | effect |
|---|---|---|
| `HINDCAST_SKIP` | unset | skip recording for this session entirely |
| `HINDCAST_CONTROL_PCT` | 10 | percentage of sessions in A/B control arm (only relevant under legacy inject) |
| `HINDCAST_LEGACY_INJECT` | unset | re-enable the deprecated context-injection mechanism for A/B or nostalgia |

`.hindcast-project` in your project root overrides the default project name (useful for monorepos and symlinked trees).

## Does this work for me?

Two ways to find out without trusting the maintainer's numbers:

- `hindcast verify` — runs a leave-one-out evaluation over your own backfilled records. Tells you what the predictor's MALR is on YOUR data, broken down by source tier. Exit code 0 on PASS, 1 on FAIL.
- `hindcast bench-cross` — runs the same predictor against public agent traces (METR HCAST runs, OpenHands SWE-bench Lite). Tells you whether the architecture generalizes off the maintainer's data.

Honest cross-corpus findings (~6,000 predictions across two public corpora):

- **Architecture (per-project chronological history) generalizes.** Group-median MALR is comparable across corpora and matches what you'd see locally.
- **The BM25 prompt-similarity mechanism is use-case-dependent.** It works well when prompts are evolving multi-turn task work (typical Claude Code use). It works poorly on independent-task corpora like SWE-bench Lite (BM25 sim picks up library-name overlap, not task overlap).
- The single-threshold gate (`sim ≥ 0.5`) tuned on the maintainer's data does NOT generalize to all corpora. v0.3.1 ships `hindcast tune` (auto-runs from the Stop hook when `health.json` is stale): each install computes its own empirical cliff via prefix-LOO and persists it. View with `hindcast show --health`. The tuned threshold is computed but not yet wired into a text-injection gate; status line stays the universal default.

If your usage looks like Claude Code (multi-turn project work), expect kNN to earn its keep. If you use Claude Code for one-shot independent tasks, the bucket/project tier may be the ceiling.

v0.4 added a learned regressor (`hindcast train`) over universal features (prompt length, task type, recent project velocity, BM25 signals as features) — both GBDT and ridge linear, trainable on your local data and benchmarked cross-corpus. **Empirical finding: linear regressor wins on unique-instance corpora (OpenHands +17–23% over kNN/group_median); kNN wins on dense-repetition data (maintainer's own).** The regressor is shipped as offline tooling and is NOT wired into the live predict ladder — the case for displacing kNN on data with within-project repetition isn't there yet. The natural follow-up is per-user adaptive tier selection, which is not in this release.

## Known limitations

- **Windows is build-only** — hooks use `syscall.Setsid` and `/tmp` conventions that are Linux/macOS only. CI verifies the binary compiles on Windows; `install` refuses to run there.
- **Status line is terminal-only.** Web app and IDE-extension users won't see predictions; the status line is a Claude Code CLI feature.
- **NFS home directories** break the O_APPEND atomicity guarantee per-project JSONL writes rely on.
- **BM25 stopwords are English-only.** Non-English prompts get less filtering → slightly noisier index, slightly weaker kNN retrieval.
- **BM25 over salt-hashed tokens loses synonym signal.** "fix" and "repair" hash differently. Privacy/retrieval tradeoff is real and bounded.
- **Schema versioning is nominal** — `schema_version: 1` in records, no migration logic yet. Breaking changes in v0.3+ will require `hindcast backfill --rebuild`.
- **Cold start is thin.** Until you have ≥20 turns per project, the predictor leans on overall-project or global fallbacks, which are weak signals. Use hindcast for a week before judging it.

## Why "hindcast"

A meteorology term: running your forecast model against historical observations to calibrate it against reality. Same move here — your own past turn durations calibrate a predictor for the next one.

*Tap. Tap. Tap.*

## License

[MIT](LICENSE)
