# hindcast

[![CI](https://github.com/justinstimatze/hindcast/actions/workflows/ci.yml/badge.svg)](https://github.com/justinstimatze/hindcast/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/justinstimatze/hindcast)](https://goreportcard.com/report/github.com/justinstimatze/hindcast)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> *"I departed London on the 2nd of October at 8:45 PM and returned on the 21st of December at 8:50 PM. Not 8:49. Not 8:51. I keep meticulous logs. You estimated an hour; it took four minutes. I find this — adjusts pocket watch — correctable."*
> — Phileas Fogg, allegedly

Claude Code's wall-clock estimates are not trustworthy for scheduling. The priors come from training — human engineers writing "refactor module = 2 days" — which is reasonable for a human and broken for an agent that finishes the same work in minutes. You can stop trusting the estimates entirely, or you can keep your own log and feed it back to Claude before it answers.

hindcast is the log. Every turn's actual wall-clock and active-compute duration gets recorded. Before each turn runs, a local k-nearest-neighbors predictor matches the current prompt against past turns and injects a one-line calibrated prior into Claude's context — gated against anchoring failures.

## How the injection avoids anchoring

A naive context-injection ([Lou et al. 2024](https://arxiv.org/abs/2412.06593)) gets Claude to obey injected numbers 20–60% of the time regardless of whether the number is well-chosen. When retrieval is right, that's calibration. When retrieval is wrong, it's confident error. hindcast's injection is gated to suppress the wrong-retrieval cases:

- **Tier gate.** Fires only when the prediction comes from a calibrated source (regressor / kNN / task-type bucket / project median). The cross-project "global sketch" tier and the "no data" tier are silent — those are the cases where anchoring would dominate signal.
- **Variance gate.** If `WallP75 / WallP25 > 3`, the injection emits the band as the headline ("wall 1m–8m, high uncertainty") in place of a point estimate. A wide interval is honest where a precise-looking number would falsely anchor.
- **Citation form.** Claude is instructed to cite predictions as `~Xm wall (P25–P75: a–b, source n=N)`, surfacing provenance so you can spot when it's leaning on a thin sample.
- **Override discipline.** Claude is told to override when the prompt has a structural reason the predictor cannot see (much larger scope, blocked on long external process), not on perceived complexity — and not to pad the override out of caution.

Disable injection any time with `HINDCAST_INJECT=0`. Re-enable the v0.1 full-bucket-table injection (no gates) with `HINDCAST_LEGACY_INJECT=1`.

## Install

```sh
go install github.com/justinstimatze/hindcast/cmd/hindcast@latest
hindcast install
```

Two commands. Pure Go, no CGO, no network, ~5 MB binary. `install` merges three hooks + one MCP server into `~/.claude/settings.json` (timestamped backup), initializes a per-install salt, and backfills priors from your `~/.claude/projects/*/` history. Restart Claude Code and the next turn's prompt gets a calibrated prior in Claude's context.

To remove: `hindcast uninstall`.

## Demo

```
> Refactor the fetcher to use the new Client type.

  [hindcast prior, injected into Claude's context]
    wall ~3m (P25–P75: 1m–6m) · active ~45s
    source: knn sim=0.42 · n=5 · task=refactor

Claude: ~3m wall (P25–P75: 1m–6m, knn n=5).
        I'll trace the fetcher's callers, update signatures, and run the test suite.

  [running…]

— actual: 4 minutes 12 seconds —
```

Without the prior, Claude's first-shot estimate tends to inflate — typical case is half-hour-to-hour ranges for work that finishes in single-digit minutes. With the prior, Claude cites the predicted number and lands inside the band.

The mechanism (anchoring) is documented in [Lou et al. 2024]; the gates above bound the failure case but do not eliminate it. The size of the calibration lift on actual production turns is reported in the next section.

## Empirical lift

A/B against the Claude API on real-session historical prompts, claude-sonnet-4-6, control (stock) vs hindcast-v0.6 inject. Three independent n=30 samples (different seeds):

| seed | control MALR | treatment MALR | lift | 95% CI    |
|------|--------------|----------------|------|-----------|
|   42 |       16.3×  |          2.5×  | 6.5× | 2.6–17.6  |
|    7 |       10.6×  |          2.7×  | 4.0× | 1.4–8.2   |
| 2026 |       17.3×  |          1.7×  | 9.9× | 6.9–28.0  |

**Stock Claude over-estimates by 10–17× median; with hindcast, median error drops to 1.7–2.7×. Lift = 4–10× across seeds (median ≈6.5×). All three CIs are above 1.0 — treatment reduces error with statistical confidence on every sample.**

Stock's worst outliers run wild — a 9-second task can get an "18000-second" answer because training-data priors are calibrated to human engineers, not agents. Hindcast clamps the central tendency to ~2× of truth with mild over-estimate bias (1.3–1.6×) — much better than stock, deliberately under-confident relative to maintainer intuition (anchoring caution).

Reproduce with `hindcast eval-api -n 30 -seed 42`. Requires `ANTHROPIC_API_KEY`. Costs <$1 in API tokens per run. Compare modes with `-inject legacy` (v0.1 bucket-table) vs `-inject v06` (default; v0.6.x gated band).

## The predictor

A BM25-weighted k-nearest-neighbors regressor over your own project's history. Given the current prompt's hashed tokens, it finds the top-k most similar past turns and takes a similarity-weighted median of their wall / active seconds. Tiers, in order:

0. **regressor (optional)** — a per-user GBDT or ridge linear model over universal features, served first ONLY when `hindcast tune` measured it to beat the ladder below by ≥15% on your 50/50 held-out split. Otherwise dormant.
1. **kNN** — at least 3 neighbors above similarity 0.15. Returns weighted median + p25/p75 band.
2. **task-type bucket** — records classified into the same task type (refactor / debug / feature / test / docs / other), n ≥ 4.
3. **overall project** — any record from the same project, n ≥ 4.
4. **global sketch** — cross-project bootstrap prior, n=1000.
5. **none** — no data yet; injection is suppressed.

Source is shown in the injected prior so Claude (and you) can see what the number is grounded in.

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

`hindcast show --accuracy` reads the per-project reconciliation log (predicted vs actual on every completed turn since hindcast started recording) and reports MALR by prediction source plus band hit rate.

```
$ hindcast show --accuracy
hindcast predictor accuracy: 142 turns across 3 project(s) (140 usable for wall MALR)

overall MALR: wall 1.84x   active 2.10x

source          n   wall MALR   active MALR        bias
knn            83      1.52x         1.73x       0.94x
bucket         41      2.10x         2.41x       1.06x
project        13      2.97x         3.15x       1.11x
global          5      4.20x         4.66x       1.40x

band hit rate (actual ∈ rendered band): 47/83  (57%)
  point-rendered  [P25, P75]:  32/52  (62%)  ← inject showed a point; band is the implied 50% interval (target ≈50%)
  variance-gated  [P10, P90]:  15/31  (48%)  ← inject showed only the band (target ≈80% if P10/P90, ≈50% if P25/P75)
```

**Two metrics, two questions:**

- **MALR** measures the *point estimate* against actual. Useful for diagnosing systematic bias, but penalizes regression-to-the-mean on tail outliers (kNN takes a weighted-median of neighbors, so a real 10-second turn matched to neighbors with median 2 minutes will look 12× off — even when the prediction was honest about the central tendency). 1.00× is perfect.
- **Band hit rate** measures whether the actual fell inside the injected `[P25, P75]` band. This is the metric the inject actually targets — when the band is wide, the variance gate suppresses the point and Claude sees the band as the headline. So band hit rate is what matches what Claude saw. 50% is what a perfectly-calibrated quartile band predicts in expectation; consistent &gt; 50% means the band is wider than the truth distribution (under-confident); &lt; 50% means it's narrower (over-confident).

The expected MALR ordering is kNN &lt; bucket &lt; project &lt; global — if yours is inverted, the predictor's stratification isn't earning its keep and you should tell me. Band fields populate from v0.6.1 forward; older entries are reported as "not yet computable."

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
hindcast install                Wire hooks + MCP + seed + backfill. Run once.
hindcast uninstall              Reverse install. Drops data unless --keep-data.
hindcast predict [PROMPT]       One-shot kNN prediction for a prompt (CLI / stdin).
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
| `HINDCAST_SKIP` | unset | skip both recording and injection for this session entirely (UserPromptSubmit short-circuits before either) |
| `HINDCAST_CONTROL_PCT` | 10 | percentage of sessions in A/B control arm (only relevant under legacy inject) |
| `HINDCAST_INJECT` | unset (on) | set to `0` to suppress hook output to Claude (predictor still records turns) |
| `HINDCAST_LEGACY_INJECT` | unset | re-enable the v0.1 ungated full-bucket-table injection (mostly for A/B research) |
| `HINDCAST_FRESHNESS_HALFLIFE_DAYS` | `60` | half-life (days) for kNN recency weighting; `0` or negative disables and treats every record as freshness-neutral |

`.hindcast-project` in your project root overrides the default project name (useful for monorepos and symlinked trees).

## Does this work for me?

Two ways to find out without trusting the maintainer's numbers:

- `hindcast verify` — runs a leave-one-out evaluation over your own backfilled records. Tells you what the predictor's MALR is on YOUR data, broken down by source tier. Exit code 0 on PASS, 1 on FAIL.
- `hindcast bench-cross` — runs the same predictor against public agent traces (METR HCAST runs, OpenHands SWE-bench Lite). Tells you whether the architecture generalizes off the maintainer's data.

Honest cross-corpus findings (~6,000 predictions across two public corpora):

- **Architecture (per-project chronological history) generalizes.** Group-median MALR is comparable across corpora and matches what you'd see locally.
- **The BM25 prompt-similarity mechanism is use-case-dependent.** It works well when prompts are evolving multi-turn task work (typical Claude Code use). It works poorly on independent-task corpora like SWE-bench Lite (BM25 sim picks up library-name overlap, not task overlap).
- The single-threshold gate (`sim ≥ 0.5`) tuned on the maintainer's data does NOT generalize to all corpora. v0.3.1 ships `hindcast tune` (auto-runs from the Stop hook when `health.json` is stale): each install computes its own empirical cliff via prefix-LOO and persists it. View with `hindcast show --health`.

If your usage looks like Claude Code (multi-turn project work), expect kNN to earn its keep. If you use Claude Code for one-shot independent tasks, the bucket/project tier may be the ceiling.

v0.5 ships per-user adaptive tier selection. `hindcast tune` (and the Stop-hook auto-refresh) measures ladder/GBDT/linear on a chronological 50/50 held-out split of your own records and writes the winner to `~/.claude/hindcast/health.json`. If a regressor variant beats the existing ladder by ≥15% on your data, `predict.Predict` serves that regressor first; otherwise the kNN→bucket→project→global ladder runs unchanged. View the comparison with `hindcast show --health`.

The regressor is the v0.4 universal-features design (prompt length, task type, recent project velocity, BM25 signals), but with the post-turn `SizeBucket` feature removed in v0.5 (it leaked into training but was always empty at predict time, biasing v0.4 numbers optimistic). Honest cross-corpus finding stands: **linear regressor still wins on unique-instance OpenHands at 1.83× MALR (vs 2.14× group_median, 2.27× kNN)**. On the maintainer's own data, the ladder remains the winner (1.59× vs gbdt 1.72× / linear 1.70× held-out) — adaptive selection correctly stays dormant.

## Known limitations

- **Windows is build-only** — hooks use `syscall.Setsid` and `/tmp` conventions that are Linux/macOS only. CI verifies the binary compiles on Windows; `install` refuses to run there.
- **Anchoring is not eliminated, only bounded.** The variance and tier gates suppress the worst wrong-retrieval cases, but a confidently-wrong kNN match above the active sim floor with a tight band still anchors. `hindcast tune` measures a per-user empirical sim cliff and persists it to `health.json`; v0.6.6+ wires that tuned threshold into `predict.Predict` as the per-user kNN admission floor (defaulting to `knnMinSim = 0.15` when health is unset). When the verdict is "never inject," kNN is suppressed entirely and the prediction falls through to bucket / project / global tiers. The variance gate is the additional per-prediction backstop. If you find Claude consistently parroting bad numbers, set `HINDCAST_INJECT=0` per-shell or open an issue with the bad-prediction example.
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
