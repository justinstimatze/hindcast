# What Didn't Work

Experiments that were tried, measured, and reverted or abandoned. Each entry includes what was tried, why it seemed promising, what actually happened, and why. Captures the dead ends so future contributors don't redo them.

_Numbers below are from specific experiment snapshots during development and may differ from the current combined results in `README.md` and `CHANGELOG.md`._

## Injection mechanism

### Inject bucket p75 directly into Claude's context (v0.1, 2026-04-22)
**Tried:** UserPromptSubmit hook prints a task-matched priors block — `feature bucket wall p75 = 3.2m`, plus an instruction "USE THAT NUMBER" — into Claude's session-start context. Claude was told to call `hindcast_estimate(wall_seconds=192)` from the priors block exactly.

**Expected:** Claude's anchoring tendency would lock onto the injected number and produce calibrated estimates.

**Result:** It worked when the bucket retrieval was right. When the bucket was wrong (small project, edge-case prompt), Claude confidently parroted a wildly off number. We could only measure lift on the maintainer's data and the lift was contingent on retrieval quality, which we couldn't prove generalizes.

**Why it was abandoned:** Per [Lou et al. 2024](https://arxiv.org/abs/2412.06593), LLMs obey injected numbers 20–60% of the time regardless of whether the number is well-chosen. That's calibration when retrieval is right and confident error when retrieval is wrong. For a personal tool whose classifier is coarse, the wrong-retrieval case dominated.

**Status:** Replaced in v0.2 by status-line-only (predictor surfaces to the human, never to Claude). Then re-introduced in v0.6 with anchoring guards (tier gate, variance gate, P10/P90 wider band on uncertain cases).

### Status line as the only surface (v0.2 – v0.5, 2026-04-23 – 2026-04-26)
**Tried:** Predictor writes a one-line `hindcast: ~3m wall / 45s active [1m–6m] · knn sim=0.42 n=5 · refactor` to a file the Claude Code statusLine reads. The human sees it; Claude doesn't.

**Expected:** No anchoring, calibrated number for the user, no harm to Claude's natural estimation behavior.

**Result:** Users (the maintainer) didn't read the status line. Claude continued to over-estimate by 5–10× on agent-shaped tasks. The dogfood signal was zero — the status line was infrastructure shipping to nobody.

**Why it was abandoned:** "I just want Claude to give me good estimates" — the human doesn't actually want to be the calibrator. The whole product has to flow into Claude's response.

**Status:** v0.6 ripped out the status line and re-introduced gated injection. The prediction code was kept; only the rendering layer flipped.

### Cmd_eval_api with legacy v0.1 inject (until v0.6.5, 2026-04-30)
**Tried:** The offline Claude API A/B harness used the v0.1 ungated bucket-table injection block to compare control vs treatment.

**Result:** Worked, but measured a mechanism we no longer ship. v0.6 production injection is the gated band, not the bucket table.

**Status:** v0.6.5 added an `-inject` flag (`v06` default, `legacy` available) so eval-api A/Bs the actually-shipped mechanism.

## Predictor mechanism

### Single-threshold global sim cliff (v0.2 – v0.3.1, 2026-04-23 – 2026-04-25)
**Tried:** kNN injection gated by hardcoded `sim ≥ 0.5`. Tuned on the maintainer's data where the cliff was clean.

**Expected:** The cliff would generalize; one threshold would work for everyone.

**Result:** Cross-corpus benchmark (v0.3) over METR HCAST + OpenHands SWE-bench Lite showed the cliff is not universal. Multi-turn project work has a cliff around 0.5; independent-task corpora don't have one at all.

**Why it failed:** The cliff is a property of the user's prompt distribution, not a constant of the BM25 retrieval algorithm.

**Status:** v0.3.1 replaced the constant with `hindcast tune` (per-user prefix-LOO empirical cliff persisted to `health.json`). v0.6.6 wired the tuned threshold into `predict.Predict` so it actually gates kNN injection per user.

### v0.4 universal-features regressor with `SizeBucket` (2026-04-26)
**Tried:** A GBDT/linear regressor trained on 21 prompt-and-context features including the post-turn `SizeBucket` (`small`/`medium`/`large` derived from `filesTouched + toolCount`).

**Expected:** Train/eval consistency and a lift over the kNN ladder.

**Result:** v0.4 train/bench numbers looked great (1.64× user GBDT MALR, 1.83× linear OpenHands MALR). Then v0.5 found the leak: `SizeBucket` is a post-turn observation not available at UserPromptSubmit, but training fed it as if it were. Production would have run with size always `{0,0,0}`.

**Why it failed:** Feature-leakage class. The model learned to depend on a feature that would always be empty in production.

**Status:** v0.5 dropped `SizeBucket` from `features.go` (21 → 18 features). Honest revised numbers: user GBDT 1.91×, user linear 2.24× (worse than the simpler kNN ladder on the maintainer's data, which is why the per-user adaptive selector keeps the regressor dormant unless it beats the ladder by ≥15% on held-out 50/50).

### Tightening empirical P25/P75 with alpha=0 (v0.6.4 first attempt)
**Tried:** Use the raw weighted-quantile P25/P75 directly in the inject's `(P25–P75: a–b)` annotation.

**Result:** Production point-rendered band hit rate landed at 31% instead of the 50% target. With only ~7 kNN neighbors, the empirical central-50% is systematically narrower than the true central-50%.

**Why it failed:** Small-sample quantile shrinkage toward the median. Standard statistical phenomenon; we re-derived it the hard way.

**Status:** v0.6.4 added `1 + alpha/sqrt(n)` shrinkage (alpha=0.5). At n=7 the bands widen by ~19%. Conservative; if production data still under-bands, alpha can tune up.

### Pending file as single per-session JSON (until v0.6.2, 2026-04-30)
**Tried:** `pending-<session>.json` overwritten on each UserPromptSubmit. One file per session bridges UserPromptSubmit → Stop.

**Result:** Rapid-fire conversational turns (typical multi-turn iterative work) overwrote the previous turn's pending before its Stop hook fired. By the time Stop read pending, `start_ts` was the *next* turn's submit time. The completed turn fell before the cutoff, got logged as `no turns since`, and was silently dropped. v0.6.1 ship snapshot showed 30+ "no turns since" entries and 0 "recorded:" across an evening of real conversation.

**Why it failed:** The single-pending-per-session design assumed one in-flight turn at a time. CC rapid-fire breaks that assumption.

**Status:** v0.6.2 keys the pending filename on `(session, start_ts_nanos)` and Stop matches the latest completed transcript turn against the closest pending by timestamp. Read-side glob includes the legacy single-file shape for upgrade compat.

## Architectural

### Auto-tune as inline goroutine inside Stop hook (until v0.6.6, 2026-05-01)
**Tried:** Spawn a goroutine inside `recordWorker` for the periodic tune+regressor refresh, since it's cheap (~500ms typical) and runs at most hourly.

**Result:** When tune runs longer (large corpus, hundreds of records per project, slow disk), it can outlive the worker's 30s timeout. Process exit kills the goroutine mid `m.Save()` rename. Stale `.tmp` files accumulate; in rare cases the regressor.gob and health.json end up out of sync.

**Why it failed:** Worker timeout doesn't bound spawned goroutines — process exit is a kill signal that arrives during gob rename.

**Status:** v0.6.6 detaches auto-tune to a `_autotune-worker` subprocess (same pattern `recordWorker` itself uses). The subprocess outlives the Stop hook's lifetime and finishes its writes independently.

### Synchronous auto-tune as the fix (v0.6.5, 2026-05-01)
**Tried:** Replace `go func() { ... }()` with `func() { ... }()` so the tune runs inside the worker timeout, theoretically bounded.

**Result:** Reduced overlap with concurrent Stop hooks but didn't eliminate the kill-mid-rename hazard — `recordWorker`'s `select`-on-timeout fires `return` from the worker, OS kills the process, mid-rename gob write dies anyway.

**Why it failed:** Misread of Go's timeout semantics. The select returns from the goroutine that ran doRecord; the binary's process kill is unrelated to whether the tune finished.

**Status:** Superseded by the v0.6.6 subprocess detach.
