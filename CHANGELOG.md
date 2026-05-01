# Changelog

## [0.6.4] — unreleased — small-sample shrinkage on empirical quantiles

### Headline

The v0.6.3 production data window showed point-rendered band hit rate
at 31% (n=29) — well under the 50% target for a calibrated [P25, P75]
interval. Diagnosis: with only ~7 kNN neighbors, the empirical P25/P75
is a noisy estimate of the true central-50% interval, and small samples
systematically produce quantiles that are too compact (biased toward
the median).

The fix is small-sample shrinkage: widen each empirical quantile away
from the median by a factor `1 + alpha/sqrt(n)`, with `alpha=0.5`.

```
n=7   → factor ≈ 1.189   (widen by ~19%)
n=20  → factor ≈ 1.112   (widen by ~11%)
n=100 → factor ≈ 1.050   (widen by ~5%)
n=∞   → factor → 1.0     (no correction needed)
```

Applied to all four wall quantiles (P10, P25, P75, P90) and all four
active quantiles. Bucket / project / global tiers are unaffected — they
have higher minimum-n floors and use a separate code path.

### Why alpha=0.5

Conservative on purpose. The principled small-sample correction for the
quantile of a normal distribution is closer to alpha=1.0 (factor ≈ 1.38
at n=7), but production turn-duration distributions are usually heavier-
tailed than normal, so true bands are even wider. Starting at 0.5 as
the floor; will tune up if the v0.6.4 production data still underbands.

### What this targets

Point-rendered hit rate (currently 31%) should drift toward 50%; that's
the regime where the variance gate didn't trip and the inject committed
to a number with `(P25–P75: a–b)` as the implied confidence interval.
Variance-gated rate (currently 55%, mid-transition toward 80%) should
also bump up, but it was already on path.

### Tests

- `TestPredictKNNAppliesShrinkage` — corrected P25 ≤ raw P25, corrected
  P75 ≥ raw P75 (shrinkage widens, never tightens).
- `TestPredictKNNShrinkageScalesWithN` — ordered quantiles preserved
  across sample sizes.
- Existing `TestPredictKNNComputesWideQuantiles` still passes — relative
  ordering invariants survive the shrinkage.

## [0.6.3] — unreleased — wider bands, freshness weighting, dynamic variance gate

### Headline

Three independent moves to handle the project-and-day-by-day calibration
drift surfaced in v0.6.2's first real data window. v0.6.2 unstuck
recording; v0.6.3 makes the inject *useful* across drifting projects
without over-fitting to one weird week.

### A — Wider quantile band when uncertain

The kNN tier now computes `WallP10/WallP90` (and Active counterparts)
alongside `WallP25/WallP75`. When the variance gate trips, the inject
renders the wider `[P10, P90]` band as the headline:

```
wall 30s–8m (high uncertainty, no point estimate; P10–P90 band)
```

A calibrated P10/P90 band hits 80% of actuals by definition; a
calibrated P25/P75 band hits 50%. Rendering the wider band when the
predictor is already uncertain means the surface to Claude is honest
about the spread without committing to a precise-looking point.

The accuracy log captures both quartile pairs; `hindcast show
--accuracy` now reports band hit rate against the *rendered* band
(P10/P90 for variance-gated, P25/P75 for point), with target hit
rates labeled in the output.

### B — Freshness weighting in kNN

`bm25.Doc.TS` is now populated when a record enters the index (live
record + happy-path Stop + fallback Stop + backfill). `predict.Predict`
multiplies each kNN match's BM25 sim by an exponential recency factor
with a 60-day half-life:

  `weight = sim × 2^(-age_days / halfLife)`

Recent turns weighted ~2x heavier than two-month-old turns; week-old
turns at ~92%. Conservative on purpose: enough to catch genuine project
drift without swinging on one anomalous week. Override via
`HINDCAST_FRESHNESS_HALFLIFE_DAYS` env var; ≤0 disables.

Pre-v0.6.3 records have zero TS and weight 1.0 (neutral) — existing
indexes don't degrade until they refresh.

### C — Dynamic variance gate threshold

The variance gate threshold (P75/P25 ratio) was a hard 3.0. v0.6.3
makes it dynamic: kNN matches with small samples (n<5) or low BM25
similarity (max_sim<0.4) drop the threshold to 2.0. The intuition:
when the underlying signal is itself uncertain (thin neighbors, weak
match), the inject should surface the band more readily rather than
commit to a point estimate the predictor can't really back.

Bucket / project / regressor tiers keep the base 3.0 threshold —
those tiers don't have a per-prediction confidence signal to act on.

### Why all three together

- **A alone** widens the band on already-uncertain cases — useful when
  the kNN's neighbor distribution does span reality but P25/P75 cut it
  off.
- **B alone** shifts the kNN's median toward recent reality, useful
  when a project's distribution has drifted (psychosis-ismyaialive
  case from the v0.6.2 data: kNN was matching against past short
  turns, missing that recent turns are 3-5× longer).
- **C alone** trips the variance gate more often on weak-signal
  predictions, which then renders A's wider band.

A + B + C compose: B fixes the *center* of the distribution under
drift, A widens *what gets shown* when uncertainty is high, C decides
*when uncertainty is high*. None of the three tunes a feature
weight against feedback — there's no model to overfit, just rendering
choices over the existing weighted-quantile data.

### Tests

- `internal/predict/predict_test.go:TestPredictKNNComputesWideQuantiles`
  — kNN populates ordered P10<P25≤median≤P75<P90.
- `TestRecencyWeight` — recencyWeight returns 1.0 for zero TS, exp
  decay for non-zero, ≥1.0 floor for clock-skew safety.
- `TestPredictKNNFreshnessShiftsMedianTowardRecent` — mixed-age
  neighbors: half from 6 months ago short, half today long → median
  pulled toward recent values.
- `cmd/hindcast/cmd_pending_test.go:TestFormatClaudeInjectionDynamicVarianceThreshold`
  — large-n + ratio 2.5 stays point; small-n + same ratio trips
  (lowered threshold); low-sim same; non-kNN unaffected.
- `TestFormatClaudeInjectionRendersWidePercentilesWhenAvailable` —
  variance gate uses P10/P90 when populated, falls back to P25/P75.

### Schema

`accuracy.jsonl` entries optionally include `predicted_wall_p10`,
`predicted_wall_p90`. `bm25.Doc` has a new `TS` field (gob-encoded;
older indexes default to zero TS, which means freshness-neutral).
Forward-compat: pre-v0.6.3 readers ignore the new fields.

## [0.6.2] — unreleased — pending-per-turn (fix rapid-fire turn drops)

### Headline

Pending file is now keyed on `(session, start_ts_nanos)` instead of
session alone. The single-pending-per-session design (`pending-<id>.json`)
was being overwritten when a follow-up prompt arrived within milliseconds
of the prior assistant finishing — by the time Stop fired for turn N, the
pending had already been clobbered by turn N+1's UserPromptSubmit, so
Stop's `since`-cutoff was set to a moment after the just-completed turn
and silently logged "no turns since" instead of recording.

This is exactly the failure mode that surfaced during rapid-fire
multi-session use the night of v0.6.1: 30+ "no turns since" entries and
zero "recorded:" entries across an evening of real conversation.

### Fix

- `store.PendingPath(sessionID, startTS)` now embeds the start timestamp
  in the filename: `pending-<sessionID>-<unixNanos>.json`. Zero startTS
  returns the legacy single-file shape so old in-flight pendings are
  still readable post-upgrade.
- `store.ListPendingForSession(sessionID)` globs both legacy and new
  filename shapes, parses each, returns `[]PendingFile` sorted oldest-
  first by `StartTS`. Corrupt files are skipped silently and cleaned by
  the TTL sweeper.
- `cmd_record.go doRecord` no longer reads a single pending. It finds
  the latest completed turn in the transcript, then matches that turn
  to the pending whose StartTS is closest to (and within ±10s of) the
  turn's PromptTS. Unmatched pendings stay in place — they correspond
  to canceled/interrupted prompts and get swept by the existing 6h TTL.
- When no pending matches the latest turn (e.g., upgrade-time orphaned
  state, missing UserPromptSubmit), doRecord falls through to the same
  transcript-only fallback path used for shadowed-hook projects. That
  path records without writing an accuracy.jsonl entry, since there's
  no prediction to compare against.

### Compat

Sessions started under a pre-v0.6.2 binary continue to write to the old
single-file pending path until they exit. Post-v0.6.2 Stop reads both
patterns, so no in-flight data is lost across upgrade.

### Tests

- `internal/store/store_test.go:TestPendingPathTimestamped` —
  PendingPath returns distinct paths for distinct startTS values; zero
  startTS returns the legacy shape.
- `TestListPendingForSessionSorts` — multi-pending session returns
  oldest-first.
- `TestListPendingForSessionPicksUpLegacy` — glob captures both pre-
  v0.6.2 and v0.6.2 pending file shapes.

## [0.6.1] — unreleased — accuracy log captures band; band-hit-rate metric

### Headline

`accuracy.jsonl` now records the predicted P25/P75 band, max BM25 sim, and
a `variance_gated` flag alongside the point estimate. `hindcast show
--accuracy` reports band hit rate (actual ∈ [P25, P75]) split by inject
mode (point-rendered vs variance-gated). This is the metric that targets
what Claude actually saw — point MALR penalizes regression-to-the-mean on
tail outliers, but the inject suppresses the point behind the variance
gate when the band is wide.

### Why this matters

Live point-MALR was 4.45× in the v0.6.0 ship snapshot, while `hindcast
verify` (prefix-LOO synthetic over backfilled records) reported 1.61×.
Decomposing by actual-duration bucket showed the gap is structural: kNN
regresses tail outliers (very-short or very-long real turns) toward the
central tendency, so a 10-second real turn matched to neighbors with
median 2 minutes scores as a 12× miss even though the prediction was
honest about the central tendency.

The variance gate suppresses point estimates in those cases (the inject
emits the band as the headline instead). But accuracy.jsonl was only
logging the point — measuring something the inject doesn't actually
surface to Claude. v0.6.1 captures the band and computes a hit-rate
metric that matches what Claude saw.

### Schema

`accuracy.jsonl` entries now optionally include:

```json
{
  "predicted_wall_p25": 30,
  "predicted_wall_p75": 90,
  "predicted_max_sim": 0.71,
  "variance_gated": true,
  ...
}
```

All four are `omitempty`, so older readers stay tolerant. Pre-v0.6.1
entries are excluded from band-hit-rate computation; the report shows
"not yet computable" until new entries accumulate.

### Tier-aware variance flag

`variance_gated` now means *the inject was rendered AND showed the band
as headline* — not just *the predictor produced a wide band*. Tiers the
inject doesn't surface (global / none) always log
`variance_gated=false` regardless of band width, because Claude never
saw them. Without this, a global-tier wide-band entry would be counted
toward "variance-gated band hit rate," measuring something other than
what was actually rendered. Caught immediately after v0.6.1's first
production entry surfaced a global-source row with `variance_gated=true`.

### Tests

- `internal/store/store_test.go:TestPendingBandFieldsRoundtrip` —
  PendingTurn round-trips the new fields through Write/Read.
- `cmd/hindcast/cmd_admin_test.go:TestWithBandRows` — filter excludes
  pre-v0.6.1 entries AND non-injecting tiers (global / none).
- `cmd/hindcast/cmd_admin_test.go:TestBandHitRateComputation` — boundary
  cases (inclusive bracket), point-vs-variance-gated split, all from
  inject-eligible source tiers.

## [0.6.0] — unreleased — gated context injection + Stop-hook fallback + status line removed

### Headline

The status line is gone. `hindcast pending` now emits a calibrated prior
into Claude's UserPromptSubmit additionalContext channel — the v0.1
inject-into-Claude path, but with anchoring guards that didn't exist
before. Claude reads the predicted band and is told to cite it in a
specific form on any wall-clock estimate.

### Anchoring guards

Two gates trip when retrieval is likely wrong:

- **Tier gate.** Injection fires only for `regressor / kNN / bucket /
  project` sources. The cross-project `global` sketch and `none` tier
  are silent — global p50 is biased short on new projects (sketch is
  dominated by maintainer/test sessions), and an anchor would flip the
  failure mode from "wildly over" to "wildly under" with the same
  magnitude error and the opposite sign.
- **Variance gate.** When `WallP75 / WallP25 > 3.0`, the injection
  emits the band as the headline ("wall 1m–8m, high uncertainty") in
  place of a point estimate. A wide interval is honest where a precise-
  looking number would falsely anchor.

The injection text additionally instructs Claude to cite predictions in
`~Xm wall (P25–P75: a–b, source n=N)` form, override only on structural
scope mismatch (and not on perceived complexity), and **not pad the
override out of caution** — the over-estimation bias hindcast is meant
to fix.

`HINDCAST_INJECT=0` disables; `HINDCAST_LEGACY_INJECT=1` re-enables the
v0.1 ungated full-bucket-table injection for A/B research.

### Stop-hook fallback for shadowed UserPromptSubmit

When a project's local `.claude/settings.json` defines its own
UserPromptSubmit hooks, Claude Code merges by **replacement**, not
union — a project-local UserPromptSubmit list shadows the global
`hindcast pending` hook entirely. Prior to v0.6, this resulted in
silent zero-record projects: Stop fired but found no pending file and
gave up.

v0.6 adds `doRecordFallback` in `cmd_record.go`. When Stop sees no
pending file, it parses the transcript tail directly, hashes the prompt
in-memory, and reconstructs the same Record the happy path would have
produced. A per-session `fallback-marker` file tracks the last
recorded `PromptTS` so re-fires don't double-count.

No accuracy log entry is written from the fallback path — there is no
prediction to compare against because `pending` never ran.

### Status line removed

`hindcast statusline` (subcommand) and `cmdStatusline` (binary) are
deleted along with the supporting `writeLastPrediction` and
`CurrentSessionPointerPath` paths. The store path builder
`LastPredictionPath` was renamed `SessionDirPath` (it was always a
directory; the old name was a vestige of the deleted prediction file).

`hindcast install` migrates pre-v0.6 settings.json by removing any
`statusLine` entry that points at `hindcast statusline`. User-customized
statusLine commands are preserved. `hindcast uninstall` mirrors the
cleanup so cruft from any version is removed on uninstall.

### Privacy invariant preserved

`formatClaudeInjection` reads only numeric `Prediction` fields. The
fallback hashes prompts before any disk write. Two new regression tests
enforce this:

- `cmd/hindcast/cmd_pending_test.go` — table-driven tier gate, variance
  gate, active-zero suppression, don't-pad guidance presence,
  per-source label rendering.
- `cmd/hindcast/cmd_record_test.go` — fallback writes no plaintext
  (sentinel-string sweep over `~/.claude/hindcast/`), and
  `fallback-marker` advances correctly to suppress duplicate recording.
- `cmd/hindcast/cmd_install_test.go` — legacy statusLine migration is
  applied to ours and not to user-customized entries.

The existing `internal/store/store_test.go:TestRecordContainsNoPromptText`
continues to pass.

### Migration

Existing users upgrading from v0.5 should run `hindcast install` (not
just `go install`). The first install merges the legacy statusLine
cleanup into settings.json. Without it, Claude Code's status bar
displays "unknown command: statusline" on every prompt.

## [0.5.0] — unreleased — per-user adaptive tier selection + feature-leak fix

### Headline

`hindcast tune` (and Stop-hook auto-refresh) now runs a chronological 50/50
held-out comparison of {ladder, gbdt, linear} on the user's data and writes
the winner to `health.json`. When a regressor variant beats the kNN-led
ladder by ≥15%, `predict.Predict` serves that regressor first. Otherwise
the existing ladder is unchanged. The maintainer's data: ladder still
wins (1.59× vs gbdt 1.72× / linear 1.70× held-out 50/50).

### Bug fix: feature leakage

v0.4 trained the regressor with `SizeBucket` (one-hot small/medium/large)
as a feature. SizeBucket is `sizes.Classify(filesTouched, toolCount)` —
both **post-turn** observations not available at UserPromptSubmit time.
Trained models therefore learned to depend on a feature that would always
be empty at production. The earlier v0.4 train/bench numbers were
optimistically biased because eval ran with size populated from completed
records, while production would have run with size always {0,0,0}.

Fix: dropped `SizeBucket` from `features.go` (21 features → 18). All
features now satisfy the package contract: derivable at UserPromptSubmit
time. Training/eval/predict now share one distribution.

**Honest revised numbers** (compare to v0.4 entry below):

| dataset            | metric          | v0.4 (with size) | v0.5 (honest) |
|--------------------|-----------------|------------------|---------------|
| user 50/50 GBDT    | held-out MALR   | 1.64×            | **1.91×**     |
| user 50/50 linear  | held-out MALR   | 1.83×            | **2.24×**     |
| METR linear        | held-out MALR   | 1.17×            | 1.17×         |
| OpenHands linear   | held-out MALR   | 1.83×            | **1.83×**     |

Cross-corpus numbers were nearly unchanged — those corpora have stable
within-group sizes so the feature carried little marginal information.
User data was where the leak mattered, because per-project task-size
distribution is heterogeneous and the model was free-riding on it.

### Added

- `health.Compute` now runs a 50/50 chronological eval section alongside
  prefix-LOO. Records ladder/gbdt/linear MALR and a `RegressorWinner`
  (`"gbdt" | "linear" | "none"`) in `health.json`.
- `predict.SourceRegressor` source name; `cmd_pending.computePrediction`
  short-circuits to the persisted winner model when health says so. Soft
  falls through to the ladder on any IO/decode error.
- `internal/regressor`: `GBDTModelPath` + `LinearModelPath` + symmetric
  `Save`/`LoadLinear`. Models persist as siblings — predict reads only
  the named winner. `regressor.MakeContext` exposes the BM25-feature
  pipeline so health and the predict path can't drift from training.
- `hindcast show --health` displays the 50/50 numbers and the winner.
- Stop-hook auto-tune additionally retrains both models on every refresh,
  keeping the on-disk gobs current with `health.json`.
- Regressor bands: train-set residual P25/P75 (log-space) persisted on
  each model and rendered as `[low–high]` on the status line. In-sample
  residuals understate held-out spread; treat as a "better than nothing"
  floor, not a calibrated interval.
- Status line distinguishes regressor variants: `regressor:linear`/
  `regressor:gbdt` instead of bare `regressor`. New `SourceDetail` field
  on `predict.Prediction` carries the variant name.
- Tests added for `internal/predict` (ladder behavior),
  `internal/regressor` (feature/Extract length parity, train + save/load
  roundtrip, insufficient-data sentinel), and `internal/health` (empty
  + flat-data RegressorWinner=none case). Catches the size-feature class
  of regression — Extract column count must match FeatureNames length.

### Behavioral notes

- For the maintainer specifically, RegressorWinner = "none" — ladder
  serves all predictions, no observable change.
- For users whose data shape doesn't have a kNN cliff, a regressor
  variant may activate automatically once `hindcast tune` runs once.
- Anti-anchoring invariant unchanged: predictions still go only to the
  human via status line, never to Claude's context.

### Migration

- Stale `regressor.gob` (v0.4 path) is no longer used; new paths are
  `regressor.gbdt.gob` and `regressor.linear.gob`. Run `hindcast train`
  or wait for the next Stop-hook refresh to populate them.
- Old gob files have a 21-feature schema and would fail length checks
  if loaded by v0.5 code paths — the new code only loads files written
  by v0.5+ via the new paths, so v0.4 files are dormant, not dangerous.

---

## [0.4.0] — unreleased — universal-features regressor (offline tooling)

### The investigation

After v0.3.1 landed per-user tuning, the open question was: can a learned
regressor over universal features (prompt length, task type, recent
project velocity, BM25 signals as features rather than the primary tier)
beat the kNN tier where the kNN cliff doesn't exist?

Built `internal/regressor` with two model classes (GBDT and ridge linear)
trained on the same 21-feature vector. Extended `hindcast bench-cross` to
evaluate both against group_median and kNN on chronological 50/50 splits
of METR HCAST and OpenHands SWE-bench Lite.

**What we found:**

- **Linear beats GBDT cross-corpus.** At our sample sizes (a few thousand
  rows), depth-3 × 100-round GBDT overfits. Ridge linear with λ=1 is more
  robust:
  - METR held-out: linear 1.17× vs GBDT 1.31×
  - OpenHands held-out: linear 1.83× vs GBDT 2.03×
- **Linear is the best predictor on OpenHands** (unique-instance regime),
  beating group_median by 17% and kNN by 23%. This is the regime where
  lexical overlap fails (each SWE-bench instance is one-shot) and
  universal features add the marginal signal.
- **Neither regressor beats kNN on the maintainer's local data.** kNN at
  tuned-sim hits 1.47× (per `hindcast tune`); held-out GBDT 1.62×, linear
  1.83×. The maintainer's data has enough same-project repetition that
  kNN is hard to beat.
- **Neither regressor beats group_median on METR** (1.07× group_median
  vs 1.17× linear). When same-task repetition dominates, "what does this
  task family typically take" is near-optimal; universal features can't
  improve.

### Conclusion

The regressor adds value in the unique-instance regime. It does NOT
beat kNN on data with within-project repetition. The empirical case for
displacing kNN as the primary predictor on user data is not there.

### Added

- `internal/regressor` — pure-Go implementation:
  - `gbdt.go` — gradient-boosted regression trees, depth-3, 100 rounds,
    lr=0.1, squared-error loss on log(wall_seconds).
  - `linear.go` — ridge regression with feature standardization, solved
    via Gauss-Jordan with partial pivoting.
  - `features.go` — 21-feature vector covering prompt-time signals
    (no leakage from post-turn observations like ToolCalls/FilesTouched
    of the current turn — those would only be available after prediction).
  - `model.go` — train/predict/save/load via gob to
    `~/.claude/hindcast/regressor.gob`.
  - `Ensemble(regressorWall, knnWall, maxSim)` — log-space blend by sim.
- `hindcast train [--warmup N] [-v]` — trains both GBDT and linear on
  backfilled records, persists the GBDT, reports in-sample MALR for
  both and a held-out 50/50 chronological MALR for honest comparison.
- `hindcast bench-cross` — extended with regressor cross-eval. Reports
  GBDT, linear, group_median, kNN, ensemble, and a "gap-fill" diagnostic
  (regressor MALR on rows where kNN didn't fire).

### Not yet wired (intentional)

- `predict.Predict` is unchanged. The kNN tier remains primary.
- The Stop hook does not auto-train. Auto-training would only be
  worthwhile if the model fed `predict.Predict`.
- Future direction: per-user adaptive tier selection. Extend
  `health.Compute` to also benchmark the regressor on user data and let
  `predict.Predict` consult health.json to pick the locally-winning tier.
  Makes the regressor opt-in by data, not by config.

## [0.3.1] — unreleased — auto-tuned per-user sim threshold

Cross-corpus benchmark in v0.3.0 showed the global `sim ≥ 0.5` cliff
doesn't generalize. v0.3.1 ships per-user tuning: each install computes
its own empirical sim threshold from local prefix-LOO, refreshed
automatically from the Stop hook when stale.

### Added

- `internal/health` — package that computes per-user predictor health
  via prefix-LOO. Finds the cliff threshold where kNN starts beating
  bucket fallback by ≥30% (per-window) AND continues to beat by ≥15%
  (aggregate above threshold). Persisted to `~/.claude/hindcast/health.json`.
- `hindcast tune [--warmup N] [-v]` — runs the cliff-finding algorithm
  manually. Reports tuned threshold, kNN MALR at threshold, and the
  per-sim-bucket breakdown. Writes health.json.
- `hindcast show --health` — displays the persisted tuned state.
- Auto-tune from Stop hook: if `health.json` is older than 1 hour, the
  Stop hook spawns a goroutine that recomputes health and saves. No
  user intervention required. Cheap (~500ms over thousands of records),
  runs at most once per hour, doesn't block the hook.

### Not yet wired (deferred, intentional)

- The tuned threshold is computed and persisted but NOT currently used
  to gate text injection. v0.3.0 shipped status-line-only by default;
  re-enabling injection based on tuned threshold is a separate explicit
  decision (would re-introduce the anchoring channel under controlled
  conditions). Future iteration may add an opt-in toggle once the tuned
  values prove stable across users in the wild.

## [0.3.0] — unreleased — cross-corpus benchmark + honest reframe

### The investigation

After v0.2 shipped status-line-only, we asked: does this generalize off the
maintainer's machine? Built `hindcast bench-cross`, ran it against two public
corpora — METR HCAST (~9k Claude agent runs across 79 task families) and
OpenHands SWE-bench Lite (~1.2k Claude trajectories with real prompts).

**What we found:**

- **Architecture generalizes.** Per-project chronological history is predictive
  cross-corpus. Group-median MALR is 1.10× on METR (highly repetitive tasks),
  1.65× on OpenHands (one-shot diverse tasks), 1.50–2.00× on the maintainer's
  own data. The "predict from prior runs in this project" tier earns its keep
  on every corpus tested.
- **BM25 mechanism does not universally generalize.** On the maintainer's data
  (multi-turn evolving project work), BM25 over prompt tokens lifts kNN MALR
  to 1.53× at sim≥0.5 — a clean cliff, 1.6× lift over bucket. On OpenHands
  (independent SWE-bench Lite tasks), the cliff doesn't exist; high-sim matches
  are vocabulary overlap, not task overlap; kNN MALR roughly matches the
  global median.
- **The gate threshold (`sim ≥ 0.5`) is your-data-specific tuning, not a
  universal constant.** The cliff in the maintainer's data reflects how their
  Claude Code use evolves task-by-task. SWE-bench-style use has no such cliff.

### Implications

- Status line stays the headline product. Universal default, no compliance
  loop required.
- Inject-text-into-Claude's-context (the v0.1 mechanism, gated behind
  `HINDCAST_LEGACY_INJECT=1`) was always going to be brittle for users whose
  prompt patterns differ from the maintainer's. Honest framing: the in-context
  injection path is best regarded as your-corpus-specific tuning.
- Future v0.3.x will auto-tune the kNN sim threshold from each user's own
  reconciliation log, instead of using a global constant.
- Future v0.4 will explore universal features (file count, prompt length,
  recent project velocity, plancheck blast-radius) to reduce dependence on
  prompt-token similarity entirely.

### Added

- `hindcast bench-cross [--corpus metr|openhands]` — cross-corpus benchmark
  against public agent trace datasets. Reports MALR per predictor tier
  (`global_median`, `group_median`, `task_id_median`, `knn_*`). On the
  OpenHands corpus, also reports MALR by max-sim bucket so users can see
  whether their cliff matches the maintainer's.
- Cross-corpus data caching at `~/.cache/hindcast-bench/` (METR runs.jsonl,
  OpenHands extraction).

### Changed

- README reframed with honest cross-corpus findings. Adds "Does this work
  for me?" section pointing users at `verify` and `bench-cross`. Documents
  status-line-CLI-only limitation, BM25 synonym-signal loss, and the
  use-case-dependent nature of the kNN tier.

### Removed

- `cmd/diag/` — was a one-off diagnostic for tuning the local sim threshold.
  Superseded by `bench-cross`.

## [0.2.0] — unreleased — pivot from context-injection to human-facing predictor

### The pivot
An adversarial review + literature survey during development flagged two
disqualifying problems with v0.1's design:

1. **The mechanism was anchoring, not calibration.** v0.1 injected bucket
   p75 into Claude's context and told it to use the number. LLMs comply
   with injected numbers regardless of whether retrieval was well-matched
   ([Lou et al. 2024](https://arxiv.org/abs/2412.06593); [Synthetic-data
   anchoring 2025](https://arxiv.org/html/2505.15392v2)). When the regex
   classifier misfired, we confidently misled Claude.

2. **Classification was running on the wrong string.** `firstLine(effective)`
   returned the oldest line of the rolling-window context, not the user's
   actual prompt. Every post-rolling-window turn (incl. the v6 eval) was
   classifying on arbitrary prior-assistant text. The headline "7.99× lift"
   was mostly Claude obeying an anchor on a degenerate task bucket.

v0.2 pivots: a local BM25-weighted kNN regressor predicts the turn's
wall/active duration, written to a per-session last-prediction file, and
rendered to **the human** via a Claude Code `statusLine` entry. Claude
never sees the prediction, so anchoring can't fire.

### Added
- `internal/predict` — BM25-weighted kNN regressor with weighted-median +
  p25/p75 band. Falls back through task-bucket → project → global sketch
  → none. Source is reported in the output so provenance is visible.
- `hindcast predict [PROMPT]` — one-shot CLI predictor (args or stdin).
- `hindcast statusline` — renders the active session's prediction for the
  status line. Wired automatically by `hindcast install` (non-destructive:
  preserves any user-installed statusLine).
- `hindcast show --accuracy` — reads the per-project `accuracy.jsonl`
  reconciliation log and reports predictor MALR by source.
- `hindcast verify` — prefix-based leave-one-out self-eval over all
  backfilled records. For each historical turn, predicts using only
  earlier turns, compares to actual, aggregates MALR by tier, emits a
  pass/fail verdict with exit code. Runs in ~0.5s over 4k predictions
  on the maintainer's machine — offline, reproducible, no API cost.
  Replaces "dogfood for a week" as the quality gate.
- Per-turn reconciliation: Stop hook appends `{predicted, actual, source}`
  to `~/.claude/hindcast/projects/<hash>/accuracy.jsonl`.

### Changed
- **UserPromptSubmit** default: emits nothing to Claude's context. Still
  logs the pending record and computes + writes a prediction for the
  status line. Paces now asserts no-stdout as an invariant.
- **SessionStart** default: silent. Legacy priors block available behind
  `HINDCAST_LEGACY_INJECT=1`.
- Classification fixes: `cmd_pending.go`, `cmd_mcp.go`, `cmd_eval_api.go`
  all now classify on the user's actual prompt, not `firstLine(effective)`.
- `PromptChars` records `len(in.Prompt)` again, not `len(effective)`;
  historical records remain comparable.

### Deprecated
- `hindcast calibrate` and `hindcast eval-api` still exist and still work,
  but now evaluate the legacy anchoring path (`HINDCAST_LEGACY_INJECT=1`).
  Kept for A/B research value; not the product surface going forward.

### Research basis
- [Lou 2024 — Anchoring Bias in LLMs](https://arxiv.org/abs/2412.06593)
- [Vacareanu 2024 — Learning vs Retrieval for LLM Regression](https://arxiv.org/html/2409.04318v1)
- [Flyvbjerg — Reference Class Forecasting](https://en.wikipedia.org/wiki/Reference_class_forecasting)
- [Shepperd & Schofield — Analogy-Based Estimation (IEEE TSE)](https://dl.acm.org/doi/10.1109/32.637387)

## [0.1.0] — initial release

### Added
- Three Claude Code hooks registered via `hindcast install`:
  - `UserPromptSubmit` → `hindcast pending` (turn start, BM25 close-match delivery)
  - `Stop` → `hindcast record` (detached worker, transcript parsing, 30s timeout)
  - `SessionStart` → `hindcast inject` (priors injected as `additionalContext`)
- MCP stdio server exposing `hindcast_prior(prompt, project?)` for plan-mode per-step lookups. Behavioral instructions delivered via the server's `initialize` response.
- Backfill subcommand that parses `~/.claude/projects/*/*.jsonl` into records + BM25 indexes + global sketch, with done-file tracking for safe restart.
- Offline benchmark (`hindcast bench`): MALR across global-median, tags-only, project+tags, project+tags+size, and BM25-blend prediction methods.
- Online A/B calibration (`hindcast calibrate`): 90/10 treatment/control split per session, MALR per arm, lift computation, tag-rate divergence warning.
- Admin commands: `show`, `status`, `forget`, `export-seed`, `uninstall`.
- Privacy invariant: no prompt text on disk. Salt-hashed BM25 tokens (per-install 32-byte random salt). Zero network calls. `HINDCAST_SKIP=1` opt-out.
- Record schema v1 with `schema_version` field for future migrations.

### Known limitations
- Windows is build-only (hooks require unix syscalls).
- NFS home directories break O_APPEND atomicity assumption.
- BM25 stopwords English-only.
- MCP instruction compliance for tag emission is unvalidated in the wild — early `hindcast calibrate` output will reveal tag rate per arm.

### Calibration and measurement
- SessionStart injection now shows median / p75 / p90 side-by-side so Claude can anchor on the range, not the midpoint.
- `hindcast bench` adds `n_raw` (raw hits without fallback) and `bias` (exp median signed log-ratio) columns.
- `hindcast calibrate` adds bootstrap 95% confidence intervals on MALR and lift.
- Tag-rate divergence warning flags biased A/B comparisons when control and treatment tag rates differ by >10 pp.

### Operational hardening
- BM25 index rebuilds automatically on JSONL log rotation so index and log stay in sync.
- Global sketch writes take a cross-project lock to prevent lost updates from concurrent Stops.
- `hook.log` rotates at 10 MB; single-generation retention.
- Lock files include process start-time alongside PID to defend against PID recycling.
- Salt generation guarded by a lock file — concurrent first-run hooks won't race.
- `hindcast rotate-salt` regenerates the BM25 salt and clears all indexes (records retained).
- Session IDs sanitized before being interpolated into filesystem paths.

### Tooling
- CI matrix: Linux + macOS × Go 1.22/1.23/1.24, plus Windows build-only and a staticcheck job.
- `scripts/smoke.sh` exercises the install/uninstall cycle against a throwaway HOME.
- `scripts/paces.sh` is the pipeline integration test — walks a synthetic
  CC session through every hook (UserPromptSubmit → Stop → SessionStart +
  MCP handshake + `hindcast_estimate` tool call) and verifies state mutations
  land where they should. Asserts the privacy invariant: no plaintext prompt
  anywhere on disk post-hook. Wired into CI as its own job.
- Release build uses `-ldflags="-s -w"` for binary slimming.

### Accuracy
v0.1 ships at **treatment MALR 2.19× with 7.99× lift [95% CI 5.40–11.46]**
against Sonnet 4.6 on n=117 historical turns, measured in the
realistic rolling-window conversational setting (both arms see the
same prior-turn context; only treatment gets hindcast's priors).

Without hindcast (control arm): MALR 17.50×, p90 126×, bias 17.50×.
With hindcast (treatment): MALR 2.19×, p90 16.45×, bias 1.85×.

The earlier "3.27× lift" number (eval-api v4) used isolated-prompt
samples, which were artificially easy on Claude's untethered arm —
context-free prompts don't trigger Claude's "this sounds substantial"
inflation the way real multi-turn conversation does. The v6 numbers
are the honest measurement.

What does the work, roughly in order of contribution:

- **Task-matched priors at UserPromptSubmit** (`hindcastPrior`): emits
  the specific bucket p75 for the prompt's task-type, not a generic
  project average.
- **Imperative MCP instructions + few-shot examples**: tells Claude to
  use the bucket number rather than reason around it, with GOOD/BAD
  anti-pattern examples.
- **Live bias correction** (`store.ComputeBiasFactor` + `internal/seed/bias.go`):
  the priors block leads with `→ Recommended hindcast_estimate: wall_seconds=N`
  where N = bucket_p75 ÷ known_model_bias. Math is shown. Shipped
  default for Sonnet 4.6 is 1.70×; live records override once ≥10
  samples accumulate.
- **`hindcastPrior` output emits p75** (previously only med + p90 — a
  bug, since the instructions reference p75).

### Round-4 polish (pre-v0.1.0)
- `hindcast eval-api`:
  - Automation filter widened — drops `<task-notification>`, shell-paste,
    bracket-prefix, and marker-phrase prompts that don't exercise
    estimation.
  - `loadAPIKey` walks up the directory tree from cwd looking for `.env`
    (no longer strictly cwd-bound) and strips surrounding quotes.
  - Retries transient failures (429 / 529 / 5xx) with exponential backoff
    up to 3 attempts.
  - Reports 95% bootstrap CI on lift alongside the point estimate.
  - Validates tool-input range: negative or absurdly large estimates
    are rejected as out-of-range.
- `scripts/paces.sh`:
  - Removed `python3` dependency; synthetic pending state is written via
    here-doc after verifying the hook's output shape.
  - Portable `md5` shim (GNU `md5sum` or BSD `md5 -q`).

### Estimate capture and offline A/B
- `hindcast_estimate` MCP tool added alongside `hindcast_prior`. Claude records
  its per-turn estimate via a structured tool call; the Stop hook folds it into
  the Record. Replaces the earlier HTML-comment tag emission (which CC's
  renderer didn't actually hide).
- `hindcast eval-api` subcommand added: samples N historical turns with known
  actual durations, calls the Anthropic Messages API once per arm (control vs
  treatment) with a forced tool-use of `hindcast_estimate`, reports MALR per
  arm + lift. Requires `ANTHROPIC_API_KEY` from shell env or `.env`. This is
  the offline counterpart to live `hindcast calibrate` — reproducible before/
  after numbers without waiting on A/B accumulation.
- SessionStart injection now includes `session_id` so Claude has the value to
  pass to `hindcast_estimate`.
- Windows install refuses to run with a clear message (hooks require Unix
  syscalls).
