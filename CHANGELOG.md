# Changelog

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
