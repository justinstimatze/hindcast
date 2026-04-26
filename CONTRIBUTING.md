# Contributing

hindcast is a small personal project maintained by [@justinstimatze](https://github.com/justinstimatze). PRs and issues welcome.

## Ground rules

- **Go stdlib only.** No third-party dependencies. If you think we need one, open an issue first.
- **No network calls, ever.** `net/http` must never appear in `go.mod`'s transitive deps. Privacy by construction is load-bearing.
- **No LLM calls in any hook path.** Hooks are fast and deterministic.
- **`defer recover()` on every hook entry.** A panic in hindcast must never break a Claude Code turn.
- **Privacy invariant: no prompt text on disk.** All on-disk fields are numeric, deterministic labels, or salt-hashed. Adding plaintext prompt storage needs a separate discussion.

## Dev setup

```sh
git clone https://github.com/justinstimatze/hindcast
cd hindcast
go test ./...
go build ./cmd/hindcast
```

Runs on Go 1.22+. Pure-stdlib; no external tools needed.

## Tests

Each pure package in `internal/` has a `*_test.go` covering the primary paths. Command glue (`cmd/hindcast/*`) is exercised via smoke tests — `go install ./cmd/hindcast && hindcast install --no-backfill && hindcast uninstall`.

Before submitting a PR:

```sh
go test ./...
go vet ./...
go build ./cmd/hindcast
```

## Commit style

Short imperative subject lines. Body explains why, not what. Reference issues as `#123`.

Don't attribute Claude / copilot authorship in commit messages unless the PR description spells out what the AI did.

## Scope discipline

If the change adds a feature, there must be a one-paragraph argument for why the feature is on-by-default or doesn't exist at all. hindcast's product promise is "install and forget"; opt-in flags on the value path are a regression.

## A/B calibration invariant

hindcast runs a live 90/10 control/treatment A/B to measure whether injected priors actually improve Claude's estimates. **Never inject priors, BM25 lines, or any other treatment content in the control arm** — that's the measurement floor. Specifically:

- `cmd_inject.go`: check `store.SessionArm(sessionID, controlPct) == store.ArmControl` and return early.
- `cmd_pending.go`: same check; skip `deliverBM25(...)` on control.
- MCP server `instructions`: delivered to both arms (tags are the measurement unit, not the treatment).

Breaking this invariant means we can't measure lift, and the bias detection in `hindcast calibrate` will silently yield garbage. Reviewers: reject PRs that cross arms.

## Release process (maintainer)

1. Update `CHANGELOG.md`.
2. Tag `vX.Y.Z` on `main`.
3. GitHub release with binaries for linux/darwin × amd64/arm64.
4. Announce.
