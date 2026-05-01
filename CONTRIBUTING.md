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

Runs on the Go version pinned in `go.mod` (currently 1.25). CI uses
`go-version-file: go.mod`; local dev should track the same. Pure-stdlib;
no external tools needed beyond `go` and (for full lint parity with CI)
`golangci-lint`.

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

## A/B calibration invariant (legacy-inject mode)

The default v0.6 inject path in `cmd_pending.go formatClaudeInjection`
is **not** wired to the session A/B arm — every Claude Code session
gets the gated band injection (subject to tier + variance gates).
The v0.1 ungated full-bucket-table inject is gated by both
`HINDCAST_LEGACY_INJECT=1` AND `arm == ArmTreatment`; the control
arm is the measurement floor *for that legacy A/B only*.

If you're touching the legacy-inject paths (`cmd_inject.go`'s
HINDCAST_LEGACY_INJECT branch, or the legacy block at the end of
`cmd_pending.cmdPending`):

- Never inject in the control arm. Check `store.SessionArm(...) ==
  store.ArmTreatment` before emitting.
- The MCP server's `instructions` are delivered regardless of arm —
  tags + tool definitions are the measurement unit, not the treatment.

For the v0.6 inject path: A/B testing is handled offline by
`hindcast eval-api` rather than via session arms. Reviewers can
ignore arm-crossing in default-mode hooks; only flag it on the legacy
path.

## Release process (maintainer)

1. Update `CHANGELOG.md`.
2. Tag `vX.Y.Z` on `main`.
3. GitHub release with binaries for linux/darwin × amd64/arm64.
4. Announce.
