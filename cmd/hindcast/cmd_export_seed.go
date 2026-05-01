package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/justinstimatze/hindcast/internal/store"
)

// cmdExportSeed dumps the current global sketch as JSON. Numeric-only:
// sample vectors of wall/active seconds. Maintainer runs this and
// commits the output to internal/seed/priors.json to refresh the
// bootstrap seed shipped with the next release.
func cmdExportSeed(args []string) {
	fl := flag.NewFlagSet("export-seed", flag.ExitOnError)
	fl.Usage = func() {
		fmt.Fprintln(os.Stderr, "export-seed — emit current global sketch as JSON (maintainers only).")
		fmt.Fprintln(os.Stderr, "\nRegenerates the bootstrap seed embedded in the next release at")
		fmt.Fprintln(os.Stderr, "internal/seed/priors.json. End users do not need to run this.")
		fmt.Fprintln(os.Stderr, "\nUsage: hindcast export-seed > internal/seed/priors.json")
	}
	_ = fl.Parse(args)
	s, err := store.LoadSketch()
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast export-seed: %s\n", err)
		os.Exit(1)
	}
	if len(s.Wall) == 0 && len(s.Active) == 0 {
		fmt.Fprintln(os.Stderr, "hindcast export-seed: no sketch data yet.")
		fmt.Fprintln(os.Stderr, "  Run `hindcast backfill` first if you have historical CC transcripts,")
		fmt.Fprintln(os.Stderr, "  or wait until you've accumulated some real-use records.")
		fmt.Fprintln(os.Stderr, "  This command is for MAINTAINERS regenerating the bootstrap seed")
		fmt.Fprintln(os.Stderr, "  embedded in the next release (internal/seed/priors.json).")
		os.Exit(1)
	}
	out := map[string]any{
		"wall_seconds":   s.Wall,
		"active_seconds": s.Active,
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "hindcast export-seed: %s\n", err)
		os.Exit(1)
	}
	_, _ = os.Stdout.Write(data)
	fmt.Println()
}
