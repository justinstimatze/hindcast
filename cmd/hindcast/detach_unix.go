//go:build unix

package main

import "syscall"

// detachAttrs returns SysProcAttr that puts the child in its own process
// group so CC's Stop hook can exit immediately without waiting on the
// worker. `Setsid: true` handles both process-group detachment and
// controlling-terminal detachment in one flag.
func detachAttrs() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setsid: true}
}
