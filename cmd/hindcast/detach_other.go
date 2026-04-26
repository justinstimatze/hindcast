//go:build !unix

package main

import "syscall"

func detachAttrs() *syscall.SysProcAttr {
	return nil
}
