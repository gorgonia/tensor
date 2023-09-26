//go:build !nounsafe
// +build !nounsafe

package stdeng

import "unsafe"

func isSameSlice[DT any](a, b []DT) bool {
	hdra := uintptr(unsafe.Pointer(&a[0]))
	hdrb := uintptr(unsafe.Pointer(&b[0]))
	return hdra == hdrb
}
