//go:build !amd64 || noasm
// +build !amd64 noasm

package gutils

func Divmod(a, b int) (q, r int) {
	q = a / b
	r = a % b
	return
}
