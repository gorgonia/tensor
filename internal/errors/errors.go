// package errors provides errors for the tensor package. It also vendors some existing packages
package errors

import (
	"runtime"

	"github.com/pkg/errors"
)

type NoOpError interface {
	NoOp()
}

type NoOp struct{}

func (err NoOp) Error() string { return "No Op" }
func (err NoOp) NoOp()         {}

var (
	Errorf = errors.Errorf
	Wrap   = errors.Wrap
	Wrapf  = errors.Wrapf
	New    = errors.New
)

const (
	// there are only so many NaNs in the numbers. Let's use one of them for Gorgonia specific errors.

	ErrorF64 float64 = 0
	ErrorF32 float32 = 0
)

// ThisFn returns the name of the function
func ThisFn(skips ...uint) string {
	c := 1
	if len(skips) > 0 {
		c += int(skips[0])
	}

	pc, _, _, ok := runtime.Caller(c)
	if !ok {
		return "UNKNOWNFUNC"
	}
	return runtime.FuncForPC(pc).Name()
}
