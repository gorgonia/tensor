package tensor

import "github.com/pkg/errors"

// Sum sums a Tensor along the given axes.
func Sum(t Tensor, along ...int) (retVal Tensor, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if sumer, ok := e.(Sumer); ok {
		return sumer.Sum(ctx, t, along...)
	}
	return nil, errors.New("Engine does not support Sum()")
}

// Prod sums a Tensor along the given axes.
func Prod(t Tensor, along ...int) (retVal Tensor, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if sumer, ok := e.(Proder); ok {
		return sumer.Prod(ctx, t, along...)
	}
	return nil, errors.New("Engine does not support Prod()")
}

// Max finds the maximum value along the given axes.
func Max(t Tensor, along ...int) (retVal Tensor, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if maxer, ok := e.(Maxer); ok {
		return maxer.Max(ctx, t, along...)
	}
	return nil, errors.New("Engine does not support Max()")
}

// Argmax finds the index of the max value along the axis provided
func Argmax(t Tensor, axis int) (retVal Tensor, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if argmaxer, ok := e.(Argmaxer); ok {
		return argmaxer.Argmax(ctx, t, axis)
	}
	return nil, errors.New("Engine does not support Argmax()")
}

// Argmin finds the index of the min value along the axis provided
func Argmin(t Tensor, axis int) (retVal Tensor, err error) {
	e := t.Engine()
	ctx := ctxFromEngine(e)
	if argminer, ok := e.(Argminer); ok {
		return argminer.Argmin(ctx, t, axis)
	}
	return nil, errors.New("Engine does not support Argmax()")
}
