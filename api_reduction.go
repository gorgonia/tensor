package tensor

import "github.com/pkg/errors"

// Sum sums a Tensor along the given axes.
func Sum(t Tensor, along ...int) (retVal Tensor, err error) {
	if sumer, ok := t.Engine().(Sumer); ok {
		return sumer.Sum(t, along...)
	}
	return nil, errors.New("Engine does not support Sum()")
}

// Max finds the maximum value along the given axes.
func Max(t Tensor, along ...int) (retVal Tensor, err error) {
	if maxer, ok := t.Engine().(Maxer); ok {
		return maxer.Max(t, along...)
	}
	return nil, errors.New("Engine does not support Max()")
}

// Argmax finds the index of the max value along the axis provided
func Argmax(t Tensor, axis int) (retVal Tensor, err error) {
	if argmaxer, ok := t.Engine().(Argmaxer); ok {
		return argmaxer.Argmax(t, axis)
	}
	return nil, errors.New("Engine does not support Argmax()")
}

// Argmin finds the index of the min value along the axis provided
func Argmin(t Tensor, axis int) (retVal Tensor, err error) {
	if argminer, ok := t.Engine().(Argminer); ok {
		return argminer.Argmin(t, axis)
	}
	return nil, errors.New("Engine does not support Argmax()")
}
