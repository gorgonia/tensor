package dense

import (
	"gorgonia.org/tensor/internal/flatiter"
)

// cons.go contains convenient construction functions for constructing a *Dense[DT]

// Ones creates a *Dense[DT] with the provided shape and data type.
func Ones[DT Num](shape ...int) *Dense[DT] {
	retVal := New[DT](WithShape(shape...))
	for i := range retVal.data {
		retVal.data[i] = DT(1)
	}
	return retVal
}

// I creates the identity matrix (usually a square) matrix with 1s across the diagonals, and zeroes elsewhere, like so:
//
//	Matrix(4,4)
//	⎡1  0  0  0⎤
//	⎢0  1  0  0⎥
//	⎢0  0  1  0⎥
//	⎣0  0  0  1⎦
//
// While technically an identity matrix is a square matrix, in attempt to keep feature parity with Numpy,
// the I() function allows you to create non square matrices, as well as an index to start the diagonals.
//
// For example:
//
//	T = I(Float64, 4, 4, 1)
//
// Yields:
//
//	⎡0  1  0  0⎤
//	⎢0  0  1  0⎥
//	⎢0  0  0  1⎥
//	⎣0  0  0  0⎦
//
// The index k can also be a negative number:
//
//	T = I(Float64, 4, 4, -1)
//
// Yields:
//
//	⎡0  0  0  0⎤
//	⎢1  0  0  0⎥
//	⎢0  1  0  0⎥
//	⎣0  0  1  0⎦
func I[DT Num](r, c, k int) *Dense[DT] {
	retVal := New[DT](WithShape(r, c))
	i := k
	if k < 0 {
		i = (-k) * c
	}
	end := c - k

	var s *Dense[DT]
	var err error
	if end > r {
		s, err = retVal.Slice(nil)
	} else {
		s, err = retVal.Slice(SR(0, end))
	}
	if err != nil {
		panic(err)
	}
	var nexts []int
	it := flatiter.New(&s.AP)
	nexts, err = it.Slice(SR(i, s.Size(), c+1))
	// It may err, but it's safe to ignore the error. If the error is not ignored then you get panics for no good reason
	// if err != nil {
	// 	panic(err)
	// }
	for _, v := range nexts {
		s.data[v] = 1
	}
	return retVal
}
