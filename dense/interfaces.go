package dense

import (
	"gorgonia.org/tensor"
)

type DenseTensor[DT any, T tensor.Tensor[DT, T]] interface {
	tensor.Tensor[DT, T]

	IsMatrix() bool

	/*
	   Instead of providing the traditional `bind` function of a monad,
	   we provide a type deconstructor, which allows one to deconstruct any DenseTensor,
	   perform the action, then reconstruct it with `FromDense`. See wrapped_test.go for examples.
	*/

	// FromDense is a constructor function/method to create a T from a *Dense[DT]
	FromDenser[DT, T]

	// Deconstructor
	Densor[DT]

	// private methods from *Dense

	//copyMetadata(srcAP AP, srcEng Engine, srcFlag MemoryFlag, srcDT dtype.Dtype)
	//	len() int
	//restore()

	// from AP

	SetDataOrder(o DataOrder)
}

type FromDenser[DT any, T tensor.Basic[DT]] interface { //
	FromDense(d *Dense[DT]) T
}

// Densor is any tensor type that can return a *Dense[DT] equivalent of itself.
//
// While that's true in principle, in practice, the primary function of `Densor[DT]` is to provide a
// deconstruction method for any type that contains a *Dense[DT]. See wrapped_test.go for examples.
type Densor[DT any] interface {
	// GetDense is a deconstructor method - it deconstruct any type can be turned into *Dense[DT]
	GetDense() *Dense[DT]
}

type sliceEqer[DT any] interface {
	SliceEq(a, b []DT) bool
}
