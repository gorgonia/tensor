package dense

import "gorgonia.org/tensor"

// Value is a Tensor-like type that only supports reading but not writing data.
//
// This allows for scalar values to be used
type Value[DT any] interface {
	Desc
	tensor.RawAccessor[DT]
	Engine() Engine
}